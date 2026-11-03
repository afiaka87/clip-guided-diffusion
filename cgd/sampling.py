"""Fast CLIP-guided sampling.

The original pipeline takes 1000 ancestral steps because its guidance needs them:
CLIP has to build the whole image while the U-Net follows a random class label
that changes every step. This sampler reaches the same prompt fidelity in about
30 model calls by changing who does the work:

- The U-Net is conditioned on a single CLIP-selected ImageNet class, so it brings
  the subject and CLIP only steers. Never average class embeddings: the mean of
  dissimilar labels is off the manifold the network trained on and it outputs
  noise. One real label is always valid.
- A second-order stochastic solver (DPM-Solver++(2M) SDE) covers the schedule in
  uniform logSNR steps. The second-order correction uses the unguided x0 only.
  Guidance jumps between steps, and extrapolating it destabilises the solver.
- Guidance is capped relative to the noise rather than scaled in absolute terms,
  so it does not depend on the number of steps. A 30 step sampler takes 33x
  larger steps than the 1000 step one, and uncapped gradients overshoot x0 by an
  order of magnitude early on, which the trajectory never recovers from.
- Guidance only runs while it is useful: x0 is meaningless at very high noise,
  and at low noise the gradient is mostly adversarial high frequency detail that
  the U-Net cleans up better on its own. Those steps also skip the U-Net
  backward pass, the single most expensive part of guidance.
- Without a class label nothing brings the subject, so CLIP also has to set the
  composition. That happens at high noise, so unconditional checkpoints are guided
  from the very start, and need twice the steps for that not to overshoot. Neither
  change helps alone, and this is still a little behind the original sampler.

The defaults were tuned on 128px and validated on prompts held out from tuning, and
the unconditional ones likewise on 256px.
"""
import math
from contextlib import contextmanager

import numpy as np
import torch as th
from guided_diffusion.unet import AttentionBlock
from tqdm.auto import tqdm

DEFAULT_STEPS = 30
UNCOND_STEPS = 60
DEFAULT_ETA = 0.5  # Fresh noise per step: 0 is an ODE, 1 is fully ancestral.
DEFAULT_GUIDANCE_RATE = 0.08  # Max RMS of the guidance relative to the noise.
GUIDANCE_WINDOW = (0.1, 0.8)  # Guide between these fractions of the schedule.
UNCOND_GUIDANCE_WINDOW = (0.1, 1.0)
GUIDANCE_SCHEDULES = ("flat", "sine", "rising")
SOLVERS = ("dpm++2m-sde", "er-sde")
ER_SDE_INTEGRATION_POINTS = 200
GUIDANCE_LIMIT_SEARCH_STEPS = 8
X0_CLAMP = 1.2  # Bound on x0. Slightly loose so guidance can still move extremes.


class NoiseLevels:
    """Continuous-time view of a discrete diffusion schedule."""

    def __init__(self, alphas_cumprod):
        self.log_alpha_bar = np.log(np.asarray(alphas_cumprod, dtype=np.float64))
        self.num_timesteps = len(self.log_alpha_bar)
        self._table_t = np.arange(self.num_timesteps, dtype=np.float64)
        self._table_log_snr = self._log_snr(self.log_alpha_bar)

    @staticmethod
    def _log_snr(log_alpha_bar):
        return 0.5 * (log_alpha_bar - np.log1p(-np.exp(log_alpha_bar)))

    def alpha_sigma(self, t: float):
        """Signal and noise scale at a possibly fractional timestep."""
        alpha_bar = math.exp(np.interp(t, self._table_t, self.log_alpha_bar))
        return math.sqrt(alpha_bar), math.sqrt(1 - alpha_bar)

    def timesteps(self, steps: int, start: float = None):
        """`steps` descending timesteps uniform in logSNR, followed by a final 0.

        With `start` the sequence joins the full schedule at that timestep, so an
        init image costs proportionally fewer model calls.
        """
        last = self.num_timesteps - 1
        log_snrs = np.linspace(self._table_log_snr[last], self._table_log_snr[0], steps + 1)
        # Invert through log(alpha_bar), the quantity alpha_sigma() interpolates, so the
        # spacing is exact. It falls as t rises and np.interp needs ascending coordinates.
        log_alpha_bars = -np.log1p(np.exp(-2 * log_snrs))
        ts = np.interp(log_alpha_bars, self.log_alpha_bar[::-1], self._table_t[::-1])
        ts[0], ts[-1] = last, 0.0
        if start is not None:
            ts = np.concatenate(([float(start)], ts[(ts < start) | (ts == 0.0)]))
        return ts.tolist()


def loss_gradient(loss: th.Tensor, x: th.Tensor) -> th.Tensor:
    """The gradient of a guidance loss with respect to x, through the U-Net.

    The U-Net backpropagates at fp16, and a spiky CLIP gradient can overflow that. One
    inf would make the sample NaN for good, which saves as a black image, so whatever
    overflowed gets no guidance instead. Scaling the loss down would prevent it, but
    any scale changes the rounding and so every image a seed makes.
    """
    return th.nan_to_num(th.autograd.grad(loss, x)[0], nan=0.0, posinf=0.0, neginf=0.0)


def _rms(tensor: th.Tensor) -> th.Tensor:
    return tensor.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()


def dynamic_threshold(x0: th.Tensor, quantile: float) -> th.Tensor:
    """Imagen's dynamic thresholding: pull each image's extreme pixels back into range.

    A clamp flattens whatever CLIP pushes past it into saturated plateaus, which the
    U-Net then has to build on. Rescaling keeps the relative contrast, so clipped pixels
    all but disappear.
    """
    threshold = th.quantile(x0.abs().float().flatten(1), quantile, dim=1).clamp(min=1).view(-1, 1, 1, 1)
    return x0.clamp(-threshold, threshold) / threshold


def limit_guidance(x0: th.Tensor, update: th.Tensor, quantile: float) -> th.Tensor:
    """x0 plus as much of `update` as keeps its pixels in range, up to `quantile` of them.

    Dynamic thresholding keeps structure but rescales the model's image along with the
    guidance, and doing that every step fades it to grey. The model's image was never
    the problem. This only holds back what CLIP adds to it.
    """
    low, high = th.zeros_like(update[:, :1, :1, :1]), th.ones_like(update[:, :1, :1, :1])
    for _ in range(GUIDANCE_LIMIT_SEARCH_STEPS):
        share = (low + high) / 2
        extreme = th.quantile((x0 + share * update).abs().float().flatten(1), quantile, dim=1).view(-1, 1, 1, 1)
        low, high = th.where(extreme > 1, low, share), th.where(extreme > 1, share, high)
    extreme = th.quantile((x0 + update).abs().float().flatten(1), quantile, dim=1).view(-1, 1, 1, 1)
    return x0 + th.where(extreme > 1, low, th.ones_like(low)) * update


@contextmanager
def perturbed_attention(model):
    """Replace the U-Net's deepest encoder attention with the identity (PAG, Ahn et al.).

    The difference between the normal and the perturbed prediction points towards
    structure, without needing a null token or a text condition.
    """
    last = None
    for name, module in model.named_modules():
        if name.startswith("input_blocks") and isinstance(module, AttentionBlock):
            last = module

    def attend_to_self(attention, inputs, _):
        qkv = inputs[0]
        batch, width, length = qkv.shape
        channels = width // (3 * attention.n_heads)
        values = qkv.reshape(batch * attention.n_heads, channels * 3, length).split(channels, dim=1)[2]
        return values.reshape(batch, -1, length)

    handle = last.attention.register_forward_hook(attend_to_self)
    try:
        yield
    finally:
        handle.remove()


def schedule_weight(schedule: str, progress: float) -> float:
    """Multiplier on the guidance rate, `progress` of the way through the window, averaging 1."""
    if schedule == "flat":
        return 1.0
    if schedule == "sine":
        return math.pi / 2 * math.sin(math.pi * progress)
    if schedule == "rising":
        return 2.0 * progress
    raise ValueError(f"Unknown guidance schedule {schedule!r}. Choose one of {list(GUIDANCE_SCHEDULES)}.")


def er_sde_noise(noise_ratio: float) -> float:
    """ER-SDE's default noise scale function, of sigma / alpha."""
    return noise_ratio * (math.exp(noise_ratio ** 0.3) + 10)


def er_sde_step(x, x0, guided_x0, slope, alpha, sigma, alpha_next, sigma_next):
    """One step of VP ER-SDE-Solver-2 (Cui et al. 2023), from t to t_next.

    slope is the change in x0 per unit of sigma / alpha, or None on the first step. As in
    the other solver it is unguided, so that the correction never extrapolates guidance.
    """
    ratio, ratio_next = sigma / alpha, sigma_next / alpha_next
    decay = er_sde_noise(ratio_next) / er_sde_noise(ratio)
    x_next = alpha_next / alpha * decay * x + alpha_next * (1 - decay) * guided_x0
    if slope is not None:
        grid = np.linspace(ratio_next, ratio, ER_SDE_INTEGRATION_POINTS)
        integral = np.trapezoid([1 / er_sde_noise(value) for value in grid], grid)
        x_next = x_next + alpha_next * (ratio_next - ratio + integral * er_sde_noise(ratio_next)) * slope
    variance = max(ratio_next ** 2 - (decay * ratio) ** 2, 0.0)
    return x_next + alpha_next * math.sqrt(variance) * th.randn_like(x)


def sample_loop_progressive(
    model,
    diffusion,
    shape,
    loss_fn=None,
    steps: int = DEFAULT_STEPS,
    eta: float = DEFAULT_ETA,
    guidance_rate: float = DEFAULT_GUIDANCE_RATE,
    guidance_window: tuple = GUIDANCE_WINDOW,
    x0_clamp: float = X0_CLAMP,
    x0_threshold: float = None,
    guidance_limit: float = None,
    guidance_schedule: str = "flat",
    eps_adjustment=None,
    guide_model=None,
    solver: str = SOLVERS[0],
    init_image: th.Tensor = None,
    start_timestep: float = None,
    model_kwargs: dict = None,
    device: str = "cpu",
    progress: bool = True,
):
    """Yield {"sample", "pred_xstart", "final"} after each model call.

    loss_fn(x, pred_xstart, fac) returns a scalar guidance loss, where fac is the
    blend between pred_xstart and x that CLIP should see. Its gradient is taken
    with respect to x, through the U-Net.

    x0_threshold is a dynamic_threshold() quantile that replaces the clamp.
    guidance_limit is a limit_guidance() quantile.
    guidance_schedule shapes the rate over the window, with the same total guidance.
    eps_adjustment(x, batch_t, eps) returns a term to add to the noise estimate during
    the guidance window, for guidance that comes from the U-Net itself. Its output is
    treated as a constant.
    guide_model(x, alpha, sigma) is a denoiser to take guidance through instead of model,
    which then runs without a gradient.
    """
    model_kwargs = model_kwargs or {}
    levels = NoiseLevels(diffusion.alphas_cumprod)
    ts = levels.timesteps(steps, start_timestep)
    scale_t = 1000.0 / levels.num_timesteps if diffusion.rescale_timesteps else 1.0
    last = levels.num_timesteps - 1
    guide_from, guide_to = (bound * last for bound in guidance_window)
    guided_run = loss_fn is not None and guidance_rate > 0

    x = th.randn(shape, device=device)
    if init_image is not None and start_timestep is not None:
        alpha, sigma = levels.alpha_sigma(ts[0])
        x = alpha * init_image + sigma * x

    previous_x0, previous_h, previous_ratio = None, None, None
    bar = tqdm(range(len(ts) - 1), disable=not progress)
    for i in bar:
        t, t_next = ts[i], ts[i + 1]
        alpha, sigma = levels.alpha_sigma(t)
        batch_t = th.full((shape[0],), t * scale_t, device=device, dtype=th.float32)

        guide = guided_run and guide_from <= t <= guide_to
        through_model = guide and guide_model is None
        x = x.detach().requires_grad_(through_model)
        with th.set_grad_enabled(through_model):
            eps = model(x, batch_t, **model_kwargs)[:, :shape[1]]
            if eps_adjustment is not None and guide_from <= t <= guide_to:
                with th.no_grad():
                    adjustment = eps_adjustment(x, batch_t, eps.detach())
                eps = eps + adjustment  # Outside no_grad, or guidance loses its path through the U-Net.
            x0 = (x - sigma * eps) / alpha

        guided_x0 = x0.detach()
        if guide:
            guide_x, guide_x0 = x, x0
            if guide_model is not None:
                guide_x = x.detach().requires_grad_()
                guide_x0 = guide_model(guide_x, alpha, sigma)
            loss = loss_fn(guide_x, guide_x0, sigma)
            bar.set_postfix(loss=f"{loss.item():.0f}", refresh=False)
            gradient = -loss_gradient(loss, guide_x)
            ratio = _rms(sigma * gradient) / _rms(eps.detach())
            rate = guidance_rate * schedule_weight(guidance_schedule, (guide_to - t) / (guide_to - guide_from))
            gradient = gradient * (rate / ratio).clamp(max=1)
            update = sigma ** 2 / alpha * gradient
            if guidance_limit is None:
                guided_x0 = guided_x0 + update
            else:
                guided_x0 = limit_guidance(guided_x0, update, guidance_limit)
        if x0_threshold is None:
            guided_x0 = guided_x0.clamp(-x0_clamp, x0_clamp)
        else:
            guided_x0 = dynamic_threshold(guided_x0, x0_threshold)
        x, x0 = x.detach(), x0.detach()

        if i == len(ts) - 2:  # Last model call: the denoised image is the result.
            yield {"sample": guided_x0, "pred_xstart": guided_x0, "final": True}
            return

        alpha_next, sigma_next = levels.alpha_sigma(t_next)
        h = math.log(alpha_next / sigma_next) - math.log(alpha / sigma)
        if solver == "er-sde":
            slope = None if previous_x0 is None else (x0 - previous_x0) / (sigma / alpha - previous_ratio)
            x_next = er_sde_step(x, x0, guided_x0, slope, alpha, sigma, alpha_next, sigma_next)
        else:
            eta_h = eta * h
            x0_weight = alpha_next * -math.expm1(-h - eta_h)
            x_next = sigma_next / sigma * math.exp(-eta_h) * x + x0_weight * guided_x0
            if previous_x0 is not None:
                x_next = x_next + 0.5 * x0_weight * (h / previous_h) * (x0 - previous_x0)
            if eta_h > 0:
                x_next = x_next + sigma_next * math.sqrt(-math.expm1(-2 * eta_h)) * th.randn_like(x)

        previous_x0, previous_h, previous_ratio = x0, h, sigma / alpha
        x = x_next
        yield {"sample": x, "pred_xstart": guided_x0, "final": False}

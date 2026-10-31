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

import numpy as np
import torch as th
from tqdm.auto import tqdm

DEFAULT_STEPS = 30
UNCOND_STEPS = 60
DEFAULT_ETA = 0.5  # Fresh noise per step: 0 is an ODE, 1 is fully ancestral.
DEFAULT_GUIDANCE_RATE = 0.08  # Max RMS of the guidance relative to the noise.
GUIDANCE_WINDOW = (0.1, 0.8)  # Guide between these fractions of the schedule.
UNCOND_GUIDANCE_WINDOW = (0.1, 1.0)
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


def _rms(tensor: th.Tensor) -> th.Tensor:
    return tensor.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()


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

    previous_x0, previous_h = None, None
    bar = tqdm(range(len(ts) - 1), disable=not progress)
    for i in bar:
        t, t_next = ts[i], ts[i + 1]
        alpha, sigma = levels.alpha_sigma(t)
        batch_t = th.full((shape[0],), t * scale_t, device=device, dtype=th.float32)

        guide = guided_run and guide_from <= t <= guide_to
        x = x.detach().requires_grad_(guide)
        with th.set_grad_enabled(guide):
            eps = model(x, batch_t, **model_kwargs)[:, :shape[1]]
            x0 = (x - sigma * eps) / alpha

        guided_x0 = x0.detach()
        if guide:
            loss = loss_fn(x, x0, sigma)
            bar.set_postfix(loss=f"{loss.item():.0f}", refresh=False)
            gradient = -th.autograd.grad(loss, x)[0]
            ratio = _rms(sigma * gradient) / _rms(eps.detach())
            gradient = gradient * (guidance_rate / ratio).clamp(max=1)
            guided_x0 = guided_x0 + sigma ** 2 / alpha * gradient
        guided_x0 = guided_x0.clamp(-x0_clamp, x0_clamp)
        x, x0 = x.detach(), x0.detach()

        if i == len(ts) - 2:  # Last model call: the denoised image is the result.
            yield {"sample": guided_x0, "pred_xstart": guided_x0, "final": True}
            return

        alpha_next, sigma_next = levels.alpha_sigma(t_next)
        h = math.log(alpha_next / sigma_next) - math.log(alpha / sigma)
        eta_h = eta * h
        x0_weight = alpha_next * -math.expm1(-h - eta_h)
        x_next = sigma_next / sigma * math.exp(-eta_h) * x + x0_weight * guided_x0
        if previous_x0 is not None:
            x_next = x_next + 0.5 * x0_weight * (h / previous_h) * (x0 - previous_x0)
        if eta_h > 0:
            x_next = x_next + sigma_next * math.sqrt(-math.expm1(-2 * eta_h)) * th.randn_like(x)

        previous_x0, previous_h = x0, h
        x = x_next
        yield {"sample": x, "pred_xstart": guided_x0, "final": False}

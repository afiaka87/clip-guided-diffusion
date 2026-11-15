"""Fast CLIP-guided sampling: DPM-Solver++(2M) SDE in uniform logSNR steps, with capped guidance.

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
  from the very start, and take twice the steps for that not to overshoot. Neither
  change helps alone.
- They also redo each guided step above 30% of the schedule (self-recurrence, from
  Universal Guidance): x goes back up to the step's own noise level with fresh noise,
  and the step runs again. Guidance gets a second say over the composition, and the
  U-Net pulls each pass back onto what it learned. Ninety plain steps, about the same
  time, follow the prompt as closely but look no better.

The defaults were tuned on 128px and validated on prompts held out from tuning, and
the unconditional ones likewise on 256px.

A schedule here is log(alpha_bar) at each integer timestep, from schedule(), and a
denoiser predicts the noise in its input.
"""

import math
from collections.abc import Callable, Iterator, Sequence
from typing import NamedTuple

import numpy as np
import torch as th
from tqdm.auto import tqdm

from cgd.losses import finite_gradient

DEFAULT_STEPS = 30
UNCONDITIONAL_STEPS = 60
ETA = 0.5  # Fresh noise per step: 0 is an ODE, 1 is fully ancestral.
DEFAULT_GUIDANCE_RATE = 0.08  # Max RMS of the guidance relative to the noise.
GUIDANCE_WINDOW = (0.1, 0.8)  # Guide between these fractions of the schedule.
UNCONDITIONAL_GUIDANCE_WINDOW = (0.1, 1.0)
UNCONDITIONAL_RECURRENCE = 2  # Times each guided step above RECUR_ABOVE runs.
RECUR_ABOVE = 0.3  # Recurrence only redoes steps above this fraction of the schedule, where the composition forms.
X0_CLAMP = 1.2  # Bound on x0. Slightly loose so guidance can still move extremes.
GUIDANCE_LIMIT_SEARCH_STEPS = 8

Denoiser = Callable[[th.Tensor, th.Tensor], th.Tensor]
"""(x, timesteps) -> the noise in x. timesteps holds one per image of the batch."""

GuidanceLoss = Callable[[th.Tensor, th.Tensor, float], th.Tensor]
"""(x, x0, sigma) -> a scalar loss, differentiable with respect to x through x0."""


class Level(NamedTuple):
    """How much signal and noise x holds at one point of a schedule: x = alpha * x0 + sigma * noise."""

    alpha: float
    sigma: float


class Previous(NamedTuple):
    """What the second-order correction needs from the step before."""

    x0: th.Tensor  # Unguided.
    h: float


def schedule(alphas_cumprod: Sequence[float] | np.ndarray) -> np.ndarray:
    """A discrete schedule's log(alpha_bar) at each timestep, in float64."""
    return np.log(np.asarray(alphas_cumprod, dtype=np.float64))


def log_snr(log_alpha_bar: np.ndarray) -> np.ndarray:
    """log(alpha / sigma) of each log(alpha_bar)."""
    return 0.5 * (log_alpha_bar - np.log1p(-np.exp(log_alpha_bar)))


def level(log_alpha_bar: np.ndarray, t: float) -> Level:
    """The noise level at a possibly fractional timestep."""
    alpha_bar = math.exp(np.interp(t, np.arange(len(log_alpha_bar), dtype=np.float64), log_alpha_bar))
    return Level(math.sqrt(alpha_bar), math.sqrt(1 - alpha_bar))


def timesteps(log_alpha_bar: np.ndarray, steps: int, start: float | None = None) -> list[float]:
    """`steps` descending timesteps uniform in logSNR, then a final 0: one per model call, and the end.

    With `start` the sequence joins the full schedule at that timestep, so an init image
    costs proportionally fewer model calls.
    """
    last = len(log_alpha_bar) - 1
    table_t = np.arange(len(log_alpha_bar), dtype=np.float64)
    table_log_snr = log_snr(log_alpha_bar)
    log_snrs = np.linspace(table_log_snr[last], table_log_snr[0], steps + 1)
    # Invert through log(alpha_bar), the quantity level() interpolates, so the spacing
    # is exact. It falls as t rises, and np.interp needs ascending coordinates.
    ts = np.interp(-np.log1p(np.exp(-2 * log_snrs)), log_alpha_bar[::-1], table_t[::-1])
    ts[0], ts[-1] = last, 0.0
    if start is not None:
        ts = np.concatenate(([float(start)], ts[(ts < start) | (ts == 0.0)]))
    return ts.tolist()


def rms(tensor: th.Tensor) -> th.Tensor:
    """The root mean square of each image of a batch, shaped to broadcast against it."""
    return tensor.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()


def cap_guidance(gradient: th.Tensor, eps: th.Tensor, sigma: float, rate: float) -> th.Tensor:
    """The gradient, scaled down for each image where sigma times it exceeds `rate` of the noise."""
    ratio = rms(sigma * gradient) / rms(eps)
    return gradient * (rate / ratio).clamp(max=1)


def limit_guidance(x0: th.Tensor, update: th.Tensor, quantile: float) -> th.Tensor:
    """x0 plus as much of `update` as keeps its pixels in range, up to `quantile` of them.

    Dynamic thresholding keeps structure but rescales the model's image along with the
    guidance, and doing that every step fades it to grey. The model's image was never
    the problem. This only holds back what CLIP adds to it, by bisecting on its share.
    """

    def too_far(share: th.Tensor) -> th.Tensor:
        extreme = th.quantile((x0 + share * update).abs().float().flatten(1), quantile, dim=1)
        return extreme.view(-1, 1, 1, 1) > 1

    low, high = th.zeros_like(update[:, :1, :1, :1]), th.ones_like(update[:, :1, :1, :1])
    for _ in range(GUIDANCE_LIMIT_SEARCH_STEPS):
        share = (low + high) / 2
        overshoots = too_far(share)
        low, high = th.where(overshoots, low, share), th.where(overshoots, share, high)
    return x0 + th.where(too_far(th.ones_like(low)), low, th.ones_like(low)) * update


def guide_x0(x0: th.Tensor, gradient: th.Tensor, now: Level, guidance_limit: float | None = None) -> th.Tensor:
    """x0 moved along the capped guidance gradient, as far as Tweedie's formula says it moves."""
    update = now.sigma**2 / now.alpha * gradient
    return x0 + update if guidance_limit is None else limit_guidance(x0, update, guidance_limit)


def step_size(now: Level, then: Level) -> float:
    """h, how much logSNR rises from one level to the next."""
    return math.log(then.alpha / then.sigma) - math.log(now.alpha / now.sigma)


def dpm_solver_step(
    x: th.Tensor,
    x0: th.Tensor,
    guided_x0: th.Tensor,
    previous: Previous | None,
    levels: tuple[Level, Level],
    *,
    noise: th.Tensor,
    eta: float = ETA,
) -> th.Tensor:
    """One DPM-Solver++(2M) SDE step of x between two levels. `noise` is standard normal.

    The first-order part moves towards the guided x0, and the second-order correction
    extrapolates the unguided x0 from the previous step's, if there was one.
    """
    now, then = levels
    h = step_size(now, then)
    eta_h = eta * h
    x0_weight = then.alpha * -math.expm1(-h - eta_h)
    x_next = then.sigma / now.sigma * math.exp(-eta_h) * x + x0_weight * guided_x0
    if previous is not None:
        x_next = x_next + 0.5 * x0_weight * (h / previous.h) * (x0 - previous.x0)
    if eta_h > 0:
        x_next = x_next + then.sigma * math.sqrt(-math.expm1(-2 * eta_h)) * noise
    return x_next


def renoised(x: th.Tensor, levels: tuple[Level, Level], noise: th.Tensor) -> th.Tensor:
    """x at the second, lower level of a step, taken back up to the first as the forward process would.

    The signal shrinks by the ratio of the alphas, and fresh standard normal `noise` makes
    up the rest of the first level's noise.
    """
    now, then = levels
    ratio = now.alpha / then.alpha
    return ratio * x + math.sqrt(max(now.sigma**2 - (ratio * then.sigma) ** 2, 0.0)) * noise


class Guidance(NamedTuple):
    """What steers sampling, and how hard."""

    loss_fn: GuidanceLoss
    rate: float = DEFAULT_GUIDANCE_RATE  # The most the gradient may move x0, relative to the noise.
    window: tuple[float, float] = GUIDANCE_WINDOW  # Where in the schedule to guide, as fractions of it.
    limit: float | None = None  # A limit_guidance() quantile.
    recurrence: int = 1  # Times each guided step above recur_above runs, re-noised back between runs.
    recur_above: float = RECUR_ABOVE  # As a fraction of the schedule.


def guides_at(guidance: Guidance | None, t: float, last: int) -> bool:
    """Whether guidance runs at timestep t of a schedule ending at `last`."""
    if guidance is None or guidance.rate <= 0:
        return False
    guide_from, guide_to = (bound * last for bound in guidance.window)
    return guide_from <= t <= guide_to


def recurrences(guidance: Guidance | None, t: float, last: int) -> int:
    """How many times the step from timestep t runs: more than once only where it is guided, above recur_above."""
    if guidance is None or not guides_at(guidance, t, last) or t < guidance.recur_above * last:
        return 1
    return guidance.recurrence


class Prediction(NamedTuple):
    """What one model call predicts the clean image is."""

    x0: th.Tensor  # The model's own prediction.
    guided_x0: th.Tensor  # That prediction moved by guidance, then clamped.
    loss: th.Tensor | None  # The guidance loss, or None without guidance.


def predict(denoise: Denoiser, x: th.Tensor, t: float, now: Level, guidance: Guidance | None) -> Prediction:
    """The denoised image of x at timestep t, before and after guidance.

    Without guidance the guided image is only clamped, the loss is None, and the denoiser
    runs without a gradient, which skips the most expensive part of a step.
    """
    x = x.detach().requires_grad_(guidance is not None)
    with th.set_grad_enabled(guidance is not None):
        eps = denoise(x, th.full((x.shape[0],), t, device=x.device, dtype=th.float32))
        x0 = (x - now.sigma * eps) / now.alpha
    guided_x0, loss = x0.detach(), None
    if guidance is not None:
        loss = guidance.loss_fn(x, x0, now.sigma)
        gradient = cap_guidance(-finite_gradient(loss, x), eps.detach(), now.sigma, guidance.rate)
        guided_x0 = guide_x0(guided_x0, gradient, now, guidance.limit)
    return Prediction(x0.detach(), guided_x0.clamp(-X0_CLAMP, X0_CLAMP), loss)


def noised(log_alpha_bar: np.ndarray, t: float, image: th.Tensor, noise: th.Tensor) -> th.Tensor:
    """`image` with standard normal `noise` added, as the forward process would have it at timestep t."""
    now = level(log_alpha_bar, t)
    return now.alpha * image + now.sigma * noise


def sample(
    denoise: Denoiser,
    log_alpha_bar: np.ndarray,
    ts: Sequence[float],
    x: th.Tensor,
    guidance: Guidance | None = None,
    *,
    eta: float = ETA,
    progress: bool = True,
) -> Iterator[th.Tensor]:
    """Yield the denoised image at each timestep, from x at ts[0] down the timesteps `ts`. The last is the result.

    x is standard normal noise at the start of the schedule, or an image noised() to a
    timestep partway down it. A step that recurs yields the image of its last run.
    """
    levels = [level(log_alpha_bar, t) for t in ts]
    last = len(log_alpha_bar) - 1
    previous = None
    progress_bar = tqdm(range(len(ts) - 1), disable=not progress)
    for i in progress_bar:
        active = guidance if guides_at(guidance, ts[i], last) else None
        if i < len(ts) - 2:
            step = levels[i], levels[i + 1]
            prediction, x = recurring_step(
                denoise, x, ts[i], step, active, previous=previous, runs=recurrences(active, ts[i], last), eta=eta
            )
            previous = Previous(prediction.x0, step_size(*step))
        else:  # The last model call, whose denoised image is the result.
            prediction = predict(denoise, x, ts[i], levels[i], active)
        if prediction.loss is not None:
            progress_bar.set_postfix(loss=f"{prediction.loss.item():.0f}", refresh=False)
        yield prediction.guided_x0


def recurring_step(
    denoise: Denoiser,
    x: th.Tensor,
    t: float,
    step: tuple[Level, Level],
    guidance: Guidance | None,
    *,
    previous: Previous | None,
    runs: int = 1,
    eta: float = ETA,
) -> tuple[Prediction, th.Tensor]:
    """The prediction at timestep t, the first level of `step`, and x after the step, taken `runs` times.

    Between runs x goes back up to the first level with fresh noise, so guidance gets
    another say at the same noise level. A rerun is first order: the x0 of the step before
    does not continue from its fresh noise. Returns the last run's prediction and x.
    """
    for run in range(runs):
        prediction = predict(denoise, x, t, step[0], guidance)
        x_next = dpm_solver_step(
            x, prediction.x0, prediction.guided_x0, None if run else previous, step, noise=th.randn_like(x), eta=eta
        )
        if run < runs - 1:
            x = renoised(x_next, step, th.randn_like(x))
    return prediction, x_next

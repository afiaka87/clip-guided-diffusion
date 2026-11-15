"""Search on the unconditional sampler: recurrence, Feynman-Kac steering and progressive pruning.

generate() takes adm.generate's options for an unconditional checkpoint, plus:

- recurrence: redo each guided step this many times while t is at least recur_from, re-noising
  the result back to the step's own level in between (Universal Guidance, TFG, FreeDoM).
  A rerun is first order unless correct_reruns, which extrapolates from the x0 of the
  step before as the first run does.
- particles: run this many samples side by side, from successive draws of the seed.
- events: call indices after whose prediction the particles are scored by the verifier on
  the guided x0, and keep[n] of them carry on. With keep None at an event, they are
  resampled with replacement to the same number, with weights softmax(lam * (r - r_last)),
  the difference potential of Feynman-Kac steering. A number prunes to the best that many.
- The result is the particle the verifier scores best at the end.
- rollout_calls: above noise rollout_above, CLIP judges x0 after this many more
  deterministic model calls instead of the one-step x0, and the gradient comes back
  through the one-step x0's Jacobian.

With particles 1 and recurrence 1 it makes exactly the image adm.generate makes.
"""

import math
from collections.abc import Iterator, Sequence
from functools import partial
from pathlib import Path

import numpy as np
import torch as th

from cgd import adm, losses, sampler
from cgd.guide import NUM_CUTOUTS, load_guide, score
from cgd.outputs import frame_path, run_name, save_frame

AUX_SEED_OFFSET = 0x5DEECE66D  # Keeps the extra noise apart from the seed's own stream.


def renoise(x: th.Tensor, then: sampler.Level, now: sampler.Level, generator: th.Generator) -> th.Tensor:
    """x at the lower noise level `then`, taken back up to `now` as the forward process would."""
    ratio = now.alpha / then.alpha
    extra = math.sqrt(max(now.sigma**2 - (ratio * then.sigma) ** 2, 0.0))
    noise = th.randn(x.shape, device=x.device, generator=generator)
    return ratio * x + extra * noise


def rollout(denoise, log_alpha_bar: np.ndarray, x: th.Tensor, t: float, x0: th.Tensor, calls: int) -> th.Tensor:
    """x0 after `calls` more deterministic DPM-Solver++(2M) calls from x at t, in uniform logSNR steps to 0.

    x0 is the model's prediction at t itself, which the first step reuses.
    """
    table = sampler.log_snr(log_alpha_bar)
    log_snrs = np.linspace(np.interp(t, np.arange(len(table)), table), table[0], calls + 2)
    grid = np.interp(log_snrs, table[::-1], np.arange(len(table), dtype=np.float64)[::-1])
    previous = None
    for j in range(calls):
        now, then = sampler.level(log_alpha_bar, grid[j]), sampler.level(log_alpha_bar, grid[j + 1])
        clamped = x0.clamp(-sampler.X0_CLAMP, sampler.X0_CLAMP)
        x = sampler.dpm_solver_step(x, x0, clamped, previous, (now, then), noise=th.zeros_like(x), eta=0.0)
        previous = sampler.Previous(x0, sampler.step_size(now, then))
        eps = denoise(x, th.full((x.shape[0],), grid[j + 1], device=x.device, dtype=th.float32))
        x0 = (x - then.sigma * eps) / then.alpha
    return x0.clamp(-sampler.X0_CLAMP, sampler.X0_CLAMP)


def rollout_loss(loss_fn, denoise, log_alpha_bar: np.ndarray, calls: int, above: float, x, x0, sigma):
    """The guidance loss with CLIP shown a sharp rollout instead of the blurred x0 above noise `above`.

    Its gradient reaches x through the one-step x0's Jacobian, which keeps the step's
    direction the model's own, as if the rollout's image had been the one-step prediction.
    """
    if sigma <= above or calls == 0:
        return loss_fn(x, x0, sigma)
    t = float(np.interp(math.log1p(-(sigma**2)), log_alpha_bar[::-1], np.arange(len(log_alpha_bar))[::-1]))
    with th.no_grad():
        sharp = rollout(denoise, log_alpha_bar, x.detach(), t, x0.detach(), calls)
    proxy, plain = sharp.requires_grad_(), x.detach().requires_grad_()
    with th.enable_grad():
        value = loss_fn(plain, proxy, sigma)
        to_proxy, to_x = th.autograd.grad(value, (proxy, plain))
    return losses.with_gradient(x0, value.item(), to_proxy) + losses.with_gradient(x, 0.0, to_x)


def predict_each(denoise, x, t, now, guidance) -> sampler.Prediction:
    """sampler.predict() on each image of the batch on its own when guided, so a backward pass holds one image."""
    if guidance is None or x.shape[0] == 1:
        return sampler.predict(denoise, x, t, now, guidance)
    parts = [sampler.predict(denoise, x[i : i + 1], t, now, guidance) for i in range(x.shape[0])]
    return sampler.Prediction(
        th.cat([p.x0 for p in parts]), th.cat([p.guided_x0 for p in parts]), th.stack([p.loss for p in parts]).sum()
    )


def generate(  # pylint: disable=too-many-locals
    prompts: Sequence[str] = (),
    image_prompts: Sequence[str] = (),
    *,
    image_size: int = 256,
    num_cutouts: int = NUM_CUTOUTS,
    steps: int = sampler.UNCONDITIONAL_STEPS,
    guidance_rate: float = sampler.DEFAULT_GUIDANCE_RATE,
    clip_model: str = adm.DEFAULTS.clip_model,
    recurrence: int = 1,
    recur_from: float = 300.0,
    correct_reruns: bool = False,
    particles: int = 1,
    events: Sequence[int] = (),
    keep: Sequence[int | None] = (),
    lam: float = 100.0,
    verifier: str = "PE-Core-L-14-336",
    rollout_calls: int = 0,
    rollout_above: float = 0.9,
    seed: int = 0,
    output_dir: str | Path = "outputs",
    save_frequency: int = 10,  # pylint: disable=unused-argument  # Only the result is saved.
    progress: bool = False,
    device: str = "cuda",
) -> Iterator[tuple[int, Path]]:
    th.manual_seed(seed)
    guide = load_guide(clip_model, prompts, image_prompts, image_size=image_size, num_cutouts=num_cutouts, device=device)
    judge = load_guide(verifier, prompts, image_prompts, device=device) if particles > 1 else None
    model, log_alpha_bar = adm.load_model(image_size, device, fp16=True, use_checkpoint=False, unconditional=True)
    ts = sampler.timesteps(log_alpha_bar, steps)
    x = th.randn((particles, 3, image_size, image_size), device=device)
    aux = th.Generator(device).manual_seed(seed + AUX_SEED_OFFSET)
    denoise = partial(adm.denoise, model, None)
    loss_fn = partial(adm.guidance_loss, guide)
    if rollout_calls:
        loss_fn = partial(rollout_loss, loss_fn, denoise, log_alpha_bar, rollout_calls, rollout_above)
    guidance = sampler.Guidance(loss_fn, guidance_rate, sampler.UNCONDITIONAL_GUIDANCE_WINDOW)
    levels = [sampler.level(log_alpha_bar, t) for t in ts]
    last = len(log_alpha_bar) - 1
    previous, rewards = None, th.zeros(particles, device=device)
    schedule = dict(zip(events, keep, strict=True))
    for i in range(len(ts) - 1):
        active = guidance if sampler.guides_at(guidance, ts[i], last) else None
        repeats = recurrence if active is not None and ts[i] >= recur_from and i < len(ts) - 2 else 1
        for repeat in range(repeats):
            prediction = predict_each(denoise, x, ts[i], levels[i], active)
            if i == len(ts) - 2:
                break
            step = levels[i], levels[i + 1]
            x_next = sampler.dpm_solver_step(
                x,
                prediction.x0,
                prediction.guided_x0,
                previous if repeat == 0 or correct_reruns else None,
                step,
                noise=th.randn_like(x) if repeat == 0 else th.randn(x.shape, device=device, generator=aux),
            )
            if repeat < repeats - 1:
                x = renoise(x_next, levels[i + 1], levels[i], aux)
        if i in schedule and x.shape[0] > 1:
            current = score(judge, prediction.guided_x0)
            if schedule[i] is None:
                weights = th.softmax(lam * (current - rewards), 0)
                chosen = th.multinomial(weights, x.shape[0], replacement=True, generator=aux)
            else:
                chosen = current.topk(schedule[i]).indices
            if progress:
                print(i, [f"{r:.3f}" for r in current.tolist()], chosen.tolist())
            rewards = current[chosen]
            x_next, prediction = x_next[chosen], sampler.Prediction(*(p[chosen] for p in prediction[:2]), None)
        if i == len(ts) - 2:
            result = prediction.guided_x0
            break
        previous = sampler.Previous(prediction.x0, sampler.step_size(*step))
        x = x_next
    best = int(score(judge, result).argmax()) if result.shape[0] > 1 else 0
    path = frame_path(output_dir, run_name(prompts, image_prompts), len(ts) - 2, 0)
    yield 0, save_frame(result[best], path)

"""Thumbnail guidance is exact where everything is known: Gaussian data, on the CPU.

    uv run python reports/evidence/uncond/check_thumbnail.py

Pixels are independent N(0, s^2), 16x16, so 4x4 thumbnails are N(0, s^2 / 16). The "prompt"
is a noisy observation m of the thumbnail. The unconditional denoiser, the thumbnail
model's guidance direction and the true posterior given the prompt all have closed forms.
Sampling with thumbnail.guided_denoise() at scale 1, adding or replacing, must then draw from that posterior:
the thumbnails' mean and variance become the posterior's, and what a thumbnail does not
hold, each pixel's difference from its block's mean, keeps the variance it had.

A classifier of noisy thumbnails gives the same direction as -noise * grad log p(prompt | y),
which is how thumbnail.noised_clip_eps() hands over a CLIP's. The exact classifier here,
log p(m | y), is checked the same way.
"""

from functools import partial

import numpy as np
import thumbnail
import torch as th
import torch.nn.functional as tf

from cgd import sampler

S = 0.5  # The spread of a pixel. Small enough that sampler.X0_CLAMP clips little.
THUMBNAIL_VARIANCE = S**2 / thumbnail.FACTOR**2
TAU = 0.05  # The spread of the observation: m = thumbnail + TAU * noise.


def main() -> None:
    th.manual_seed(0)
    log_alpha_bar = sampler.schedule(np.cumprod(1 - np.linspace(1e-4, 0.02, 1000, dtype=np.float64)))
    observed = th.linspace(-0.3, 0.4, 16).view(1, 1, 4, 4).expand(1, 3, 4, 4)
    posterior_mean = observed * THUMBNAIL_VARIANCE / (THUMBNAIL_VARIANCE + TAU**2)
    posterior_variance = THUMBNAIL_VARIANCE * TAU**2 / (THUMBNAIL_VARIANCE + TAU**2)

    def model(x: th.Tensor, t: th.Tensor) -> th.Tensor:
        """The exact unconditional noise prediction."""
        now = sampler.level(log_alpha_bar, float(t[0]))
        x0 = now.alpha * S**2 / (now.alpha**2 * S**2 + now.sigma**2) * x
        return (x - now.alpha * x0) / now.sigma

    def predictions(y: th.Tensor, signal: float, noise: float) -> tuple[th.Tensor, th.Tensor]:
        """The exact eps(y | prompt) and eps(y) of a noisy thumbnail."""
        without = signal * THUMBNAIL_VARIANCE / (signal**2 * THUMBNAIL_VARIANCE + noise**2) * y
        gain = signal * posterior_variance / (signal**2 * posterior_variance + noise**2)
        given = posterior_mean + gain * (y - signal * posterior_mean)
        return (y - signal * given) / noise, (y - signal * without) / noise

    def classifier(y: th.Tensor, signal: float, noise: float) -> tuple[th.Tensor, th.Tensor]:
        """-noise * grad log p(m | y) and zero: the observation is normal around the thumbnail's posterior mean given y."""
        gain = signal * THUMBNAIL_VARIANCE / (signal**2 * THUMBNAIL_VARIANCE + noise**2)
        spread = THUMBNAIL_VARIANCE * noise**2 / (signal**2 * THUMBNAIL_VARIANCE + noise**2) + TAU**2
        return -noise * gain * (observed - gain * y) / spread, th.zeros_like(y)

    cases = ((0.0, False, predictions), (1.0, False, predictions), (1.0, True, predictions), (1.0, False, classifier))
    for scale, replace, source in cases:
        ts = sampler.timesteps(log_alpha_bar, 60)
        x = th.randn(8192, 3, 16, 16, dtype=th.float64)
        denoise = partial(thumbnail.guided_denoise, model, log_alpha_bar, source, scale, 0.0, replace=replace)
        *_, x0 = sampler.sample(denoise, log_alpha_bar, ts, x, None, progress=False)
        thumbnails = tf.avg_pool2d(x0, thumbnail.FACTOR)
        detail = x0 - tf.interpolate(thumbnails, scale_factor=thumbnail.FACTOR, mode="nearest")
        want_mean = posterior_mean if scale else th.zeros_like(posterior_mean)
        want_variance = posterior_variance if scale else THUMBNAIL_VARIANCE
        mean_error = float((thumbnails.mean(0) - want_mean[0]).abs().max())
        variance = float(thumbnails.var(0).mean())
        detail_variance = float(detail.var(0).mean())
        replacing = " replacing" if replace else " from the classifier" if source is classifier else ""
        print(
            f"scale {scale}{replacing}: thumbnail mean off by at most {mean_error:.4f}, variance {variance:.5f} "
            f"(want {want_variance:.5f}), detail variance {detail_variance:.5f} (want {S**2 * 15 / 16:.5f})"
        )
        assert mean_error < 0.01
        assert abs(variance / want_variance - 1) < 0.05
        assert abs(detail_variance / (S**2 * 15 / 16) - 1) < 0.05


if __name__ == "__main__":
    main()

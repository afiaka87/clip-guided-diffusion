"""Which noise levels to train at, how much each counts, and the noise put on a CLIP embedding.

A noise level here is `cgd.sampler`'s log(alpha / sigma), half the log of the signal to noise ratio. The
sampler takes steps uniform in it over the whole schedule, at fractional timesteps, so training draws it the
same way and not a uniform integer timestep. These are pure functions of tensors and a schedule.
"""

import numpy as np
import torch as th
from torch.nn import functional as tf

from cgd import sampler


def sample_log_snr(log_alpha_bar: np.ndarray, count: int, bias: float, generator: th.Generator) -> th.Tensor:
    """`count` noise levels within the schedule's range, in float64.

    With `bias` 1 they are uniform, as the sampler's steps are. Above 1 they lean towards high noise, where the
    condition decides what the image is of: the fraction of the way from the noisiest level is `u ** bias`.
    """
    table = sampler.log_snr(log_alpha_bar)
    fraction = th.rand(count, generator=generator, dtype=th.float64) ** bias
    return table[-1] + (table[0] - table[-1]) * fraction


def timestep_of(log_alpha_bar: np.ndarray, log_snr: th.Tensor) -> th.Tensor:
    """The fractional timestep of each noise level, the inverse of `sampler.level` as `sampler.timesteps` has it."""
    table_t = np.arange(len(log_alpha_bar), dtype=np.float64)
    wanted = -np.log1p(np.exp(-2 * log_snr.numpy()))  # Its log(alpha_bar), which falls as t rises.
    return th.from_numpy(np.interp(wanted, log_alpha_bar[::-1], table_t[::-1]))


def level_of(log_snr: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
    """alpha and sigma of each noise level: how much image and how much noise the noisy image holds."""
    return th.sigmoid(2 * log_snr).sqrt(), th.sigmoid(-2 * log_snr).sqrt()


def min_snr_weight(log_snr: th.Tensor, gamma: float, floor: float = 0.0) -> th.Tensor:
    """The weight of a noise prediction loss at each level (Min-SNR, Hang et al. 2023).

    The plain loss counts every level alike, which in terms of the image is its signal to noise ratio: nearly
    clean images dominate. This caps that ratio at `gamma`, so levels noisier than it keep weight 1 and cleaner
    ones count for less.

    `floor` does the opposite at the noisy end. There the noise is nearly the input, so its prediction error is
    tiny however wrong the image behind it, and those levels hardly move the weights, yet they are where a
    condition says the most about the image. With a floor the ratio is never counted as less than it, which
    multiplies the loss at levels noisier than `floor` by `floor / snr`.
    """
    snr = th.exp(2 * log_snr)
    return snr.clamp(min=floor, max=gamma) / snr


def noisy_embedding(embedding: th.Tensor, level: th.Tensor, generator: th.Generator | None = None) -> th.Tensor:
    """Unit embeddings moved off their direction by noise of relative length `level`, unit length again.

    A model trained on image embeddings is sampled with text embeddings, which sit apart from them. Noise on
    the image embedding teaches it not to rely on exactly where the embedding is (LAFITE, Zhou et al. 2022).
    `level` is one value per embedding, [N, 1].
    """
    noise = th.randn(embedding.shape, generator=generator, dtype=embedding.dtype)
    return tf.normalize(embedding + level * tf.normalize(noise, dim=-1), dim=-1)

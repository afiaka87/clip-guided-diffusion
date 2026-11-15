"""Thumbnail guidance, on Gaussian images whose exact denoisers are known and small stand-ins for its models.

When an image's pixels are independent Gaussians, so are its thumbnail's, and a prompt
that shifts the mean of every block of pixels shifts the thumbnail's mean by as much.
The exact noise prediction given the prompt is then known at both sizes, which checks
that the thumbnail's correction is the right one at the right noise level.
tests/integration runs the real models.
"""

import math

import numpy as np
import pytest
import torch as th
import torch.nn.functional as tf
from torch import nn

from cgd import sampler, thumbnails
from cgd.checks import ConfigError
from cgd.thumbnails import SIZE, Thumbnail

ALPHAS_CUMPROD = np.cumprod(1 - np.linspace(1e-4, 0.02, 1000))
SPREAD = 0.3
TOKENS = 5  # How long the stand-ins' prompts are.


@pytest.fixture(name="log_alpha_bar", scope="module")
def fixture_log_alpha_bar() -> np.ndarray:
    """The log signal level of ADM's linear schedule, at each of its timesteps."""
    return sampler.schedule(ALPHAS_CUMPROD)


def gaussian_eps(mean: th.Tensor | float, spread: float, signal: float, noise: float, y: th.Tensor) -> th.Tensor:
    """The exact noise in y = signal * image + noise * standard normal, for pixels that are N(mean, spread^2)."""
    denoised = mean + signal * spread**2 / (signal**2 * spread**2 + noise**2) * (y - signal * mean)
    return (y - signal * denoised) / noise


class TestGuidedDenoise:
    @pytest.mark.parametrize("image_size", [128, 256, 512])
    @pytest.mark.parametrize("t", [950.0, 500.5, 40.0])
    def test_an_exact_thumbnail_model_gives_the_exact_prediction_for_the_prompt(
        self, log_alpha_bar: np.ndarray, image_size: int, t: float
    ) -> None:
        factor = image_size // SIZE
        shift = th.randn(2, 3, SIZE, SIZE, dtype=th.float64)  # What the prompt adds to the mean of each block.
        now = sampler.level(log_alpha_bar, t)
        x = th.randn(2, 3, image_size, image_size, dtype=th.float64)

        def denoise(x: th.Tensor, _t: th.Tensor) -> th.Tensor:
            return gaussian_eps(0.0, SPREAD, now.alpha, now.sigma, x)

        def change(y: th.Tensor, signal: float, noise: float) -> th.Tensor:
            """A thumbnail's pixels are means of `factor`^2 of the image's, so they spread 1 / `factor` as far."""
            small = SPREAD / factor
            return gaussian_eps(shift, small, signal, noise, y) - gaussian_eps(0.0, small, signal, noise, y)

        guided = thumbnails.guided_denoise(denoise, log_alpha_bar, Thumbnail(change, 1.0), x, th.full((2,), t))
        exact = gaussian_eps(tf.interpolate(shift, scale_factor=factor), SPREAD, now.alpha, now.sigma, x)
        assert guided.dtype == th.float64
        assert th.allclose(guided, exact, rtol=1e-4, atol=1e-5)

    def test_the_scale_multiplies_what_the_thumbnail_model_adds(self, log_alpha_bar: np.ndarray) -> None:
        x, t = th.randn(1, 3, 256, 256), th.full((1,), 700.0)

        def nothing(x: th.Tensor, _t: th.Tensor) -> th.Tensor:
            return th.zeros_like(x)

        def ones(y: th.Tensor, _signal: float, _noise: float) -> th.Tensor:
            return th.ones_like(y)

        guided = thumbnails.guided_denoise(nothing, log_alpha_bar, Thumbnail(ones, 10.0), x, t)
        assert th.allclose(guided, th.full_like(x, 10.0 / 4))

    def test_it_moves_whole_blocks_and_keeps_the_gradient_of_the_denoiser(self, log_alpha_bar: np.ndarray) -> None:
        x, t = th.randn(1, 3, 256, 256, requires_grad=True), th.full((1,), 700.0)

        def nothing(x: th.Tensor, _t: th.Tensor) -> th.Tensor:
            return th.zeros_like(x)

        def twice(x: th.Tensor, _t: th.Tensor) -> th.Tensor:
            return 2 * x

        def of_the_thumbnail(y: th.Tensor, _signal: float, _noise: float) -> th.Tensor:
            return y.square()

        guidance = Thumbnail(of_the_thumbnail, 3.0)
        added = thumbnails.guided_denoise(nothing, log_alpha_bar, guidance, x, t)
        assert th.equal(added, tf.interpolate(added[:, :, ::4, ::4], scale_factor=4))
        assert added.std() > 0
        guided = thumbnails.guided_denoise(twice, log_alpha_bar, guidance, x, t)
        assert th.equal(th.autograd.grad(guided.sum(), x)[0], th.full_like(x, 2.0))


@pytest.mark.parametrize(("signal", "noise"), [(0.05, 0.25), (0.6, 0.2), (0.999, 0.01)])
def test_a_model_is_asked_at_the_timestep_whose_signal_to_noise_ratio_is_the_thumbnails(
    signal: float, noise: float
) -> None:
    cosine = np.cos((np.arange(1000) / 1000 + 0.008) / 1.008 * math.pi / 2) ** 2
    model_schedule = sampler.schedule(cosine / cosine[0] * 0.9999)
    y = th.randn(2, 3, SIZE, SIZE)
    asked: list[tuple[th.Tensor, float]] = []

    def change(y: th.Tensor, t: float) -> th.Tensor:
        asked.append((y, t))
        return y

    result = thumbnails.matched(change, model_schedule, y, signal, noise)
    scaled, t = asked[0]
    now = sampler.level(model_schedule, t)
    assert now.alpha / now.sigma == pytest.approx(signal / noise, rel=1e-6)
    assert th.allclose(scaled, y * (now.alpha / signal))  # So its signal is the one that timestep has.
    assert result is scaled


class TestPrompts:
    def test_the_weights_sum_to_one_as_a_guides_do(self) -> None:
        texts, weights = thumbnails.weighted_texts(["a fox:3", "snow", "blurry:-2"], "cpu")
        assert texts == ["a fox", "snow", "blurry"]
        assert th.allclose(weights, th.tensor([1.5, 0.5, -1.0]))

    def test_a_single_prompt_weighs_exactly_one_whatever_its_weight(self) -> None:
        assert thumbnails.weighted_texts(["a fox:0.3"], "cpu")[1].tolist() == [1.0]
        assert thumbnails.weighted_texts(["a fox:-4"], "cpu")[1].tolist() == [-1.0]

    def test_weights_that_cancel_out_are_explained(self) -> None:
        with pytest.raises(ConfigError, match="must not sum to 0"):
            thumbnails.weighted_texts(["cat:1", "dog:-1"], "cpu")

    def test_each_prompt_changes_the_prediction_by_its_weight(self) -> None:
        unconditional = th.randn(2, 3, 4, 4)
        first, second = th.randn(2, 3, 4, 4), th.randn(2, 3, 4, 4)
        change = thumbnails.weighted_change(th.cat([first, second]), unconditional, th.tensor([1.5, -0.5]))
        assert th.allclose(change, 1.5 * (first - unconditional) - 0.5 * (second - unconditional))

    def test_one_prompt_of_weight_one_changes_it_by_exactly_the_difference(self) -> None:
        conditional, unconditional = th.randn(2, 3, 4, 4), th.randn(2, 3, 4, 4)
        change = thumbnails.weighted_change(conditional, unconditional, th.tensor([1.0]))
        assert th.equal(change, conditional - unconditional)


class TokenDenoiser(nn.Module):
    """A denoiser with GLIDE's interface whose prediction is a known function of its input, timestep and prompt."""

    def __init__(self) -> None:
        super().__init__()
        with th.random.fork_rng():
            th.manual_seed(0)
            self.prompt = nn.Embedding(10, 6)

    def forward(self, x: th.Tensor, timesteps: th.Tensor, tokens: th.Tensor, mask: th.Tensor) -> th.Tensor:
        """Six channels, as GLIDE gives: the noise and its variance."""
        said = (self.prompt(tokens) * mask[..., None]).sum(1)
        return x.repeat(1, 2, 1, 1) * timesteps.view(-1, 1, 1, 1) + said[:, :, None, None]


def tokens_of(prompts: list[list[int]]) -> dict[str, th.Tensor]:
    """Each prompt's tokens, then the empty prompt, padded as GLIDE's tokenizer pads them."""
    padded = [prompt + [9] * (TOKENS - len(prompt)) for prompt in [*prompts, []]]
    masks = [[True] * len(prompt) + [False] * (TOKENS - len(prompt)) for prompt in [*prompts, []]]
    return {"tokens": th.tensor(padded), "mask": th.tensor(masks)}


def test_glide_is_asked_about_every_prompt_and_no_prompt_for_each_image_of_the_batch() -> None:
    model, y = TokenDenoiser(), th.randn(3, 3, 8, 8)
    prompts, weights = [[1, 2], [3]], th.tensor([2.0, -1.0])
    change = thumbnails.glide_change(model, tokens_of(prompts), weights, y, 0.5)
    said = [model.prompt(th.tensor(prompt)).sum(0)[:3] for prompt in prompts]
    # The image's own share of the prediction cancels, which leaves what each prompt said, by its weight.
    expected = (2.0 * said[0] - 1.0 * said[1])[None, :, None, None].expand(3, 3, 8, 8)
    assert th.allclose(change, expected, atol=1e-6)


class LinearNoisyClip(nn.Module):  # pylint: disable=abstract-method  # Guidance never calls its forward().
    """The noisy CLIP's image interface over a random projection of the thumbnail, scaled by its timestep."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        with th.random.fork_rng():
            th.manual_seed(seed)
            self.projection = nn.Linear(3 * 8 * 8, 4).requires_grad_(False)
        self.logit_scale = th.tensor(math.log(100.0))

    def image_embeddings(self, images: th.Tensor, timesteps: th.Tensor) -> th.Tensor:
        """One feature vector per noisy image, as the noisy CLIP gives."""
        return self.projection(images.flatten(1)) * (1 + timesteps[:, None])


class TestNoisyClip:
    def test_it_adds_the_gradient_of_its_logit_times_the_noise_level(self, log_alpha_bar: np.ndarray) -> None:
        model, target, y = LinearNoisyClip(0), th.randn(1, 4), th.randn(2, 3, 8, 8)
        change = thumbnails.clip_change(model, target, log_alpha_bar, y, 300.0)
        gradient = (100.0 * (1 + 300) * (target @ model.projection.weight)).view(1, 3, 8, 8).expand(2, 3, 8, 8)
        assert th.allclose(change, -sampler.level(log_alpha_bar, 300).sigma * gradient, rtol=1e-4)

    @pytest.mark.parametrize(("t", "step"), [(300.4, 300.0), (-3.0, 0.0), (1e6, 999.0)])
    def test_its_timesteps_are_the_whole_ones_of_its_schedule(
        self, log_alpha_bar: np.ndarray, t: float, step: float
    ) -> None:
        model, target, y = LinearNoisyClip(0), th.randn(1, 4), th.randn(1, 3, 8, 8)
        asked, whole = (thumbnails.clip_change(model, target, log_alpha_bar, y, at) for at in (t, step))
        assert th.equal(asked, whole)

    def test_it_takes_its_gradient_even_where_sampling_takes_none(self, log_alpha_bar: np.ndarray) -> None:
        model, target, y = LinearNoisyClip(1), th.randn(1, 4), th.randn(1, 3, 8, 8)
        with th.no_grad():
            change = thumbnails.clip_change(model, target, log_alpha_bar, y, 500.0)
        assert change.abs().sum() > 0
        assert not change.requires_grad


def test_a_model_built_in_the_forked_stream_leaves_the_runs_random_numbers_alone() -> None:
    th.manual_seed(0)
    expected = th.randn(4)
    th.manual_seed(0)
    with thumbnails.forked_rng("cpu"):
        nn.Linear(8, 8)
    assert th.equal(th.randn(4), expected)

"""The fast sampler, on ADM's linear schedule and a Gaussian image distribution whose exact denoiser is known.

When the data is Gaussian, N(0, SPREAD^2) in every pixel, the denoised image is a known
linear function of the noisy one, and the probability flow ODE only rescales its input.
That makes the solver's accuracy and the sampler's output checkable without a model.
"""

import math
from collections.abc import Callable
from itertools import pairwise

import numpy as np
import pytest
import torch as th

from cgd import sampler
from cgd.sampler import Guidance, Level, Previous

ALPHAS_CUMPROD = np.cumprod(1 - np.linspace(1e-4, 0.02, 1000))
SPREAD = 0.3  # Small enough that X0_CLAMP never touches a sample.


@pytest.fixture(name="log_alpha_bar", scope="module")
def fixture_log_alpha_bar() -> np.ndarray:
    """The log signal level of the toy schedule, at each of its timesteps."""
    return sampler.schedule(ALPHAS_CUMPROD)


def gaussian_x0(now: Level, x: th.Tensor) -> th.Tensor:
    """E[x0 | x] when x0 ~ N(0, SPREAD^2)."""
    return now.alpha * SPREAD**2 / (now.alpha**2 * SPREAD**2 + now.sigma**2) * x


def gaussian_denoiser(log_alpha_bar: np.ndarray) -> Callable[[th.Tensor, th.Tensor], th.Tensor]:
    """The exact noise prediction for Gaussian data."""

    def denoise(x: th.Tensor, t: th.Tensor) -> th.Tensor:
        now = sampler.level(log_alpha_bar, float(t[0]))
        return (x - now.alpha * gaussian_x0(now, x)) / now.sigma

    return denoise


def marginal_std(now: Level) -> float:
    """The spread of x at a level, for Gaussian data."""
    return math.sqrt(now.alpha**2 * SPREAD**2 + now.sigma**2)


def towards_half(_x: th.Tensor, x0: th.Tensor, _sigma: float) -> th.Tensor:
    """A guidance loss that pulls every pixel of the denoised image towards 0.5."""
    return (x0 - 0.5).square().sum()


def nowhere(_x: th.Tensor, x0: th.Tensor, _sigma: float) -> th.Tensor:
    """A guidance loss with no gradient, so guided steps run but nothing pulls them."""
    return (x0 * 0).sum()


class TestSchedule:
    @pytest.mark.parametrize("t", [0, 137, 999])
    def test_level_matches_the_table_at_integer_timesteps(self, log_alpha_bar: np.ndarray, t: int) -> None:
        now = sampler.level(log_alpha_bar, t)
        assert now.alpha**2 == pytest.approx(ALPHAS_CUMPROD[t], rel=1e-12)
        assert now.alpha**2 + now.sigma**2 == pytest.approx(1.0, rel=1e-12)

    def test_level_interpolates_log_alpha_bar_between_timesteps(self, log_alpha_bar: np.ndarray) -> None:
        now = sampler.level(log_alpha_bar, 137.25)
        expected = 0.75 * log_alpha_bar[137] + 0.25 * log_alpha_bar[138]
        assert math.log(now.alpha**2) == pytest.approx(expected, rel=1e-12)

    def test_timesteps_are_descending_and_uniform_in_log_snr(self, log_alpha_bar: np.ndarray) -> None:
        ts = sampler.timesteps(log_alpha_bar, 30)
        assert len(ts) == 31
        assert (ts[0], ts[-1]) == (999, 0.0)
        assert all(a > b for a, b in pairwise(ts))
        spacing = np.diff([math.log(now.alpha / now.sigma) for now in (sampler.level(log_alpha_bar, t) for t in ts)])
        assert spacing.std() / spacing.mean() < 0.01

    def test_start_joins_the_full_schedule(self, log_alpha_bar: np.ndarray) -> None:
        full = sampler.timesteps(log_alpha_bar, 30)
        ts = sampler.timesteps(log_alpha_bar, 30, start=400)
        assert ts[0] == 400
        assert ts[1:] == [t for t in full if t < 400]

    def test_step_size_is_the_rise_in_log_snr(self) -> None:
        assert sampler.step_size(Level(0.6, 0.8), Level(0.8, 0.6)) == pytest.approx(2 * math.log(4 / 3))

    def test_noised_is_the_forward_process(self, log_alpha_bar: np.ndarray) -> None:
        image, noise = th.rand(1, 3, 8, 8), th.randn(1, 3, 8, 8)
        now = sampler.level(log_alpha_bar, 500)
        assert th.equal(sampler.noised(log_alpha_bar, 500, image, noise), now.alpha * image + now.sigma * noise)


class TestGuidanceCap:
    def test_a_small_gradient_is_left_alone(self) -> None:
        eps, gradient = th.randn(2, 3, 8, 8), th.randn(2, 3, 8, 8) * 0.01
        assert th.equal(sampler.cap_guidance(gradient, eps, 1.0, 0.08), gradient)

    def test_a_large_gradient_is_scaled_to_the_rate_for_its_own_image_only(self) -> None:
        eps, gradient = th.randn(2, 3, 8, 8), th.randn(2, 3, 8, 8)
        gradient[1] *= 0.001
        capped = sampler.cap_guidance(gradient, eps, 0.5, 0.08)
        ratio = sampler.rms(0.5 * capped) / sampler.rms(eps)
        assert ratio[0].item() == pytest.approx(0.08, rel=1e-5)
        assert th.equal(capped[1], gradient[1])

    def test_guide_x0_moves_x0_by_tweedies_formula(self) -> None:
        x0, gradient, now = th.rand(1, 3, 8, 8), th.randn(1, 3, 8, 8), Level(0.6, 0.8)
        assert th.allclose(sampler.guide_x0(x0, gradient, now), x0 + 0.8**2 / 0.6 * gradient)


class TestLimitGuidance:
    def test_an_update_that_stays_in_range_is_kept(self) -> None:
        x0, update = th.full((1, 3, 8, 8), 0.2), th.full((1, 3, 8, 8), 0.3)
        assert th.equal(sampler.limit_guidance(x0, update, 0.99), x0 + update)

    def test_it_holds_back_only_as_much_as_it_must_for_each_image(self) -> None:
        x0, update = th.full((2, 3, 8, 8), 0.5), th.full((2, 3, 8, 8), 1.0)
        update[1] = 0.1
        result = sampler.limit_guidance(x0, update, 0.99)
        assert 0.99 <= result[0].max().item() <= 1.0
        assert result[1].max().item() == pytest.approx(0.6)

    def test_guide_x0_applies_the_limit(self) -> None:
        x0, gradient, now = th.full((1, 3, 8, 8), 0.5), th.ones(1, 3, 8, 8), Level(0.6, 0.8)
        assert sampler.guide_x0(x0, gradient, now, guidance_limit=0.99).max().item() <= 1.0


class TestSolverStep:
    STEP = Level(0.6, 0.8), Level(0.8, 0.6)

    @pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
    def test_with_the_true_x0_it_follows_the_forward_process(self, eta: float) -> None:
        """The noise it keeps and the noise it adds always make up exactly the next level's noise."""
        (now, then), h = self.STEP, sampler.step_size(*self.STEP)
        x0, eps, noise = th.rand(1, 3, 16, 16), th.randn(1, 3, 16, 16), th.randn(1, 3, 16, 16)
        x = now.alpha * x0 + now.sigma * eps
        result = sampler.dpm_solver_step(x, x0, x0, None, self.STEP, noise=noise, eta=eta)
        kept = math.exp(-eta * h)
        expected = then.alpha * x0 + then.sigma * (kept * eps + math.sqrt(1 - kept**2) * noise)
        assert th.allclose(result, expected, atol=1e-6)

    def test_the_correction_vanishes_when_x0_has_not_changed(self) -> None:
        x0, x, noise = th.rand(1, 3, 8, 8), th.randn(1, 3, 8, 8), th.randn(1, 3, 8, 8)
        first_order = sampler.dpm_solver_step(x, x0, x0, None, self.STEP, noise=noise)
        assert th.equal(sampler.dpm_solver_step(x, x0, x0, Previous(x0, 0.3), self.STEP, noise=noise), first_order)

    def test_the_correction_extrapolates_the_unguided_x0_only(self) -> None:
        """Guidance jumps between steps, and extrapolating it destabilises the solver."""
        x0, previous, x, noise = (
            th.rand(1, 3, 8, 8),
            Previous(th.rand(1, 3, 8, 8), 0.3),
            th.randn(1, 3, 8, 8),
            th.zeros(1),
        )

        def correction(guided_x0: th.Tensor) -> th.Tensor:
            corrected = sampler.dpm_solver_step(x, x0, guided_x0, previous, self.STEP, noise=noise, eta=0.0)
            return corrected - sampler.dpm_solver_step(x, x0, guided_x0, None, self.STEP, noise=noise, eta=0.0)

        assert th.allclose(correction(x0), correction(x0 + 0.5), atol=1e-6)

    def test_the_second_order_correction_follows_the_ode_more_closely(self, log_alpha_bar: np.ndarray) -> None:
        ts = sampler.timesteps(log_alpha_bar, 10)
        levels = [sampler.level(log_alpha_bar, t) for t in ts[:-1]]
        start = th.randn(1, 1, 64, 64, dtype=th.float64)
        exact = start * marginal_std(levels[-1]) / marginal_std(levels[0])

        def solve(second_order: bool) -> th.Tensor:
            x, previous = start, None
            for now, then in pairwise(levels):
                x0 = gaussian_x0(now, x)
                x = sampler.dpm_solver_step(x, x0, x0, previous, (now, then), noise=th.zeros(1), eta=0.0)
                previous = Previous(x0, sampler.step_size(now, then)) if second_order else None
            return x

        first_error, second_error = ((solve(order) - exact).abs().max().item() for order in (False, True))
        assert second_error < first_error / 3


class TestGuidedPrediction:
    @pytest.mark.parametrize(
        ("guidance", "t", "expected"),
        [
            (None, 500, False),
            (Guidance(towards_half, rate=0.0), 500, False),
            (Guidance(towards_half), 0.1 * 999, True),
            (Guidance(towards_half), 0.1 * 999 - 1, False),
            (Guidance(towards_half), 0.8 * 999, True),
            (Guidance(towards_half), 0.8 * 999 + 1, False),
        ],
    )
    def test_guides_at_only_inside_the_window(self, guidance: Guidance | None, t: float, expected: bool) -> None:
        assert sampler.guides_at(guidance, t, 999) is expected

    def test_without_guidance_it_only_denoises_and_clamps(self, log_alpha_bar: np.ndarray) -> None:
        now, x = sampler.level(log_alpha_bar, 500), th.randn(1, 3, 8, 8) * 3
        x0, guided_x0, loss = sampler.predict(gaussian_denoiser(log_alpha_bar), x, 500, now, None)
        assert loss is None
        assert not x0.requires_grad
        assert th.allclose(x0, gaussian_x0(now, x), atol=1e-6)
        assert th.equal(guided_x0, x0.clamp(-sampler.X0_CLAMP, sampler.X0_CLAMP))

    def test_guidance_moves_the_guided_image_and_leaves_the_unguided_one(self, log_alpha_bar: np.ndarray) -> None:
        now, x, denoise = sampler.level(log_alpha_bar, 500), th.randn(1, 3, 8, 8), gaussian_denoiser(log_alpha_bar)
        plain = sampler.predict(denoise, x, 500, now, None).x0
        x0, guided_x0, loss = sampler.predict(denoise, x, 500, now, Guidance(towards_half))
        assert th.equal(x0, plain)
        assert loss is not None
        assert loss.item() == pytest.approx(towards_half(x, plain, now.sigma).item(), rel=1e-5)
        assert (guided_x0 - 0.5).abs().mean() < (x0 - 0.5).abs().mean()


class TestSample:
    def test_it_yields_one_image_per_model_call(self, log_alpha_bar: np.ndarray) -> None:
        ts = sampler.timesteps(log_alpha_bar, 6)
        x = th.randn(1, 1, 8, 8)
        assert len(list(sampler.sample(gaussian_denoiser(log_alpha_bar), log_alpha_bar, ts, x, progress=False))) == 6

    def test_it_draws_the_spread_of_the_data(self, log_alpha_bar: np.ndarray) -> None:
        th.manual_seed(0)
        ts = sampler.timesteps(log_alpha_bar, 30)
        x = th.randn(1, 3, 128, 128)
        result = list(sampler.sample(gaussian_denoiser(log_alpha_bar), log_alpha_bar, ts, x, progress=False))[-1]
        assert result.std().item() == pytest.approx(SPREAD, rel=0.03)
        assert abs(result.mean().item()) < 0.01

    def test_guidance_pulls_the_samples_towards_its_target_harder_at_a_higher_rate(
        self, log_alpha_bar: np.ndarray
    ) -> None:
        ts = sampler.timesteps(log_alpha_bar, 30)

        def mean(rate: float | None) -> float:
            th.manual_seed(0)
            x = th.randn(1, 3, 32, 32)
            guidance = None if rate is None else Guidance(towards_half, rate)
            samples = sampler.sample(gaussian_denoiser(log_alpha_bar), log_alpha_bar, ts, x, guidance, progress=False)
            return list(samples)[-1].mean().item()

        unguided, gentle, default, strong = mean(None), mean(0.02), mean(sampler.DEFAULT_GUIDANCE_RATE), mean(0.32)
        assert unguided < gentle < default < strong
        assert default - unguided > 0.1 * SPREAD

    def test_a_seed_reproduces_the_samples(self, log_alpha_bar: np.ndarray) -> None:
        ts = sampler.timesteps(log_alpha_bar, 8)

        def draw() -> th.Tensor:
            th.manual_seed(3)
            x = th.randn(1, 3, 16, 16)
            return list(sampler.sample(gaussian_denoiser(log_alpha_bar), log_alpha_bar, ts, x, progress=False))[-1]

        assert th.equal(draw(), draw())


class TestRecurrence:
    STEP = Level(0.6, 0.8), Level(0.8, 0.6)
    UNCONDITIONAL = Guidance(towards_half, window=sampler.UNCONDITIONAL_GUIDANCE_WINDOW, recurrence=2)

    def test_renoised_shrinks_the_signal_back_to_the_first_level(self) -> None:
        (now, then), x0 = self.STEP, th.rand(1, 3, 8, 8, dtype=th.float64)
        assert th.allclose(sampler.renoised(then.alpha * x0, self.STEP, th.zeros_like(x0)), now.alpha * x0)

    def test_renoised_makes_up_exactly_the_first_levels_noise(self) -> None:
        """The noise x kept from the lower level and the fresh noise together have the first level's spread."""
        th.manual_seed(0)
        (now, then), (eps, noise) = self.STEP, th.randn(2, 1, 512, 512, dtype=th.float64)
        assert sampler.renoised(then.sigma * eps, self.STEP, noise).std().item() == pytest.approx(now.sigma, rel=0.01)

    @pytest.mark.parametrize(
        ("guidance", "t", "expected"),
        [
            (UNCONDITIONAL, 999, 2),
            (UNCONDITIONAL, sampler.RECUR_ABOVE * 999, 2),
            (UNCONDITIONAL, sampler.RECUR_ABOVE * 999 - 1, 1),
            (Guidance(towards_half, window=(0.1, 0.5), recurrence=2), 600, 1),
            (Guidance(towards_half, rate=0.0, recurrence=2), 500, 1),
            (None, 500, 1),
        ],
    )
    def test_only_guided_steps_above_recur_above_rerun(
        self, guidance: Guidance | None, t: float, expected: int
    ) -> None:
        assert sampler.recurrences(guidance, t, 999) == expected

    def test_a_step_that_recurs_calls_the_model_once_per_run_and_yields_once(self, log_alpha_bar: np.ndarray) -> None:
        exact, calls = gaussian_denoiser(log_alpha_bar), []

        def denoise(x: th.Tensor, t: th.Tensor) -> th.Tensor:
            calls.append(float(t[0]))
            return exact(x, t)

        ts = sampler.timesteps(log_alpha_bar, 12)
        guidance = self.UNCONDITIONAL._replace(recurrence=3)
        recurring = [t for t in ts[:-2] if sampler.recurrences(guidance, t, 999) == 3]
        images = list(sampler.sample(denoise, log_alpha_bar, ts, th.randn(1, 3, 8, 8), guidance, progress=False))
        assert len(images) == 12
        assert 0 < len(recurring) < 11
        expected = [t for t in ts[:-1] for _ in range(3 if t in recurring else 1)]
        assert calls == pytest.approx(expected, rel=1e-6)  # The model gets float32 timesteps.

    def test_reruns_keep_the_spread_of_the_data(self, log_alpha_bar: np.ndarray) -> None:
        """Taking x back up a level is the forward process, so with nothing pulling, reruns change no statistics."""
        th.manual_seed(0)
        ts = sampler.timesteps(log_alpha_bar, 30)
        guidance = Guidance(nowhere, window=sampler.UNCONDITIONAL_GUIDANCE_WINDOW, recurrence=3, recur_above=0.0)
        x = th.randn(1, 3, 128, 128)
        result = list(sampler.sample(gaussian_denoiser(log_alpha_bar), log_alpha_bar, ts, x, guidance, progress=False))
        assert result[-1].std().item() == pytest.approx(SPREAD, rel=0.03)
        assert abs(result[-1].mean().item()) < 0.01

    @pytest.mark.parametrize("i", [2, 10, 20])
    def test_each_run_gives_guidance_another_say(self, log_alpha_bar: np.ndarray, i: int) -> None:
        """A rerun starts from where the last run's guidance took x, and guides it again.

        The pull is the guided step's mean less the unguided one's from the same noise. The
        denoiser is linear here, so that difference is guidance's effect alone.
        """
        ts = sampler.timesteps(log_alpha_bar, 30)
        step = sampler.level(log_alpha_bar, ts[i]), sampler.level(log_alpha_bar, ts[i + 1])

        def pull(runs: int) -> float:
            means = []
            for loss in (towards_half, nowhere):
                th.manual_seed(0)
                x = th.randn(1, 3, 128, 128) * marginal_std(step[0])
                denoise = gaussian_denoiser(log_alpha_bar)
                _, x_next = sampler.recurring_step(denoise, x, ts[i], step, Guidance(loss), previous=None, runs=runs)
                means.append(x_next.mean().item())
            return means[0] - means[1]

        assert 0 < pull(1) < pull(2) < pull(3)


class TestBlend:
    @staticmethod
    def counted(level: float, calls: list[float]) -> Callable[[th.Tensor, th.Tensor], th.Tensor]:
        """A denoiser that predicts `level` everywhere, and records that it was called."""

        def denoise(x: th.Tensor, _t: th.Tensor) -> th.Tensor:
            calls.append(level)
            return th.full_like(x, level)

        return denoise

    @pytest.mark.parametrize(
        ("share", "called", "level"), [(0.0, [1.0], 1.0), (1.0, [3.0], 3.0), (0.25, [1.0, 3.0], 1.5)]
    )
    def test_it_is_that_share_of_the_way_between_two_predictions_and_calls_only_what_it_needs(
        self, share: float, called: list[float], level: float
    ) -> None:
        calls: list[float] = []
        x, t = th.zeros(1, 3, 4, 4), th.full((1,), 500.0)
        eps = sampler.blend(self.counted(1.0, calls), self.counted(3.0, calls), share, x, t)
        assert calls == called
        assert th.allclose(eps, th.full_like(x, level))

    def test_a_denoiser_blended_with_itself_is_called_once(self) -> None:
        calls: list[float] = []
        denoise = self.counted(2.0, calls)
        sampler.blend(denoise, denoise, 0.5, th.zeros(1, 3, 4, 4), th.full((1,), 500.0))
        assert calls == [2.0]


class TestRedraw:
    @staticmethod
    def image() -> tuple[th.Tensor, th.Tensor]:
        """An image and noise for it."""
        image, noise = th.randn(2, 1, 3, 16, 16, generator=th.Generator().manual_seed(0))
        return image * SPREAD, noise

    def test_from_almost_no_noise_it_is_the_image(self, log_alpha_bar: np.ndarray) -> None:
        """To within the noise added, which no denoiser can tell from an image that is Gaussian itself."""
        image, noise = self.image()
        redrawn = sampler.redraw(gaussian_denoiser(log_alpha_bar), log_alpha_bar, image, noise, t=2.0, steps=30)
        assert (redrawn - image).square().mean().sqrt().item() < 1.5 * sampler.level(log_alpha_bar, 2.0).sigma

    def test_the_more_noise_the_more_of_it_is_the_denoisers_own(self, log_alpha_bar: np.ndarray) -> None:
        image, noise = self.image()
        denoise = gaussian_denoiser(log_alpha_bar)
        far = [
            (sampler.redraw(denoise, log_alpha_bar, image, noise, t=t, steps=30) - image).square().mean().item()
            for t in (50.0, 300.0, 600.0, 900.0)
        ]
        assert far == sorted(far)
        assert far[0] < 0.2 * far[-1]

    def test_it_takes_the_share_of_the_steps_below_its_timestep(self, log_alpha_bar: np.ndarray) -> None:
        exact, calls = gaussian_denoiser(log_alpha_bar), []

        def denoise(x: th.Tensor, t: th.Tensor) -> th.Tensor:
            calls.append(float(t[0]))
            return exact(x, t)

        image, noise = self.image()
        sampler.redraw(denoise, log_alpha_bar, image, noise, t=500.0, steps=30)
        below = [t for t in sampler.timesteps(log_alpha_bar, 30) if 0 < t < 500]
        assert calls == pytest.approx([500.0, *below], rel=1e-6)
        assert 5 < len(calls) < 30

    def test_it_draws_no_random_numbers(self, log_alpha_bar: np.ndarray) -> None:
        image, noise = self.image()
        th.manual_seed(0)
        expected = th.randn(4)
        th.manual_seed(0)
        first = sampler.redraw(gaussian_denoiser(log_alpha_bar), log_alpha_bar, image, noise, t=600.0, steps=30)
        assert th.equal(th.randn(4), expected)
        again = sampler.redraw(gaussian_denoiser(log_alpha_bar), log_alpha_bar, image, noise, t=600.0, steps=30)
        assert th.equal(first, again)

    def test_a_small_change_of_the_image_or_the_timestep_changes_it_a_little(self, log_alpha_bar: np.ndarray) -> None:
        image, noise = self.image()
        denoise = gaussian_denoiser(log_alpha_bar)
        redrawn = sampler.redraw(denoise, log_alpha_bar, image, noise, t=600.0, steps=30)
        moved = sampler.redraw(denoise, log_alpha_bar, image + 0.001, noise, t=600.0, steps=30)
        later = sampler.redraw(denoise, log_alpha_bar, image, noise, t=601.0, steps=30)
        assert (moved - redrawn).abs().max() < 0.002
        assert (later - redrawn).abs().max() < 0.01

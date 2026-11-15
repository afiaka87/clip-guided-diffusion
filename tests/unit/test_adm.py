"""A sequence of images, on a small class-conditional U-Net of ADM's kind with random weights.

It draws nothing recognisable, but it is the real network on the real sampler, so which
frames a sequence saves, what each image is and what lies between two of them are all checked.
"""

from functools import partial

import numpy as np
import pytest
import torch as th
from guided_diffusion.unet import UNetModel
from torch import nn

from cgd import adm, sampler, slide

STEPS = 6
SIZE = 16


@pytest.fixture(name="model", scope="module")
def fixture_model() -> UNetModel:
    """A U-Net that draws one of 4 classes, with random weights, since it starts out predicting zero."""
    with th.random.fork_rng():
        th.manual_seed(0)
        model = UNetModel(SIZE, 3, 32, 6, 1, attention_resolutions=(1, 2), channel_mult=(1, 2), num_classes=4)
        for parameter in model.parameters():
            nn.init.normal_(parameter, std=0.2)
    return model.eval().requires_grad_(False)


@pytest.fixture(name="log_alpha_bar", scope="module")
def fixture_log_alpha_bar() -> np.ndarray:
    """ADM's linear schedule."""
    return sampler.schedule(np.cumprod(1 - np.linspace(1e-4, 0.02, 1000)))


def towards(level: float, _x: th.Tensor, x0: th.Tensor, _sigma: float) -> th.Tensor:
    """A guidance loss that pulls every pixel of the denoised image towards `level`."""
    return (x0 - level).square().sum()


def drawings(model: UNetModel, classes: tuple[int, ...]) -> list[adm.Drawing]:
    """What draws one image per class, each guided towards a grey level of its own."""
    denoisers = adm.class_denoisers(model, [th.tensor([label]) for label in classes])
    return [
        adm.Drawing(unet, unet, sampler.Guidance(partial(towards, 0.2 * place))) for place, unet in enumerate(denoisers)
    ]


def frames(
    model: UNetModel,
    log_alpha_bar: np.ndarray,
    classes: tuple[int, ...],
    transition_frames: int = 5,
    loop: bool = False,
) -> list[tuple[int, th.Tensor]]:
    """Every (step, images) a sequence of one image per class saves, from seed 0, with every model call saved."""
    th.manual_seed(0)
    noise = th.randn(1, 3, SIZE, SIZE)
    ts = sampler.timesteps(log_alpha_bar, STEPS)
    settings = {"steps": STEPS, "save_frequency": 1, "transition_frames": transition_frames, "progress": False}
    return list(adm.sampled_frames(drawings(model, classes), log_alpha_bar, ts, noise, noise, loop=loop, **settings))


class TestClassDenoisers:
    def test_images_of_one_class_share_a_denoiser(self, model: UNetModel) -> None:
        first, second, third = adm.class_denoisers(model, [th.tensor([2, 2]), th.tensor([3, 3]), th.tensor([2, 2])])
        assert first is third
        assert first is not second

    def test_without_classes_every_image_shares_one(self, model: UNetModel) -> None:
        first, second = adm.class_denoisers(model, [None, None])
        assert first is second

    def test_each_draws_its_own_class(self, model: UNetModel) -> None:
        x, t = th.randn(1, 3, SIZE, SIZE), th.full((1,), 500.0)
        first, second = adm.class_denoisers(model, [th.tensor([0]), th.tensor([1])])
        assert th.equal(first(x, t), adm.denoise(model, th.tensor([0]), x, t))
        assert not th.equal(first(x, t), second(x, t))


class TestSampledFrames:
    def test_one_image_saves_its_model_calls_and_nothing_more(
        self, model: UNetModel, log_alpha_bar: np.ndarray
    ) -> None:
        assert [step for step, _ in frames(model, log_alpha_bar, (0,))] == list(range(STEPS))

    def test_each_next_image_adds_the_frames_that_slide_to_it(
        self, model: UNetModel, log_alpha_bar: np.ndarray
    ) -> None:
        saved = frames(model, log_alpha_bar, (0, 1, 2), transition_frames=5)
        assert [step for step, _ in saved] == list(range(STEPS + 2 * 5))

    def test_every_image_is_the_one_drawn_alone_from_the_same_seed(
        self, model: UNetModel, log_alpha_bar: np.ndarray
    ) -> None:
        """The sampler draws fresh noise each step, and every image of a sequence has to draw the same."""
        saved = frames(model, log_alpha_bar, (0, 1, 2), transition_frames=5)
        for place, last in enumerate((STEPS - 1, STEPS + 4, STEPS + 9)):
            th.manual_seed(0)
            noise = th.randn(1, 3, SIZE, SIZE)
            ts = sampler.timesteps(log_alpha_bar, STEPS)
            drawing = drawings(model, (0, 1, 2))[place]
            samples = sampler.sample(drawing.unet, log_alpha_bar, ts, noise, drawing.guidance, progress=False)
            alone = sampler.final(samples)
            assert th.equal(saved[last][1].clamp(-1, 1), alone.clamp(-1, 1))

    def test_a_sequence_is_the_same_every_time(self, model: UNetModel, log_alpha_bar: np.ndarray) -> None:
        first, second = (frames(model, log_alpha_bar, (0, 1)) for _ in range(2))
        assert all(th.equal(a, b) for (_, a), (_, b) in zip(first, second, strict=True))

    def test_a_loop_begins_at_the_first_image_and_slides_back_to_it(
        self, model: UNetModel, log_alpha_bar: np.ndarray
    ) -> None:
        """Without the model calls that made the first image, so the last frame leads into the first."""
        saved = frames(model, log_alpha_bar, (0, 1, 2), transition_frames=5, loop=True)
        assert [step for step, _ in saved] == list(range(STEPS - 1, STEPS + 3 * 5))
        assert th.equal(saved[-1][1], saved[0][1].clamp(-1, 1))
        assert not th.equal(saved[-2][1], saved[-1][1])
        once = frames(model, log_alpha_bar, (0, 1, 2), transition_frames=5)[STEPS - 1 :]
        assert all(th.equal(a, b) for (_, a), (_, b) in zip(saved, once, strict=False))  # The same until it turns back.


class TestBetween:
    @pytest.fixture(name="ends")
    def fixture_ends(self) -> tuple[th.Tensor, th.Tensor]:
        """Two images to slide between."""
        first, second = th.rand(2, 1, 3, SIZE, SIZE, generator=th.Generator().manual_seed(1)).mul(2).sub(1)
        return first, second

    @pytest.fixture(name="draw")
    def fixture_draw(
        self, model: UNetModel, log_alpha_bar: np.ndarray, ends: tuple[th.Tensor, th.Tensor]
    ) -> slide.Draw:
        """The frame at a position between the two images, one of class 0 and one of class 1."""
        denoisers = adm.class_denoisers(model, [th.tensor([0]), th.tensor([1])])
        noise = th.randn(1, 3, SIZE, SIZE, generator=th.Generator().manual_seed(2))
        return partial(adm.between, tuple(denoisers), log_alpha_bar, noise, ends, steps=30)

    def test_next_to_either_image_it_is_that_image(self, draw: slide.Draw, ends: tuple[th.Tensor, th.Tensor]) -> None:
        assert slide.change(draw(0.001), ends[0]) < 0.02
        assert slide.change(draw(0.999), ends[1]) < 0.02

    def test_halfway_it_is_neither(self, draw: slide.Draw, ends: tuple[th.Tensor, th.Tensor]) -> None:
        assert slide.change(draw(0.5), ends[0]) > 0.2
        assert slide.change(draw(0.5), ends[1]) > 0.2

    def test_it_depends_on_the_position_alone_and_draws_no_random_numbers(self, draw: slide.Draw) -> None:
        th.manual_seed(0)
        expected = th.randn(4)
        th.manual_seed(0)
        first = draw(0.4)
        assert th.equal(th.randn(4), expected)
        assert th.equal(draw(0.4), first)

    def test_a_small_move_changes_it_a_little(self, draw: slide.Draw) -> None:
        assert slide.change(draw(0.4), draw(0.401)) < 0.1 * slide.change(draw(0.4), draw(0.6))

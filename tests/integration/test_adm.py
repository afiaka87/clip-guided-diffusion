"""CLIP guidance of the ImageNet diffusion models, end to end on the GPU."""

import math
from functools import partial
from pathlib import Path
from typing import Any

import pytest
import torch as th
import torchvision.transforms.functional as tvf
from PIL import Image

from cgd import adm, sampler
from cgd.guide import load_guide, score

pytestmark = pytest.mark.cuda

PROMPT = "a photograph of a red fox in the snow"
OTHER_PROMPT = "a bowl of ramen on a wooden table"


def generate(output_dir: Path, **settings: Any) -> list[tuple[int, Path]]:
    """Every (batch index, path) a run saves, with only the result saved unless asked otherwise."""
    options: dict[str, Any] = {
        "prompts": [PROMPT],
        "save_frequency": 1000,
        "progress": False,
        "device": "cuda",
        "output_dir": output_dir,
    }
    return list(adm.generate(**options | settings))


def load(path: Path) -> th.Tensor:
    """A saved image, in [-1, 1]."""
    return tvf.to_tensor(Image.open(path).convert("RGB")).unsqueeze(0).mul(2).sub(1)


def prefers(path: Path, prompt: str, other: str) -> bool:
    """Whether a CLIP model scores the image closer to `prompt` than to `other`."""
    image = load(path).cuda()
    return score(load_guide("ViT-B/32", [prompt], device="cuda"), image) > score(
        load_guide("ViT-B/32", [other], device="cuda"), image
    )


def test_it_draws_the_prompt_without_blowing_out(workdir: Path) -> None:
    (_, path), *_ = reversed(generate(workdir))
    assert path.name == "0029.png"
    pixels = tvf.to_tensor(Image.open(path).convert("RGB"))
    clipped = ((pixels <= 1 / 255) | (pixels >= 254 / 255)).float().mean().item()
    assert clipped < 0.25
    assert pixels.std().item() > 0.1
    assert prefers(path, PROMPT, OTHER_PROMPT)


def test_it_saves_every_frame_asked_for_and_the_result(workdir: Path) -> None:
    names = [path.name for _, path in generate(workdir, image_size=64, steps=8, save_frequency=3)]
    assert names == ["0000.png", "0003.png", "0006.png", "0007.png"]
    assert (workdir / "current.png").read_bytes() == (
        workdir / "a_photograph_of_a_red_fox_in_the_snow/00/0007.png"
    ).read_bytes()


def test_512_fits_with_guidance(workdir: Path) -> None:
    """Gradient checkpointing is what lets it fit in 8 GB."""
    assert generate(workdir, image_size=512, steps=6)[-1][1].name == "0005.png"


def test_gradient_checkpointing_keeps_the_gradient() -> None:
    image, timestep, label = (
        th.randn(1, 3, 64, 64, device="cuda"),
        th.tensor([500.0], device="cuda"),
        th.tensor([278], device="cuda"),
    )
    gradients = []
    for use_checkpoint in (False, True):
        model, _ = adm.load_model(64, "cuda", True, use_checkpoint)
        pixels = image.clone().requires_grad_()
        gradients.append(th.autograd.grad(model(pixels, timestep, y=label)[:, :3].square().sum(), pixels)[0])
    # 16 bit weights set the tolerance, as the two paths round in a different order.
    assert ((gradients[0] - gradients[1]).abs().max() / gradients[0].abs().max()).item() < 0.01


def test_each_checkpoint_keeps_the_schedule_it_was_trained_with() -> None:
    """Sampling a model on a schedule it was not trained on produces garbage."""
    first_beta = {size: 1 - math.exp(adm.load_model(size, "cpu", False)[1][0]) for size in (64, 128)}
    assert first_beta[64] < 1e-4 / 2  # Cosine starts far below linear's 1e-4.
    assert first_beta[128] == pytest.approx(1e-4)


def test_a_batch_holds_distinct_images_of_every_prompt(workdir: Path) -> None:
    results = generate(workdir, prompts=["a red fox", "a yellow lemon:0.5"], image_size=64, steps=6, batch_size=3)
    finals = results[-3:]
    assert [index for index, _ in finals] == [0, 1, 2]
    first, second = (load(path) for _, path in finals[:2])
    assert (first - second).abs().mean().item() > 0.01


class TestInitImage:
    def test_it_shortens_the_schedule_and_stays_close_to_the_init(self, workdir: Path, photon: Path) -> None:
        results = generate(workdir, image_size=64, init_image=str(photon), skip_timesteps=700, save_frequency=1)
        assert len(results) < 30
        result = load(results[-1][1]).add(1).div(2)
        init = tvf.to_tensor(Image.open(photon).convert("RGB").resize((64, 64))).unsqueeze(0)
        assert (result - init).abs().mean() < (th.rand_like(init) - init).abs().mean()

    def test_without_skip_timesteps_it_starts_halfway(self, workdir: Path, photon: Path) -> None:
        _, log_alpha_bar = adm.load_model(64, "cuda", fp16=True, use_checkpoint=False)  # As generate() loads it.
        full = sampler.timesteps(log_alpha_bar, sampler.DEFAULT_STEPS, adm.NUM_TIMESTEPS // 2 - 1)
        assert len(generate(workdir, image_size=64, init_image=str(photon), save_frequency=1)) == len(full) - 1

    def test_it_is_stretched_to_the_requested_shape(self, workdir: Path, photon: Path) -> None:
        results = generate(workdir, image_size=64, init_image=str(photon), steps=6, width_offset=32)
        assert Image.open(results[-1][1]).size == (96, 64)

    def test_a_perceptual_init_scale_keeps_the_image_closer_to_it(self, workdir: Path, mushroom: Path) -> None:
        lpips = adm.load_lpips("cuda")
        init = adm.load_init_image(str(mushroom), 64, 64, "cuda")

        def distance(init_scale: float) -> float:
            path = generate(workdir / str(init_scale), image_size=64, init_image=str(mushroom), init_scale=init_scale)
            with th.no_grad():
                return lpips(load(path[-1][1]).cuda(), init).item()

        assert distance(1000) < distance(0) - 0.03  # 0.23 against 0.30 when measured.


class TestImagePrompts:
    def test_an_image_prompt_alone_steers_the_image(self, workdir: Path) -> None:
        fox = generate(workdir / "fox")[-1][1]
        result = generate(workdir / "from_image", prompts=[], image_prompts=[str(fox)])[-1][1]
        assert prefers(result, "a photo of a fox", "a photo of a car")

    def test_a_run_from_an_image_prompt_is_named_after_it(self, workdir: Path, photon: Path) -> None:
        results = generate(workdir, prompts=[], image_prompts=[f"{photon}:2"], image_size=64, steps=4)
        assert results[-1][1] == workdir / "photon" / "00" / "0003.png"


def test_a_gradient_that_overflows_fp16_does_not_black_out_the_image() -> None:
    """The U-Net backpropagates at half precision. Whatever overflows gets no guidance instead of turning NaN."""
    model, log_alpha_bar = adm.load_model(64, "cuda", fp16=True, use_checkpoint=False)
    guide = load_guide("ViT-B/32", [PROMPT], image_size=64, device="cuda")
    labels = th.full((1,), 277, device="cuda")

    def denoise(x: th.Tensor, t: th.Tensor) -> th.Tensor:
        return model(x, t, y=labels)[:, :3]

    def overflowing_loss(x: th.Tensor, x0: th.Tensor, sigma: float) -> th.Tensor:
        return adm.guidance_loss(guide, x, x0, sigma) * 1e30

    th.manual_seed(0)
    x = th.randn(1, 3, 64, 64, device="cuda")
    ts = sampler.timesteps(log_alpha_bar, 6)
    result = list(sampler.sample(denoise, log_alpha_bar, ts, x, sampler.Guidance(overflowing_loss), progress=False))[-1]
    assert result.isfinite().all()
    assert result.std().item() > 0.1


def test_running_out_of_memory_raises_instead_of_ending_the_run_early(workdir: Path) -> None:
    with pytest.raises(th.cuda.OutOfMemoryError):
        generate(workdir, image_size=64, batch_size=4096, save_frequency=1)
    th.cuda.empty_cache()


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"clip_model": "ViT-B/32+PE-Core-B-16"},
        {"clip_model": "PE-Core-B-16", "clip_fp16": True},
        {"guidance_limit": 0.99},
        {"aesthetic_scale": 0.2},
        {"init_scale": 1000, "skip_timesteps": 300},
    ],
    ids=["plain", "ensemble", "fp16", "limit", "aesthetic", "lpips"],
)
def test_a_seed_reproduces_the_image_exactly(workdir: Path, mushroom: Path, settings: dict[str, Any]) -> None:
    if "init_scale" in settings:
        settings = settings | {"init_image": str(mushroom)}
    run = partial(generate, steps=12, seed=3, **settings)
    assert run(workdir / "a")[-1][1].read_bytes() == run(workdir / "b")[-1][1].read_bytes()


def test_a_different_seed_changes_the_image(workdir: Path) -> None:
    first, second = (generate(workdir / str(seed), image_size=64, steps=6, seed=seed)[-1][1] for seed in (3, 4))
    assert first.read_bytes() != second.read_bytes()

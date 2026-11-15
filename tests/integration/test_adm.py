"""CLIP guidance of the ImageNet diffusion models, end to end on the GPU.

Every run here is a few steps, of the 64px model unless the test is about another one:
enough to go through each part once. tests/unit checks what the parts compute.
"""

import math
from functools import partial
from pathlib import Path
from typing import Any

import pytest
import torch as th
import torchvision.transforms.functional as tvf
from PIL import Image

from cgd import adm, thumbnails

pytestmark = pytest.mark.cuda

PROMPT = "a photograph of a red fox in the snow"
OTHER_PROMPT = "a bowl of ramen on a wooden table"
SEQUENCE = (PROMPT, f">{OTHER_PROMPT}")
# One CLIP model, which loads once. Four steps, because fewer clip nearly every pixel to black or white at 256,
# and such images are the same whatever made them.
UNCONDITIONAL = {"unconditional": True, "image_size": 256, "clip_model": "ViT-B/32", "steps": 4}


def generate(output_dir: Path, **settings: Any) -> list[tuple[int, Path]]:
    """Every (batch index, path) a run saves, with only the first and last frame saved unless asked otherwise."""
    options: dict[str, Any] = {
        "prompts": [PROMPT],
        "image_size": 64,
        "steps": 2,
        "save_frequency": 1000,
        "progress": False,
        "device": "cuda",
        "output_dir": output_dir,
    }
    return list(adm.generate(**options | settings))


def results(output_dir: Path, **settings: Any) -> list[bytes]:
    """The finished image of each image of a run's batch."""
    last = dict(generate(output_dir, **settings))
    return [last[index].read_bytes() for index in sorted(last)]


# The 64px model last, so the tests after this one find it loaded.
@pytest.mark.parametrize(
    ("image_size", "unconditional"), [(128, False), (256, False), (512, False), (512, True), (256, True), (64, False)]
)
def test_every_checkpoint_loads_on_its_schedule_and_draws_at_its_size(
    workdir: Path, image_size: int, unconditional: bool
) -> None:
    """Loading is strict, so flags that disagreed with a checkpoint's weights would fail here."""
    settings = {"image_size": image_size, "unconditional": unconditional, "thumbnail": "none"}
    path = generate(workdir, clip_model="ViT-B/32", steps=1, **settings)[-1][1]
    assert path.name == "0000.png"
    image = Image.open(path)
    assert image.size == (image_size, image_size)
    assert tvf.to_tensor(image.convert("RGB")).std().item() > 0.02
    _, log_alpha_bar = adm.load_model(  # As generate() loaded it, so this is the same model.
        image_size,
        "cuda",
        fp16=True,
        use_checkpoint=image_size**2 > adm.GRADIENT_CHECKPOINT_PIXELS,
        unconditional=unconditional,
    )
    # Sampling a model on a schedule it was not trained on produces garbage. Cosine starts far below linear's 1e-4.
    linear = 1 - math.exp(log_alpha_bar[0]) == pytest.approx(1e-4)
    assert linear == (image_size != 64)


def test_it_saves_every_frame_asked_for_and_the_result(workdir: Path) -> None:
    names = [path.name for _, path in generate(workdir, steps=5, save_frequency=3)]
    assert names == ["0000.png", "0003.png", "0004.png"]
    assert (workdir / "current.png").read_bytes() == (
        workdir / "a_photograph_of_a_red_fox_in_the_snow/00/0004.png"
    ).read_bytes()


def test_a_batch_holds_distinct_images_of_every_prompt(workdir: Path) -> None:
    first, second = results(workdir, prompts=["a red fox", "a yellow lemon:0.5"], batch_size=2)
    assert first != second


def test_an_init_image_is_stretched_to_the_requested_shape_and_skips_the_start_of_the_schedule(
    workdir: Path, photon: Path
) -> None:
    settings = {"init_image": str(photon), "skip_timesteps": 700, "width_offset": 32}
    saved = generate(workdir, steps=6, save_frequency=1, **settings)
    assert 1 <= len(saved) < 6
    assert Image.open(saved[-1][1]).size == (96, 64)


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"clip_model": "ViT-B/32+PE-Core-B-16"},
        {"clip_model": "PE-Core-B-16", "clip_fp16": True},
        {"init_scale": 1000, "skip_timesteps": 300},
    ],
    ids=["plain", "ensemble", "fp16", "lpips"],
)
def test_a_seed_reproduces_the_image_exactly(workdir: Path, mushroom: Path, settings: dict[str, Any]) -> None:
    """At 128 over six steps, where the U-Net's backward pass made three images of one seed when cuDNN chose it."""
    if "init_scale" in settings:
        settings = settings | {"init_image": str(mushroom)}
    run = partial(results, image_size=128, steps=6, seed=3, **settings)
    assert run(workdir / "a") == run(workdir / "b")


def test_a_different_seed_changes_the_image(workdir: Path) -> None:
    assert results(workdir / "3", seed=3) != results(workdir / "4", seed=4)


def test_every_way_to_draw_without_a_class_makes_its_own_images_and_the_same_ones_again(workdir: Path) -> None:
    """Each thumbnail model, none, and perturbed attention, on a batch with weighted prompts.

    The first run of each loads its thumbnail model and the second finds it loaded, which must not move the seed.
    """
    runs: dict[str, dict[str, Any]] = {name: {"thumbnail": name} for name in (adm.NO_THUMBNAIL, *thumbnails.MODELS)}
    runs["perturbed attention"] = {"perturbed_attention": True}
    draw = partial(results, prompts=["a red fox in the snow:2", "a blurry photo:-1"], batch_size=2, **UNCONDITIONAL)
    thumbnails.load_model.cache_clear()
    images = {}
    for name, settings in runs.items():
        images[name] = draw(workdir / name, **settings)
        assert draw(workdir / f"{name} again", **settings) == images[name], name
    thumbnails.load_model.cache_clear()  # The tests after this one need the VRAM more.
    assert len({image for batch in images.values() for image in batch}) == 2 * len(runs)


def test_gradient_checkpointing_keeps_the_gradient() -> None:
    image, timestep, label = (
        th.randn(1, 3, 64, 64, device="cuda"),
        th.tensor([500.0], device="cuda"),
        th.tensor([278], device="cuda"),
    )
    gradients = []
    for use_checkpoint in (False, True):
        model, _ = adm.load_model(64, "cuda", fp16=True, use_checkpoint=use_checkpoint, unconditional=False)
        pixels = image.clone().requires_grad_()
        gradients.append(th.autograd.grad(model(pixels, timestep, y=label)[:, :3].square().sum(), pixels)[0])
    # 16 bit weights set the tolerance, as the two paths round in a different order.
    assert ((gradients[0] - gradients[1]).abs().max() / gradients[0].abs().max()).item() < 0.01


class TestSequence:
    def test_each_image_is_the_one_its_prompt_makes_alone_and_a_seed_reproduces_every_frame(
        self, workdir: Path
    ) -> None:
        run = partial(generate, prompts=SEQUENCE, transition_frames=3, seed=3)
        saved = run(workdir / "a")
        assert [path.name for _, path in saved] == [f"{step:04}.png" for step in range(5)]
        frames = [path.read_bytes() for _, path in saved]
        assert frames == [path.read_bytes() for _, path in run(workdir / "b")]
        assert len(set(frames)) == len(frames)
        alone = [results(workdir / prompt[:6], prompts=[prompt], seed=3)[0] for prompt in (PROMPT, OTHER_PROMPT)]
        assert [frames[1], frames[-1]] == alone

    def test_an_unconditional_checkpoint_slides_between_images_made_with_thumbnail_guidance(
        self, workdir: Path
    ) -> None:
        run = generate(workdir / "a", prompts=SEQUENCE, transition_frames=2, **UNCONDITIONAL)
        saved = [path.read_bytes() for _, path in run]
        assert len(saved) == 4  # The first frame and the result of the first image, then the frames to the second.
        assert saved[-1] == results(workdir / "b", prompts=[OTHER_PROMPT], **UNCONDITIONAL)[0]
        assert saved[-2] != saved[-1]

    def test_a_batch_of_image_prompts_slides_each_of_its_images(self, workdir: Path, photon: Path, goose: Path) -> None:
        settings = {"prompts": [], "image_prompts": [str(photon), f">{goose}"], "batch_size": 2}
        saved = generate(workdir, transition_frames=2, **settings)
        assert [index for index, _ in saved] == [0, 1] * 4
        assert len({path.read_bytes() for _, path in saved}) == 8

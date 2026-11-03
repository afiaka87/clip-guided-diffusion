"""CLIP correcting a one-step model's noise, end to end on the GPU."""

from pathlib import Path
from typing import Any

import pytest
import torch as th
import torchvision.transforms.functional as tvf
from PIL import Image

from cgd import backbone
from cgd.guide import guide_loss, load_guide

pytestmark = pytest.mark.cuda

PROMPT = "a photograph of a red fox in the snow"


def generate(output_dir: Path, **settings: Any) -> list[tuple[int, Path]]:
    """Every (batch index, path) a run saves, with only the first and last saved unless asked otherwise."""
    options: dict[str, Any] = {
        "prompts": [PROMPT],
        "image_size": 256,
        "clip_model": "ViT-B/32",
        "iterations": 4,
        "save_frequency": 1000,
        "progress": False,
        "device": "cuda",
        "output_dir": output_dir,
    }
    return list(backbone.generate(**options | settings))


def test_optimising_the_noise_lowers_the_clip_loss_and_keeps_the_best() -> None:
    """Over the default iterations. The loss rises for the first ten or so, while the momentum builds."""
    model = backbone.load_backbone("sd-turbo", PROMPT, "cuda")
    # The whole frame is the only cutout, so the loss is a function of the noise alone.
    guide = load_guide("ViT-B/32", [PROMPT], image_size=256, num_cutouts=1, min_cut_fraction=1.0, device="cuda")
    conditioning = backbone.unet_conditioning(model, 256, 256)
    th.manual_seed(0)
    start = th.randn(backbone.latent_shape(256, 256), device="cuda")

    def draw(z: th.Tensor) -> th.Tensor:
        return backbone.preview(model, conditioning, z)

    def loss(images: th.Tensor) -> th.Tensor:
        return guide_loss(guide, images)

    with th.backends.cudnn.flags(enabled=True, deterministic=True):  # As generate() runs it.
        steps = list(backbone.optimise_noise(draw, loss, start, progress=False))
    values = [step.loss for step in steps]
    assert min(values) < values[0] - 0.03  # 0.70 against 0.77 when measured.
    assert abs(steps[-1].best_z.norm().item() - start.norm().item()) < 0.1 * start.norm().item()
    assert th.equal(steps[-1].best_z, steps[values.index(min(values))].best_z)


def test_zero_iterations_is_the_plain_model(workdir: Path) -> None:
    assert [path.name for _, path in generate(workdir, iterations=0)] == ["0000.png"]


def test_it_saves_the_frames_asked_for_and_the_final_image(workdir: Path) -> None:
    assert [path.name for _, path in generate(workdir, save_frequency=2)] == ["0000.png", "0002.png", "0004.png"]


@pytest.mark.parametrize(("settings", "size"), [({"width_offset": 64}, (320, 256)), ({"image_size": 136}, (136, 136))])
def test_the_image_has_the_requested_shape(workdir: Path, settings: dict[str, Any], size: tuple[int, int]) -> None:
    assert Image.open(generate(workdir, iterations=1, **settings)[-1][1]).size == size


def test_a_batch_holds_distinct_images(workdir: Path) -> None:
    finals = [path for _, path in generate(workdir, batch_size=2, iterations=1) if path.name == "0001.png"]
    assert len(finals) == 2
    assert finals[0].read_bytes() != finals[1].read_bytes()


def test_an_image_prompt_alone_runs(workdir: Path, photon: Path) -> None:
    assert generate(workdir, prompts=[], image_prompts=[str(photon)], iterations=1)[-1][1].name == "0001.png"


@pytest.mark.parametrize("name", ["vega", "sdxl-lightning"])
def test_the_sdxl_backbones_draw_at_512(workdir: Path, name: str) -> None:
    path = generate(workdir, backbone=name, image_size=512, iterations=1)[-1][1]
    image = Image.open(path)
    assert image.size == (512, 512)
    assert tvf.to_tensor(image.convert("RGB")).std().item() > 0.05


@pytest.mark.parametrize(
    "settings",
    [{}, {"aesthetic_scale": 0.2}, {"clip_model": "ViT-B/32+PE-Core-B-16"}],
    ids=["plain", "aesthetic", "ensemble"],
)
def test_a_seed_reproduces_the_image_exactly(workdir: Path, settings: dict[str, Any]) -> None:
    first, second = (generate(workdir / run, seed=3, **settings)[-1][1] for run in ("a", "b"))
    assert first.read_bytes() == second.read_bytes()


def test_a_different_seed_changes_the_image(workdir: Path) -> None:
    first, second = (generate(workdir / str(seed), seed=seed)[-1][1] for seed in (3, 4))
    assert first.read_bytes() != second.read_bytes()

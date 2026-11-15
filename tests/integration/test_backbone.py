"""CLIP correcting a one-step model's noise, end to end on the GPU.

Every run loads its backbone from disk, which is most of what one costs, so there are few
of them, and each is one iteration on a small image. tests/unit checks what the parts compute.
"""

from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import pytest
import torchvision.transforms.functional as tvf
from PIL import Image

from cgd import backbone

pytestmark = pytest.mark.cuda

PROMPT = "a photograph of a red fox in the snow"
OTHER = "a sailboat on a calm lake at sunset"
SEQUENCE = (PROMPT, f">{OTHER}")


def generate(output_dir: Path, **settings: Any) -> list[tuple[int, Path]]:
    """Every (batch index, path) a run saves, with only the first and last saved unless asked otherwise."""
    options: dict[str, Any] = {
        "prompts": [PROMPT],
        "image_size": 128,
        "clip_model": "ViT-B/32",
        "iterations": 1,
        "save_frequency": 1000,
        "progress": False,
        "device": "cuda",
        "output_dir": output_dir,
    }
    return list(backbone.generate(**options | settings))


def test_each_image_of_a_sequence_is_the_one_its_prompt_makes_alone_and_a_seed_reproduces_every_frame(
    workdir: Path, changes: Callable[[Sequence[Path]], list[float]]
) -> None:
    run = partial(generate, prompts=SEQUENCE, transition_frames=12, seed=3)
    saved = [path for _, path in run(workdir / "a")]
    # The plain model's image, the corrected one, then the frames to the second image.
    assert [path.name for path in saved] == [f"{step:04}.png" for step in range(14)]
    frames = [path.read_bytes() for path in saved]
    assert frames == [path.read_bytes() for _, path in run(workdir / "b")]
    alone = [generate(workdir / prompt[:9], prompts=[prompt], seed=3)[-1][1] for prompt in (PROMPT, OTHER)]
    assert [frames[1], frames[-1]] == [path.read_bytes() for path in alone]
    assert max(changes(saved[1:])) < 0.75 * changes(alone)[0]  # No frame jumps: 0.42 against 0.71 when measured.


def test_the_plain_model_slides_each_image_of_a_batch_without_clip_at_any_shape(workdir: Path) -> None:
    settings = {"prompts": SEQUENCE, "iterations": 0, "batch_size": 2, "width_offset": 64}
    saved = generate(workdir, transition_frames=2, **settings)
    assert [index for index, _ in saved] == [0] * 3 + [1] * 3
    assert [path.name for _, path in saved[:3]] == ["0000.png", "0001.png", "0002.png"]
    assert len({path.read_bytes() for _, path in saved}) == 6
    assert Image.open(saved[-1][1]).size == (192, 128)


def test_an_image_prompt_alone_guides_with_an_ensemble_and_saves_the_frames_asked_for(
    workdir: Path, photon: Path
) -> None:
    settings = {"prompts": [], "image_prompts": [str(photon)], "clip_model": "ViT-B/32+PE-Core-B-16"}
    saved = generate(workdir, iterations=2, save_frequency=1, **settings)
    assert [path.name for _, path in saved] == ["0000.png", "0001.png", "0002.png"]
    assert len({path.read_bytes() for _, path in saved}) == 3


@pytest.mark.parametrize("name", ["vega", "sdxl-lightning"])
def test_the_sdxl_backbones_draw_at_512_and_slide_their_pooled_embedding_too(workdir: Path, name: str) -> None:
    saved = generate(workdir, backbone=name, image_size=512, prompts=SEQUENCE, transition_frames=2)
    assert len({path.read_bytes() for _, path in saved}) == 4
    for _, path in saved[1:]:
        image = Image.open(path)
        assert image.size == (512, 512)
        assert tvf.to_tensor(image.convert("RGB")).std().item() > 0.05

"""Does the unconditional pipeline only lack a composition? Hand it one and see.

Each image starts from another arm's image of the same prompt and seed, shrunk to a few
pixels a side so only its layout and colours survive, then noised to a timestep partway
down the schedule. From there it is adm.generate() as usual: the unconditional U-Net and
CLIP draw everything the layout does not hold. With SD-Turbo's images as the source this
is an upper bound on what a text-conditioned prior over tiny layouts could add, and with
the pipeline's own images it is a second pass over its own composition.

    uv run python run_arm.py turbo32_t600 layout:generate \
        '{"source": "turbo", "side": 32, "start": 600, "clip_model": "ViT-B/32+PE-Core-B-16"}'
"""

import json
from collections.abc import Iterator, Sequence
from pathlib import Path

from common import IMAGES, located, prompt_of
from PIL import Image

from cgd import adm


def source_image(arm: str, prompt: str, seed: int) -> Path:
    """The image another arm made of this prompt at this seed."""
    for line in (IMAGES / arm / "runs.jsonl").read_text().splitlines():
        record = json.loads(line)
        if (prompt_of(record), record["seed"]) == (prompt, seed):
            return located(record["path"])
    raise LookupError(f"{arm} has no image of {prompt!r} at seed {seed}.")


def layout(image: Path, side: int, size: int = 256) -> Image.Image:
    """The image averaged down to `side` pixels a side, and smoothly back up to `size`."""
    small = Image.open(image).convert("RGB").resize((side, side), Image.BOX)
    return small.resize((size, size), Image.BICUBIC)


def generate(
    prompts: Sequence[str] = (),
    image_prompts: Sequence[str] = (),
    *,
    source: str,
    side: int,
    start: int,
    seed: int = 0,
    output_dir: str | Path = "outputs",
    **options,
) -> Iterator[tuple[int, Path]]:
    """adm.generate() on the unconditional checkpoint, from `source`'s layout noised to timestep `start`."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    init = output_dir / f"layout_{side}.png"
    layout(source_image(source, prompts[0], seed), side).save(init)
    yield from adm.generate(
        prompts,
        image_prompts,
        unconditional=True,
        init_image=str(init),
        skip_timesteps=adm.NUM_TIMESTEPS - 1 - start,
        seed=seed,
        output_dir=output_dir,
        **options,
    )

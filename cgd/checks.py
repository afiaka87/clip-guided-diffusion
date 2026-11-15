"""Rejecting a configuration that cannot work, before anything slow runs."""

from collections.abc import Iterable, Sequence
from pathlib import Path

from cgd.download import is_url
from cgd.prompts import by_image, loops, parse_prompt, scenes


class ConfigError(ValueError):
    """The requested run cannot work. The message says what to change."""


def require(condition: bool, message: str) -> None:
    """Raise a ConfigError saying `message` unless `condition` holds."""
    if not condition:
        raise ConfigError(message)


def check_common(
    prompts: Sequence[str],
    image_prompts: Sequence[str],
    *,
    batch_size: int,
    num_cutouts: int,
    save_frequency: int,
    aesthetic_scale: float,
    transition_frames: int | None = None,
    other_images: Iterable[str] = (),
) -> None:
    """The checks every pipeline shares: something to draw for each image of a sequence, and images that exist.

    `transition_frames` is None for a run that did not set it.
    """
    require(bool(prompts or image_prompts), "Provide at least one prompt, text or image.")
    lengths = {len(by_image(prompts)), len(by_image(image_prompts))} - {1}
    require(
        len(lengths) <= 1,
        "The text prompts and the image prompts are sequences of different lengths. "
        "Give both as many images, or make one of them a single image's, to go with every image.",
    )
    require(
        transition_frames is None or len(scenes(prompts, image_prompts)) > 1,
        "--transition-frames needs a sequence: begin the prompts of each image after the first with >, "
        "as in 'a castle|>a fox'.",
    )
    require(transition_frames is None or transition_frames >= 1, "--transition-frames must be at least 1.")
    require(
        not loops(prompts, image_prompts) or len(scenes(prompts, image_prompts)) > 1,
        "A > at the end slides back to the first image, which takes another image to come back from, "
        "as in 'a castle|>a fox|>'.",
    )
    require(batch_size >= 1, "--batch-size must be at least 1.")
    require(num_cutouts >= 1, "--num-cutouts must be at least 1.")
    require(save_frequency >= 1, "--save-frequency must be at least 1.")
    require(aesthetic_scale >= 0, "--aesthetic-scale must not be negative.")
    listed = (parse_prompt(prompt)[0] for image in by_image(image_prompts) for prompt in image)
    for image in (*listed, *other_images):
        require(is_url(image) or Path(image).is_file(), f"No such image: {image}")

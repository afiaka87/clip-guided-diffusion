"""Rejecting a configuration that cannot work, before anything slow runs."""

from collections.abc import Iterable, Sequence
from pathlib import Path

from cgd.download import is_url
from cgd.prompts import parse_prompt


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
    save_frequency: int,
    aesthetic_scale: float,
    other_images: Iterable[str] = (),
) -> None:
    """The checks every pipeline shares: something to draw, and images that exist."""
    require(bool(prompts or image_prompts), "Provide at least one prompt, text or image.")
    require(batch_size >= 1, "--batch-size must be at least 1.")
    require(save_frequency >= 1, "--save-frequency must be at least 1.")
    require(aesthetic_scale >= 0, "--aesthetic-scale must not be negative.")
    for image in (*(parse_prompt(prompt)[0] for prompt in image_prompts), *other_images):
        require(is_url(image) or Path(image).is_file(), f"No such image: {image}")

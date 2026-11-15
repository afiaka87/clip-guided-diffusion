"""Prompts: a text, or an image's path or URL, with an optional weight after a colon.

A prompt that starts with > begins the next image of a sequence, so "a castle|>a fox" is
two images, and the animation slides from the first to the second.
"""

import math
from collections.abc import Sequence
from typing import NamedTuple

NEXT_IMAGE = ">"


class Scene(NamedTuple):
    """What one image of a sequence is made from."""

    prompts: tuple[str, ...]
    image_prompts: tuple[str, ...]


def parse_prompt(prompt: str) -> tuple[str, float]:
    """Split "<text or image>:<weight>" into its parts. Without a weight, it is 1."""
    text, separator, weight = prompt.rpartition(":")
    if not separator:
        return prompt, 1.0
    try:
        value = float(weight)
    except ValueError:  # That colon was part of the prompt or URL, not a weight.
        return prompt, 1.0
    return (text, value) if math.isfinite(value) else (prompt, 1.0)


def split_prompts(prompts: str) -> tuple[str, ...]:
    """The prompts of a string that joins them with |, leaving out empty ones."""
    return tuple(prompt for prompt in prompts.split("|") if prompt)


def by_image(prompts: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    """The prompts of each image of a sequence, without the > that begins every image after the first.

    Always at least one image, which has no prompts when there are none at all.
    """
    images: list[list[str]] = [[]]
    for prompt in prompts:
        text = prompt
        if prompt.startswith(NEXT_IMAGE):
            images.append([])
            text = prompt[len(NEXT_IMAGE) :].strip()
        if text:
            images[-1].append(text)
    return tuple(tuple(image) for image in images if image) or ((),)


def scenes(prompts: Sequence[str], image_prompts: Sequence[str] = ()) -> tuple[Scene, ...]:
    """Each image of a sequence, in order: the text prompts and the image prompts at the same place in theirs.

    Text or image prompts that are not a sequence themselves go with every image. If both
    are sequences they must be as long as each other, which checks.check_common() checks.
    """
    texts, images = by_image(prompts), by_image(image_prompts)
    count = max(len(texts), len(images))
    return tuple(Scene(texts[place % len(texts)], images[place % len(images)]) for place in range(count))

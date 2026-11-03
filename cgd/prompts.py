"""Prompts: a text, or an image's path or URL, with an optional weight after a colon."""

import math


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

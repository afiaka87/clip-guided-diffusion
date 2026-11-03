"""Parsing weighted prompts."""

import pytest

from cgd.prompts import parse_prompt, split_prompts


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("a red fox", ("a red fox", 1.0)),
        ("a red fox:2", ("a red fox", 2.0)),
        ("blurry:-0.5", ("blurry", -0.5)),
        ("1984", ("1984", 1.0)),
        ("a sign that says: hello", ("a sign that says: hello", 1.0)),
        ("https://example.com:8080/cat.png", ("https://example.com:8080/cat.png", 1.0)),
        ("https://example.com:8080/cat.png:0.5", ("https://example.com:8080/cat.png", 0.5)),
        ("cat:nan", ("cat:nan", 1.0)),
        ("cat:inf", ("cat:inf", 1.0)),
    ],
)
def test_parse_prompt(prompt: str, expected: tuple[str, float]) -> None:
    assert parse_prompt(prompt) == expected


@pytest.mark.parametrize(
    ("prompts", "expected"), [("", ()), ("a fox", ("a fox",)), ("a fox:2||blurry:-1|", ("a fox:2", "blurry:-1"))]
)
def test_split_prompts_leaves_out_empty_ones(prompts: str, expected: tuple[str, ...]) -> None:
    assert split_prompts(prompts) == expected

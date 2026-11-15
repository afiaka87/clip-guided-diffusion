"""Parsing weighted prompts."""

import pytest

from cgd.prompts import Scene, by_image, parse_prompt, scenes, split_prompts


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


@pytest.mark.parametrize(
    ("prompts", "expected"),
    [
        ((), ((),)),
        (("a fox", "blurry:-1"), (("a fox", "blurry:-1"),)),
        (("a castle", ">a fox", "snow:0.5", "> a sailboat"), (("a castle",), ("a fox", "snow:0.5"), ("a sailboat",))),
        ((">a castle", ">", ">a fox"), (("a castle",), ("a fox",))),
    ],
)
def test_a_prompt_that_starts_with_the_mark_begins_the_next_image(
    prompts: tuple[str, ...], expected: tuple[tuple[str, ...], ...]
) -> None:
    assert by_image(prompts) == expected


def test_on_the_command_line_the_next_image_follows_a_bar_and_the_mark() -> None:
    assert by_image(split_prompts("a castle|sunset:0.5|>a fox:2|>a sailboat")) == (
        ("a castle", "sunset:0.5"),
        ("a fox:2",),
        ("a sailboat",),
    )


def test_each_image_of_a_sequence_has_the_prompts_at_its_place() -> None:
    assert scenes(["a castle", ">a fox"], ["castle.png", ">fox.png"]) == (
        Scene(("a castle",), ("castle.png",)),
        Scene(("a fox",), ("fox.png",)),
    )


def test_prompts_that_are_not_a_sequence_go_with_every_image() -> None:
    assert scenes(["a castle", ">a fox"], ["style.png:0.5"]) == (
        Scene(("a castle",), ("style.png:0.5",)),
        Scene(("a fox",), ("style.png:0.5",)),
    )
    assert scenes(["oil painting"], ["castle.png", ">fox.png"]) == (
        Scene(("oil painting",), ("castle.png",)),
        Scene(("oil painting",), ("fox.png",)),
    )
    assert scenes(["a fox"]) == (Scene(("a fox",), ()),)

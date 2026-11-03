"""The command line: which pipeline its options choose, and how it explains a bad run. Nothing here loads a model."""

import argparse
from pathlib import Path

import pytest
import torch as th

from cgd import adm, backbone
from cgd.checks import ConfigError
from cgd.cli import build_parser, collect, is_out_of_memory, main, out_of_memory_advice, pipeline


def parse(*argv: str) -> argparse.Namespace:
    """The options the command reads from `argv`."""
    return build_parser().parse_args(argv)


class TestPipeline:
    def test_the_imagenet_model_gets_only_the_options_given_and_its_own(self) -> None:
        generate, settings = pipeline(parse("a fox", "-n", "12", "-clip", "RN50", "-q"))
        assert generate is adm.generate
        assert settings == {
            "width_offset": 0,
            "height_offset": 0,
            "batch_size": 1,
            "clip_model": "RN50",
            "clip_fp16": False,
            "steps": 12,
            "save_frequency": 10,
            "progress": False,
        }

    def test_a_backbone_gets_its_own_options(self) -> None:
        generate, settings = pipeline(parse("a fox", "-bb", "vega", "-iters", "3", "-size", "512", "-aes", "0"))
        assert generate is backbone.generate
        assert settings["backbone"] == "vega"
        assert (settings["iterations"], settings["image_size"], settings["aesthetic_scale"]) == (3, 512, 0.0)
        assert settings["progress"] is True

    def test_an_animation_saves_every_frame_unless_told_otherwise(self) -> None:
        assert pipeline(parse("a fox", "-gif"))[1]["save_frequency"] == 1
        assert pipeline(parse("a fox", "-mp4", "-freq", "3"))[1]["save_frequency"] == 3

    @pytest.mark.parametrize(
        ("argv", "message"),
        [
            (("-bb", "sd-turbo", "-n", "4"), "--steps only applies without --backbone."),
            (("-bb", "sd-turbo", "-rate", "0.1", "-init", "x.png"), "--guidance-rate, --init-image only apply without"),
            (("-iters", "4"), "--iterations only applies with --backbone."),
        ],
    )
    def test_an_option_of_the_other_pipeline_is_explained(self, argv: tuple[str, ...], message: str) -> None:
        with pytest.raises(ConfigError, match=message):
            pipeline(parse("a fox", *argv))

    def test_the_short_and_long_spellings_are_the_same_option(self) -> None:
        short = parse("a", "-size", "64", "-wd", "8", "-ht", "16", "-bs", "2", "-seed", "3", "-clip", "RN50", "-fp16")
        long = parse(
            "a",
            "--image-size=64",
            "--width-offset=8",
            "--height-offset=16",
            "--batch-size=2",
            "--seed=3",
            "--clip-model=RN50",
            "--clip-fp16",
        )
        assert short == long


def test_collect_groups_each_images_frames_in_order() -> None:
    saved = [(0, Path("a/00/0000.png")), (1, Path("a/01/0000.png")), (0, Path("a/00/0005.png"))]
    assert collect(saved) == {0: [Path("a/00/0000.png"), Path("a/00/0005.png")], 1: [Path("a/01/0000.png")]}


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (("-q",), "at least one prompt"),
        (("a fox", "-best", "0"), "--candidates must be at least 1"),
        (("a fox", "-size", "100"), "No checkpoint for size 100"),
        (("a fox", "-clip", "ViT-Z/99"), "Unknown CLIP model"),
        (("a fox", "-dev", "gpu"), "Unknown device"),
        (("a fox", "-bb", "sd-turbo", "-size", "100"), "multiples of 8"),
        (("a fox", "-bb", "sd-turbo", "-skip", "10"), "--skip-timesteps only applies without --backbone"),
    ],
)
def test_a_bad_run_exits_with_its_reason_and_no_traceback(
    argv: tuple[str, ...], message: str, workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 2
    error = capsys.readouterr().err
    assert message in error
    assert "Traceback" not in error
    assert not (workdir / "outputs").exists()


def test_an_unknown_backbone_is_an_invalid_choice(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        parse("a fox", "-bb", "dall-e")
    assert "invalid choice" in capsys.readouterr().err


def test_both_of_pytorchs_memory_errors_count_as_out_of_memory() -> None:
    assert is_out_of_memory(th.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 14.00 MiB."))
    assert is_out_of_memory(th.AcceleratorError("CUDA error: out of memory", 2))
    assert not is_out_of_memory(RuntimeError("mat1 and mat2 shapes cannot be multiplied"))


@pytest.mark.parametrize(
    ("argv", "fp16_advice", "iterations_advice"),
    [
        (("a",), False, False),
        (("a", "-clip", "PE-Core-B-16"), True, False),
        (("a", "-clip", "PE-Core-B-16", "-fp16"), False, False),
        (("a", "-bb", "sd-turbo"), True, True),
    ],
)
def test_out_of_memory_advice_fits_the_run(argv: tuple[str, ...], fp16_advice: bool, iterations_advice: bool) -> None:
    advice = out_of_memory_advice(parse(*argv))
    assert advice.startswith("Ran out of GPU memory.")
    assert ("--clip-fp16" in advice) is fp16_advice
    assert ("--iterations" in advice) is iterations_advice

"""The command line: which pipeline its options choose, and how it explains a bad run. Nothing here loads a model."""

import argparse
from pathlib import Path

import pytest
import torch as th

from cgd import adm, backbone, finetuned
from cgd.checks import ConfigError
from cgd.cli import (
    build_parser,
    collect,
    is_out_of_memory,
    is_sequence,
    main,
    out_of_memory_advice,
    pipeline,
    watched,
)


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

    @pytest.mark.parametrize("backbone_argv", [(), ("-bb", "sd-turbo"), ("-uncond",)])
    def test_every_pipeline_gets_the_cutouts_asked_for_and_otherwise_picks_its_own(
        self, backbone_argv: tuple[str, ...]
    ) -> None:
        assert "num_cutouts" not in pipeline(parse("a fox", *backbone_argv))[1]
        assert pipeline(parse("a fox", "-cutn", "8", *backbone_argv))[1]["num_cutouts"] == 8

    def test_unconditional_is_passed_only_when_asked_for(self) -> None:
        assert "unconditional" not in pipeline(parse("a fox"))[1]
        generate, settings = pipeline(parse("a fox", "-uncond", "-size", "512"))
        assert generate is adm.generate
        assert (settings["unconditional"], settings["image_size"]) == (True, 512)

    def test_recurrence_is_passed_only_when_asked_for(self) -> None:
        """Unset, adm.generate picks it from the checkpoint."""
        assert "recurrence" not in pipeline(parse("a fox", "-uncond"))[1]
        assert pipeline(parse("a fox", "-recur", "3"))[1]["recurrence"] == 3

    def test_unconditional_alone_leaves_every_setting_to_its_defaults(self) -> None:
        _, settings = pipeline(parse("a fox", "-uncond", "-q"))
        assert settings == {
            "width_offset": 0,
            "height_offset": 0,
            "batch_size": 1,
            "clip_fp16": False,
            "unconditional": True,
            "save_frequency": 10,
            "progress": False,
        }

    def test_each_option_given_overrides_a_default_of_unconditional(self) -> None:
        argv = ("-clip", "RN50", "-cutn", "16", "-n", "30", "-recur", "1", "-thumb", "none", "-size", "512")
        _, settings = pipeline(parse("a fox", "-uncond", *argv))
        chosen = {name: settings[name] for name in adm.UNCONDITIONAL_DEFAULTS._fields}
        assert chosen == {
            "image_size": 512,
            "steps": 30,
            "recurrence": 1,
            "clip_model": "RN50",
            "num_cutouts": 16,
            "thumbnail": "none",
        }

    @pytest.mark.parametrize("name", ["clip", "glide", "laionide", "karlo", "none"])
    def test_a_thumbnail_model_is_passed_only_when_asked_for(self, name: str) -> None:
        assert "thumbnail" not in pipeline(parse("a fox", "-uncond"))[1]
        assert pipeline(parse("a fox", "-uncond", "-thumb", name))[1]["thumbnail"] == name

    def test_perturbed_attention_is_passed_only_when_asked_for(self) -> None:
        assert "perturbed_attention" not in pipeline(parse("a fox", "-uncond"))[1]
        assert pipeline(parse("a fox", "-uncond", "-pag"))[1]["perturbed_attention"] is True

    def test_a_fine_tune_is_passed_only_when_asked_for(self) -> None:
        assert "finetuned" not in pipeline(parse("a fox"))[1]
        generate, settings = pipeline(parse("-ft", "a fox"))
        assert generate is adm.generate
        assert settings["finetuned"] == str(finetuned.DEFAULT_PATH)  # The prompt is not taken for a path.
        assert "finetuned_path" not in settings
        assert pipeline(parse("a fox", "-ftpath", "cgd.pt"))[1]["finetuned"] == "cgd.pt"
        assert pipeline(parse("a fox", "-ft", "-ftpath", "cgd.pt"))[1]["finetuned"] == "cgd.pt"

    def test_an_animation_saves_every_frame_unless_told_otherwise(self) -> None:
        assert pipeline(parse("a fox", "-gif"))[1]["save_frequency"] == 1
        assert pipeline(parse("a fox", "-mp4", "-freq", "3"))[1]["save_frequency"] == 3

    @pytest.mark.parametrize("backbone_argv", [(), ("-bb", "sd-turbo"), ("-uncond",)])
    def test_every_pipeline_gets_the_transition_frames_asked_for_and_otherwise_picks_its_own(
        self, backbone_argv: tuple[str, ...]
    ) -> None:
        assert "transition_frames" not in pipeline(parse("a castle|>a fox", *backbone_argv))[1]
        assert pipeline(parse("a castle|>a fox", "-frames", "12", *backbone_argv))[1]["transition_frames"] == 12

    @pytest.mark.parametrize(
        ("argv", "message"),
        [
            (("-bb", "sd-turbo", "-n", "4"), "--steps only applies without --backbone."),
            (("-bb", "sd-turbo", "-uncond"), "--unconditional only applies without --backbone."),
            (("-bb", "sd-turbo", "-recur", "2"), "--recurrence only applies without --backbone."),
            (("-bb", "sd-turbo", "-thumb", "glide"), "--thumbnail only applies without --backbone."),
            (("-bb", "sd-turbo", "-pag"), "--perturbed-attention only applies without --backbone."),
            (("-bb", "sd-turbo", "-ft"), "--finetuned only applies without --backbone."),
            (("-bb", "sd-turbo", "-ftpath", "cgd.pt"), "--finetuned-path only applies without --backbone."),
            (("-bb", "sd-turbo", "-rate", "0.1", "-init", "x.png"), "--guidance-rate, --init-image only apply without"),
            (("-iters", "4"), "--iterations only applies with --backbone."),
        ],
    )
    def test_an_option_of_the_other_pipeline_is_explained(self, argv: tuple[str, ...], message: str) -> None:
        with pytest.raises(ConfigError, match=message):
            pipeline(parse("a fox", *argv))

    def test_the_short_and_long_spellings_are_the_same_option(self) -> None:
        short = parse(
            "a",
            "-size",
            "256",
            "-wd",
            "8",
            "-ht",
            "16",
            "-bs",
            "2",
            "-seed",
            "3",
            "-clip",
            "RN50",
            "-fp16",
            "-cutn",
            "4",
            "-frames",
            "12",
            "-uncond",
            "-recur",
            "3",
            "-thumb",
            "karlo",
            "-pag",
        )
        long = parse(
            "a",
            "--image-size=256",
            "--width-offset=8",
            "--height-offset=16",
            "--batch-size=2",
            "--seed=3",
            "--clip-model=RN50",
            "--clip-fp16",
            "--num-cutouts=4",
            "--transition-frames=12",
            "--unconditional",
            "--recurrence=3",
            "--thumbnail=karlo",
            "--perturbed-attention",
        )
        assert short == long


class TestSequence:
    @pytest.mark.parametrize(
        ("argv", "expected"),
        [
            (("a castle|>a fox",), True),
            (("a castle|>a fox|>",), True),
            (("a castle|>",), False),
            (("a castle|a fox",), False),
            (("a fox", "-imgs", "castle.png|>fox.png"), True),
            (("",), False),
        ],
    )
    def test_a_sequence_is_several_images_of_text_or_of_image_prompts(
        self, argv: tuple[str, ...], expected: bool
    ) -> None:
        assert is_sequence(parse(*argv)) is expected

    def test_a_sequence_is_made_into_a_gif_unless_it_asks_for_a_video(self) -> None:
        assert watched(parse("a castle|>a fox")).gif is True
        assert watched(parse("a castle|>a fox", "-mp4")).gif is False
        assert watched(parse("a castle|>a fox", "-mp4", "-gif")).gif is True
        assert watched(parse("a fox")).gif is False

    def test_so_it_saves_every_frame_of_its_first_image(self) -> None:
        assert pipeline(watched(parse("a castle|>a fox")))[1]["save_frequency"] == 1
        assert pipeline(watched(parse("a castle|>a fox", "-freq", "3")))[1]["save_frequency"] == 3


def test_collect_groups_each_images_frames_in_order() -> None:
    saved = [(0, Path("a/00/0000.png")), (1, Path("a/01/0000.png")), (0, Path("a/00/0005.png"))]
    assert collect(saved) == {0: [Path("a/00/0000.png"), Path("a/00/0005.png")], 1: [Path("a/01/0000.png")]}


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (("-q",), "at least one prompt"),
        (("a fox", "-best", "0"), "--candidates must be at least 1"),
        (("a castle|>",), "takes another image to come back from"),
        (("a fox", "-size", "100"), "No checkpoint for size 100"),
        (("a fox", "-uncond", "-size", "128"), "No unconditional checkpoint for size 128"),
        (("a fox", "-thumb", "glide"), "--thumbnail only applies with --unconditional"),
        (("a fox", "-uncond", "-wd", "32", "-thumb", "clip"), "--thumbnail needs a square image"),
        (("a fox", "-pag"), "--perturbed-attention only applies with --unconditional at 256"),
        (
            ("a fox", "-uncond", "-size", "512", "-pag"),
            "--perturbed-attention only applies with --unconditional at 256",
        ),
        (("a fox", "-cutn", "0"), "--num-cutouts must be at least 1"),
        (("a fox", "-bb", "sd-turbo", "-cutn", "0"), "--num-cutouts must be at least 1"),
        (("a fox", "-clip", "ViT-Z/99"), "Unknown CLIP model"),
        (("a fox", "-dev", "gpu"), "Unknown device"),
        (("a fox", "-bb", "sd-turbo", "-size", "100"), "multiples of 8"),
        (("a fox", "-bb", "sd-turbo", "-skip", "10"), "--skip-timesteps only applies without --backbone"),
        (("a castle|>a fox", "-best", "2"), "--candidates keeps the best of several images"),
        (("a fox", "-frames", "12"), "--transition-frames needs a sequence"),
        (("a fox", "-bb", "sd-turbo", "-frames", "12"), "--transition-frames needs a sequence"),
        (("a castle|>a fox", "-frames", "0"), "--transition-frames must be at least 1"),
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


@pytest.mark.parametrize("argv", [("-bb", "dall-e"), ("-uncond", "-thumb", "dall-e")])
def test_an_unknown_backbone_or_thumbnail_model_is_an_invalid_choice(
    argv: tuple[str, ...], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        parse("a fox", *argv)
    assert "invalid choice" in capsys.readouterr().err


def test_both_of_pytorchs_memory_errors_count_as_out_of_memory() -> None:
    assert is_out_of_memory(th.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 14.00 MiB."))
    assert is_out_of_memory(th.AcceleratorError("CUDA error: out of memory", 2))
    assert not is_out_of_memory(RuntimeError("mat1 and mat2 shapes cannot be multiplied"))


@pytest.mark.parametrize(
    ("argv", "cutouts", "fp16_advice", "iterations_advice", "thumbnail_advice"),
    [
        (("a",), 16, False, False, False),
        (("a", "-clip", "PE-Core-B-16"), 16, True, False, False),
        (("a", "-clip", "PE-Core-B-16", "-fp16"), 16, False, False, False),
        (("a", "-bb", "sd-turbo"), 16, True, True, False),
        (("a", "-bb", "sd-turbo", "-cutn", "2"), 2, True, True, False),
        (("a", "-uncond"), 4, True, False, False),
        (("a", "-uncond", "-cutn", "8", "-clip", "ViT-B/32", "-thumb", "none"), 8, False, False, False),
        (("a", "-uncond", "-thumb", "karlo"), 4, True, False, True),
        (("a", "-ft"), 16, True, False, False),
    ],
)
def test_out_of_memory_advice_fits_the_run(
    argv: tuple[str, ...], cutouts: int, fp16_advice: bool, iterations_advice: bool, thumbnail_advice: bool
) -> None:
    advice = out_of_memory_advice(parse(*argv))
    assert advice.startswith("Ran out of GPU memory.")
    assert f"--num-cutouts/-cutn (currently {cutouts})" in advice
    assert ("--clip-fp16" in advice) is fp16_advice
    assert ("--iterations" in advice) is iterations_advice
    assert ("--thumbnail/-thumb clip or none needs less than karlo." in advice) is thumbnail_advice

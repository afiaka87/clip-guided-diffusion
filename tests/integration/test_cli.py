"""The cgd command, end to end: in this process where that is enough, in a new one where the process itself matters."""

import os
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from PIL import Image

from cgd.cli import judged_candidates, main

pytestmark = pytest.mark.cuda

PROMPT = "Loose seal."
FAST = ["-size", "64", "-n", "6", "-clip", "RN50", "-q"]  # The quickest run there is: six calls of the 64px model.
REPO = Path(__file__).resolve().parents[2]
COUNT_FRAMES = "ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0"


def frames(workdir: Path, name: str = "Loose_seal") -> list[str]:
    """The names of the frames left of the first image of the batch."""
    return sorted(path.name for path in (workdir / "outputs" / name / "00").glob("*.png"))


def run(*argv: str, capsys: pytest.CaptureFixture[str]) -> str:
    """Run the command in this process, and return what it printed."""
    main(argv)
    return capsys.readouterr().out


def run_process(*argv: str, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run `python -m cgd` in a new process."""
    environment = {**os.environ, "PYTHONPATH": str(REPO), **(env or {})}
    command = [sys.executable, "-m", "cgd", *argv]
    return subprocess.run(command, cwd=cwd, env=environment, capture_output=True, text=True, check=False)


class TestProcess:
    def test_it_runs_as_a_module_and_reports_the_saved_image(self, workdir: Path) -> None:
        result = run_process(PROMPT, *FAST, cwd=workdir)
        assert result.returncode == 0, result.stderr
        assert "Saved outputs/Loose_seal/00/0005.png" in result.stdout
        assert (workdir / "current.png").exists()

    def test_the_installed_script_runs(self) -> None:
        result = subprocess.run(
            [Path(sys.executable).parent / "cgd", "-h"], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.startswith("usage: cgd")

    def test_an_open_clip_model_runs_without_noise_from_its_libraries(self, workdir: Path) -> None:
        result = run_process(PROMPT, *FAST, "-clip", "PE-Core-B-16", "-fp16", cwd=workdir)
        assert result.returncode == 0, result.stderr
        assert "Saved outputs/Loose_seal/00/0005.png" in result.stdout
        assert "HF Hub" not in result.stderr

    @staticmethod
    def without_allocator_setting() -> dict[str, str]:
        """This environment without either name of the allocator setting."""
        return {
            name: value
            for name, value in os.environ.items()
            if name not in {"PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"}
        }

    @pytest.mark.parametrize("name", ["PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"])
    def test_the_allocator_setting_leaves_the_users_own_alone(self, name: str) -> None:
        code = "import os, cgd; print(os.environ.get('PYTORCH_ALLOC_CONF'), os.environ.get('PYTORCH_CUDA_ALLOC_CONF'))"
        own = "max_split_size_mb:64"
        environment = {**self.without_allocator_setting(), name: own, "PYTHONPATH": str(REPO)}
        result = subprocess.run(
            [sys.executable, "-c", code], env=environment, capture_output=True, text=True, check=False
        )
        expected = f"{own} None" if name == "PYTORCH_ALLOC_CONF" else f"None {own}"
        assert result.stdout.strip() == expected, result.stderr

    def test_the_allocator_setting_reaches_the_allocator_without_a_warning(self) -> None:
        # Only expandable segments can serve the third block from the memory of the first two.
        code = (
            "import cgd, torch as th\n"
            "first, second = (th.empty(100 << 20, dtype=th.uint8, device='cuda') for _ in range(2))\n"
            "del first, second\n"
            "third = th.empty(200 << 20, dtype=th.uint8, device='cuda')\n"
            "print(th.cuda.memory_reserved() >> 20)\n"
        )
        environment = {**self.without_allocator_setting(), "PYTHONPATH": str(REPO)}
        result = subprocess.run(
            [sys.executable, "-c", code], env=environment, capture_output=True, text=True, check=False
        )
        assert result.stdout.strip() == "200", result.stderr
        assert "deprecated" not in result.stderr


class TestErrors:
    def test_weights_that_cancel_out_are_explained(self, workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exit_info:
            main(["cat:1|dog:-1", *FAST])
        assert exit_info.value.code == 2
        assert "must not sum to 0" in capsys.readouterr().err
        assert not (workdir / "outputs").exists()

    def test_running_out_of_memory_fails_with_advice(self, workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exit_info:
            main([PROMPT, *FAST, "-bs", "4096"])
        assert exit_info.value.code == 1
        error = capsys.readouterr().err
        assert error.startswith("Ran out of GPU memory.")
        assert "Traceback" not in error
        assert workdir.exists()


class TestOutputs:
    def test_the_output_folder_can_be_new_and_nested(self, workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
        assert "Saved a/b/c/Loose_seal/00/0005.png" in run(PROMPT, *FAST, "-dir", "a/b/c", capsys=capsys)
        assert (workdir / "a/b/c/Loose_seal/00/0005.png").exists()

    def test_a_run_from_an_image_prompt_is_named_after_it(
        self, workdir: Path, photon: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        run("-imgs", f"{photon}:2", *FAST, capsys=capsys)
        assert frames(workdir, "photon") == ["0000.png", "0005.png"]

    def test_a_gif_holds_every_saved_frame_and_leaves_other_runs_frames(
        self, workdir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        run(PROMPT, *FAST, "-n", "8", "-freq", "1", capsys=capsys)
        assert len(frames(workdir)) == 8
        assert "Created outputs/Loose_seal/00.gif" in run(PROMPT, *FAST, "-freq", "2", "-gif", capsys=capsys)
        gif = workdir / "outputs/Loose_seal/00.gif"
        count = subprocess.run([*COUNT_FRAMES.split(), str(gif)], capture_output=True, text=True, check=True)
        assert count.stdout.strip() == "4"  # Steps 0, 2, 4 and the final step 5.
        assert frames(workdir) == ["0001.png", "0003.png", "0006.png", "0007.png"]  # The earlier run's.

    def test_a_video_replaces_the_frames(self, workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
        run(PROMPT, *FAST, "-mp4", capsys=capsys)
        assert (workdir / "outputs/Loose_seal/00.mp4").stat().st_size > 0
        assert not (workdir / "outputs/Loose_seal/00").exists()

    def test_the_frames_survive_without_ffmpeg(
        self, workdir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        empty = workdir / "empty"
        empty.mkdir()
        monkeypatch.setenv("PATH", str(empty))
        output = run(PROMPT, *FAST, "-gif", capsys=capsys)
        assert "ffmpeg not found" in output
        assert "Saved outputs/Loose_seal/00/0005.png" in output
        assert len(frames(workdir)) == 6

    def test_upscaling_saves_a_larger_copy_beside_the_image(
        self, workdir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert "Saved outputs/Loose_seal/00/0005_x4.png" in run(PROMPT, *FAST, "-up", capsys=capsys)
        assert Image.open(workdir / "outputs/Loose_seal/00/0005_x4.png").size == (256, 256)

    def test_the_upscaled_copy_outlives_an_animation(self, workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
        run(PROMPT, *FAST, "-up", "-gif", capsys=capsys)
        assert frames(workdir) == ["0005_x4.png"]

    def test_candidates_report_the_winning_seed(self, workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
        output = run(PROMPT, *FAST, "-best", "2", "-seed", "7", capsys=capsys)
        assert re.search(r"Saved outputs/seed_[78]/Loose_seal/00/0005.png", output)
        assert (workdir / "outputs/seed_7").is_dir()
        assert (workdir / "outputs/seed_8").is_dir()

    def test_a_backbone_draws_at_any_multiple_of_eight(self, workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
        output = run(PROMPT, "-bb", "sd-turbo", "-size", "136", "-iters", "2", "-clip", "ViT-B/32", "-q", capsys=capsys)
        assert "Saved outputs/Loose_seal/00/0002.png" in output
        assert Image.open(workdir / "outputs/Loose_seal/00/0002.png").size == (136, 136)


def test_candidates_keep_the_image_the_judges_prefer(tmp_path: Path, goose: Path) -> None:
    def fake_run(seed: int, output_dir: Path) -> Iterator[tuple[int, Path]]:
        """Seed 1 draws a goose, the others a gray square."""
        output_dir.mkdir(parents=True)
        path = output_dir / "0000.png"
        image = (
            Image.open(goose).convert("RGB").resize((128, 128)) if seed == 1 else Image.new("RGB", (128, 128), "gray")
        )
        image.save(path)
        yield 0, path

    winner = judged_candidates(
        fake_run, 3, seed=0, output_dir=tmp_path, prompts=["a photograph of a goose"], image_prompts=[]
    )
    assert winner == {0: [tmp_path / "seed_1" / "0000.png"]}

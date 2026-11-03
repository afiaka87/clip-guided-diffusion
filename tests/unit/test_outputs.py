"""Where images are saved, and the GIFs and videos made from them, through a real ffmpeg."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import torch as th
import torchvision.transforms.functional as tvf
from PIL import Image

from cgd.outputs import MAX_NAME_LENGTH, frame_path, run_name, save_frame, save_frames, save_gif, save_video

needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="needs ffmpeg")


class TestNames:
    def test_a_run_is_named_after_its_text_prompts(self) -> None:
        assert run_name(["Loose seal.", "a red fox"]) == "Loose_seal_a_red_fox"

    def test_without_text_it_is_named_after_its_images(self) -> None:
        assert run_name([], ["images/photon.png:2", "https://example.com/cat.jpg"]) == "photon_cat"

    def test_a_name_fits_in_a_file_system(self) -> None:
        assert len(run_name(["a" * 1000])) == MAX_NAME_LENGTH

    def test_frames_go_in_a_folder_per_image_of_the_batch(self) -> None:
        assert frame_path("outputs", "a_fox", 42, 3) == Path("outputs/a_fox/03/0042.png")


def test_save_frame_writes_the_image_and_current_png(workdir: Path) -> None:
    image = th.linspace(-1, 1, 3 * 4 * 4).view(3, 4, 4)
    path = save_frame(image, workdir / "a" / "b" / "0001.png")
    saved = tvf.to_tensor(Image.open(path)).mul(2).sub(1)
    assert (saved - image).abs().max().item() <= 2 / 255  # One 8 bit step, in [-1, 1].
    assert (workdir / "current.png").read_bytes() == path.read_bytes()


def test_save_frames_keeps_every_nth_step_and_the_final_one(workdir: Path) -> None:
    batches = (th.full((2, 3, 4, 4), step / 10 - 0.5) for step in range(8))
    saved = list(save_frames(batches, workdir, "run", save_frequency=3, final_step=7))
    assert saved == [(index, frame_path(workdir, "run", step, index)) for step in (0, 3, 6, 7) for index in (0, 1)]
    assert tvf.to_tensor(Image.open(saved[-1][1])).mean().item() == pytest.approx(0.6, abs=1 / 255)


@needs_ffmpeg
class TestAnimations:
    @pytest.fixture
    def frames(self, tmp_path: Path) -> list[Path]:
        """Three frames of a size no codec takes as it is, so the video has to pad them."""
        paths = [tmp_path / f"{index}.png" for index in range(3)]
        for path, color in zip(paths, ("red", "green", "blue"), strict=True):
            Image.new("RGB", (17, 19), color).save(path)
        return paths

    @staticmethod
    def probe(path: Path) -> dict[str, str | int]:
        """What ffprobe says of the first video stream, its frames counted by decoding them."""
        entries = "stream=codec_name,pix_fmt,width,height,nb_read_frames"
        command = ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", entries]
        result = subprocess.run([*command, "-of", "json", str(path)], check=True, capture_output=True, text=True)
        return json.loads(result.stdout)["streams"][0]

    def test_a_video_holds_every_frame_and_pads_odd_dimensions(self, frames: list[Path], tmp_path: Path) -> None:
        output = tmp_path / "animation.mp4"
        assert save_video(frames, output) == output
        stream = self.probe(output)
        assert (stream["codec_name"], stream["pix_fmt"]) == ("h264", "yuv420p")
        assert (stream["width"], stream["height"]) == (18, 20)
        assert stream["nb_read_frames"] == "3"
        assert all(frame.exists() for frame in frames)
        assert not list(tmp_path.glob(".*.tmp.mp4"))

    def test_a_gif_holds_every_frame(self, frames: list[Path], tmp_path: Path) -> None:
        output = tmp_path / "animation.gif"
        assert save_gif(frames, output) == output
        assert self.probe(output)["nb_read_frames"] == "3"

    def test_a_failed_encode_keeps_the_old_output_and_the_frames(self, tmp_path: Path) -> None:
        output, invalid = tmp_path / "animation.mp4", tmp_path / "invalid.png"
        output.write_bytes(b"previous export")
        invalid.write_bytes(b"not a PNG")
        assert save_video([invalid], output) is None
        assert output.read_bytes() == b"previous export"
        assert invalid.exists()
        assert not list(tmp_path.glob(".*.tmp.mp4"))

    def test_without_ffmpeg_it_explains_and_keeps_the_frames(
        self, frames: list[Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.setenv("PATH", str(empty))
        assert save_video(frames, tmp_path / "out.mp4") is None
        assert "ffmpeg not found" in capsys.readouterr().out
        assert all(frame.exists() for frame in frames)

    @pytest.mark.parametrize(("count", "fps"), [(0, 10), (3, 0)])
    def test_no_frames_or_no_frame_rate_is_an_error(
        self, frames: list[Path], tmp_path: Path, count: int, fps: int
    ) -> None:
        with pytest.raises(ValueError, match="at least one frame"):
            save_video(frames[:count], tmp_path / "out.mp4", fps=fps)

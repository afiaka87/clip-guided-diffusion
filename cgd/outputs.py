"""Where a run's images go, and the GIFs and videos made from them."""

import os
import re
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import torch as th
import torchvision.transforms.functional as tvf

from cgd.prompts import parse_prompt

MAX_NAME_LENGTH = 255
DEFAULT_FPS = 10
ENCODE_TIMEOUT = 120  # Seconds ffmpeg may take after the last frame.
ERROR_TAIL = 4096  # How much of ffmpeg's error output to show.
# fmt: off
GIF_ARGS = (
    "-filter_complex",
    "split[frames][copy];[copy]palettegen=max_colors=256:stats_mode=full[palette];"
    "[frames][palette]paletteuse=dither=floyd_steinberg:diff_mode=rectangle",
    "-loop", "0",
)
VIDEO_ARGS = (
    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # H.264 in yuv420p needs even dimensions.
    "-c:v", "libx264", "-preset", "slow", "-crf", "18",
    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
)
# fmt: on


def run_name(prompts: Sequence[str], image_prompts: Sequence[str] = ()) -> str:
    """The folder a run's images go in: its text prompts, or else its image prompts' file names."""
    names = list(prompts) or [Path(parse_prompt(image)[0]).stem for image in image_prompts]
    return "_".join(re.sub(r"[^\w\s]", "", name).replace(" ", "_") for name in names)[:MAX_NAME_LENGTH]


def frame_path(output_dir: str | Path, name: str, step: int, batch_index: int) -> Path:
    """Where the image of `step` goes, for one image of the batch."""
    return Path(output_dir) / name / f"{batch_index:02}" / f"{step:04}.png"


def save_frame(image: th.Tensor, path: Path) -> Path:
    """Save an image in [-1, 1] to `path`, and copy it to current.png in the working directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tvf.to_pil_image(image.add(1).div(2).clamp(0, 1)).save(path)
    shutil.copyfile(path, Path.cwd() / "current.png")
    return path


def save_frames(
    batches: Iterable[th.Tensor], output_dir: str | Path, name: str, save_frequency: int, final_step: int
) -> Iterator[tuple[int, Path]]:
    """Save every `save_frequency`th batch of images and the one at `final_step`, yielding (batch index, path)."""
    for step, images in enumerate(batches):
        if step % save_frequency == 0 or step == final_step:
            for index, image in enumerate(images):
                yield index, save_frame(image, frame_path(output_dir, name, step, index))


def save_gif(frames: Sequence[str | Path], output: str | Path, fps: int = DEFAULT_FPS) -> Path | None:
    """A GIF of the frames, with a palette made from the frames themselves."""
    return _encode(frames, output, fps, GIF_ARGS)


def save_video(frames: Sequence[str | Path], output: str | Path, fps: int = DEFAULT_FPS) -> Path | None:
    """An H.264 MP4 of the frames."""
    return _encode(frames, output, fps, VIDEO_ARGS)


def _encode(frames: Sequence[str | Path], output: str | Path, fps: int, encoder_args: Sequence[str]) -> Path | None:
    """Stream the frames through ffmpeg into `output`, which is only replaced once that worked.

    Returns the path of the output, or None after printing why ffmpeg failed. A failure
    leaves the frames and any earlier output as they were.
    """
    if not frames or fps <= 0:
        raise ValueError("Export requires at least one frame and a positive frame rate")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{uuid.uuid4().hex}.tmp{destination.suffix}")
    command = ["ffmpeg", "-y", "-f", "image2pipe", "-framerate", str(fps), "-i", "-", *encoder_args, str(temporary)]
    try:
        error = _run_ffmpeg(command, frames)
        if error is None and (not temporary.is_file() or temporary.stat().st_size == 0):
            error = "ffmpeg did not produce an output file."
        if error is not None:
            print(error)
            return None
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Created {destination}")
    return destination


def _run_ffmpeg(command: Sequence[str], frames: Sequence[str | Path]) -> str | None:
    """Pipe the frames to ffmpeg. None if it succeeded, or else what went wrong."""
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        with (
            tempfile.TemporaryFile() as errors,
            subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors, creationflags=creation_flags
            ) as process,
        ):
            try:
                with process.stdin as stream:
                    for frame in frames:
                        with Path(frame).open("rb") as source:
                            shutil.copyfileobj(source, stream, length=64 * 1024)
                code = process.wait(timeout=ENCODE_TIMEOUT)
            except BaseException:
                process.kill()
                raise
            if code:
                errors.seek(max(0, errors.seek(0, os.SEEK_END) - ERROR_TAIL))
                return f"ffmpeg error: {errors.read().decode(errors='replace')}"
    except FileNotFoundError as error:
        return f"ffmpeg not found or an input frame was missing: {error}"
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"ffmpeg export failed: {error}"
    return None

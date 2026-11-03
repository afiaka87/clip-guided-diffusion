"""Downloading checkpoints into the cache, and reading images from files or URLs."""

import hashlib
import io
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import requests
from tqdm.auto import tqdm

CACHE_PATH = Path.home() / ".cache" / "clip-guided-diffusion"
CHUNK_SIZE = 64 * 1024
TIMEOUT = (30, 120)  # Connect, then read. A slow connection can stall near the end of a file.
ATTEMPTS = 3
IMAGE_TIMEOUT = 30

T = TypeVar("T")


def is_url(path: str | Path) -> bool:
    """Whether `path` is an http or https URL rather than a file."""
    return str(path).startswith(("http://", "https://"))


def fetch(url_or_path: str | Path) -> io.BytesIO:
    """The bytes of an image prompt or init image, from a URL or a file."""
    if is_url(url_or_path):
        response = requests.get(str(url_or_path), timeout=IMAGE_TIMEOUT)
        response.raise_for_status()
        return io.BytesIO(response.content)
    return io.BytesIO(Path(url_or_path).read_bytes())


def download(url: str, filename: str, root: Path = CACHE_PATH) -> Path:
    """The cached copy of `url` in `root`, downloaded first if it is not there yet."""
    target = Path(root) / filename
    if target.exists() and not target.is_file():
        raise RuntimeError(f"{target} exists and is not a regular file")
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        return retry(lambda: _download_to(url, target), ATTEMPTS)
    except (requests.RequestException, RuntimeError) as error:
        raise RuntimeError(f"Download failed after {ATTEMPTS} attempts: {error}") from error


def download_verified(url: str, filename: str, sha256: str, root: Path = CACHE_PATH) -> Path:
    """download(), refusing a file that is not the one these results were measured with."""
    path = download(url, filename, root)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha256:
        raise RuntimeError(f"{path} has sha256 {digest}, expected {sha256}. Delete it to download it again.")
    return path


def retry(action: Callable[[], T], attempts: int, sleep: Callable[[float], object] = time.sleep) -> T:
    """`action()`, tried up to `attempts` times, waiting 1s, 2s, 4s... between them. The last failure raises."""
    for attempt in range(1, attempts):
        try:
            return action()
        except (requests.RequestException, RuntimeError) as error:
            delay = 2 ** (attempt - 1)
            print(f"Download failed (attempt {attempt}/{attempts}): {error}\nRetrying in {delay}s...")
            sleep(delay)
    return action()


def _download_to(url: str, target: Path) -> Path:
    """Stream `url` to a temporary file, then move it to `target` once it is complete."""
    partial = target.with_suffix(".tmp")
    try:
        with requests.get(url, stream=True, timeout=TIMEOUT) as response:
            response.raise_for_status()
            expected = int(response.headers.get("Content-Length", 0))
            with (
                partial.open("wb") as output,
                tqdm(total=expected, ncols=80, unit="B", unit_scale=True, unit_divisor=1024) as progress_bar,
            ):
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    output.write(chunk)
                    progress_bar.update(len(chunk))
                output.flush()
                os.fsync(output.fileno())
        size = partial.stat().st_size
        if expected and size != expected:
            raise RuntimeError(f"Download incomplete: expected {expected} bytes, got {size} bytes")
        partial.rename(target)
        return target
    finally:
        partial.unlink(missing_ok=True)

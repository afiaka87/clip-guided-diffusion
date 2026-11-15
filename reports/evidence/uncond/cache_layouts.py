"""Shrink webdataset shards to 64x64 thumbnails with their captions, for training a layout prior quickly.

    uv run python reports/evidence/uncond/cache_layouts.py 'D:/laion2b-en-aesthetic/*.tar' D:/laion-cache

Decoding 5.6M JPEGs every epoch would be the bottleneck of a model that small, so this does
it once, on the CPU alone. Each shard becomes OUT/SHARD.npy, uint8 [n, 64, 64, 3], and
OUT/SHARD.txt, a caption per line in the same order. It keeps the images
train_conditioning.py keeps: not likely unsafe, not likely watermarked. Shards already done
are skipped, so it resumes.
"""

import glob
import io
import json
import sys
import tarfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

SIDE = 64
MAX_UNSAFE, MAX_WATERMARK = 0.1, 0.5  # train_conditioning.py's defaults.


def thumbnail(jpeg: bytes) -> np.ndarray:
    """The centre square of a JPEG, averaged down to SIDE pixels a side."""
    image = Image.open(io.BytesIO(jpeg))
    image.draft("RGB", (2 * SIDE, 2 * SIDE))  # Decode at a fraction of the size, never below twice SIDE.
    image = image.convert("RGB")
    width, height = image.size
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    image = image.resize((SIDE, SIDE), Image.LANCZOS, box=(left, top, left + side, top + side), reducing_gap=None)
    return np.asarray(image, dtype=np.uint8)


def samples(path: str):
    """(jpeg, caption) of each image of a shard that passes the filters."""
    group: dict[str, bytes] = {}
    key = None

    def complete():
        if "jpg" not in group or "txt" not in group:
            return
        meta = json.loads(group.get("json", b"{}"))
        if (meta.get("punsafe") or 0) > MAX_UNSAFE or (meta.get("pwatermark") or 0) > MAX_WATERMARK:
            return
        yield group["jpg"], " ".join(group["txt"].decode("utf-8", "replace").split())

    with tarfile.open(path, "r|") as tar:
        for member in tar:
            if not member.isfile():
                continue
            stem, _, extension = member.name.rpartition(".")
            if stem != key:
                yield from complete()
                group, key = {}, stem
            group[extension] = tar.extractfile(member).read()
    yield from complete()


def convert(job: tuple[str, str]) -> tuple[str, int]:
    path, out = job
    stem = Path(path).stem
    images_path, captions_path = Path(out) / f"{stem}.npy", Path(out) / f"{stem}.txt"
    if images_path.exists() and captions_path.exists():
        return stem, -1
    images, captions = [], []
    try:
        for jpeg, caption in samples(path):
            try:
                images.append(thumbnail(jpeg))
            except (OSError, ValueError):
                continue
            captions.append(caption)
    except (tarfile.TarError, OSError) as error:
        print(f"Stopped reading {path}: {error}", flush=True)
    if not images:
        return stem, 0
    partial = images_path.with_suffix(".partial")
    with partial.open("wb") as file:
        np.save(file, np.stack(images))
    captions_path.write_text("\n".join(captions) + "\n", encoding="utf-8")
    partial.replace(images_path)  # Last, so a shard with both files is complete.
    return stem, len(images)


def main() -> None:
    paths, out = sorted(glob.glob(sys.argv[1])), sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    Path(out).mkdir(parents=True, exist_ok=True)
    total = 0
    with Pool(workers) as pool:
        for done, (stem, count) in enumerate(pool.imap_unordered(convert, [(p, out) for p in paths]), 1):
            total += max(count, 0)
            if done % 25 == 0 or done == len(paths):
                print(f"{done}/{len(paths)} shards, {total} new images, last {stem}", flush=True)


if __name__ == "__main__":
    main()

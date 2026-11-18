r"""Step 1: one table per dataset, a row per image, so the loader can filter on metadata and seek to any image.

    uv run python -m finetune.index                  # The LAION best tier and the ImageNet train split
    uv run python -m finetune.index F:/some/dataset  # Any folder of NNNNN.tar shards with jpg, json and txt members

It writes `finetune/index.parquet` into each dataset's folder. A row holds the shard, the sample's key, where its JPEG's
bytes are in the tar, and every scalar of its JSON: LAION's `aesthetic`, `punsafe`, `pwatermark`, `similarity`,
`url` and `alt_text`, ImageNet's `label`, `class_name` and `wnid`, the generated caption, and the filter's
scores (`quality`, `wm_yolo`, `colorfulness` and the rest). Rows are in shard order, then the tar's order, and
that order is the one `finetune.embed` writes its embeddings in.

It only reads tar headers and JSON, so it needs no GPU and takes a few minutes.
"""

import argparse
import json
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from finetune.data import IMAGENET_ROOT, INDEX_NAME, LAION_ROOT

# What the JSON repeats from elsewhere, or that says nothing about a sample that passed the filters.
DROPPED = frozenset(
    ("key", "status", "error_message", "exif", "WIDTH", "HEIGHT", "width", "height", "sha256", "caption_model")
)
RENAMED = {"caption": "alt_text"}  # LAION's caption is the alt text. `caption` here means the generated one.


def flatten(metadata: dict) -> dict:
    """A sample's JSON as one flat row: the filter's scores join the rest, and repeats are dropped."""
    row = {RENAMED.get(name, name): value for name, value in metadata.items() if name not in DROPPED}
    scores = row.pop("cgd_filter", {})
    return {**row, **scores}


def scan_shard(path: Path) -> list[dict]:
    """A row per sample of one shard, in the tar's order."""
    rows: dict[str, dict] = {}
    with tarfile.open(path, "r:") as tar:
        for member in tar:
            key, _, extension = member.name.rpartition(".")
            row = rows.setdefault(key, {"shard": int(path.stem), "key": key})
            if extension == "jpg":
                row["offset"], row["size"] = member.offset_data, member.size
            elif extension == "json":
                row.update(flatten(json.load(tar.extractfile(member))))
    return list(rows.values())


def build_index(root: Path, workers: int = 16) -> pd.DataFrame:
    """Scan every shard of `root` and write its index."""
    shards = sorted(root.glob("[0-9]*.tar"))
    if not shards:
        raise SystemExit(f"No shards in {root}.")
    with ProcessPoolExecutor(workers) as pool:
        index = pd.DataFrame([row for rows in pool.map(scan_shard, shards) for row in rows])
    missing = int(index["offset"].isna().sum())
    if missing:
        raise SystemExit(f"{missing} samples of {root} have no jpg.")
    index = index.astype({"offset": "int64", "size": "int64"})
    partial = root / (INDEX_NAME + ".tmp")
    partial.parent.mkdir(exist_ok=True)
    index.to_parquet(partial)
    partial.replace(root / INDEX_NAME)
    return index


def main() -> None:
    """Index the datasets named on the command line."""
    parser = argparse.ArgumentParser(description="Index datasets of webdataset shards for the fine-tune's loader.")
    parser.add_argument("roots", nargs="*", type=Path, default=[LAION_ROOT, IMAGENET_ROOT])
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    for root in args.roots:
        index = build_index(root, args.workers)
        print(f"{root}: {len(index)} images in {index['shard'].nunique()} shards, columns {list(index.columns)}")


if __name__ == "__main__":
    main()

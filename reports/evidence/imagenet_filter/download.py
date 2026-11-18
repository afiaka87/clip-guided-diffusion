# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub>=1.0", "hf_xet"]
# ///
"""Download ILSVRC 2012 (ImageNet-1k) as webdataset shards from timm/imagenet-1k-wds on Hugging Face.

    uv run reports/evidence/imagenet_filter/download.py              # train and validation to F:/imagenet-1k-wds
    uv run reports/evidence/imagenet_filter/download.py --sha256     # also hash every file against the repo

The repository is gated: accept ImageNet's terms on its page and log in (`hf auth login`) first. Run it again to
resume: files already complete are skipped, and a file cut off part way continues where it stopped. Each sample is
`KEY.jpg` (the original JPEG), `KEY.cls` (the class index) and `KEY.json` (label, width, height, original filename).
"""

import argparse
import hashlib
import os
import shutil
import sys
import time
from pathlib import Path

# Plain HTTPS from Hugging Face's CDN, not the Xet protocol. Measured on 2026-11-15: Xet stalled on its first shard,
# with 68 MB of 108 in ten minutes, while HTTPS gave about 1 MB/s for one shard and about 30 MB/s for eight at once.
# Read when huggingface_hub is imported; set HF_HUB_DISABLE_XET=0 to try Xet.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import HfApi, auth_check, snapshot_download

REPO = "timm/imagenet-1k-wds"
SHARDS = {"train": 1024, "validation": 64}  # The card says 63 validation shards; the repository holds 64.
META = ["README.md", "_info.json"]
ATTEMPTS = 20


def expected_files(api, splits):
    """The repository's files for these splits, with their sizes and hashes."""
    wanted = {f"imagenet1k-{s}-" for s in splits}
    files = [f for f in api.list_repo_tree(REPO, repo_type="dataset") if hasattr(f, "size")]
    files = [f for f in files if f.path in META or any(f.path.startswith(w) for w in wanted)]
    for s in splits:
        n = sum(f.path.startswith(f"imagenet1k-{s}-") for f in files)
        if n != SHARDS[s]:
            sys.exit(f"The repository lists {n} {s} shards, not {SHARDS[s]}. Check it before downloading.")
    return files


def missing(files, dest):
    """Files not yet on disk at their full size."""
    return [f for f in files if not (dest / f.path).is_file() or (dest / f.path).stat().st_size != f.size]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(2**24):
            h.update(chunk)
    return h.hexdigest()


def download(files, dest, splits, workers):
    """snapshot_download, retried: each retry resumes from what is on disk."""
    patterns = META + [f"imagenet1k-{s}-*.tar" for s in splits]
    for attempt in range(1, ATTEMPTS + 1):
        try:
            snapshot_download(REPO, repo_type="dataset", local_dir=dest, allow_patterns=patterns, max_workers=workers)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:  # A dropped connection or a server error.
            print(f"attempt {attempt} of {ATTEMPTS} failed: {type(e).__name__}: {e}", flush=True)
            time.sleep(min(300, 15 * attempt))
            continue
        if not missing(files, dest):
            return


def check_hashes(files, dest):
    """Hash every file against the repository, and set aside any that differ so a rerun fetches them again."""
    bad = []
    for i, f in enumerate(files, 1):
        if f.lfs is not None and sha256(dest / f.path) != f.lfs.sha256:
            bad.append(f.path)
        if i % 50 == 0 or i == len(files):
            print(f"hashed {i} of {len(files)}, {len(bad)} wrong", flush=True)
    stamp = int(time.time())
    for p in bad:
        (dest / p).rename(dest / f"{p}.bad{stamp}")
    if bad:
        sys.exit(f"{len(bad)} files did not match and were renamed to .bad{stamp}: {bad[:5]}. Run again to refetch.")
    print("Every file matches the repository's SHA-256.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    ap.add_argument("--dest", type=Path, default=Path("F:/imagenet-1k-wds"))
    ap.add_argument("--splits", nargs="+", choices=sorted(SHARDS), default=["train", "validation"])
    ap.add_argument("--workers", type=int, default=16, help="files downloaded at once; one alone gets about 1 MB/s")
    ap.add_argument("--meta-only", action="store_true", help="download only the card and _info.json, to test access")
    ap.add_argument("--dry-run", action="store_true", help="say what would be downloaded, and download nothing")
    ap.add_argument("--sha256", action="store_true", help="after downloading, hash every file against the repository")
    args = ap.parse_args()

    auth_check(REPO, repo_type="dataset")  # Fails here, with a clear message, if the terms are not accepted.
    splits = [] if args.meta_only else args.splits
    files = expected_files(HfApi(), splits)
    args.dest.mkdir(parents=True, exist_ok=True)

    need = sum(f.size for f in missing(files, args.dest))
    free = shutil.disk_usage(args.dest).free
    total = sum(f.size for f in files) / 1e9
    print(f"{REPO}: {len(files)} files, {total:.1f} GB for {', '.join(splits) or 'the metadata'}")
    print(f"{need / 1e9:.1f} GB still to download to {args.dest}, where {free / 1e9:.0f} GB is free")
    if need > free * 0.95:
        sys.exit("Not enough free space.")
    if args.dry_run:
        return

    t0 = time.time()
    download(files, args.dest, splits, args.workers)
    secs = time.time() - t0
    todo = missing(files, args.dest)
    if todo:
        sys.exit(f"{len(todo)} files are still missing or short, for example {todo[0].path}. Run again to resume.")
    print(f"All {len(files)} files are present at full size: {need / 1e9:.1f} GB in {secs / 60:.1f} min.")
    if args.sha256:
        check_hashes(files, args.dest)


if __name__ == "__main__":
    main()

r"""Step 1: crop ImageNet to 256 as img2dataset cropped the LAION set, and measure every crop.

    python prepare.py train        # F:\imagenet-1k-wds\imagenet1k-train-*.tar      -> D:\imagenet-filter\train
    python prepare.py validation   # F:\imagenet-1k-wds\imagenet1k-validation-*.tar -> D:\imagenet-filter\validation

Per source shard it writes `crops256\NNNNN.tar` (jpg and json per kept image, keyed by the image's position in the
source shard) and `scan\NNNNN.parquet` (one row per source image, kept or not). A shard already scanned is skipped.

An image is not cropped at all when its short side is under 256 (it would be upscaled) or it is wider than 1.8:1.
The rest are decoded with PIL (which ignores EXIF rotation, as guided-diffusion's loader did), resized so the short
side is 256 with OpenCV's area filter, centre cropped, and saved as JPEG quality 95, 4:2:0: img2dataset's
`center_crop` at 256. The statistics are `laion_filter/scan.py`'s, so the same classical tests apply.
"""
import glob
import hashlib
import importlib.util
import io
import json
import os
import sys
import tarfile
import time
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "laion_filter"))
from scan import stats  # noqa: E402

SRC = r"F:\imagenet-1k-wds"
OUT = r"D:\imagenet-filter"
N = 256
MAX_ASPECT = 1.8


def class_names():
    spec = importlib.util.spec_from_file_location("c", os.path.join(HERE, "..", "..", "..", "cgd", "imagenet_classes.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.IMAGENET_CLASSES


def crop256(im):
    """img2dataset's center_crop: short side to 256 with an area filter, then the centre square."""
    a = np.asarray(im.convert("RGB"))
    h, w = a.shape[:2]
    s = N / min(h, w)
    if min(h, w) != N:
        a = cv2.resize(a, (max(N, round(w * s)), max(N, round(h * s))), interpolation=cv2.INTER_AREA)
    h, w = a.shape[:2]
    y, x = (h - N) // 2, (w - N) // 2
    return np.ascontiguousarray(a[y:y + N, x:x + N])


def one(args):
    path, shard, split, names = args
    scan = os.path.join(OUT, split, "scan", f"{shard:05d}.parquet")
    if os.path.exists(scan):
        return shard, -1, -1
    dst = os.path.join(OUT, split, "crops256", f"{shard:05d}.tar")
    rows, order, kept = {}, [], 0
    with tarfile.open(path, "r:") as tf, tarfile.open(dst + ".tmp", "w") as out:
        for m in tf:
            stem, ext = m.name.rsplit(".", 1)
            if stem not in rows:
                rows[stem] = {"shard": shard, "key": f"{len(order):06d}", "imagenet_key": stem, "pre_reason": ""}
                order.append(stem)
            r = rows[stem]
            data = tf.extractfile(m).read()
            if ext == "json":
                j = json.loads(data)
                r["label"], r["filename"] = int(j["label"]), j["filename"]
                r["wnid"], r["class_name"] = j["filename"].split("/")[0], names[int(j["label"])]
            elif ext == "jpg":
                r["src_bytes"], r["sha256"] = len(data), hashlib.sha256(data).hexdigest()
                try:
                    im = Image.open(io.BytesIO(data))
                    r["mode"] = im.mode
                    r["original_width"], r["original_height"] = im.size
                    lo, hi = min(im.size), max(im.size)
                    if lo < N:
                        r["pre_reason"] = "source_under_256"
                    elif hi / lo > MAX_ASPECT:
                        r["pre_reason"] = "aspect_over_1.8"
                    else:
                        a = crop256(im)
                        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(a, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
                        r["jpg"] = enc.tobytes()
                        r["jpg_bytes"] = len(r["jpg"])
                        r.update(stats(a.astype(np.float32)))
                    r["decode_ok"] = True
                except Exception as ex:  # a file PIL cannot read
                    r["decode_ok"], r["pre_reason"], r["mode"] = False, "undecodable", repr(ex)[:80]
        for stem in order:  # after the loop, so each json has its label whatever the member order
            r = rows[stem]
            jpg = r.pop("jpg", None)
            if jpg is None:
                continue
            meta = {"key": r["key"], "split": split, "label": r["label"], "class_name": r["class_name"],
                    "wnid": r["wnid"], "filename": r["filename"], "original_width": r["original_width"],
                    "original_height": r["original_height"], "width": N, "height": N, "sha256": r["sha256"]}
            for e, d in (("jpg", jpg), ("json", json.dumps(meta).encode())):
                ti = tarfile.TarInfo(f"{r['key']}.{e}")
                ti.size, ti.mtime = len(d), 0
                out.addfile(ti, io.BytesIO(d))
            kept += 1
    os.replace(dst + ".tmp", dst)
    # A DataFrame, not Table.from_pylist: that takes its columns from the first row, which has none of the
    # statistics when the shard's first image was not cropped.
    pd.DataFrame(list(rows.values())).to_parquet(scan + ".tmp")
    os.replace(scan + ".tmp", scan)
    return shard, len(rows), kept


def main(split, limit=None):
    for d in ("scan", "crops256"):
        os.makedirs(os.path.join(OUT, split, d), exist_ok=True)
    tars = sorted(glob.glob(os.path.join(SRC, f"imagenet1k-{split}-*.tar")))[:limit]
    names = class_names()
    jobs = [(p, int(os.path.basename(p).rsplit("-", 1)[1][:-4]), split, names) for p in tars]
    t, n, k = time.time(), 0, 0
    with Pool(int(os.environ.get("PROCS", 28))) as p:
        for i, (_, rows, kept) in enumerate(p.imap_unordered(one, jobs)):
            n += max(rows, 0)
            k += max(kept, 0)
            if i % 50 == 0 or i == len(jobs) - 1:
                el = time.time() - t
                print(f"{split}: {i + 1}/{len(jobs)} shards, {n} images, {k} cropped, {el:.0f}s, {n / el:.0f}/s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)

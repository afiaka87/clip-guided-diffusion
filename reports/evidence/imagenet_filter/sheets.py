r"""Contact sheets of what each stage-C test removed, and of what was kept, labelled with class and score.

    python sheets.py validation   # reads D:\imagenet-filter\validation\selection\scored.parquet
"""
import collections
import io
import os
import sys
import tarfile

import pandas as pd
from PIL import Image, ImageDraw

OUT = r"D:\imagenet-filter"
SCORE = {"watermark_yolo": "wm_yolo", "zeroshot_text": "zs_text", "zeroshot_collage": "zs_collage",
         "zeroshot_screenshot": "zs_screenshot", "zeroshot_logo": "zs_logo", "zeroshot_nsfw": "zs_nsfw"}


def sheet(split, rows, labels, out, cols=8, cell=256):
    by = collections.defaultdict(set)
    for s, k in zip(rows.shard, rows.key):
        by[int(s)].add(k)
    ims = {}
    for s, keys in by.items():
        with tarfile.open(os.path.join(OUT, split, "crops256", f"{s:05d}.tar")) as tf:
            for m in tf:
                if m.name.endswith(".jpg") and m.name[:-4] in keys:
                    ims[(s, m.name[:-4])] = Image.open(io.BytesIO(tf.extractfile(m).read())).convert("RGB")
    n = len(rows)
    sh = Image.new("RGB", (cols * (cell + 4), ((n + cols - 1) // cols) * (cell + 18)), (255, 255, 255))
    d = ImageDraw.Draw(sh)
    for i, (s, k) in enumerate(zip(rows.shard, rows.key)):
        x, y = (i % cols) * (cell + 4), (i // cols) * (cell + 18)
        sh.paste(ims[(int(s), k)], (x + 2, y + 2))
        d.text((x + 2, y + cell + 3), str(labels[i])[:42], fill=(0, 0, 0))
    sh.save(out, quality=88)


def main(split, n=40):
    a = pd.read_parquet(os.path.join(OUT, split, "selection", "scored.parquet"))
    out = os.path.join(OUT, "sheets")
    os.makedirs(out, exist_ok=True)
    for reason, g in a.groupby("stage_c_reason"):
        g = g.sample(min(n, len(g)), random_state=0)
        c = SCORE.get(reason, "quality")
        sheet(split, g, [f"{x:.3f} {y}" for x, y in zip(g[c], g.class_name)],
              os.path.join(out, f"{split}_{reason or 'kept'}.jpg"))
        print(reason or "kept", len(g))


if __name__ == "__main__":
    main(sys.argv[1])

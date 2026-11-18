r"""Check a finished split: every index against the selection, every tar's members, the labels, a sample of
decoded images, and caption statistics.

    python verify.py train   # F:\imagenet-1k-clean-256\train against D:\imagenet-filter\train\selection\clean.parquet
"""
import glob
import io
import json
import os
import re
import sys
import tarfile
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

split = sys.argv[1]
OUT = rf"F:\imagenet-1k-clean-256\{split}"
SEL = rf"D:\imagenet-filter\{split}\selection\clean.parquet"

sel = pd.read_parquet(SEL, columns=["shard", "key", "label", "class_name"])
sel["id"] = [int(s) * 10**9 + int(k) for s, k in zip(sel.shard, sel.key)]
idx_files = sorted(glob.glob(os.path.join(OUT, "*.parquet")))
tars = sorted(glob.glob(os.path.join(OUT, "*.tar")))
print("tars", len(tars), "indexes", len(idx_files), "tmp", glob.glob(os.path.join(OUT, "*.tmp")))
idx = pd.concat([pd.read_parquet(f).assign(file=os.path.basename(f)[:5]) for f in idx_files], ignore_index=True)
print("rows", len(idx), "unique keys", idx.key.nunique(), "unique sources", idx.source_id.nunique())
print("selected", len(sel), "equal to selection", set(idx.source_id) == set(sel.id))
keys = idx.key.astype(int).to_numpy()
print("keys consecutive", bool((np.sort(keys) == np.arange(len(keys))).all()))
print("key in its shard", bool((keys // 5000 == idx.file.astype(int)).all()))

label_of = dict(zip(sel.id, sel.label))
bad, wrong_label, wrong_caption, n_json = [], 0, 0, 0
rng = np.random.default_rng(0)
decoded, wrong_image = 0, 0
for t in tars:
    sub = idx[idx.file == os.path.basename(t)[:5]]
    source = dict(zip(sub.key, sub.source_id))
    caption = dict(zip(sub.key, sub.txt))
    names = []
    with tarfile.open(t) as tf:
        for m in tf:
            names.append(m.name)
            stem, ext = m.name.rsplit(".", 1)
            if ext == "json":
                j = json.load(tf.extractfile(m))
                n_json += 1
                wrong_label += j["label"] != label_of[source[stem]]
                wrong_caption += j["generated_caption"] != caption[stem]
            elif ext == "txt":
                wrong_caption += tf.extractfile(m).read().decode() != caption[stem]
            elif ext == "jpg" and rng.random() < 0.01:
                im = Image.open(io.BytesIO(tf.extractfile(m).read()))
                im.load()
                decoded += 1
                wrong_image += im.size != (256, 256) or im.mode != "RGB"
    by = Counter(n.rsplit(".", 1)[0] for n in names)
    exts = Counter(n.rsplit(".", 1)[1] for n in names)
    if len(by) != len(sub) or set(by) != set(sub.key) or set(by.values()) != {3} or set(exts) != {"jpg", "txt", "json"}:
        bad.append(os.path.basename(t))
print("tars with wrong members", bad)
print("json read", n_json, "wrong labels", int(wrong_label), "captions differing from the index", int(wrong_caption))
print("images decoded", decoded, "not 256x256 RGB", int(wrong_image))

txt = idx.txt.fillna("")
words = txt.str.split().str.len()
print("empty captions", int((txt.str.strip() == "").sum()))
print("words p1/p50/p99/max", np.percentile(words, [1, 50, 99]).tolist(), int(words.max()))
print("start 'The image shows'", round(txt.str.startswith("The image shows").mean(), 4), "others", int((~txt.str.startswith("The image shows")).sum()))
print("not ending in . ! ? \"", round((~txt.str.rstrip().str.contains(r"[.!?\"']$")).mean(), 4))
print("mentions text", round(txt.str.contains(r"\b(?:text|words|reads|written|logo)\b", flags=re.I).mean(), 4))
print("exact duplicate captions", int(txt.duplicated().sum()), round(txt.duplicated().mean(), 4))
print("most common captions:")
print(txt.value_counts().head(5).to_string())

# How often the caption names the class: any word of the class name, as a rough lower bound on agreement.
m = idx.merge(sel[["id", "class_name"]], left_on="source_id", right_on="id")
hit = [any(w in c.lower() for w in re.findall(r"[a-z]{4,}", n.lower())) for n, c in zip(m.class_name, m.txt)]
print("caption contains a word of the class name", round(float(np.mean(hit)), 4))
per = m.groupby("class_name").size()
print("classes", len(per), "min", int(per.min()), per.idxmin(), "median", int(per.median()), "under 500:", int((per < 500).sum()), "under 300:", int((per < 300).sum()))

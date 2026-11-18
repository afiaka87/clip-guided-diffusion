"""Check the finished best tier: every index against best1m.parquet, every tar's members, caption stats."""
import glob
import os
import re
import tarfile
from collections import Counter

import numpy as np
import pandas as pd

OUT = r"F:\laion-aesthetic-best1m-256"
SEL = r"D:\laion2b-filter\selection\best1m.parquet"

sel = pd.read_parquet(SEL, columns=["shard", "key"])
want = set(int(s) * 10**9 + int(k) for s, k in zip(sel.shard, sel.key))
idx_files = sorted(glob.glob(os.path.join(OUT, "*.parquet")))
tars = sorted(glob.glob(os.path.join(OUT, "*.tar")))
print("tars", len(tars), "indexes", len(idx_files), "tmp", glob.glob(os.path.join(OUT, "*.tmp")))
idx = pd.concat([pd.read_parquet(f).assign(file=os.path.basename(f)[:5]) for f in idx_files], ignore_index=True)
print("rows", len(idx), "unique keys", idx.key.nunique(), "unique sources", idx.source_id.nunique())
print("selected", len(want), "equal to selection", set(idx.source_id) == want)
keys = idx.key.astype(int).to_numpy()
print("keys consecutive", bool((np.sort(keys) == np.arange(len(keys))).all()))
print("key in its shard", bool((keys // 5000 == idx.file.astype(int)).all()))

bad = []
for t in tars:
    with tarfile.open(t) as tf:
        names = tf.getnames()
    by = Counter(n.rsplit(".", 1)[0] for n in names)
    exts = Counter(n.rsplit(".", 1)[1] for n in names)
    sub = idx[idx.file == os.path.basename(t)[:5]]
    if len(by) != len(sub) or set(by) != set(sub.key) or set(by.values()) != {3} or set(exts) != {"jpg", "txt", "json"}:
        bad.append(os.path.basename(t))
print("tars with wrong members", bad)

txt = idx.txt.fillna("")
words = txt.str.split().str.len()
print("empty captions", int((txt.str.strip() == "").sum()))
print("words p1/p50/p99/max", np.percentile(words, [1, 50, 99]).tolist(), int(words.max()))
print("start 'The image shows'", round(txt.str.startswith("The image shows").mean(), 4))
print("not ending in . ! ? \"", round((~txt.str.rstrip().str.contains(r"[.!?\"']$")).mean(), 4))
print("mentions text", round(txt.str.contains(r"\b(text|words|reads|written|logo)\b", flags=re.I).mean(), 4))
print("exact duplicate captions", int(txt.duplicated().sum()))
print("most common captions:")
print(txt.value_counts().head(5).to_string())

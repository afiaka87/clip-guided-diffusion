r"""Contact sheet of near-duplicates stage C dropped, each beside the better-scored image it matched, with the cosine."""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sheet_keys import sheet  # noqa: E402

SEL = r"D:\laion2b-filter\selection"
STAGEB = r"D:\laion2b-filter\stageB"

ids, emb = [], []
for f in sorted(glob.glob(os.path.join(STAGEB, "part_*.npz"))):
    d = np.load(f)
    ids.append(d["ids"]); emb.append(d["emb"])
ids = np.concatenate(ids); emb = np.concatenate(emb)
pos = pd.Series(np.arange(len(ids)), index=ids)
s = pd.read_parquet(os.path.join(SEL, "scored.parquet"), columns=["id", "shard", "key", "leaf", "quality", "stage_c_reason"])
dups = s[s.stage_c_reason == "near_duplicate"].sample(16, random_state=0)
items, labels = [], []
for _, r in dups.iterrows():
    mates = s[(s.leaf == r.leaf) & (s.quality > r.quality)]
    sims = emb[pos.loc[mates.id].values].astype(np.float32) @ emb[pos.loc[r.id]].astype(np.float32)
    m = mates.iloc[int(sims.argmax())]
    items += [(int(m.shard), m.key), (int(r.shard), r.key)]
    labels += [f"kept q={m.quality:.2f}", f"dropped cos={sims.max():.3f}"]
sheet(items, sys.argv[1], cols=8, labels=labels)

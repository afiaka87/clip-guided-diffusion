r"""A sheet of near-duplicates stage C dropped, each right of the better-scored image it matched, with the cosine.

    python dup_pairs.py train
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

from sheets import OUT, sheet


def main(split, n=20):
    ids, emb = [], []
    for f in sorted(glob.glob(os.path.join(OUT, split, "stageB", "part_*.npz"))):
        d = np.load(f)
        ids.append(d["ids"])
        emb.append(d["emb"])
    ids, emb = np.concatenate(ids), np.concatenate(emb)
    pos = pd.Series(np.arange(len(ids)), index=ids)
    s = pd.read_parquet(os.path.join(OUT, split, "selection", "scored.parquet"),
                        columns=["id", "shard", "key", "leaf", "quality", "stage_c_reason", "class_name"])
    rows, labels, same = [], [], 0
    dups = s[s.stage_c_reason == "near_duplicate"]
    n = min(n, len(dups))
    for _, r in dups.sample(n, random_state=0).iterrows():
        mates = s[(s.leaf == r.leaf) & (s.quality >= r.quality) & (s.id != r.id)]
        sims = emb[pos.loc[mates.id].values].astype(np.float32) @ emb[pos.loc[r.id]].astype(np.float32)
        m = mates.iloc[int(sims.argmax())]
        same += m.class_name == r.class_name
        rows += [m, r]
        labels += [f"kept {m.class_name}", f"dropped {sims.max():.3f} {r.class_name}"]
    sheet(split, pd.DataFrame(rows), labels, os.path.join(OUT, "sheets", f"{split}_near_dup_pairs.jpg"))
    print(f"{same} of {n} pairs share a class")


if __name__ == "__main__":
    main(sys.argv[1])

r"""Step 2: the classical filters, the same tests and thresholds as `laion_filter/pipeline/stage_a_select.py` where
they apply. ImageNet has no LAION scores, placeholders or re-fetched files, so those tests are gone.

    python select_a.py train   # D:\imagenet-filter\train\scan -> D:\imagenet-filter\train\selection\stage_a.parquet
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

OUT = r"D:\imagenet-filter"


def main(split):
    m = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(OUT, split, "scan", "*.parquet")))],
                  ignore_index=True)
    tests = [
        ("undecodable", m.decode_ok != True),  # noqa: E712
        ("source_under_256", m.pre_reason == "source_under_256"),
        ("aspect_over_1.8", m.pre_reason == "aspect_over_1.8"),
        ("blank", m.lum_std < 4),
        ("exact_duplicate", m.sha256.duplicated()),
        ("white_ring_over_half", m.ring_white >= 0.5),
        ("greyscale", m.chroma_mean < 4),
        ("under_256_colours", m.n_colors < 256),
    ]
    reason = pd.Series("", index=m.index)
    for name, t in tests:
        reason[np.asarray(t, bool) & (reason.values == "")] = name
    keep = reason == ""
    m["stage_a_reason"] = reason
    sel = os.path.join(OUT, split, "selection")
    os.makedirs(sel, exist_ok=True)
    m[keep].reset_index(drop=True).to_parquet(os.path.join(sel, "stage_a.parquet"))
    m.loc[~keep, ["shard", "key", "imagenet_key", "label", "class_name", "stage_a_reason"]].to_parquet(
        os.path.join(sel, "stage_a_dropped.parquet"))
    print(f"{split}: kept {int(keep.sum())} of {len(m)} ({keep.mean():.1%})")
    print(reason[~keep].value_counts().to_string())
    per = m.assign(keep=keep).groupby("class_name").keep.agg(["mean", "sum"])
    print("classes kept least:")
    print(per.sort_values("mean").head(15).round(3).to_string())
    print("smallest class after the filters:", int(per["sum"].min()), "| median", int(per["sum"].median()))


if __name__ == "__main__":
    main(sys.argv[1])

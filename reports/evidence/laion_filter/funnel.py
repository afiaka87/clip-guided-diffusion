"""Hard-filter funnel over the scan, in the order a pipeline would apply it."""
import numpy as np, pandas as pd
from load import load
import sys
m = load(columns=None)
PADCHECK = len(sys.argv) > 1
if PADCHECK:
    p = pd.read_parquet("D:/laion2b-filter/analysis/padcheck.parquet")
    P = p[["p_top","p_bottom","p_left","p_right"]].to_numpy(); S = p[["p_top_step","p_bottom_step","p_left_step","p_right_step"]].to_numpy()
    p["strict_pad"] = ((P >= 3) & (P < 256) & (S > 12)).any(axis=1)
    m = m.merge(p[["shard","key","strict_pad"]], on=["shard","key"], how="inner")
    print("restricted to the", m.shard.nunique(), "shards of the padding check")
else:
    m["strict_pad"] = False
n0 = len(m); print("all samples", n0)
ss = np.minimum(m.original_width, m.original_height); ls_ = np.maximum(m.original_width, m.original_height)
ar = ls_ / ss
B = m[["b_top", "b_bottom", "b_left", "b_right"]].to_numpy()
sha_n = m.sha256.map(m.sha256.value_counts())
dh_n = m.dhash.map(m.dhash.value_counts())
dh_caps = m.groupby("dhash").caption.nunique(); dh_capn = m.dhash.map(dh_caps)
ar_l = m.WIDTH / m.HEIGHT; ar_o = m.original_width / m.original_height
steps = [
    ("decodes", m.decode_ok == True),
    ("not blank (grey std >= 4)", m.lum_std >= 4),
    ("not a placeholder (sha256 seen < 3 times)", sha_n < 3),
    ("not a placeholder (dHash seen < 5 times with different captions)", ~((dh_n >= 5) & (dh_capn >= 3))),
    ("first copy of an exact duplicate", ~m.sha256.duplicated()),
    ("aspect matches LAION's record (same image as was scored)", np.abs(np.log(ar_l / ar_o)) <= 0.05),
    ("source short side >= 256 (not upscaled)", ss >= 256),
    ("aspect <= 1.8 (crop keeps >= 56% of the long side)", ar <= 1.8),
    ("no flat edge band >= 3px ending in a sharp step (padding or cutout on flat colour)", ~m.strict_pad),
    ("outer ring < 50% white", m.ring_white < .5),
    ("colour (chroma_mean >= 4)", m.chroma_mean >= 4),
    (">= 256 colours", m.n_colors >= 256),
    ("pwatermark < 0.5", m.pwatermark < .5),
    ("punsafe < 0.1", m.punsafe < .1),
]
keep = np.ones(n0, bool)
for name, s in steps:
    s = np.asarray(s, bool)
    print(f"{name:66s} alone drops {100*(~s).mean():5.2f}%   cumulative keep {int((keep & s).sum()):>9,} ({100*(keep & s).mean():5.1f}%)")
    keep &= s
k = m[keep]
print("survivors", len(k))
print("survivor aesthetic >=", {t: int((k.aesthetic >= t).sum()) for t in [7, 7.25, 7.5, 7.75, 8]})
print("survivor short side >= 384:", int((np.minimum(k.original_width, k.original_height) >= 384).sum()), ">= 512:", int((np.minimum(k.original_width, k.original_height) >= 512).sum()))

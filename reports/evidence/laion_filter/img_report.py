import numpy as np, pandas as pd
from load import load
from sheet_keys import sheet
m = load(); m = m[m.decode_ok == True]
n = len(m); print("samples", n)
pct = lambda s: f"{s.mean()*100:.2f}% ({int(s.sum())})"
B = m[["b_top", "b_bottom", "b_left", "b_right"]]
bm = m[["b_top_mean", "b_bottom_mean", "b_left_mean", "b_right_mean"]].values
blank = m.lum_std < 4
print("blank/uniform image (grey std<4)", pct(blank))
nb = ~blank
for t in [2, 4, 8, 16, 32, 64]:
    print(f"any edge band >= {t}px", pct(nb & (B.max(axis=1) >= t)))
for t in [4, 16]:
    has = (B.values >= t)
    white = has & (bm > 235); black = has & (bm < 20) & (bm >= 0); other = has & ~((bm > 235) | ((bm < 20) & (bm >= 0)))
    print(f"  >= {t}px  white band {pct(nb & white.any(1))}  black band {pct(nb & black.any(1))}  other colour {pct(nb & other.any(1))}")
    lb = nb & (m.b_top >= t) & (m.b_bottom >= t); pb = nb & (m.b_left >= t) & (m.b_right >= t)
    print(f"  >= {t}px  letterbox top+bottom {pct(lb)}  pillarbox left+right {pct(pb)}  all four {pct(nb & (B.min(axis=1) >= t))}")
print("ring (outer 4px) >= 90% white", pct(nb & (m.ring_white >= .9)), " >=50%", pct(nb & (m.ring_white >= .5)))
print("ring >= 90% black", pct(nb & (m.ring_black >= .9)))
print("white_frac >= 0.5", pct(m.white_frac >= .5), " >= 0.25", pct(m.white_frac >= .25))
print("greyscale (chroma_mean < 4)", pct(nb & (m.chroma_mean < 4)), " near-grey (<8)", pct(nb & (m.chroma_mean < 8)))
print("few colours (<256 at 5 bits)", pct(nb & (m.n_colors < 256)), " <512", pct(nb & (m.n_colors < 512)))
ss = np.minimum(m.original_width, m.original_height)
print("hf3 (energy > 0.3 cyc/px) median by original short side:")
for lo, hi in [(0, 128), (128, 192), (192, 256), (256, 384), (384, 768), (768, 1e9)]:
    s = (ss >= lo) & (ss < hi)
    print(f"  [{lo},{hi}) n={s.sum()} hf3 med={m.hf3[s].median():.4f} hf4 med={m.hf4[s].median():.4f} lap_var med={m.lap_var[s].median():.0f} jpgKB={m.jpg_bytes[s].median()/1e3:.1f}")
dh = m.dhash.value_counts()
print("dHash exact groups >1:", (dh > 1).sum(), "samples:", dh[dh > 1].sum())
sha = m.sha256.value_counts()
print("sha256 groups >1:", (sha > 1).sum(), "samples:", sha[sha > 1].sum(), " groups>=3:", (sha >= 3).sum(), "samples:", sha[sha >= 3].sum())
ar_l = m.WIDTH / m.HEIGHT; ar_o = m.original_width / m.original_height
print("aspect mismatch vs LAION's WIDTH/HEIGHT >5%:", pct(np.abs(np.log(ar_l / ar_o)) > 0.05))
print("original short side < 32 (icons, 1x1 placeholders):", pct(ss < 32))
m.to_parquet(r"D:\laion2b-filter\analysis\partial_cache.parquet") if False else None

import os
def show(mask, name, k=24, sort=None, label=None):
    if not os.environ.get("SHEETS"): return
    x = m[mask]
    x = x.sample(min(k, len(x)), random_state=1) if sort is None else x.sort_values(sort).head(k)
    lab = [f"{label}={v:.3g}" for v in x[label]] if label else None
    sheet(list(zip(x.shard, x.key)), f"s_{name}.png", labels=lab)
    print("sheet", name, len(m[mask]))
show(nb & (B.max(axis=1) >= 8) & (B.max(axis=1) < 200), "bands", label="b_top")
show(nb & (m.ring_white >= .9), "ringwhite", label="white_frac")
show(nb & (m.n_colors < 400), "fewcolors", label="n_colors")
show(nb & (ss >= 256) & (m.hf3 < m.hf3[ss >= 256].quantile(.03)), "soft_highres", label="hf3")
show(nb & (m.chroma_mean < 4), "grey", label="chroma_mean")
show(m.pwatermark > .6, "watermark", label="pwatermark")
show(nb & (m.edge_frac > m.edge_frac.quantile(.97)), "edgy", label="edge_frac")
print("lum_mean mean", m.lum_mean.mean(), "median", m.lum_mean.median(), "(ImageNet mean luma ~117)")
print("lum_mean quantiles", m.lum_mean.quantile([.05,.25,.5,.75,.95]).round(1).tolist())
print("sat_mean median", m.sat_mean.median(), "colorfulness median", m.colorfulness.median())
print("bright high-key (lum_mean > 200)", pct(m.lum_mean > 200), " dark (< 50)", pct(m.lum_mean < 50))

import numpy as np, pandas as pd
from sheet_keys import sheet
p = pd.read_parquet(r"D:\laion2b-filter\analysis\padcheck.parquet")
n = len(p); print("samples", n)
pct = lambda s: f"{s.mean()*100:.2f}% ({int(s.sum())})"
E = ["top", "bottom", "left", "right"]
P = p[[f"p_{e}" for e in E]].to_numpy(); M = p[[f"p_{e}_mean" for e in E]].to_numpy(); S = p[[f"p_{e}_step" for e in E]].to_numpy()
full = (P == 256).any(1)
pad = (P >= 3) & (P < 256) & (S > 12)  # flat band ending in a clear step
print("blank (a flat edge spans the whole image)", pct(full))
for t in [3, 8, 16, 32]:
    q = (P >= t) & (P < 256) & (S > 12)
    print(f"strict pad band >= {t}px on any side {pct(q.any(1))}   white {pct((q & (M > 235)).any(1))}  black {pct((q & (M < 20)).any(1))}  other {pct((q & (M >= 20) & (M <= 235)).any(1))}")
    print(f"   letterbox (top & bottom) {pct(q[:,0] & q[:,1])}   pillarbox (left & right) {pct(q[:,2] & q[:,3])}   frame (all four) {pct(q.all(1))}   one side only {pct(q.sum(1) == 1)}")
print("collage gutters (interior flat full lines) h>=1 or v>=1:", pct((p.gut_h >= 1) | (p.gut_v >= 1)), " both:", pct((p.gut_h >= 1) & (p.gut_v >= 1)))
q8 = (P >= 8) & (P < 256) & (S > 12)
x = p[q8.any(1)].sample(32, random_state=2)
sheet(list(zip(x.shard, x.key)), "strict_pad.png", labels=[f"t{a} b{b} l{c} r{d}" for a, b, c, d in x[["p_top","p_bottom","p_left","p_right"]].to_numpy()])
w = p[(q8 & (M > 235)).any(1)].sample(24, random_state=3)
sheet(list(zip(w.shard, w.key)), "strict_pad_white.png", labels=[f"t{a} b{b} l{c} r{d}" for a, b, c, d in w[["p_top","p_bottom","p_left","p_right"]].to_numpy()])
g = p[(p.gut_h >= 1) & (p.gut_v >= 1) & ~q8.any(1)].sample(24, random_state=4)
sheet(list(zip(g.shard, g.key)), "gutters.png", labels=[f"h{a} v{b}" for a, b in g[["gut_h","gut_v"]].to_numpy()])

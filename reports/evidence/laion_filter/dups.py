from load import load
import pandas as pd, numpy as np
from sheet_keys import sheet
m = load(columns=["shard","key","sha256","url","caption","original_width","original_height","WIDTH","HEIGHT","dhash","jpg_bytes"])
vc = m.sha256.value_counts()
print("groups with >1:", (vc > 1).sum(), "samples in them:", vc[vc > 1].sum())
print("size distribution of dup groups:", vc[vc>1].value_counts().sort_index().head(10).to_dict())
top = vc.head(16)
items, labels = [], []
for h, c in top.items():
    r = m[m.sha256 == h].iloc[0]
    dom = r.url.split("/")[2]
    print(c, r.original_width, r.original_height, dom, "|", r.caption[:60])
    items.append((r.shard, r.key)); labels.append(f"{c}x {dom}")
sheet(items, "dups_top.png", cols=8, labels=labels)
mis = (m.WIDTH != m.original_width) | (m.HEIGHT != m.original_height)
ar_l = m.WIDTH / m.HEIGHT; ar_o = m.original_width / m.original_height
print("size mismatch", mis.mean(), "aspect mismatch >5%", (np.abs(np.log(ar_l / ar_o)) > 0.05).mean())
print("mismatch & same aspect (rescaled copy)", (mis & (np.abs(np.log(ar_l / ar_o)) <= 0.05)).mean())
x = m[np.abs(np.log(ar_l / ar_o)) > 0.05].sample(16, random_state=0)
sheet(list(zip(x.shard, x.key)), "aspect_mismatch.png", labels=[f"{a}x{b}->{c}x{d}" for a,b,c,d in zip(x.WIDTH,x.HEIGHT,x.original_width,x.original_height)])

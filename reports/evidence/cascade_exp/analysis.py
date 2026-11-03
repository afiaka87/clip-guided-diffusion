"""Scratch: paired differences with bootstrap intervals, medians of cost."""
import json, os, statistics as st
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
rows = json.load(open("scores.json"))
extra = {(r["pipe"], r["prompt"]): r["crop_b16"] for r in json.load(open("crops.json"))} if os.path.exists("crops.json") else {}
table = {}
for r in rows:
    r["crop_b16"] = extra.get((r["pipe"], r["prompt"]), float("nan"))
    table.setdefault(r["pipe"], {})[r["prompt"]] = r
J = ["ViT-B/16", "ViT-H", "SigLIP2-L", "crop_b16"]
rng = np.random.default_rng(0)

def ci(diff):
    idx = rng.integers(0, len(diff), (10000, len(diff)))
    means = diff[idx].mean(1)
    return np.percentile(means, [2.5, 97.5])

def summary(name):
    t = table[name]
    v = list(t.values())
    return dict(n=len(v), sec=round(st.median(x["sec"] for x in v), 2), min_sec=round(min(x["sec"] for x in v), 2), alloc=max(x["peak_alloc_gib"] for x in v), res=max(x["peak_reserved_gib"] for x in v),
                **{k: round(float(np.mean([x[k] for x in v])), 4) for k in J + ["clipped_pct", "saturation", "laplacian_var"]}, others=max(x["others"] for x in v))

for name in table:
    print(name, summary(name))
print()
def compare(a, b):
    common = sorted(set(table[a]) & set(table[b]))
    parts = []
    for k in J:
        d = np.array([table[a][p][k] - table[b][p][k] for p in common])
        if np.isnan(d).all():
            continue
        lo, hi = ci(d)
        parts.append(f"{k} {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}]")
    print(f"{a} - {b} (n={len(common)}):", "; ".join(parts))

for a, b in [("d64", "d256"), ("d128", "d256"), ("sr256_u", "d256"), ("sr256_g", "d256"), ("sr256_u", "d64"), ("sr256_g", "sr256_u"), ("i2i256_s500", "d256"), ("i2i256_s700", "d256"),
             ("esrgan256", "d256"), ("ultrasharp256", "d256"), ("swinirL256", "d256"), ("hat256", "d256"), ("compact256", "d256"), ("esrgan256", "d64"), ("sr256_u", "esrgan256"), ("sr256_n10", "sr256_u"), ("sr256_n60", "sr256_u"), ("sr256_blur06", "sr256_u"), ("sr256_blur10", "sr256_u"),
             ("d128", "d512"), ("d256", "d512"), ("sr512_u", "d512"), ("sr512_g", "d512"), ("sr512_u", "d128"), ("sr512_g", "sr512_u"), ("i2i512_s700", "d512"), ("i2i512_s700", "d256"), ("esrgan512", "d512"), ("ultrasharp512", "d512"), ("swinirL512", "d512"), ("hat512", "d512"), ("compact512", "d512"), ("esrgan512", "d128"), ("sr512_u", "esrgan512"),
             ("v_base", "d256"), ("v_unguided", "v_base"), ("v_secondary", "v_base"), ("v_dt995", "v_base"), ("v_orth", "v_base"), ("v_freeu13", "v_base"), ("v_freeu11", "v_base"), ("pp_hm100", "v_base"), ("pp_hm50", "v_base"), ("pp_sat085", "v_base"),
             ("v512_base", "d512"), ("v512_secondary", "v512_base"), ("sel8_sr256_u", "sr256_u"), ("sel8_sr256_u", "d256")]:
    if a in table and b in table:
        compare(a, b)
print()
seeds = ["d64"] + [f"sel64_s{s}" for s in range(1, 8)]
if all(s in table for s in seeds):
    prompts = sorted(table["d64"])
    for selector in ["ViT-B/16", "ViT-H", "SigLIP2-L"]:
        for k in (2, 4, 8):
            line = []
            for judge in ["ViT-B/16", "ViT-H", "SigLIP2-L"]:
                gains = []
                for p in prompts:
                    pool = [table[s][p] for s in seeds[:k]]
                    pick = max(pool, key=lambda r: r[selector])
                    gains.append(pick[judge] - table["d64"][p][judge])
                gains = np.array(gains)
                lo, hi = ci(gains)
                line.append(f"{judge} {gains.mean():+.3f} [{lo:+.3f},{hi:+.3f}]")
            print(f"select by {selector}, best of {k}, gain over seed 0:", "; ".join(line))
    # Within-prompt agreement between judges across seeds.
    for a, b in [("ViT-B/16", "ViT-H"), ("ViT-B/16", "SigLIP2-L"), ("ViT-H", "SigLIP2-L")]:
        rs = []
        for p in prompts:
            x = [table[s][p][a] for s in seeds]; y = [table[s][p][b] for s in seeds]
            rs.append(np.corrcoef(x, y)[0, 1])
        print("within-prompt seed correlation", a, b, round(float(np.mean(rs)), 3))

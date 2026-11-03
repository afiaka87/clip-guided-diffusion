"""Held-out -best: pick by ViT-B/16 + SigLIP2 (the repo's judges), evaluate on ViT-H, which took no part in the pick."""
import json, os, sys
import numpy as np
S = os.environ["SP"]; db = json.load(open(f"{S}/eval/db.json"))
def key(p): return f"{p}|{int(os.path.getmtime(p))}"
def run(name, path, arm, seeds):
    d = json.load(open(path)); rows = []
    for i in range(12):
        c = [db[key(d[f"{arm} s{s}"][i])] for s in seeds]
        pick = max(range(len(c)), key=lambda j: c[j]["b16"] + c[j]["sig2"])
        rows.append((c[pick]["vith"] - np.mean([x["vith"] for x in c]), c[pick]["aesb16"] - np.mean([x["aesb16"] for x in c])))
    r = np.array(rows); m, ci = r.mean(0), 1.96 * r.std(0, ddof=1) / 12 ** .5
    print(f"{name}: best of {len(seeds)}: held-out ViT-H {m[0]:+.4f} +-{ci[0]:.4f} ({(r[:, 0] > 0).sum()}/12), aes16 {m[1]:+.3f} +-{ci[1]:.3f}")
run("ADM 128", f"{S}/results/adm128_paths.json", "base", (1, 2, 3, 4))
run("ADM 128 ens", f"{S}/results/adm128_paths.json", "ens B32+PE", (1, 2, 3, 4))
run("Turbo tiny, no aes", f"{S}/results/aes_sd-turbo512.json", "base", (1, 2, 3))
run("Turbo aes .1", f"{S}/results/aes_sd-turbo512.json", "aes .1", (1, 2, 3))

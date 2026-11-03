import json, sys
import numpy as np
K = ["1", "2", "4", "8", "16", "32"]
for path in sys.argv[1:]:
    data = json.load(open(path))
    steps = list(data["iid bicubic"])
    cells = min(len(data[n].get(s, [])) for n in data for s in steps[:1])
    print("\n", path.split("/")[-1], "cells per scheme", {n: sum(len(v) for v in d.values()) for n, d in data.items()})
    def get(name, k, field, judge=None):
        out = {}
        for s in steps:
            for i, row in enumerate(data[name].get(s, [])):
                out[s, i] = row[k]["judge"][judge] if judge else row[k][field]
        return out
    judges = list(data["iid bicubic"][steps[0]][0]["1"]["judge"])
    def diff(a, b, k, field, judge=None):
        x, y = get(a, k, field, judge), get(b, k, field, judge)
        d = np.array([x[c] - y[c] for c in x if c in y])
        return d.mean(), 1.96 * d.std(ddof=1) / np.sqrt(len(d)), len(d)
    for a, b in (("LHS bicubic", "iid bicubic"), ("R3 bicubic", "iid bicubic"), ("antithetic bicubic", "iid bicubic"), ("overview+inner bicubic", "iid bicubic"), ("LHS area", "iid area (repo)"), ("iid area (repo)", "iid bicubic")):
        line = []
        for k in ("2", "4", "8", "16", "32"):
            m, ci, n = diff(a, b, k, "pair")
            line.append(f"k{k} {m:+.3f}+-{ci:.3f}")
        print(f"pair {a:24s} vs {b:16s} n={n} " + " ".join(line))
        for j in judges:
            print(f"   transfer {j:14s} " + " ".join("k%s %+.4f+-%.4f" % ((k,) + diff(a, b, k, None, j)[:2]) for k in ("4", "32")))
    for name in ("iid area (repo)", "iid bicubic", "LHS bicubic"):
        print(f"{name:20s} pair by k", [round(float(np.mean(list(get(name, k, 'pair').values()))), 3) for k in K])
        for j in judges:
            print(f"   {j:14s} transfer by k", [round(float(np.mean(list(get(name, k, None, j).values()))), 3) for k in K])

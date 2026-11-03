"""Reduce e3 results: does repeatability follow Spearman-Brown, what do schemes gain, where does transfer top out?"""
import json
import sys

import numpy as np

path = sys.argv[1]
data = json.load(open(path))
K = ["1", "2", "4", "8", "16", "32"]
steps = list(next(iter(data.values())))


def table(name, key, judge=None):
    """prompts x steps x counts."""
    out = []
    for step in steps:
        rows = data[name][step]
        out.append([[(row[k]["judge"][judge] if judge else row[k][key]) for k in K] for row in rows])
    return np.array(out).transpose(1, 0, 2)


def spearman_brown(c, k):
    return k * c / (1 + (k - 1) * c)


def effective(pair, c):
    return pair * (1 - c) / (c * (1 - pair))


judges = list(next(iter(data.values()))[steps[0]][0]["1"]["judge"])
print("judges", judges, "| steps", steps)
print("\n-- Spearman-Brown: measured pair against a one parameter fit --")
counts = np.array([1, 2, 4, 8, 16, 32])
for name in ("iid area (repo)", "iid bicubic", "LHS bicubic"):
    mean = table(name, "pair").mean((0, 1))
    grid = np.linspace(0.005, 0.5, 2000)
    c = grid[np.argmin([((spearman_brown(g, counts) - mean) ** 2).sum() for g in grid])]
    print(f"{name:20s} c={c:.3f} measured {np.round(mean, 3)} fit {np.round(spearman_brown(c, counts), 3)}")

print("\n-- pair by step for k=1, 4, 32 and k needed for pair .5 and .8 --")
for name in ("iid area (repo)", "iid bicubic"):
    t = table(name, "pair")
    for i, step in enumerate(steps):
        mean = t[:, i].mean(0)
        c = mean[2] / (4 - 3 * mean[2])  # invert Spearman-Brown at k=4
        print(f"{name:20s} {step:7s} k1 {mean[0]:.3f} k4 {mean[2]:.3f} k32 {mean[5]:.3f} c(from k4) {c:.3f} k(.5) {1 / c:.0f} k(.8) {4 * (1 - c) / c:.0f}")

print("\n-- paired difference in pair against iid bicubic, over 16 prompt-step cells --")
base = table("iid bicubic", "pair")
base_c = spearman_brown  # for readability below
for name in ("LHS bicubic", "R3 bicubic", "antithetic bicubic", "iid lanczos3", "iid bicubic flip", "iid area (repo)"):
    t = table(name, "pair")
    line = []
    for index, k in ((1, 2), (2, 4), (3, 8), (4, 16), (5, 32)):
        d = (t[..., index] - base[..., index]).flatten()
        c = 0.11
        ratio = effective(t[..., index].mean(), c) / effective(base[..., index].mean(), c)
        line.append(f"k{k}: {d.mean():+.3f}+-{1.96 * d.std(ddof=1) / np.sqrt(len(d)):.3f} (x{ratio:.2f} cuts)")
    print(f"{name:20s} " + "  ".join(line))

print("\n-- transfer: cosine with another model's 256-cut mean, and the limit j / sqrt(pair) --")
for judge in judges:
    print("judge", judge)
    for name in ("iid area (repo)", "iid bicubic", "LHS bicubic", "overview+inner bicubic", "whole frame jitter bicubic", "iid bicubic flip", "iid bicubic noise .02", "iid bicubic noise .05", "whole frame noise .02"):
        t = table(name, None, judge)
        p = table(name, "pair")
        per_step = t.mean(0)
        limit = per_step[:, 5] / np.sqrt(np.clip(p.mean(0)[:, 5], 1e-6, None))
        cells = " ".join(f"{step}:{per_step[i, 2]:.3f}>{per_step[i, 5]:.3f}" for i, step in enumerate(steps))
        print(f"  {name:28s} k4>k32 {cells}  limit {np.round(limit, 3)}")

print("\n-- paired transfer difference against iid bicubic at k=4 and k=32, pooled over judges, prompts, steps --")
for name in ("iid area (repo)", "LHS bicubic", "R3 bicubic", "antithetic bicubic", "overview+inner bicubic", "whole frame jitter bicubic", "iid bicubic flip", "iid lanczos3", "iid bicubic noise .02", "iid bicubic noise .05", "whole frame noise .02", "whole frame noise .05"):
    line = []
    for index, k in ((2, 4), (5, 32)):
        ds = np.stack([(table(name, None, j) - table("iid bicubic", None, j))[..., index] for j in judges])
        d = ds.mean(0).flatten()
        per_judge = " ".join(f"{x:+.3f}" for x in ds.mean((1, 2)))
        line.append(f"k{k}: {d.mean():+.3f}+-{1.96 * d.std(ddof=1) / np.sqrt(len(d)):.3f} [{per_judge}]")
    print(f"{name:28s} " + "  ".join(line))

print("\n-- gain from k=4 to other counts, iid area (repo) --")
for judge in judges:
    t = table("iid area (repo)", None, judge).mean((0, 1))
    print(judge, np.round(t, 3), "ratio to k4", np.round(t / t[2], 2))

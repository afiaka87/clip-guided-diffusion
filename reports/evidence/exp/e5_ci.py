import json
import numpy as np
rows = json.load(open("results/e5_blend.json"))
for step in sorted({r["step"] for r in rows}):
    part = [r for r in rows if r["step"] == step]
    cell = []
    for key in ("pair_raw", "pair_unet", "pair_blend", "hf_blend", "identity_over_unet"):
        v = np.array([r[key] for r in part])
        cell.append(f"{key} {v.mean():.3f}+-{2.571 * v.std(ddof=1) / np.sqrt(len(v)):.3f}")
    print(step, round(part[0]["sigma"], 3), len(part), " ".join(cell))
data = json.load(open("results/e5_resampler.json"))
for name, row in data.items():
    print(name, " ".join(f"{k} {np.mean(v):.3f}+-{2.571 * np.std(v, ddof=1) / np.sqrt(len(v)):.3f}" for k, v in row.items()))

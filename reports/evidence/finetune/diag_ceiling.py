"""How much of an image can its CLIP embeddings predict at all? A ridge regression to the downsampled image."""

import numpy as np
import torch as th

from finetune import data
from finetune.embed import DEFAULT_MODELS

th.manual_seed(0)
rng = np.random.default_rng(0)
out = {}
for name, root in (("laion", data.LAION_ROOT), ("imagenet", data.IMAGENET_ROOT)):
    source = data.open_source(name, root, DEFAULT_MODELS, split="all")
    picks = np.sort(rng.choice(len(source.dataset), 7000, replace=False))
    images, feats = [], {k: [] for k in ("image", "caption", "short")}
    for i in picks:
        item = source.dataset[int(i)]
        images.append(item["image"])
        for k in feats:
            feats[k].append(th.cat([item["embeds"][m][k] for m in DEFAULT_MODELS]))
    out[name] = (th.stack(images).float().div(127.5).sub(1), {k: th.stack(v) for k, v in feats.items()})
    print(name, "loaded", flush=True)
images = th.cat([out[n][0] for n in out])
feats = {k: th.cat([out[n][1][k] for n in out]) for k in ("image", "caption", "short")}
order = th.randperm(len(images))
train, test = order[:12000], order[12000:]
total = images[test].sub(images[train].mean(0)).square().mean().item()
print(f"pixel variance of a held-out image around the mean image: {total:.4f}")
for size in (4, 8, 16, 32):
    low = th.nn.functional.adaptive_avg_pool2d(images, size)
    up = th.nn.functional.interpolate(low, 256, mode="bilinear", align_corners=False)
    kept = 1 - (images[test] - up[test]).square().mean().item() / total
    y = low.flatten(1)
    line = f"{size:2d}x{size:<2d} holds {kept:5.1%} of the variance;"
    for kind, x in feats.items():
        mu, sd = x[train].mean(0), x[train].std(0) + 1e-6
        xs = th.cat([(x - mu) / sd, th.ones(len(x), 1)], dim=1).double()
        best = -1.0
        for lam in (30.0, 100.0, 300.0, 1000.0, 3000.0):
            a = xs[train].T @ xs[train] + lam * th.eye(xs.shape[1], dtype=th.float64)
            w = th.linalg.solve(a, xs[train].T @ y[train].double())
            pred = (xs[test] @ w).float().reshape(-1, 3, size, size)
            pred_up = th.nn.functional.interpolate(pred, 256, mode="bilinear", align_corners=False)
            r2 = 1 - (images[test] - pred_up).square().mean().item() / total
            best = max(best, r2)
        line += f"  {kind} embeddings predict {best:5.1%} of the full image"
    print(line, flush=True)

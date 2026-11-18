"""How large is the condition the U-Net receives, next to the timestep embedding it is added to?"""

from pathlib import Path

import numpy as np
import torch as th
from guided_diffusion.nn import timestep_embedding

from finetune import data
from finetune.embed import DEFAULT_MODELS

th.manual_seed(0)
ckpt = th.load("D:/cgd-finetune/uncond256-v2/adapter.pt", map_location="cpu", weights_only=True)
weights = {k: v.float() for k, v in ckpt["weights"].items()}
print("checkpoint step", ckpt["steps"])
base = th.load(
    Path.home() / ".cache/clip-guided-diffusion/256x256_diffusion_uncond.pt",
    map_location="cpu",
    mmap=True,
    weights_only=True,
)
rows = np.sort(np.random.default_rng(0).choice(900000, 2000, replace=False))
embeds = {}
for name in DEFAULT_MODELS:
    e = np.load(data.embed_path(data.LAION_ROOT, name, "image"), mmap_mode="r")[rows].astype(np.float32)
    embeds[name] = th.from_numpy(e)
    mean = e.mean(0)
    print(
        f"{name}: per-dimension std {e.std(0).mean():.4f}, norm of the mean embedding {np.linalg.norm(mean):.3f}, "
        f"norm of an image minus the mean {np.linalg.norm(e - mean, axis=1).mean():.3f}"
    )
hidden = 0
for name in DEFAULT_MODELS:
    w, b = (
        weights[f"conditioner.inputs.{data.slug(name)}.weight"],
        weights[f"conditioner.inputs.{data.slug(name)}.bias"],
    )
    pre = embeds[name] @ w.T + b
    hidden = hidden + th.nn.functional.silu(pre)
    print(f"{name}: hidden pre-activation std {pre.std():.4f}")
out = hidden @ weights["conditioner.to_emb.weight"].T + weights["conditioner.to_emb.bias"]
print(f"hidden: mean abs {hidden.abs().mean():.4f}; across images std {hidden.std(0).mean():.4f}")
shared, varying = out.mean(0), out - out.mean(0)
print(
    f"condition added to the timestep embedding: norm {out.norm(dim=1).mean():.4f}, "
    f"shared part {shared.norm():.4f}, part that differs between images {varying.norm(dim=1).mean():.4f}"
)
for t in (999.0, 800.0, 500.0, 100.0):
    s = timestep_embedding(th.tensor([t]), 256)
    h = th.nn.functional.silu(s @ base["time_embed.0.weight"].float().T + base["time_embed.0.bias"].float())
    emb = h @ base["time_embed.2.weight"].float().T + base["time_embed.2.bias"].float()
    print(f"timestep embedding at t={t:.0f}: norm {emb.norm():.3f}, per-entry std {emb.std():.4f}")
tall = th.cat(
    [
        (
            lambda s: (
                th.nn.functional.silu(s @ base["time_embed.0.weight"].float().T + base["time_embed.0.bias"].float())
                @ base["time_embed.2.weight"].float().T
                + base["time_embed.2.bias"].float()
            )
        )(timestep_embedding(th.tensor([float(t)]), 256))
        for t in range(0, 1000, 20)
    ]
)
print(
    f"timestep embedding: how far it moves between timesteps 20 apart {(tall[1:] - tall[:-1]).norm(dim=1).mean():.3f}"
)
tokens = hidden @ weights["conditioner.to_tokens.weight"].T + weights["conditioner.to_tokens.bias"]
tokens = tokens.reshape(len(tokens), 4, -1)
print(
    f"tokens: norm {tokens.norm(dim=-1).mean():.3f}, part that differs between images {(tokens - tokens.mean(0)).norm(dim=-1).mean():.3f}"
)

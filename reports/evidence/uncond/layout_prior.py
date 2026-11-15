"""A small text-conditioned model of 64x64 thumbnails: a prior over compositions for the unconditional U-Net.

The unconditional pipeline's weak point is the start of sampling, where the U-Net's denoised
image is a grey blur and CLIP has nothing to read a layout from. What that stage decides is
only the coarse arrangement of shapes and colours, and that is a small problem: this is a
transformer over 4x4 pixel patches (16x16 tokens) that draws a thumbnail from the prompt's
CLIP text embeddings. Its thumbnail, enlarged and noised partway down the schedule, is where
the U-Net and CLIP guidance start from, so they only have to draw what a thumbnail cannot hold.

It is a rectified flow: x_t = (1 - t) * image + t * noise, the network predicts
noise - image, and sampling integrates that from t = 1 (noise) to t = 0 with Euler steps
and classifier-free guidance.
"""

import math
from collections.abc import Sequence

import torch as th
import torch.nn.functional as tf
from torch import nn

SIDE = 64


class Block(nn.Module):
    """A transformer block whose normalisations are scaled and shifted by the condition (adaLN-zero, from DiT)."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(approximate="tanh"), nn.Linear(4 * width, width))
        self.modulation = nn.Linear(width, 6 * width)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: th.Tensor, condition: th.Tensor) -> th.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(condition).unsqueeze(1).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale1) + shift1
        batch, tokens, width = h.shape
        q, k, v = self.qkv(h).view(batch, tokens, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        h = tf.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(batch, tokens, width)
        x = x + gate1 * self.proj(h)
        return x + gate2 * self.mlp(self.norm2(x) * (1 + scale2) + shift2)


class LayoutPrior(nn.Module):
    """Predicts the velocity of a noisy thumbnail, given the time and the prompt's CLIP text embeddings."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        text_dims: Sequence[int],
        *,
        width: int = 512,
        depth: int = 12,
        heads: int = 8,
        patch: int = 4,
        context_tokens: int = 8,
    ):
        super().__init__()
        self.config = {
            "text_dims": list(text_dims), "width": width, "depth": depth, "heads": heads, "patch": patch,
            "context_tokens": context_tokens,
        }  # fmt: skip
        self.patch, self.width, self.context_tokens = patch, width, context_tokens
        grid = SIDE // patch
        self.embed = nn.Conv2d(3, width, patch, patch)
        self.position = nn.Parameter(th.randn(1, grid * grid, width) * 0.02)
        self.time = nn.Sequential(nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width))
        text_dim = sum(text_dims)
        self.text_tokens = nn.Linear(text_dim, context_tokens * width)
        self.text_pooled = nn.Sequential(nn.Linear(text_dim, width), nn.SiLU(), nn.Linear(width, width))
        # What stands in for a prompt that was dropped, so classifier-free guidance has an unconditional model.
        self.null_tokens = nn.Parameter(th.randn(1, context_tokens, width) * 0.02)
        self.null_pooled = nn.Parameter(th.zeros(1, width))
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.out_modulation = nn.Linear(width, 2 * width)
        self.out = nn.Linear(width, patch * patch * 3)
        for layer in (self.out_modulation, self.out):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: th.Tensor, t: th.Tensor, text: th.Tensor, present: th.Tensor | None = None) -> th.Tensor:
        """The velocity of x, [batch, 3, 64, 64], at times t in [0, 1].

        `text` is each image's CLIP text embeddings, each normalised, side by side. `present`
        is a [batch] mask of the images that have a prompt, all of them when it is None.
        """
        batch = x.shape[0]
        tokens = self.text_tokens(text).view(batch, self.context_tokens, self.width)
        pooled = self.text_pooled(text)
        if present is not None:
            tokens = th.where(present[:, None, None], tokens, self.null_tokens.to(tokens.dtype))
            pooled = th.where(present[:, None], pooled, self.null_pooled.to(pooled.dtype))
        condition = tf.silu(self.time(timestep_features(t).to(pooled.dtype)) + pooled)
        h = self.embed(x).flatten(2).transpose(1, 2) + self.position
        h = th.cat([tokens, h], dim=1)
        for block in self.blocks:
            h = block(h, condition)
        h = h[:, self.context_tokens :]
        shift, scale = self.out_modulation(condition).unsqueeze(1).chunk(2, dim=-1)
        h = self.out(self.norm(h) * (1 + scale) + shift)
        grid = SIDE // self.patch
        h = h.view(batch, grid, grid, self.patch, self.patch, 3).permute(0, 5, 1, 3, 2, 4)
        return h.reshape(batch, 3, SIDE, SIDE)


def timestep_features(t: th.Tensor, features: int = 256) -> th.Tensor:
    """Sinusoidal features of times in [0, 1]."""
    frequencies = th.exp(-math.log(10_000) * th.arange(features // 2, device=t.device) / (features // 2))
    angles = 1000 * t[:, None].float() * frequencies
    return th.cat([angles.cos(), angles.sin()], dim=-1)


def join_text(embeddings: Sequence[th.Tensor]) -> th.Tensor:
    """Several models' text embeddings of the same prompts, each normalised, side by side."""
    return th.cat([tf.normalize(e.float(), dim=-1) for e in embeddings], dim=-1)


@th.no_grad()
def sample(
    model: LayoutPrior,
    text: th.Tensor,
    *,
    scale: float = 4.0,
    steps: int = 50,
    generator: th.Generator | None = None,
) -> th.Tensor:
    """Thumbnails in [-1, 1] of each row of `text`, drawn with classifier-free guidance of `scale`."""
    device = text.device
    x = th.randn((text.shape[0], 3, SIDE, SIDE), device=device, generator=generator)
    absent = th.zeros(text.shape[0], dtype=th.bool, device=device)
    times = th.linspace(1, 0, steps + 1, device=device)
    for now, then in zip(times[:-1], times[1:], strict=True):
        t = now.expand(text.shape[0])
        with th.autocast(device.type, dtype=th.bfloat16, enabled=device.type == "cuda"):
            conditional = model(x, t, text).float()
            unconditional = model(x, t, text, absent).float() if scale != 1 else conditional
        velocity = unconditional + scale * (conditional - unconditional)
        x = x + (then - now) * velocity
    return x.clamp(-1, 1)


def load(path: str, device: str) -> tuple[LayoutPrior, dict]:
    """A trained prior with its EMA weights, frozen, and the checkpoint's settings."""
    state = th.load(path, map_location="cpu", weights_only=False)
    model = LayoutPrior(**state["config"])
    model.load_state_dict(state["ema"])
    return model.to(device).eval().requires_grad_(False), state["args"]

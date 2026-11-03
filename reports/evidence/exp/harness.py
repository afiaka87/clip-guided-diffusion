"""Cutout and gradient-quality variants on the real sampler, without touching the repo.

Everything plugs into sampling.sample_loop_progressive through its loss_fn hook. Per
cutout gradients are taken in x_in space, then aggregated, and handed to the sampler
as a surrogate scalar whose gradient through the U-Net is the aggregated gradient.
With the defaults this computes the same thing as guidance_loss() in cgd/cgd.py.
"""
import dataclasses
import json
import math
import os
import sys
import time

import numpy as np
import torch as th
import torch.nn.functional as tf
import torchvision.transforms.functional as tvf

REPO = "/home/sam/GitHub/clip-guided-diffusion"
sys.path.insert(0, REPO)
from cgd import clip_util, losses, sampling, script_util  # noqa: E402
from cgd.modules import adaptive_avg_pool  # noqa: E402

device = "cuda"
OUT = os.path.dirname(os.path.abspath(__file__)) + "/results"
os.makedirs(OUT, exist_ok=True)

PROMPTS = [
    "a red fox in the snow",
    "a pencil sketch of an old man with a beard",
    "a plate of sushi on a black slate",
    "a lighthouse on a rocky coast at sunset",
    "an oil painting of a sailboat in a storm",
    "a bowl of ramen with chopsticks",
    "a snow covered mountain above a green valley",
    "a vintage red bicycle leaning against a brick wall",
    "a watercolor painting of a hummingbird and flowers",
    "a steam locomotive crossing a stone bridge",
    "a golden retriever puppy playing in autumn leaves",
    "a futuristic city skyline at night with neon lights",
]


@dataclasses.dataclass
class Config:
    n: int = 4
    min_frac: float = 0.7
    cut_pow: float = 1.0
    layout: str = "iid"
    resample: str = "area"
    noise: float = 0.0  # Per-cutout std after resampling, on the [0, 1] scale.
    flip: bool = False
    gray: float = 0.0
    pure_x0: bool = False
    agg: str = "mean"
    blur: float = 0.0  # Gaussian sigma on the x_in gradient, in pixels.
    ema: float = 0.0
    rate: float = sampling.DEFAULT_GUIDANCE_RATE


def _kernel(name):
    if name == "bilinear":
        return lambda d: (1 - d.abs()).clamp(min=0), 1

    def bicubic(d, a=-0.5):
        d = d.abs()
        return ((a + 2) * d**3 - (a + 3) * d**2 + 1) * (d <= 1) + (a * d**3 - 5 * a * d**2 + 8 * a * d - 4 * a) * ((d > 1) & (d < 2))

    if name == "bicubic":
        return bicubic, 2
    if name == "lanczos3":
        return lambda d: th.sinc(d) * th.sinc(d / 3) * (d.abs() < 3), 3
    raise ValueError(name)


_MATRICES = {}


def resample_matrix(in_size, out_size, name):
    key = (in_size, out_size, name)
    if key not in _MATRICES:
        kernel, support = _kernel(name)
        scale = in_size / out_size
        stretch = max(scale, 1.0)  # Widen the kernel when shrinking, which antialiases.
        centers = (th.arange(out_size, device=device, dtype=th.float32) + 0.5) * scale - 0.5
        positions = th.arange(in_size, device=device, dtype=th.float32)
        distance = (positions[None] - centers[:, None]) / stretch
        weights = kernel(distance) * (distance.abs() < support)
        _MATRICES[key] = weights / weights.sum(1, keepdim=True)
    return _MATRICES[key]


def resample(cut, size, name):
    if name == "area":
        return adaptive_avg_pool(cut, size)
    w = resample_matrix(cut.shape[-1], size, name)
    return w @ cut @ w.T


def blur_matrix(side, sigma):
    positions = th.arange(side, device=device, dtype=th.float32)
    w = th.exp(-((positions[None] - positions[:, None]) ** 2) / (2 * sigma**2))
    return w / w.sum(1, keepdim=True)


PHI3 = 1.2207440846057596


class Views:
    """Draws one step's cutouts. All randomness comes from `gen`, not the global stream."""

    def __init__(self, cut_size, cfg, gen):
        self.cut_size, self.cfg, self.gen = cut_size, cfg, gen
        self.counter = 0
        self.shift = th.rand(3, generator=gen)

    def _uniform(self, n):
        cfg = self.cfg
        r = lambda count: th.rand(count, generator=self.gen)  # noqa: E731
        if cfg.layout == "lhs":
            return [(th.randperm(n, generator=self.gen) + r(n)) / n for _ in range(3)]
        if cfg.layout == "r3":
            index = self.counter + th.arange(n)
            self.counter += n
            return [(self.shift[d] + index / PHI3 ** (d + 1)) % 1 for d in range(3)]
        if cfg.layout == "antithetic":
            half = (n + 1) // 2
            us, ux, uy = r(half), r(half), r(half)
            return [us.repeat(2)[:n], th.cat((ux, 1 - ux))[:n], th.cat((uy, 1 - uy))[:n]]
        return [r(n), r(n), r(n)]

    def inner(self, side, n):
        cfg = self.cfg
        lo = min(int(side * cfg.min_frac), self.cut_size)
        us, ux, uy = self._uniform(n)
        out = []
        for i in range(n):
            size = int(us[i] ** cfg.cut_pow * (side - lo) + lo)
            room = side - size
            out.append((size, int(ux[i] * (room + 1)), int(uy[i] * (room + 1))))
        return out

    def draw(self, side, sigma):
        """[(size, x, y, flip, gray, dx, dy)] for this step."""
        cfg, n = self.cfg, self.cfg.n
        if cfg.layout == "whole":
            boxes = [(side, 0, 0)] * n
        elif cfg.layout == "jitter":
            boxes = [(side, 0, 0)] * n
        elif cfg.layout == "overview":
            boxes = [(side, 0, 0)] + self.inner(side, n - 1)
        elif cfg.layout == "disco":
            overview = max(1, round(n * (0.75 if sigma >= 0.988 else 0.25)))
            boxes = [(side, 0, 0)] * overview + self.inner(side, n - overview)
        else:
            boxes = self.inner(side, n)
        flips = th.rand(n, generator=self.gen) < 0.5 if cfg.flip else th.zeros(n, dtype=th.bool)
        if cfg.layout == "disco":
            flips = th.tensor([False, True, False][: n] + [False] * max(0, n - 3))[:n] if overview > 1 else flips
        grays = [i < math.ceil(cfg.gray * n) for i in range(n)]
        reach = side // 14 if cfg.layout == "jitter" else 0
        shifts = th.randint(-reach, reach + 1, (n, 2), generator=self.gen) if reach else th.zeros(n, 2, dtype=th.long)
        return [(*box, bool(flips[i]), grays[i], int(shifts[i, 0]), int(shifts[i, 1])) for i, box in enumerate(boxes)]

    def cut(self, leaves, plan):
        """One cutout per leaf: leaves[i] goes through plan[i]."""
        side = leaves.shape[-1]
        cuts = []
        for i, (size, x, y, flip, gray, dx, dy) in enumerate(plan):
            cut = leaves[i:i + 1]
            if dx or dy:  # Translate, smearing the border, with a reproducible backward.
                cut = cut[..., (th.arange(side, device=device) + dy).clamp(0, side - 1), :]
                cut = cut[..., (th.arange(side, device=device) + dx).clamp(0, side - 1)]
            cut = cut[..., y:y + size, x:x + size]
            cut = resample(cut, self.cut_size, self.cfg.resample)
            if flip:
                cut = cut.flip(-1)
            if gray:
                cut = tvf.rgb_to_grayscale(cut, 3)
            cuts.append(cut)
        cuts = th.cat(cuts)
        if self.cfg.noise:
            cuts = cuts + self.cfg.noise * th.randn(cuts.shape, generator=self.gen).to(cuts)
        return cuts


def per_cut_gradients(clip_model, target, views, x01, plan, chunk=32):
    """dLoss/dx01 for each cutout separately, [n, 3, H, W], plus the mean loss."""
    grads, total = [], 0.0
    for start in range(0, len(plan), chunk):
        part = plan[start:start + chunk]
        leaves = x01.detach().repeat(len(part), 1, 1, 1).requires_grad_(True)
        embeds = clip_model.encode_image(clip_util.preprocess(clip_model, views.cut(leaves, part))).float()
        dists = losses.spherical_dist_loss(embeds, target)
        grads.append(th.autograd.grad(dists.sum(), leaves)[0])
        total += dists.sum().item()
    return th.cat(grads), total / len(plan)


def aggregate(grads, cfg, gen=None):
    n = len(grads)
    if cfg.agg == "mean":
        return grads.mean(0, keepdim=True)
    if cfg.agg == "norm":  # Every cutout gets the same vote, at the average length.
        norms = grads.flatten(1).norm(dim=1)[:, None, None, None]
        return (grads / norms).mean(0, keepdim=True) * norms.mean()
    if cfg.agg == "mom":  # Median of 3 group means, per pixel.
        groups = th.stack([part.mean(0) for part in grads.chunk(3)])
        return groups.median(0).values[None]
    if cfg.agg == "reject":  # Drop the quarter least aligned with the rest.
        flat = tf.normalize(grads.flatten(1), dim=1)
        others = tf.normalize(flat.sum(0, keepdim=True) - flat, dim=1)
        scores = (flat * others).sum(1)
        keep = scores.topk(max(1, n - max(1, n // 4))).indices
        return grads[keep].mean(0, keepdim=True)
    raise ValueError(cfg.agg)


def make_loss(clip_model, target, cfg, clip_size, gen, scale=1000.0, tv_scale=150.0, range_scale=50.0, hook=None):
    views = Views(clip_size, cfg, gen)
    state = {"ema": None, "count": 0, "step": 0}

    def loss_fn(x, pred_xstart, fac):
        fac = 1.0 if cfg.pure_x0 else fac
        x_in = pred_xstart * fac + x * (1 - fac)
        if hook is not None:
            hook(state["step"], x, pred_xstart, x_in, fac, views)
        x01 = x_in.detach().add(1).div(2)
        plan = views.draw(x.shape[-1], fac if not cfg.pure_x0 else 1.0)
        grads, _ = per_cut_gradients(clip_model, target, views, x01, plan)
        g = aggregate(grads, cfg)
        if cfg.blur:
            w = blur_matrix(x.shape[-1], cfg.blur)
            g = w @ g @ w.T
        if cfg.ema:
            state["ema"] = g if state["ema"] is None else cfg.ema * state["ema"] + (1 - cfg.ema) * g
            state["count"] += 1
            g = state["ema"]
        state["step"] += 1
        surrogate = (x_in * g.detach()).sum() * 0.5 * scale  # d(x01)/d(x_in) = 0.5
        return surrogate + losses.range_loss(pred_xstart).sum() * range_scale + losses.tv_loss(x_in).sum() * tv_scale

    return loss_fn


class Pipeline:
    """One diffusion model and guide. Models stay loaded across samples."""

    def __init__(self, size=128, clip_name="ViT-B/32", fp16=False):
        self.size, self.clip_name, self.fp16 = size, clip_name, fp16
        path = script_util.download_guided_diffusion(image_size=size, class_cond=True)
        self.model, self.diffusion = script_util.load_guided_diffusion(
            checkpoint_path=path, image_size=size, class_cond=True, timestep_respacing="", use_fp16=True, device=device,
            use_checkpoint=size ** 2 > 256 ** 2)
        self.clip, self.clip_size = clip_util.load_clip(clip_name, device, fp16)
        self.targets = {}

    def target(self, prompt):
        if prompt not in self.targets:
            text, _ = clip_util.encode_text_prompt(prompt, 1.0, self.clip_name, device, self.fp16)
            query = tf.normalize(text, dim=-1)
            index = clip_util.imagenet_top_n(query, device=device, n=1, clip_model_name=self.clip_name, fp16=self.fp16)
            self.targets[prompt] = text, index
        return self.targets[prompt]

    def sample(self, cfg, prompt, seed, hook=None):
        text, class_index = self.target(prompt)
        th.manual_seed(seed)  # Only the sampler draws from this stream, so variants are paired.
        gen = th.Generator().manual_seed(seed + 1000)
        loss_fn = make_loss(self.clip, text, cfg, self.clip_size, gen, hook=hook)
        y = th.zeros(1, device=device, dtype=th.long)
        y[:] = class_index
        result = None
        for out in sampling.sample_loop_progressive(self.model, self.diffusion, (1, 3, self.size, self.size), loss_fn=loss_fn,
                                                    steps=sampling.DEFAULT_STEPS, guidance_rate=cfg.rate, model_kwargs={"y": y},
                                                    device=device, progress=False):
            result = out["pred_xstart"]
        return result.detach().add(1).div(2).clamp(0, 1)[0].mul(255).round().byte().cpu()


def wait_for_gpu(max_minutes=25):
    """Share the card with whoever else is measuring: never start on top of them."""
    import subprocess
    deadline = time.time() + max_minutes * 60
    while time.time() < deadline:
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        if not apps:
            return True
        time.sleep(10)
    return False


def save_json(name, obj):
    with open(f"{OUT}/{name}.json", "w") as handle:
        json.dump(obj, handle, indent=1)


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    t = {11: 2.201, 23: 2.069, 35: 2.030}.get(n - 1, 1.96)
    return float(values.mean()), float(t * values.std(ddof=1) / math.sqrt(n))

"""Follow-up to e1: isolate why conv-with-pad failed, and map interpolate by ratio."""
import sys

import torch as th
import torch.nn.functional as tf

sys.path.insert(0, "/tmp/claude-1000/-home-sam-GitHub-clip-guided-diffusion/34d680cf-d9a7-412c-9f94-d042cb1cd642/scratchpad/exp")
device = "cuda"
REPEATS = 30
th.manual_seed(0)


def image(side, n=4):
    x = tf.interpolate(th.randn(n, 3, 8, 8, device=device), size=side, mode="bicubic", align_corners=False)
    return ((x + 0.3 * th.randn(n, 3, side, side, device=device)) * 0.2 + 0.5).clamp(0, 1).detach()


def check(name, fn, x0):
    y0 = fn(x0)
    w = th.randn_like(y0)
    base, differing, worst = None, 0, 0.0
    for _ in range(REPEATS):
        x = x0.clone().requires_grad_(True)
        g = th.autograd.grad((fn(x) * w).sum(), x)[0].detach()
        if base is None:
            base = g.clone()
        elif not th.equal(g, base):
            differing += 1
            worst = max(worst, (g - base).abs().max().item())
    print(f"{name:60s} {'identical' if not differing else f'DIFFERS {differing}/{REPEATS - 1}, {worst / base.abs().mean().item():.1e}'}", flush=True)


crop = image(224)
k1 = th.tensor([1, 4, 6, 4, 1.0], device=device) / 16
k2 = (k1[None] * k1[:, None]).expand(3, 1, 5, 5)
print("== is it the conv or the pad? ==")
check("conv2d groups=3, no pad", lambda v: tf.conv2d(v, k2, groups=3), crop)
check("conv2d groups=3, built-in padding=2 (zeros)", lambda v: tf.conv2d(v, k2, padding=2, groups=3), crop)
check("conv2d groups=3, stride 2, no pad", lambda v: tf.conv2d(v, k2, stride=2, groups=3), crop)
check("conv2d groups=3 + tf.pad constant", lambda v: tf.conv2d(tf.pad(v, (2, 2, 2, 2), value=0.0), k2, groups=3), crop)
check("conv2d groups=3 + tf.pad circular", lambda v: tf.conv2d(tf.pad(v, (2, 2, 2, 2), mode="circular"), k2, groups=3), crop)
check("conv2d groups=3 + tf.pad reflect", lambda v: tf.conv2d(tf.pad(v, (2, 2, 2, 2), mode="reflect"), k2, groups=3), crop)
check("conv2d groups=3 + tf.pad replicate", lambda v: tf.conv2d(tf.pad(v, (2, 2, 2, 2), mode="replicate"), k2, groups=3), crop)

print("== tf.pad alone, by width (a border pixel gets width+1 adds) ==")
for mode in ("replicate", "reflect"):
    for width in (1, 2, 4, 16):
        check(f"tf.pad {mode} width {width}", lambda v: tf.pad(v, (width,) * 4, mode=mode), crop)


def edge_index(side, width, mode):
    index = th.arange(-width, side + width, device=device)
    if mode == "replicate":
        return index.clamp(0, side - 1)
    period = 2 * (side - 1)
    index = index.remainder(period)
    return th.where(index < side, index, period - index)


def pad_index(v, width, mode):
    i = edge_index(v.shape[-1], width, mode)
    return v[..., i, :][..., :, i]


print("== reflect and replicate through advanced indexing ==")
for mode in ("replicate", "reflect"):
    for width in (2, 16):
        check(f"advanced index pad {mode} width {width}", lambda v: pad_index(v, width, mode), crop)
check("conv2d + advanced index reflect pad", lambda v: tf.conv2d(pad_index(v, 2, "reflect"), k2, groups=3), crop)


def separable_blur_matrix(side, sigma):
    """Blur as W @ x @ W.T, renormalised rows so the edge has replicate-like behaviour."""
    positions = th.arange(side, device=device, dtype=th.float32)
    w = th.exp(-((positions[None] - positions[:, None]) ** 2) / (2 * sigma**2))
    return w / w.sum(1, keepdim=True)


b = separable_blur_matrix(224, 1.5)
check("gaussian blur as W @ x @ W.T", lambda v: b @ v @ b.T, crop)

print("== tf.interpolate by input side, to 224 ==")
for mode in ("bilinear", "bicubic"):
    for side in (89, 112, 128, 179, 223, 224, 225, 256, 358, 448, 512):
        x = image(side)
        check(f"{mode} {side}->224", lambda v: tf.interpolate(v, size=224, mode=mode, align_corners=False), x)
print("== tf.interpolate to 448 ==")
for mode in ("bilinear", "bicubic"):
    for side in (128, 256, 512):
        x = image(side)
        check(f"{mode} {side}->448", lambda v: tf.interpolate(v, size=448, mode=mode, align_corners=False), x)

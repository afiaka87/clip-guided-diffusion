"""Which resampling and augmentation ops give a bit-identical gradient on CUDA?

Same input, same cotangent, REPEATS backward passes. No seeds involved: any op that
is not identical here cannot reproduce an image, because the sampler amplifies it.
"""
import inspect
import math
import sys

import torch as th
import torch.nn.functional as tf
import torchvision.transforms.functional as tvf
from torchvision.transforms import InterpolationMode as IM

sys.path.insert(0, "/home/sam/GitHub/clip-guided-diffusion")
from cgd.modules import adaptive_avg_pool  # noqa: E402
from cgd.ResizeRight import resize_right  # noqa: E402
from cgd.ResizeRight.interp_methods import cubic, lanczos3  # noqa: E402

device = "cuda"
REPEATS = 30
th.manual_seed(0)


def image(side, n=4):
    """Natural-ish image: low frequency content plus detail, in [0, 1]."""
    x = th.randn(n, 3, 8, 8, device=device)
    x = tf.interpolate(x, size=side, mode="bicubic", align_corners=False)
    x = x + 0.3 * th.randn(n, 3, side, side, device=device)
    return (x * 0.2 + 0.5).clamp(0, 1).detach()


def check(name, fn, x0):
    try:
        y0 = fn(x0)
    except Exception as error:  # A missing option is a result, not a crash.
        print(f"{name:52s} ERROR {type(error).__name__}: {str(error)[:60]}", flush=True)
        return
    w = th.randn_like(y0)
    base = None
    worst = 0.0
    identical = True
    differing = 0
    for _ in range(REPEATS):
        x = x0.clone().requires_grad_(True)
        y = fn(x)
        g = th.autograd.grad((y * w).sum(), x)[0].detach()
        if base is None:
            base = g.clone()
            continue
        if not th.equal(g, base):
            identical = False
            differing += 1
            worst = max(worst, (g - base).abs().max().item())
    scale = base.abs().mean().item() or 1.0
    verdict = "identical" if identical else f"DIFFERS {differing}/{REPEATS - 1} runs, max diff/mean|g|={worst / scale:.2e}"
    print(f"{name:52s} {verdict}", flush=True)


def kernel_matrix(in_size, out_size, kernel, support, antialias=True, ref=None):
    """Rows are output pixels, columns are input pixels. W @ x @ W.T resamples."""
    scale = in_size / out_size
    stretch = max(scale, 1.0) if antialias else 1.0
    centers = (th.arange(out_size, device=device, dtype=th.float32) + 0.5) * scale - 0.5
    positions = th.arange(in_size, device=device, dtype=th.float32)
    distance = (positions[None] - centers[:, None]) / stretch
    weights = kernel(distance) * (distance.abs() < support)
    return weights / weights.sum(1, keepdim=True)


def bicubic_kernel(d, a=-0.5):
    d = d.abs()
    near = (a + 2) * d**3 - (a + 3) * d**2 + 1
    far = a * d**3 - 5 * a * d**2 + 8 * a * d - 4 * a
    return near * (d <= 1) + far * ((d > 1) & (d < 2))


def lanczos_kernel(d, a=3):
    return th.sinc(d) * th.sinc(d / a) * (d.abs() < a)


def bilinear_kernel(d):
    return (1 - d.abs()).clamp(min=0)


def matrix_resample(x, size, kernel, support, antialias=True):
    w = kernel_matrix(x.shape[-1], size, kernel, support, antialias)
    return w @ x @ w.T


def rotation_grid(n, side, angle=7.0, shift=0.05):
    a = math.radians(angle)
    theta = th.tensor([[math.cos(a), -math.sin(a), shift], [math.sin(a), math.cos(a), -shift]], device=device)
    return tf.affine_grid(theta[None].expand(n, -1, -1), (n, 3, side, side), align_corners=False)


class FixedPointBilinear(th.autograd.Function):
    """Bilinear warp whose backward accumulates in int64, so sum order cannot matter."""

    BITS = 32

    @staticmethod
    def forward(ctx, x, grid):
        n, c, h, w = x.shape
        px = (grid[..., 0] + 1) * w / 2 - 0.5
        py = (grid[..., 1] + 1) * h / 2 - 0.5
        x0, y0 = px.floor().long(), py.floor().long()
        fx, fy = (px - x0).unsqueeze(1), (py - y0).unsqueeze(1)
        out = 0
        corners = []
        for dy, dx, weight in ((0, 0, (1 - fx) * (1 - fy)), (0, 1, fx * (1 - fy)), (1, 0, (1 - fx) * fy), (1, 1, fx * fy)):
            xi, yi = (x0 + dx).clamp(0, w - 1), (y0 + dy).clamp(0, h - 1)
            index = (yi * w + xi).flatten(1).unsqueeze(1).expand(n, c, -1)
            out = out + x.flatten(2).gather(2, index).view(n, c, *grid.shape[1:3]) * weight
            corners.append((index, weight))
        ctx.corners, ctx.shape = corners, x.shape
        return out

    @staticmethod
    def backward(ctx, cotangent):
        n, c, h, w = ctx.shape
        total = th.zeros(n, c, h * w, device=cotangent.device, dtype=th.int64)
        for index, weight in ctx.corners:
            term = (cotangent * weight).flatten(2).double() * 2.0**FixedPointBilinear.BITS
            total.scatter_add_(2, index, term.round().long())
        return (total.double() / 2.0**FixedPointBilinear.BITS).float().view(n, c, h, w), None


print("torch", th.__version__, "cudnn.deterministic", th.backends.cudnn.deterministic, "benchmark", th.backends.cudnn.benchmark)
print("\n== torch docs: functions carrying the backward reproducibility note ==")
for name, fn in inspect.getmembers(tf, callable):
    doc = getattr(fn, "__doc__", "") or ""
    if "nondeterministic" in doc and "backward" in doc:
        print("  torch.nn.functional." + name)

up = image(128)
down = image(512)
sized = {"up 128->224": (up, 224), "down 512->224": (down, 224)}

print("\n== resamplers ==")
for label, (x, size) in sized.items():
    print(f"-- {label}")
    check("tf.adaptive_avg_pool2d [control, known bad]", lambda v: tf.adaptive_avg_pool2d(v, size), x)
    check("cgd.modules.adaptive_avg_pool [control, known good]", lambda v: adaptive_avg_pool(v, size), x)
    check("tf.interpolate area", lambda v: tf.interpolate(v, size=size, mode="area"), x)
    check("tf.interpolate nearest", lambda v: tf.interpolate(v, size=size, mode="nearest"), x)
    check("tf.interpolate nearest-exact", lambda v: tf.interpolate(v, size=size, mode="nearest-exact"), x)
    check("tf.interpolate bilinear", lambda v: tf.interpolate(v, size=size, mode="bilinear", align_corners=False), x)
    check("tf.interpolate bilinear antialias", lambda v: tf.interpolate(v, size=size, mode="bilinear", align_corners=False, antialias=True), x)
    check("tf.interpolate bicubic", lambda v: tf.interpolate(v, size=size, mode="bicubic", align_corners=False), x)
    check("tf.interpolate bicubic antialias", lambda v: tf.interpolate(v, size=size, mode="bicubic", align_corners=False, antialias=True), x)
    check("ResizeRight lanczos3 (default)", lambda v: resize_right.resize(v, out_shape=(size, size), interp_method=lanczos3, support_sz=None, antialiasing=True, by_convs=False, scale_tolerance=None), x)
    check("ResizeRight cubic (Disco's call)", lambda v: resize_right.resize(v, out_shape=(size, size)), x)
    check("ResizeRight cubic by_convs", lambda v: resize_right.resize(v, out_shape=(size, size), interp_method=cubic, support_sz=None, antialiasing=True, by_convs=True, scale_tolerance=None), x)
    check("matrix bilinear W@x@W.T", lambda v: matrix_resample(v, size, bilinear_kernel, 1), x)
    check("matrix bicubic W@x@W.T", lambda v: matrix_resample(v, size, bicubic_kernel, 2), x)
    check("matrix lanczos3 W@x@W.T", lambda v: matrix_resample(v, size, lanczos_kernel, 3), x)

print("\n== pooling at integer factors ==")
x512 = image(448)
check("tf.avg_pool2d 2", lambda v: tf.avg_pool2d(v, 2), x512)
check("tf.max_pool2d 2", lambda v: tf.max_pool2d(v, 2), x512)
check("tf.adaptive_max_pool2d 224", lambda v: tf.adaptive_max_pool2d(v, 224), x512)
check("tf.pixel_unshuffle 2", lambda v: tf.pixel_unshuffle(v, 2), x512)

print("\n== geometric warps ==")
crop = image(224)
n = crop.shape[0]
grid = rotation_grid(n, 224)
check("tf.grid_sample bilinear border", lambda v: tf.grid_sample(v, grid, mode="bilinear", padding_mode="border", align_corners=False), crop)
check("tf.grid_sample bicubic border", lambda v: tf.grid_sample(v, grid, mode="bicubic", padding_mode="border", align_corners=False), crop)
check("tf.grid_sample nearest border", lambda v: tf.grid_sample(v, grid, mode="nearest", padding_mode="border", align_corners=False), crop)
check("fixed point bilinear warp (custom backward)", lambda v: FixedPointBilinear.apply(v, grid), crop)
check("tvf.affine bilinear", lambda v: tvf.affine(v, angle=7, translate=(3, 2), scale=1.0, shear=0, interpolation=IM.BILINEAR), crop)
check("tvf.affine nearest (torchvision default)", lambda v: tvf.affine(v, angle=7, translate=(3, 2), scale=1.0, shear=0), crop)
start = [[0, 0], [223, 0], [223, 223], [0, 223]]
end = [[12, 5], [210, 9], [219, 214], [3, 220]]
check("tvf.perspective bilinear", lambda v: tvf.perspective(v, start, end, interpolation=IM.BILINEAR), crop)
check("tvf.resized_crop bilinear antialias", lambda v: tvf.resized_crop(v, 20, 20, 150, 150, [224, 224], antialias=True), crop)

print("\n== exact geometry: slices, pads, flips, rolls ==")
check("slice crop", lambda v: v[:, :, 10:170, 20:180], crop)
check("tf.pad reflect + slice", lambda v: tf.pad(v, (16, 16, 16, 16), mode="reflect")[:, :, 5:229, 27:251], crop)
check("tf.pad replicate + slice", lambda v: tf.pad(v, (16, 16, 16, 16), mode="replicate")[:, :, 5:229, 27:251], crop)
check("tf.pad circular + slice", lambda v: tf.pad(v, (16, 16, 16, 16), mode="circular")[:, :, 5:229, 27:251], crop)
check("tf.pad constant + slice", lambda v: tf.pad(v, (16, 16, 16, 16), value=0.5)[:, :, 5:229, 27:251], crop)
check("th.roll", lambda v: th.roll(v, (5, -7), (2, 3)), crop)
check("flip", lambda v: v.flip(-1), crop)
index = th.randint(0, 224, (224,), device=device)
check("index_select with repeats", lambda v: v.index_select(3, index), crop)
check("advanced index [..., i] with repeats", lambda v: v[..., index], crop)

print("\n== photometric and filters ==")
check("gaussian blur conv2d groups=3 (tvf.gaussian_blur)", lambda v: tvf.gaussian_blur(v, 5, 1.0), crop)
check("tvf.rgb_to_grayscale", lambda v: tvf.rgb_to_grayscale(v, 3), crop)
check("tvf.adjust_brightness", lambda v: tvf.adjust_brightness(v, 1.1), crop)
check("tvf.adjust_contrast", lambda v: tvf.adjust_contrast(v, 1.1), crop)
check("tvf.adjust_saturation", lambda v: tvf.adjust_saturation(v, 1.1), crop)
check("tvf.adjust_hue", lambda v: tvf.adjust_hue(v, 0.05), crop)
check("tvf.adjust_sharpness", lambda v: tvf.adjust_sharpness(v, 1.5), crop)
check("torch.fft rfft2 low-pass irfft2", lambda v: th.fft.irfft2(th.fft.rfft2(v) * th.exp(-(th.fft.fftfreq(224, device=device)[:, None] ** 2 + th.fft.rfftfreq(224, device=device)[None] ** 2) * 40), s=(224, 224)), crop)

print("\n== Gaussian blur of a gradient and pyramids ==")
check("tf.avg_pool2d + tf.interpolate bilinear (laplacian level)", lambda v: tf.interpolate(tf.avg_pool2d(v, 2), scale_factor=2, mode="bilinear", align_corners=False), crop)
down_kernel = th.tensor([1, 4, 6, 4, 1.0], device=device) / 16
kernel2d = (down_kernel[None] * down_kernel[:, None]).expand(3, 1, 5, 5)
check("conv2d 5-tap with stride 2 (pyr down)", lambda v: tf.conv2d(tf.pad(v, (2, 2, 2, 2), mode="reflect"), kernel2d, stride=2, groups=3), crop)
check("conv_transpose2d stride 2 (pyr up)", lambda v: tf.conv_transpose2d(v, 4 * kernel2d, stride=2, padding=2, output_padding=1, groups=3), crop)

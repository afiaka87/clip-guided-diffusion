"""Scratch: secondary model, FreeU, dynamic thresholding, orthogonal guidance and colour post-processing at 256."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import math, types
import numpy as np
import torch.nn.functional as tf
import torch.fft as fft
from PIL import Image
import torchvision.transforms.functional as tvf
from guided_diffusion.nn import timestep_embedding
from cgd import clip_util, losses, sampling, script_util
from cgd.cgd import FAST_NUM_CUTOUTS, FAST_MIN_CUT_FRACTION
from secondary import SecondaryDiffusionImageNet2

DEVICE = "cuda"


def rms(tensor):
    return tensor.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()


def loop(model, diffusion, shape, loss_fn, kwargs, secondary=None, dyn=None, orth=False, steps=30, eta=0.5, rate=0.08, window=(0.1, 0.8), clamp=1.2):
    levels = sampling.NoiseLevels(diffusion.alphas_cumprod)
    ts = levels.timesteps(steps, None)
    last = levels.num_timesteps - 1
    scale_t = 1000.0 / levels.num_timesteps if diffusion.rescale_timesteps else 1.0
    guide_from, guide_to = (bound * last for bound in window)
    x = th.randn(shape, device=DEVICE)
    previous_x0, previous_h = None, None
    for i in range(len(ts) - 1):
        t, t_next = ts[i], ts[i + 1]
        alpha, sigma = levels.alpha_sigma(t)
        batch_t = th.full((shape[0],), t * scale_t, device=DEVICE, dtype=th.float32)
        guide = loss_fn is not None and guide_from <= t <= guide_to
        through_unet = guide and secondary is None
        x = x.detach().requires_grad_(through_unet)
        with th.set_grad_enabled(through_unet):
            eps = model(x, batch_t, **kwargs)[:, :shape[1]]
            x0 = (x - sigma * eps) / alpha
        guided_x0 = x0.detach()
        if guide:
            if secondary is None:
                loss = loss_fn(x, x0, sigma)
                gradient = -sampling.loss_gradient(loss, x)
            else:
                x_g = x.detach().requires_grad_(True)
                cosine_t = th.full((shape[0],), math.atan2(sigma, alpha) * 2 / math.pi, device=DEVICE)
                pred = secondary(x_g, cosine_t).pred
                gradient = -sampling.loss_gradient(loss_fn(x_g, pred, sigma), x_g)
            ratio = rms(sigma * gradient) / rms(eps.detach())
            gradient = gradient * (rate / ratio).clamp(max=1)
            update = sigma ** 2 / alpha * gradient
            if orth:
                dims = (1, 2, 3)
                parallel = (update * guided_x0).sum(dims, keepdim=True) / guided_x0.square().sum(dims, keepdim=True).clamp(min=1e-8) * guided_x0
                update = update - parallel
            guided_x0 = guided_x0 + update
        if dyn is None:
            guided_x0 = guided_x0.clamp(-clamp, clamp)
        else:
            s = th.quantile(guided_x0.abs().float().flatten(1), dyn, dim=1).clamp(min=1).view(-1, 1, 1, 1)
            guided_x0 = guided_x0.clamp(-s, s) / s
        x, x0 = x.detach(), x0.detach()
        if i == len(ts) - 2:
            return guided_x0
        alpha_next, sigma_next = levels.alpha_sigma(t_next)
        h = math.log(alpha_next / sigma_next) - math.log(alpha / sigma)
        eta_h = eta * h
        x0_weight = alpha_next * -math.expm1(-h - eta_h)
        x_next = sigma_next / sigma * math.exp(-eta_h) * x + x0_weight * guided_x0
        if previous_x0 is not None:
            x_next = x_next + 0.5 * x0_weight * (h / previous_h) * (x0 - previous_x0)
        if eta_h > 0:
            x_next = x_next + sigma_next * math.sqrt(-math.expm1(-2 * eta_h)) * th.randn_like(x)
        previous_x0, previous_h = x0, h
        x = x_next


def fourier_filter(x, threshold, scale):
    dtype = x.dtype
    freq = fft.fftshift(fft.fftn(x.float(), dim=(-2, -1)), dim=(-2, -1))
    _, _, H, W = freq.shape
    mask = th.ones_like(freq.real)
    crow, ccol = H // 2, W // 2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    return fft.ifftn(fft.ifftshift(freq * mask, dim=(-2, -1)), dim=(-2, -1)).real.type(dtype)


def freeu_forward(b1, b2, s1, s2):
    def forward(self, x, timesteps, y=None):
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            emb = emb + self.label_emb(y)
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        deep, mid = self.model_channels * 4, self.model_channels * 2
        for module in self.output_blocks:
            skip = hs.pop()
            if h.shape[1] in (deep, mid):
                b, s = (b1, s1) if h.shape[1] == deep else (b2, s2)
                mean = h.float().mean(1, keepdim=True)
                low = mean.flatten(1).min(-1).values.view(-1, 1, 1, 1)
                high = mean.flatten(1).max(-1).values.view(-1, 1, 1, 1)
                mean = (mean - low) / (high - low)
                half = h.shape[1] // 2
                h = th.cat([h[:, :half] * ((b - 1) * mean + 1).type(h.dtype), h[:, half:]], dim=1)
                skip = fourier_filter(skip, 1, s)
            h = module(th.cat([h, skip], dim=1), emb)
        return self.out(h.type(x.dtype))
    return forward


def make_loss(clip_model, clip_size, target):
    cutouts = clip_util.MakeCutouts(cut_size=clip_size, num_cutouts=FAST_NUM_CUTOUTS, cutout_size_power=1.0, use_augs=False, min_cut_fraction=FAST_MIN_CUT_FRACTION)

    def loss_fn(x, pred_xstart, fac):
        x_in = pred_xstart * fac + x * (1 - fac)
        embeds = clip_model.encode_image(clip_util.preprocess(clip_model, cutouts(x_in.add(1).div(2)))).float()
        embeds = embeds.view(-1, x.shape[0], embeds.shape[-1])
        clip_loss = losses.spherical_dist_loss(embeds.unsqueeze(2), target).sum(2).mean(0).sum() * 1000
        return clip_loss + losses.tv_loss(x_in).sum() * 150 + losses.range_loss(pred_xstart).sum() * 50
    return loss_fn


def save(name, prompt, image):
    folder = os.path.join(HERE, "out", name)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, script_util.alphanumeric_filter(prompt) + ".png")
    if isinstance(image, th.Tensor):
        tvf.to_pil_image(image[0].add(1).div(2).clamp(0, 1).float().cpu()).save(path)
    else:
        Image.fromarray(image).save(path)
    return path


def run_variant(name, model, diffusion, size, guided=True, **options):
    clip_model, clip_size = clip_util.load_clip("ViT-B/32", DEVICE, False)
    for prompt in PROMPTS:
        target, _ = clip_util.encode_text_prompt(prompt, 1.0, "ViT-B/32", DEVICE, False)
        y = clip_util.imagenet_top_n(tf.normalize(target, dim=-1), device=DEVICE, n=1, clip_model_name="ViT-B/32", fp16=False).reshape(1)
        loss_fn = make_loss(clip_model, clip_size, target) if guided else None

        def run():
            th.manual_seed(0)
            return loop(model, diffusion, (1, 3, size, size), loss_fn, {"y": y}, **options)
        image, stats = measured(run)
        record(name, prompt, save(name, prompt, image), stats)


def histogram_match(source, reference):
    out = np.zeros_like(source)
    for c in range(3):
        src_values, src_counts = np.unique(source[..., c], return_counts=True)
        ref_values, ref_counts = np.unique(reference[..., c], return_counts=True)
        mapped = np.interp(np.cumsum(src_counts) / source[..., c].size, np.cumsum(ref_counts) / reference[..., c].size, ref_values)
        out[..., c] = np.interp(source[..., c], src_values, mapped)
    return out


secondary = SecondaryDiffusionImageNet2().eval().requires_grad_(False)
secondary.load_state_dict(th.load(os.path.join(SP, "weights_secondary", "secondary_model_imagenet_2.pth"), map_location="cpu", weights_only=True))
print("secondary params", sum(p.numel() for p in secondary.parameters()), flush=True)
secondary = secondary.to(DEVICE)

path256 = script_util.download_guided_diffusion(image_size=256, class_cond=True)
model, diffusion = script_util.load_guided_diffusion(path256, 256, True, "", True, DEVICE, False)
original_forward = type(model).forward
run_variant("v_base", model, diffusion, 256)
run_variant("v_unguided", model, diffusion, 256, guided=False)
run_variant("v_secondary", model, diffusion, 256, secondary=secondary)
run_variant("v_dt995", model, diffusion, 256, dyn=0.995)
run_variant("v_orth", model, diffusion, 256, orth=True)
model.forward = types.MethodType(freeu_forward(1.3, 1.4, 0.9, 0.2), model)
run_variant("v_freeu13", model, diffusion, 256)
model.forward = types.MethodType(freeu_forward(1.1, 1.2, 0.9, 0.2), model)
run_variant("v_freeu11", model, diffusion, 256)
del model.forward
print("256 variants done", flush=True)

# Colour post-processing, against the unguided image from the same seed and class.
base, reference = paths("v_base"), paths("v_unguided")
for prompt in PROMPTS:
    a = np.asarray(Image.open(base[prompt]).convert("RGB")).astype(np.float64)
    b = np.asarray(Image.open(reference[prompt]).convert("RGB")).astype(np.float64)
    start = time.perf_counter()
    matched = histogram_match(a, b)
    elapsed = time.perf_counter() - start
    stats = dict(sec=round(elapsed, 3), peak_alloc_gib=0, peak_reserved_gib=0, others_before=0, others_after=0)
    record("pp_hm100", prompt, save("pp_hm100", prompt, matched.round().clip(0, 255).astype(np.uint8)), stats)
    record("pp_hm50", prompt, save("pp_hm50", prompt, ((a + matched) / 2).round().clip(0, 255).astype(np.uint8)), stats)
    hue, chroma, value = Image.open(base[prompt]).convert("HSV").split()
    desaturated = np.asarray(Image.merge("HSV", (hue, chroma.point(lambda level: round(level * 0.85)), value)).convert("RGB"))
    record("pp_sat085", prompt, save("pp_sat085", prompt, desaturated), stats)

del model, diffusion
script_util.load_guided_diffusion.cache_clear(); gc.collect(); th.cuda.empty_cache()
path512 = script_util.download_guided_diffusion(image_size=512, class_cond=True)
model, diffusion = script_util.load_guided_diffusion(path512, 512, True, "", True, DEVICE, True)
run_variant("v512_base", model, diffusion, 512)
run_variant("v512_secondary", model, diffusion, 512, secondary=secondary)
print("all variants done", flush=True)

"""How good are the cheap CLIP gradients? Real SD 1.5, real TAESD, real ViT-B/32.

Walks unguided 20 step DDIM trajectories. At several timesteps it computes the gradient
of the CLIP loss four ways, and reports the cosine between them in latent space:

  xt_vae    d loss / d x_t through the U-Net and the full VAE: the expensive reference
  xt_tiny   d loss / d x_t through the U-Net and TAESD
  x0_vae    d loss / d x0 through the full VAE only, no U-Net backward (the MPGD shortcut)
  x0_tiny   d loss / d x0 through TAESD only: the cheapest

Cosines are given raw, and low passed to 8x8 latents, because the coarse structure is the
part of the gradient that steers composition. Also counts non-finite values at fp16.
"""
import json
import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch as th
import torch.nn.functional as tf
from diffusers import AutoencoderKL, AutoencoderTiny, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from cgd import clip_util, losses

REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_cache")
PROMPTS = ["a red bicycle leaning against a stone wall", "an oil painting of a lighthouse in a storm",
           "a macro photograph of a dragonfly on a leaf", "two cats playing chess in a library"]
TARGET_STEPS = (4, 8, 12, 16)  # Of 20: from high noise to low noise.
SEEDS = (0, 1, 2)
GUIDANCE = 7.5
DEVICE = "cuda"
FP16 = th.float16


def cosine(a, b):
    return tf.cosine_similarity(a.flatten(1).double(), b.flatten(1).double()).item()


def low(g):
    return tf.avg_pool2d(g.float(), 8)


def main():
    unet = UNet2DConditionModel.from_pretrained(REPO, subfolder="unet", torch_dtype=FP16, local_files_only=True).to(DEVICE).requires_grad_(False).eval()
    vae = AutoencoderKL.from_pretrained(REPO, subfolder="vae", torch_dtype=FP16, local_files_only=True).to(DEVICE).requires_grad_(False).eval()
    tiny = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=FP16, cache_dir=SCRATCH).to(DEVICE).requires_grad_(False).eval()
    clip_model, clip_size = clip_util.load_clip("ViT-B/32", DEVICE, False)
    tokenizer = CLIPTokenizer.from_pretrained(REPO, subfolder="tokenizer", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(REPO, subfolder="text_encoder", local_files_only=True)
    scheduler = DDIMScheduler.from_pretrained(REPO, subfolder="scheduler", local_files_only=True)
    scheduler.config.prediction_type = "epsilon"
    alphas = scheduler.alphas_cumprod.to(DEVICE)
    scale = vae.config.scaling_factor

    def embed(text):
        ids = tokenizer(text, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
        with th.no_grad():
            return text_encoder(ids)[0].to(DEVICE, FP16)

    uncond = embed("")

    def clip_loss(image, target):
        batch = clip_util.preprocess(clip_model, tf.interpolate(image.float().add(1).div(2), size=clip_size, mode="bilinear", align_corners=False))
        return losses.spherical_dist_loss(clip_model.encode_image(batch).float(), target).sum()

    def full_vae(z):
        return vae.decode(z / scale).sample

    def taesd(z):
        return tiny.decode(z).sample

    rows = []
    for prompt in PROMPTS:
        cond = embed(prompt)
        target, _ = clip_util.encode_text_prompt(prompt, 1.0, "ViT-B/32", DEVICE, False)
        target = target.detach().float()
        for seed in SEEDS:
            th.manual_seed(seed)
            scheduler.set_timesteps(20)
            x = th.randn(1, 4, 64, 64, device=DEVICE, dtype=FP16)
            for index, t in enumerate(scheduler.timesteps):
                if index in TARGET_STEPS:
                    alpha, sigma = alphas[t].sqrt().to(FP16), (1 - alphas[t]).sqrt().to(FP16)
                    grads = {}
                    for name, decode in (("xt_vae", full_vae), ("xt_tiny", taesd)):
                        x_grad = x.detach().requires_grad_()
                        eps = unet(x_grad, t, encoder_hidden_states=cond).sample
                        x0 = (x_grad - sigma * eps) / alpha
                        grads[name] = th.autograd.grad(clip_loss(decode(x0), target), x_grad)[0]
                        del x_grad, eps, x0
                    with th.no_grad():
                        x0_fixed = (x - sigma * unet(x, t, encoder_hidden_states=cond).sample) / alpha
                    for name, decode in (("x0_vae", full_vae), ("x0_tiny", taesd)):
                        z = x0_fixed.detach().requires_grad_()
                        grads[name] = th.autograd.grad(clip_loss(decode(z), target), z)[0]
                    bad = {k: int((~v.isfinite()).sum()) for k, v in grads.items()}
                    grads = {k: th.nan_to_num(v.float()) for k, v in grads.items()}
                    reference = grads["xt_vae"]
                    rows.append(dict(
                        prompt=prompt, seed=seed, step=index, t=int(t), nonfinite=bad,
                        tiny_vs_vae_at_x0=cosine(grads["x0_tiny"], grads["x0_vae"]),
                        tiny_vs_vae_at_x0_low=cosine(low(grads["x0_tiny"]), low(grads["x0_vae"])),
                        **{f"{k}_vs_full": cosine(grads[k], reference) for k in ("xt_tiny", "x0_vae", "x0_tiny")},
                        **{f"{k}_vs_full_low": cosine(low(grads[k]), low(reference)) for k in ("xt_tiny", "x0_vae", "x0_tiny")},
                        rms={k: float(v.square().mean().sqrt()) for k, v in grads.items()},
                    ))
                    print(json.dumps(rows[-1]), flush=True)
                    th.cuda.empty_cache()
                with th.no_grad():
                    eps_u = unet(x, t, encoder_hidden_states=uncond).sample
                    eps_c = unet(x, t, encoder_hidden_states=cond).sample
                    x = scheduler.step(eps_u + GUIDANCE * (eps_c - eps_u), t, x).prev_sample
    keys = [k for k in rows[0] if k.endswith(("_full", "_full_low", "_at_x0", "_at_x0_low"))]
    print("SUMMARY by step (mean over prompts and seeds)")
    for step in TARGET_STEPS:
        subset = [r for r in rows if r["step"] == step]
        print(json.dumps(dict(step=step, t=subset[0]["t"], n=len(subset), **{k: round(sum(r[k] for r in subset) / len(subset), 3) for k in keys})))
    print("TOTAL nonfinite", json.dumps({k: sum(r["nonfinite"][k] for r in rows) for k in rows[0]["nonfinite"]}))


if __name__ == "__main__":
    main()

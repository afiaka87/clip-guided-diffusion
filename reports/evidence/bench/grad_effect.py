"""Are the cheap CLIP gradients any use? Real SD 1.5, TAESD, ViT-B/32, scored by PE-Core-B-16.

grad_agree.py found near zero cosine between TAESD and full VAE gradients. That alone
proves little, because two draws of cutouts through the same decoder may not agree either.
So this measures the floor, then measures the outcome.

Part A, at x0 only: gradients through each decoder for six draws of the project's cutouts.
Reports the cosine between draws through the same decoder (the floor) and across decoders,
for one draw and for the mean of three.

Part B, guided DDIM runs that differ only in the gradient path, at one relative strength:
  none, xt_vae (through U-Net and VAE), xt_tiny, x0_vae, x0_tiny.
Each is decoded with the full VAE and scored with the guiding CLIP and a held-out model.
"""
import json
import os
import sys

sys.path[:0] = ["/tmp/claude-1000/-home-sam-GitHub-clip-guided-diffusion/34d680cf-d9a7-412c-9f94-d042cb1cd642/scratchpad/pylib",
                "/home/sam/GitHub/clip-guided-diffusion"]
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch as th
import torch.nn.functional as tf
import torchvision.utils as tvu
from diffusers import AutoencoderKL, AutoencoderTiny, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from cgd import clip_util, losses

th.set_num_threads(int(os.environ.get("BENCH_THREADS", "12")))
REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.path.join(HERE, "hf_cache")
OUT_DIR = os.path.join(HERE, "effect")
PROMPTS = ["a red bicycle leaning against a stone wall", "an oil painting of a lighthouse in a storm",
           "a macro photograph of a dragonfly on a leaf", "two cats playing chess in a library"]
DEVICE = os.environ.get("BENCH_DEVICE", "cpu")
DTYPE = th.float16 if DEVICE == "cuda" else th.float32
PARTS = os.environ.get("BENCH_PARTS", "A")
N_PROMPTS = int(os.environ.get("BENCH_PROMPTS", "3"))
START_PROMPT = int(os.environ.get("BENCH_START_PROMPT", "0"))
SEEDS = tuple(int(s) for s in os.environ.get("BENCH_SEEDS", "0").split(","))
RATE = float(os.environ.get("BENCH_RATE", "0.2"))
GUIDANCE = 7.5
STEPS = 20
WINDOW = (3, 14)  # Guided steps of the 20, inclusive.
A_STEPS = (4, 10, 16)
PATHS = ("none", "xt_vae", "xt_tiny", "x0_vae", "x0_tiny")


def cosine(a, b):
    return tf.cosine_similarity(a.flatten(1).double(), b.flatten(1).double()).item()


def low(g):
    return tf.avg_pool2d(g.float(), 8)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    unet = UNet2DConditionModel.from_pretrained(REPO, subfolder="unet", torch_dtype=DTYPE, local_files_only=True).to(DEVICE).requires_grad_(False).eval()
    vae = AutoencoderKL.from_pretrained(REPO, subfolder="vae", torch_dtype=DTYPE, local_files_only=True).to(DEVICE).requires_grad_(False).eval()
    if DEVICE == "cpu":  # The same gradients in less memory.
        unet.enable_gradient_checkpointing()
        unet.train()
        vae.enable_gradient_checkpointing()
        vae.train()
    tiny = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=DTYPE, cache_dir=SCRATCH).to(DEVICE).requires_grad_(False).eval()
    clip_model, clip_size = clip_util.load_clip("ViT-B/32", DEVICE, False)
    make_cutouts = clip_util.MakeCutouts(cut_size=clip_size, num_cutouts=4, cutout_size_power=1.0, min_cut_fraction=0.7)
    tokenizer = CLIPTokenizer.from_pretrained(REPO, subfolder="tokenizer", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(REPO, subfolder="text_encoder", local_files_only=True)
    scheduler = DDIMScheduler.from_pretrained(REPO, subfolder="scheduler", local_files_only=True)
    alphas = scheduler.alphas_cumprod.to(DEVICE)
    scale = vae.config.scaling_factor

    def embed(text):
        ids = tokenizer(text, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
        with th.no_grad():
            return text_encoder(ids)[0].to(DEVICE, DTYPE)

    uncond = embed("")

    def clip_loss(image, target, draw):
        th.manual_seed(draw)
        batch = clip_util.preprocess(clip_model, make_cutouts(image.float().add(1).div(2)))
        return losses.spherical_dist_loss(clip_model.encode_image(batch).float(), target).mean()

    def full_vae(z):
        return vae.decode(z / scale).sample

    def taesd(z):
        return tiny.decode(z).sample

    def x0_grad(x0, decode, target, draw):
        z = x0.detach().requires_grad_()
        return th.nan_to_num(th.autograd.grad(clip_loss(decode(z), target, draw), z)[0].float())

    def predict(x, t, cond, sigma, alpha):
        with th.no_grad():
            eps = unet(x, t, encoder_hidden_states=uncond).sample
            eps = eps + GUIDANCE * (unet(x, t, encoder_hidden_states=cond).sample - eps)
        return eps, (x - sigma * eps) / alpha

    rows = []
    if "A" in PARTS:
        for prompt in PROMPTS[:N_PROMPTS]:
            cond = embed(prompt)
            target = clip_util.encode_text_prompt(prompt, 1.0, "ViT-B/32", DEVICE, False)[0].detach().float()
            th.manual_seed(0)
            scheduler.set_timesteps(STEPS)
            x = th.randn(1, 4, 64, 64, device=DEVICE, dtype=DTYPE)
            for index, t in enumerate(scheduler.timesteps):
                alpha, sigma = alphas[t].sqrt().to(DTYPE), (1 - alphas[t]).sqrt().to(DTYPE)
                eps, x0 = predict(x, t, cond, sigma, alpha)
                if index in A_STEPS:
                    vae_grads = [x0_grad(x0, full_vae, target, 1000 + k) for k in range(6)]
                    tiny_grads = [x0_grad(x0, taesd, target, 1000 + k) for k in range(6)]
                    v1, v2 = sum(vae_grads[:3]), sum(vae_grads[3:])
                    t1, t2 = sum(tiny_grads[:3]), sum(tiny_grads[3:])
                    pairs = dict(vae_vs_vae_one=(vae_grads[0], vae_grads[1]), tiny_vs_tiny_one=(tiny_grads[0], tiny_grads[1]),
                                 vae_vs_tiny_same_draw=(vae_grads[0], tiny_grads[0]), vae_vs_tiny_other_draw=(vae_grads[0], tiny_grads[1]),
                                 vae_vs_vae_mean3=(v1, v2), tiny_vs_tiny_mean3=(t1, t2), vae_vs_tiny_mean3=(v1, t2))
                    row = dict(part="A", prompt=prompt, step=index, t=int(t))
                    for name, (a, b) in pairs.items():
                        row[name] = round(cosine(a, b), 4)
                        row[name + "_low"] = round(cosine(low(a), low(b)), 4)
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                x = scheduler.step(eps, t, x).prev_sample
        keys = [k for k in rows[0] if k.startswith(("vae_", "tiny_"))]
        for step in A_STEPS:
            subset = [r for r in rows if r["step"] == step]
            print("SUMMARY_A", json.dumps(dict(step=step, n=len(subset), **{k: round(sum(r[k] for r in subset) / len(subset), 3) for k in keys})), flush=True)

    if "B" in PARTS:
        held_out, held_size = clip_util.load_clip("PE-Core-B-16", DEVICE, False)
        results = []
        for prompt_index, prompt in enumerate(PROMPTS[START_PROMPT:N_PROMPTS], START_PROMPT):
            cond = embed(prompt)
            target = clip_util.encode_text_prompt(prompt, 1.0, "ViT-B/32", DEVICE, False)[0].detach().float()
            held_target = clip_util.encode_text_prompt(prompt, 1.0, "PE-Core-B-16", DEVICE, False)[0].detach().float()
            for seed in SEEDS:
                for path in PATHS:
                    decode = taesd if path.endswith("tiny") else full_vae
                    th.manual_seed(seed)
                    scheduler.set_timesteps(STEPS)
                    x = th.randn(1, 4, 64, 64, device=DEVICE, dtype=DTYPE)
                    for index, t in enumerate(scheduler.timesteps):
                        alpha, sigma = alphas[t].sqrt().to(DTYPE), (1 - alphas[t]).sqrt().to(DTYPE)
                        guided = path != "none" and WINDOW[0] <= index <= WINDOW[1]
                        draw = 100000 * seed + 100 * prompt_index + index
                        with th.no_grad():
                            eps_u = unet(x, t, encoder_hidden_states=uncond).sample
                        if guided and path.startswith("xt"):
                            x_grad = x.detach().requires_grad_()
                            eps_c = unet(x_grad, t, encoder_hidden_states=cond).sample
                            x0 = (x_grad - sigma * eps_c) / alpha
                            grad = th.nan_to_num(th.autograd.grad(clip_loss(decode(x0), target, draw), x_grad)[0].float())
                            eps_c = eps_c.detach()
                            del x_grad, x0
                            eps = eps_u + GUIDANCE * (eps_c - eps_u)
                        else:
                            with th.no_grad():
                                eps_c = unet(x, t, encoder_hidden_states=cond).sample
                            eps = eps_u + GUIDANCE * (eps_c - eps_u)
                            if guided:
                                grad = x0_grad((x - sigma * eps) / alpha, decode, target, draw)
                        if guided:
                            size = eps.float().square().mean().sqrt() / grad.square().mean().sqrt().clamp_min(1e-12)
                            eps = eps + (RATE * size * grad).to(eps.dtype)
                        x = scheduler.step(eps, t, x).prev_sample
                    with th.no_grad():
                        image = full_vae(x).float().add(1).div(2).clamp(0, 1)
                        small = tf.interpolate(image, size=clip_size, mode="bicubic", antialias=True, align_corners=False).clamp(0, 1)
                        held = tf.interpolate(image, size=held_size, mode="bicubic", antialias=True, align_corners=False).clamp(0, 1)
                        guide_sim = tf.cosine_similarity(clip_model.encode_image(clip_util.preprocess(clip_model, small)).float(), target).item()
                        held_sim = tf.cosine_similarity(held_out.encode_image(clip_util.preprocess(held_out, held)).float(), held_target).item()
                    tvu.save_image(image, os.path.join(OUT_DIR, f"{prompt_index}_{seed}_{path}_r{RATE}.png"))
                    result = dict(part="B", prompt=prompt_index, seed=seed, path=path, rate=RATE, guide_clip=round(guide_sim, 4), held_out=round(held_sim, 4))
                    results.append(result)
                    print(json.dumps(result), flush=True)
        for path in PATHS:
            subset = [r for r in results if r["path"] == path]
            base = [r for r in results if r["path"] == "none"]
            n = len(subset)
            print("SUMMARY_B", json.dumps(dict(path=path, n=n, guide_clip=round(sum(r["guide_clip"] for r in subset) / n, 4), held_out=round(sum(r["held_out"] for r in subset) / n, 4),
                                               guide_delta=round(sum(a["guide_clip"] - b["guide_clip"] for a, b in zip(subset, base)) / n, 4),
                                               held_delta=round(sum(a["held_out"] - b["held_out"] for a, b in zip(subset, base)) / n, 4))), flush=True)


if __name__ == "__main__":
    main()

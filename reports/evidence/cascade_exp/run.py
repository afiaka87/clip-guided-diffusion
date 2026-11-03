"""Scratch experiment: cascades and pixel-space hires fix against direct guided sampling."""
import gc, json, os, subprocess, sys, time
REPO = "/home/sam/GitHub/clip-guided-diffusion"
sys.path.insert(0, REPO)
HERE = os.path.dirname(os.path.abspath(__file__))
SP = os.path.dirname(HERE)
os.chdir(HERE)  # log_image copies to ./current.png
import numpy as np
import torch as th
import torch.nn.functional as tf
from PIL import Image
import torchvision.transforms.functional as tvf
from guided_diffusion.script_util import sr_model_and_diffusion_defaults, sr_create_model_and_diffusion
from cgd import clip_util, losses, sampling, script_util
from cgd.cgd import clip_guided_diffusion, FAST_NUM_CUTOUTS, FAST_MIN_CUT_FRACTION

DEVICE = "cuda"
PROMPTS = [
    "a red fox in the snow", "a plate of sushi on a black slate",
    "a pencil sketch of an old man with a beard", "a lighthouse on a rocky coast at sunset",
    "a bowl of ramen with a soft boiled egg", "an oil painting of a sailing ship in a storm",
    "a golden retriever puppy on a wooden floor", "a steam locomotive crossing a stone bridge",
    "a watercolor of cherry blossoms by a river", "a red vintage sports car on a desert road",
    "a snow covered cabin in a pine forest at night", "a macro photograph of a dew covered dragonfly",
]
LOG = open(os.path.join(HERE, "results.jsonl"), "a", buffering=1)


def others_mib():
    used = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])
    return used - th.cuda.memory_reserved() / 2**20


def gate(limit=1500, patience=180):
    start = time.time()
    while others_mib() > limit and time.time() - start < patience:
        time.sleep(3)


def measured(fn, retries=30):
    for attempt in range(retries):
        gate()
        before = others_mib()
        th.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        try:
            result = fn()
        except th.cuda.OutOfMemoryError:
            gc.collect(); th.cuda.empty_cache(); time.sleep(20)
            continue
        th.cuda.synchronize()
        return result, dict(sec=round(time.perf_counter() - start, 3), peak_alloc_gib=round(th.cuda.max_memory_allocated() / 2**30, 3),
                            peak_reserved_gib=round(th.cuda.max_memory_reserved() / 2**30, 3), others_before=round(before), others_after=round(others_mib()))
    raise RuntimeError("gave up")


def record(pipe, prompt, path, stats):
    LOG.write(json.dumps(dict(pipe=pipe, prompt=prompt, path=path, **stats)) + "\n")


def direct(name, size, **kwargs):
    for prompt in PROMPTS:
        def run():
            return list(clip_guided_diffusion(prompts=[prompt], image_size=size, seed=0, prefix_path=os.path.join(HERE, "out", name),
                                              save_frequency=1000, progress=False, device=DEVICE, **{k: (v(prompt) if callable(v) else v) for k, v in kwargs.items()}))[-1][1]
        path, stats = measured(run)
        record(name, prompt, path, stats)


def paths(name):
    return {r["prompt"]: r["path"] for r in map(json.loads, open(os.path.join(HERE, "results.jsonl"))) if r["pipe"] == name}


def load_sr(filename, large, small, **flags):
    config = sr_model_and_diffusion_defaults()
    config.update(large_size=large, small_size=small, class_cond=True, learn_sigma=True, diffusion_steps=1000, noise_schedule="linear",
                  num_channels=192, num_res_blocks=2, resblock_updown=True, use_fp16=True, use_scale_shift_norm=True, timestep_respacing="",
                  use_checkpoint=large * large > 256 ** 2, **flags)
    with th.device("meta"):
        model, diffusion = sr_create_model_and_diffusion(**config)
    state_dict = th.load(os.path.join(SP, "weights_sr", filename), map_location="cpu", mmap=True, weights_only=True)
    missing = model.load_state_dict(state_dict, assign=True)  # strict: raises on any mismatch
    params = sum(p.numel() for p in model.parameters())
    print(filename, "strict load ok", missing, "params", params, flush=True)
    model.requires_grad_(False).eval()
    model.convert_to_fp16()
    return model.to(DEVICE), diffusion, params


def upsample(name, model, diffusion, large, base_paths, guided):
    clip_model, clip_size = clip_util.load_clip("ViT-B/32", DEVICE, False)
    cutouts = clip_util.MakeCutouts(cut_size=clip_size, num_cutouts=FAST_NUM_CUTOUTS, cutout_size_power=1.0, use_augs=False, min_cut_fraction=FAST_MIN_CUT_FRACTION)
    for prompt in PROMPTS:
        target, _ = clip_util.encode_text_prompt(prompt, 1.0, "ViT-B/32", DEVICE, False)
        class_index = clip_util.imagenet_top_n(tf.normalize(target, dim=-1), device=DEVICE, n=1, clip_model_name="ViT-B/32", fp16=False)
        low = tvf.to_tensor(Image.open(base_paths[prompt]).convert("RGB")).unsqueeze(0).to(DEVICE).mul(2).sub(1)
        kwargs = {"y": class_index.reshape(1), "low_res": low}

        def loss_fn(x, pred_xstart, fac):
            x_in = pred_xstart * fac + x * (1 - fac)
            embeds = clip_model.encode_image(clip_util.preprocess(clip_model, cutouts(x_in.add(1).div(2)))).float()
            embeds = embeds.view(-1, x.shape[0], embeds.shape[-1])
            clip_loss = losses.spherical_dist_loss(embeds.unsqueeze(2), target).sum(2).mean(0).sum() * 1000
            return clip_loss + losses.tv_loss(x_in).sum() * 150 + losses.range_loss(pred_xstart).sum() * 50

        def run():
            th.manual_seed(0)
            final = None
            for sample in sampling.sample_loop_progressive(model, diffusion, (1, 3, large, large), loss_fn=loss_fn if guided else None,
                                                           model_kwargs=kwargs, device=DEVICE, progress=False):
                final = sample["pred_xstart"]
            return final
        image, stats = measured(run)
        out = os.path.join(HERE, "out", name)
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, script_util.alphanumeric_filter(prompt) + ".png")
        tvf.to_pil_image(image[0].add(1).div(2).clamp(0, 1).cpu().float()).save(path)
        record(name, prompt, path, stats)


def free():
    script_util.load_guided_diffusion.cache_clear()
    gc.collect(); th.cuda.empty_cache()


for name, size in (("d64", 64), ("d128", 128), ("d256", 256), ("d512", 512)):
    direct(name, size); free()
print("direct done", flush=True)

model, diffusion, params = load_sr("64_256_upsampler.pt", 256, 64, attention_resolutions="32,16,8", num_heads=4)
LOG.write(json.dumps(dict(info="64_256", params=params)) + "\n")
upsample("sr256_u", model, diffusion, 256, paths("d64"), False)
upsample("sr256_g", model, diffusion, 256, paths("d64"), True)
del model, diffusion; free()
print("sr256 done", flush=True)

direct("i2i256_s500", 256, init_image=lambda p: paths("d128")[p], skip_timesteps=500); free()
direct("i2i256_s700", 256, init_image=lambda p: paths("d128")[p], skip_timesteps=700); free()
direct("i2i512_s700", 512, init_image=lambda p: paths("d256")[p], skip_timesteps=700); free()
print("i2i done", flush=True)

model, diffusion, params = load_sr("128_512_upsampler.pt", 512, 128, attention_resolutions="32,16", num_head_channels=64)
LOG.write(json.dumps(dict(info="128_512", params=params)) + "\n")
upsample("sr512_u", model, diffusion, 512, paths("d128"), False)
upsample("sr512_g", model, diffusion, 512, paths("d128"), True)
print("all done", flush=True)

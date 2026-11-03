"""Scratch: the cascade stages run.py never reached, plus low-res seed selection before upsampling."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import torch.nn.functional as tf
from PIL import Image
import torchvision.transforms.functional as tvf
from guided_diffusion.script_util import sr_model_and_diffusion_defaults, sr_create_model_and_diffusion
from cgd import clip_util, losses, sampling, script_util
from cgd.cgd import clip_guided_diffusion, FAST_NUM_CUTOUTS, FAST_MIN_CUT_FRACTION

DEVICE = "cuda"


def direct(name, size, seed=0, **kwargs):
    done = paths(name)
    for prompt in PROMPTS:
        if prompt in done:
            continue

        def run():
            return list(clip_guided_diffusion(prompts=[prompt], image_size=size, seed=seed, prefix_path=os.path.join(HERE, "out", name),
                                              save_frequency=1000, progress=False, device=DEVICE, **{k: (v(prompt) if callable(v) else v) for k, v in kwargs.items()}))[-1][1]
        path, stats = measured(run)
        record(name, prompt, path, stats)


def load_sr(filename, large, small, **flags):
    config = sr_model_and_diffusion_defaults()
    config.update(large_size=large, small_size=small, class_cond=True, learn_sigma=True, diffusion_steps=1000, noise_schedule="linear",
                  num_channels=192, num_res_blocks=2, resblock_updown=True, use_fp16=True, use_scale_shift_norm=True, timestep_respacing="",
                  use_checkpoint=large * large > 256 ** 2, **flags)
    with th.device("meta"):
        model, diffusion = sr_create_model_and_diffusion(**config)
    state_dict = th.load(os.path.join(SP, "weights_sr", filename), map_location="cpu", mmap=True, weights_only=True)
    result = model.load_state_dict(state_dict, assign=True)
    params = sum(p.numel() for p in model.parameters())
    print(filename, "strict load ok", result, "params", params, flush=True)
    model.requires_grad_(False).eval()
    model.convert_to_fp16()
    return model.to(DEVICE), diffusion, params


def upsample(name, model, diffusion, large, base_paths, guided):
    done = paths(name)
    clip_model, clip_size = clip_util.load_clip("ViT-B/32", DEVICE, False)
    cutouts = clip_util.MakeCutouts(cut_size=clip_size, num_cutouts=FAST_NUM_CUTOUTS, cutout_size_power=1.0, use_augs=False, min_cut_fraction=FAST_MIN_CUT_FRACTION)
    for prompt in PROMPTS:
        if prompt in done:
            continue
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


def choose_best():
    """Keep the seed ViT-B/16 likes most, on the CPU. Guidance used ViT-B/32, so this judge did not steer."""
    judge, size = clip_util.load_clip("ViT-B/16", "cpu")
    text = tf.normalize(clip_util.encode_texts(PROMPTS, "ViT-B/16", "cpu"), dim=-1)
    tables = [paths("d64")] + [paths(f"sel64_s{seed}") for seed in range(1, 8)]
    best, start = {}, time.perf_counter()
    for index, prompt in enumerate(PROMPTS):
        ranked = []
        for table in tables:
            image = tvf.to_tensor(Image.open(table[prompt]).convert("RGB")).unsqueeze(0)
            with th.no_grad():
                x = tf.interpolate(image, size=(size, size), mode="bicubic", antialias=True).clamp(0, 1)
                feature = tf.normalize(judge.encode_image(clip_util.preprocess(judge, x)).float(), dim=-1)
            ranked.append((float(feature[0] @ text[index]), table[prompt]))
        best[prompt] = max(ranked)[1]
    print("judged 8 candidates per prompt on the CPU, sec per image", (time.perf_counter() - start) / (8 * len(PROMPTS)), flush=True)
    return best


for seed in range(1, 8):
    direct(f"sel64_s{seed}", 64, seed=seed)
free()
best = choose_best()
print("candidates done", flush=True)

model, diffusion, params = load_sr("64_256_upsampler.pt", 256, 64, attention_resolutions="32,16,8", num_heads=4)
upsample("sr256_u", model, diffusion, 256, paths("d64"), False)
upsample("sr256_g", model, diffusion, 256, paths("d64"), True)
upsample("sel8_sr256_u", model, diffusion, 256, best, False)
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

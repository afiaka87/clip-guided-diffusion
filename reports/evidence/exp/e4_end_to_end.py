"""End to end: full sampler runs per variant, scored by models that did not guide them.

Same sampler noise for every variant of a prompt and seed, so differences are paired.
Scores follow the README: CLIP similarity to the prompt under held-out judges.
"""
import os
import sys
import time

import numpy as np
import open_clip
import torch as th
import torch.nn.functional as tf
from PIL import Image

from harness import OUT, PROMPTS, Config, Pipeline, clip_util, device, mean_ci, save_json, wait_for_gpu

name = sys.argv[1]  # "ofat", or a variants file key
size = int(sys.argv[2]) if len(sys.argv) > 2 else 128
clip_name = sys.argv[3] if len(sys.argv) > 3 else "ViT-B/32"
fp16 = len(sys.argv) > 4 and sys.argv[4] == "fp16"
seeds = (0, 1)
bic = dict(resample="bicubic")

SETS = {
    "ofat": {
        "base": Config(),
        "bilinear": Config(resample="bilinear"),
        "bicubic": Config(**bic),
        "lanczos3": Config(resample="lanczos3"),
        "lhs": Config(layout="lhs"),
        "r3": Config(layout="r3"),
        "antithetic": Config(layout="antithetic"),
        "overview+3": Config(layout="overview"),
        "disco schedule": Config(layout="disco"),
        "1 whole": Config(layout="whole", n=1),
        "1 whole jitter": Config(layout="jitter", n=1),
        "4 whole jitter": Config(layout="jitter"),
        "min .4": Config(min_frac=0.4),
        "flip": Config(flip=True),
        "gray .25": Config(gray=0.25),
        "noise .03": Config(noise=0.03),
        "noise .1": Config(noise=0.1),
        "pure x0": Config(pure_x0=True),
        "per-cut norm": Config(agg="norm"),
        "blur 1": Config(blur=1.0),
        "blur 2": Config(blur=2.0),
        "ema .5": Config(ema=0.5),
        "ema .8": Config(ema=0.8),
        "n8": Config(n=8),
        "n16": Config(n=16),
        "n32": Config(n=32),
        "n8 mom": Config(n=8, agg="mom"),
        "n8 reject": Config(n=8, agg="reject"),
    },
}
exec(open(os.path.dirname(os.path.abspath(__file__)) + "/e4_sets.py").read()) if os.path.exists(os.path.dirname(os.path.abspath(__file__)) + "/e4_sets.py") else None
variants = SETS[name]
tag = f"e4_{name}_{size}_{clip_name.replace('/', '-')}{'_fp16' if fp16 else ''}"
image_dir = f"{OUT}/{tag}"
os.makedirs(image_dir, exist_ok=True)

print("waiting for gpu:", wait_for_gpu(), flush=True)
pipe = Pipeline(size, clip_name, fp16)
images = {}
times = {}
th.cuda.reset_peak_memory_stats()
for variant, cfg in variants.items():
    start = time.time()
    for prompt_index, prompt in enumerate(PROMPTS):
        for seed in seeds:
            image = pipe.sample(cfg, prompt, seed)
            images[variant, prompt_index, seed] = image
            Image.fromarray(image.permute(1, 2, 0).numpy()).save(f"{image_dir}/{variant.replace(' ', '_')}_{prompt_index:02d}_{seed}.png")
    times[variant] = (time.time() - start) / (len(PROMPTS) * len(seeds))
    print(f"{variant:20s} {times[variant]:.2f} s/image", flush=True)
peak = th.cuda.max_memory_allocated() / 2**30
del pipe
clip_util.load_clip.cache_clear()
th.cuda.empty_cache()


def judge_scores(judge, pretrained):
    model, _, preprocess = open_clip.create_model_and_transforms(judge, pretrained=pretrained, cache_dir=clip_util.CLIP_CACHE_PATH)
    model = model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(judge, cache_dir=clip_util.CLIP_CACHE_PATH)
    with th.no_grad():
        texts = tf.normalize(model.encode_text(tokenizer(PROMPTS).to(device)).float(), dim=-1)
        scores = {}
        keys = list(images)
        for start in range(0, len(keys), 16):
            part = keys[start:start + 16]
            batch = th.stack([preprocess(Image.fromarray(images[key].permute(1, 2, 0).numpy())) for key in part]).to(device)
            features = tf.normalize(model.encode_image(batch).float(), dim=-1)
            for key, feature in zip(part, features):
                scores[key] = (feature @ texts[key[1]]).item()
    del model
    th.cuda.empty_cache()
    return scores


JUDGES = [("ViT-B-16", "openai"), ("ViT-H-14-quickgelu", "dfn5b"), ("ViT-L-16-SigLIP2-256", "webli")]
if clip_name == "ViT-B/16":
    JUDGES = [("ViT-B-32", "openai")] + JUDGES[1:]
if clip_name == "ViT-L-16-SigLIP2-256":
    JUDGES = JUDGES[:2]
all_scores = {f"{j}": judge_scores(j, p) for j, p in JUDGES}


def high_frequency(image):
    gray = image.float().mean(0) / 255
    power = th.fft.rfft2(gray - gray.mean()).abs() ** 2
    fy = th.fft.fftfreq(gray.shape[0])[:, None]
    fx = th.fft.rfftfreq(gray.shape[1])[None]
    return (power[(fy**2 + fx**2).sqrt() > 0.25].sum() / power.sum()).item()


summary = {}
print(f"\npeak VRAM of sampling {peak:.2f} GiB. Paired difference from base, mean over 12 prompts of the mean over {len(seeds)} seeds, +- 95% CI, (prompts improved)")
print(f"{'variant':20s} {'s/img':>6s} " + " ".join(f"{j[:18]:>24s}" for j, _ in JUDGES) + f" {'all judges':>24s} {'hf':>7s}")
baseline = next(iter(variants))
for variant in variants:
    row = {}
    per_judge = []
    cells = []
    for judge, _ in JUDGES:
        scores = all_scores[judge]
        diffs = [np.mean([scores[variant, p, s] - scores[baseline, p, s] for s in seeds]) for p in range(len(PROMPTS))]
        absolute = np.mean([scores[variant, p, s] for p in range(len(PROMPTS)) for s in seeds])
        mean, ci = mean_ci(diffs)
        row[judge] = {"mean": mean, "ci": ci, "wins": int(np.sum(np.array(diffs) > 0)), "absolute": float(absolute)}
        per_judge.append(diffs)
        cells.append(f"{mean:+.4f}+-{ci:.4f} ({row[judge]['wins']:2d})")
    pooled = np.mean(per_judge, axis=0)
    mean, ci = mean_ci(pooled)
    row["all"] = {"mean": mean, "ci": ci, "wins": int(np.sum(pooled > 0))}
    hf = np.mean([high_frequency(images[variant, p, s]) for p in range(len(PROMPTS)) for s in seeds])
    row["hf"] = float(hf)
    row["seconds"] = times[variant]
    summary[variant] = row
    print(f"{variant:20s} {times[variant]:6.2f} " + " ".join(f"{c:>24s}" for c in cells) + f" {mean:+.4f}+-{ci:.4f} ({row['all']['wins']:2d}) {hf:7.4f}", flush=True)
save_json(tag, summary)

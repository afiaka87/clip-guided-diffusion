"""How repeatable and how transferable is the guidance gradient, per scheme and count?

At real states of a real guided run, draw independent k-cutout gradient estimates.
- pair: mean cosine between independent estimates. 1 means fully repeatable.
- self: cosine with the same model's 256-cutout average gradient.
- judge: cosine with a *different* model's 256-cutout average. A model's own quirks do
  not transfer, so this is the part of the gradient that is about the image.
"""
import itertools
import sys

import torch as th

from harness import Config, Pipeline, PROMPTS, Views, aggregate, blur_matrix, clip_util, device, per_cut_gradients, resample_matrix, save_json, wait_for_gpu

clip_name = sys.argv[1] if len(sys.argv) > 1 else "ViT-B/32"
size = int(sys.argv[2]) if len(sys.argv) > 2 else 128
fp16 = len(sys.argv) > 3 and sys.argv[3] == "fp16"
tag = f"{clip_name.replace('/', '-')}_{size}{'_fp16' if fp16 else ''}"
COUNTS = (1, 2, 4, 8, 16, 32)
GROUPS = 8
STEPS = {8: "s=.997", 13: "s=.92", 16: "s=.67", 19: "s=.33"}
SCHEMES = {
    "iid area (repo)": Config(),
    "iid bicubic": Config(resample="bicubic"),
    "iid lanczos3": Config(resample="lanczos3"),
    "iid bicubic flip": Config(resample="bicubic", flip=True),
    "LHS bicubic": Config(resample="bicubic", layout="lhs"),
    "R3 bicubic": Config(resample="bicubic", layout="r3"),
    "antithetic bicubic": Config(resample="bicubic", layout="antithetic"),
    "overview+inner bicubic": Config(resample="bicubic", layout="overview"),
    "whole frame jitter bicubic": Config(resample="bicubic", layout="jitter"),
    "whole frame noise .02": Config(resample="bicubic", layout="whole", noise=0.02),
    "whole frame noise .05": Config(resample="bicubic", layout="whole", noise=0.05),
    "iid bicubic noise .02": Config(resample="bicubic", noise=0.02),
    "iid bicubic noise .05": Config(resample="bicubic", noise=0.05),
}
if size >= 256:
    SCHEMES["iid wide (min .4)"] = Config(resample="bicubic", min_frac=0.4)

print("waiting for gpu:", wait_for_gpu(), flush=True)
pipe = Pipeline(size, clip_name, fp16)
states = []


def recorder(store):
    def hook(step, x, x0, x_in, fac, views):
        if step + 6 in STEPS:  # Guidance starts at model call 6.
            store.append((step + 6, x_in.detach().clone(), float(fac)))
    return hook


prompts = PROMPTS[:4]
for prompt in prompts:
    store = []
    pipe.sample(Config(), prompt, 0, hook=recorder(store))
    states.append((prompt, store))
    print("captured", prompt, [s[0] for s in store], flush=True)


def big_average(clip_model, clip_size, text, x01, seed, n=256):
    gen = th.Generator().manual_seed(seed)
    views = Views(clip_size, Config(resample="bicubic", n=n), gen)
    plan = views.draw(x01.shape[-1], 0.5)
    grads, _ = per_cut_gradients(clip_model, text, views, x01, plan)
    return grads.mean(0).flatten()


def cosine(a, b):
    return th.nn.functional.cosine_similarity(a, b, dim=-1)


judges = {}
for judge_name, judge_fp16 in (("ViT-B/16", False), ("PE-Core-B-16", True)):
    if judge_name == clip_name:
        continue
    clip_util.load_clip.cache_clear()
    model, judge_size = clip_util.load_clip(judge_name, device, judge_fp16)
    judges[judge_name] = (model, judge_size, judge_fp16)
clip_util.load_clip.cache_clear()
guide, guide_size = clip_util.load_clip(clip_name, device, fp16)

results = {}
for (prompt, store), (step, x_in, fac) in ((p, s) for p in states for s in p[1]):
    x01 = x_in.add(1).div(2)
    text, _ = pipe.target(prompt)
    truth = big_average(guide, guide_size, text, x01, 7)
    truth2 = big_average(guide, guide_size, text, x01, 8)
    judge_truth = {}
    for judge_name, (model, judge_size, judge_fp16) in judges.items():
        judge_text, _ = clip_util.encode_text_prompt(prompt, 1.0, judge_name, device, judge_fp16)
        judge_truth[judge_name] = big_average(model, judge_size, judge_text, x01, 9)
    print(f"{prompt[:30]:30s} step {step} {STEPS[step]} truth self-check cos={cosine(truth, truth2).item():.3f}", flush=True)
    for name, cfg in SCHEMES.items():
        row = {}
        for k in COUNTS:
            cfg_k = Config(**{**cfg.__dict__, "n": k})
            gen = th.Generator().manual_seed(abs(hash((name, k))) % 2**31)
            views = Views(guide_size, cfg_k, gen)
            estimates = []
            for group in range(GROUPS):
                plan = views.draw(x01.shape[-1], fac)
                grads, _ = per_cut_gradients(guide, text, views, x01, plan)
                estimates.append(aggregate(grads, cfg_k).flatten())
            estimates = th.stack(estimates)
            pair = th.stack([cosine(a, b) for a, b in itertools.combinations(estimates, 2)]).mean().item()
            row[k] = {"pair": pair, "self": cosine(estimates, truth).mean().item(),
                      "judge": {j: cosine(estimates, t).mean().item() for j, t in judge_truth.items()},
                      "norm": estimates.norm(dim=1).mean().item()}
        results.setdefault(name, {}).setdefault(STEPS[step], []).append(row)
    save_json(f"e3_{tag}", results)

print("\n=== mean over prompts ===")
for name, by_step in results.items():
    for step_name, rows in by_step.items():
        line = []
        for k in COUNTS:
            pair = sum(r[k]["pair"] for r in rows) / len(rows)
            self_cos = sum(r[k]["self"] for r in rows) / len(rows)
            judge = {j: sum(r[k]["judge"][j] for r in rows) / len(rows) for j in judges}
            line.append(f"k{k}: p{pair:.3f} s{self_cos:.3f} " + " ".join(f"j{v:.3f}" for v in judge.values()))
        print(f"{name:28s} {step_name:7s} " + " | ".join(line), flush=True)

"""Aesthetic guidance through the repo's ClipGuide. argv: <SP> <REPO> [arms]; env SEEDS, BACKBONE."""
import json, os, sys, time
S, REPO = sys.argv[1], sys.argv[2]; ONLY = sys.argv[3].split(";") if len(sys.argv) > 3 else None
sys.path.insert(0, REPO)
import torch as th
from PIL import Image
from cgd import clip_util, modern
exec(open(f"{S}/scripts/prompts.py").read())
SEEDS = [int(v) for v in os.environ.get("SEEDS", "1,2,3").split(",")]
NAME = os.environ.get("BACKBONE", "sd-turbo"); dev = "cuda"; SIZE = int(os.environ.get("SIZE", modern.BACKBONES[NAME].size)); TAG = f"{NAME}{SIZE}"
EMB = th.load(f"{S}/turbo/emb_{NAME}.pt", weights_only=False)
real_load = clip_util.load_clip; loaded = {}
def cached(name="ViT-B/32", device="cpu", fp16=False):
    if (name, fp16) not in loaded:
        real_load.cache_clear(); loaded[(name, fp16)] = real_load(name, device, fp16)
    return loaded[(name, fp16)]
clip_util.load_clip = cached
B = modern.load_backbone(NAME, "", dev, True)
V = {"plain": dict(aes=0., plain=True), "base": dict(aes=0.), "aes .05": dict(aes=.05), "aes .1": dict(aes=.1), "aes .2": dict(aes=.2), "aes .4": dict(aes=.4), "aes .8": dict(aes=.8)}
ENS = os.environ.get("ENS", "ViT-B/32+PE-Core-B-16")
out_json = f"{S}/results/aes_{TAG}.json"
paths = json.load(open(out_json)) if os.path.exists(out_json) else {}
LIMIT = int(os.environ.get("LIMIT", 12)); PROMPTS = PROMPTS[:LIMIT]
if "LIMIT" in os.environ: out_json = out_json.replace(".json", "_probe.json")
for arm, cfg in V.items():
    if ONLY and arm not in ONLY: continue
    for seed in SEEDS:
        key = f"{arm} s{seed}"
        if key in paths: continue
        d = f"{S}/turbo/aes_{TAG}/{key.replace(' ', '_')}"; os.makedirs(d, exist_ok=True); got = []
        th.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
        for i, p in enumerate(PROMPTS):
            B.text, pooled = EMB[p]
            if pooled is not None: B.pooled = pooled
            shape = B.latent_shape(SIZE, SIZE)
            while True:
                try:
                    th.manual_seed(seed)
                    guide = clip_util.ClipGuide(ENS, [p], (), SIZE, 4, dev, min_cut_fraction=.7, aesthetic_scale=cfg["aes"])
                    z = th.randn(shape, device=dev)
                    if not cfg.get("plain"):
                        for step in modern.optimise_noise(B, guide, z, progress=False):
                            pass
                        z = step["best_z"]
                    with th.no_grad(): img = modern.render(B, z, 1)
                    break
                except th.cuda.OutOfMemoryError:
                    print("OOM, waiting", flush=True); th.cuda.empty_cache(); time.sleep(20)
            Image.fromarray(img[0].add(1).div(2).clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()).save(f"{d}/{i}.png"); got.append(f"{d}/{i}.png")
        paths[key] = got; json.dump(paths, open(out_json, "w"), indent=0)
        print(key, f"{(time.perf_counter() - t0) / LIMIT:.1f}s/img peak {th.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
print("done", flush=True)

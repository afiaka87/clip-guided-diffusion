"""Cutout schemes on ADM through the repo pipeline. argv: <SP> <REPO> [arms;]; env SEEDS, SIZE."""
import json, os, sys, time
S, REPO = sys.argv[1], sys.argv[2]; ONLY = sys.argv[3].split(";") if len(sys.argv) > 3 else None
sys.path.insert(0, REPO); sys.path.insert(0, f"{S}/cuts")
import torch as th
from cgd import clip_util
import cutlib
clip_util.MakeCutouts = cutlib.Scheme
from cgd.cgd import clip_guided_diffusion
exec(open(f"{S}/scripts/prompts.py").read())
SEEDS = [int(v) for v in os.environ.get("SEEDS", "1,2").split(",")]
SIZE = int(os.environ.get("SIZE", 128))
V = {
 "c random4": dict(crops=4),
 "c whole": dict(whole=1),
 "c whole+3 .44-.7": dict(whole=1, crops=3, low=.44, high=.7),
 "c whole+3 .7-1": dict(whole=1, crops=3),
 "c whole+7 .44-.7": dict(whole=1, crops=7, low=.44, high=.7),
 "c 16": dict(crops=16),
 "c strat4": dict(crops=4, strat=True),
 "c pad4": dict(crops=4, pad=True),
 "c strat+pad4": dict(crops=4, pad=True, strat=True),
 "c flip2": dict(crops=2, flip=True),
 "c anneal 1-.44": dict(crops=4, anneal=(.8, 1., .44, .7)),
 "c anneal .44-1": dict(crops=4, anneal=(.44, .7, .8, 1.)),
 "c whole+strat+pad3 .44-.7": dict(whole=1, crops=3, low=.44, high=.7, pad=True, strat=True),
 "c 8": dict(crops=8), "c 32": dict(crops=32), "c 64": dict(crops=64), "c strat16": dict(crops=16, strat=True),
 "c ens 16": dict(crops=16, kw=dict(clip_model_name="ViT-B/32+PE-Core-B-16")),
 "c ensb 16": dict(crops=16, kw=dict(clip_model_name="ViT-B/32+PE-Core-B-16")),
 "c ens 4": dict(crops=4, kw=dict(clip_model_name="ViT-B/32+PE-Core-B-16")),
}
out_json = f"{S}/results/admcut{SIZE}.json"
paths = json.load(open(out_json)) if os.path.exists(out_json) else {}
for name, config in V.items():
    if ONLY and name not in ONLY: continue
    for seed in SEEDS:
        key = f"{name} s{seed}"
        if key in paths: continue
        cfg = dict(config); kw = cfg.pop("kw", {}); cutlib.CONFIG.clear(); cutlib.CONFIG.update(cfg)
        th.cuda.reset_peak_memory_stats(); t0 = time.perf_counter(); got = []
        for i, p in enumerate(PROMPTS):
            d = f"{S}/cuts/out{SIZE}/{key.replace(' ', '_').replace('+', '_')}/{i}"
            while True:
                try:
                    last = None
                    for _, path in clip_guided_diffusion(prompts=[p], image_size=SIZE, seed=seed, prefix_path=d, progress=False, save_frequency=1000, **kw):
                        last = path
                    break
                except th.cuda.OutOfMemoryError:
                    print("OOM, waiting", flush=True); th.cuda.empty_cache(); time.sleep(20)
            got.append(str(last))
        paths[key] = got
        json.dump(paths, open(out_json, "w"), indent=0)
        print(key, f"{(time.perf_counter() - t0) / len(PROMPTS):.1f}s/img peak {th.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
print("done", flush=True)

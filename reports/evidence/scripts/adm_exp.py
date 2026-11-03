"""Fresh seed test of the ADM options through the repo's own pipeline. argv: <SP> <REPO> [arm,arm]; env SEEDS."""
import json, os, sys, time
S, REPO = sys.argv[1], sys.argv[2]; ONLY = sys.argv[3].split(",") if len(sys.argv) > 3 else None
sys.path.insert(0, REPO)
import torch as th
from cgd.cgd import clip_guided_diffusion
exec(open(f"{S}/scripts/prompts.py").read())
SEEDS = [int(v) for v in os.environ.get("SEEDS", "1,2,3,4").split(",")]
SIZE = int(os.environ.get("SIZE", 128))
B32PE = "ViT-B/32+PE-Core-B-16"
V = {
 "base": {},
 "dt .995": dict(x0_threshold=.995), "dt .999": dict(x0_threshold=.999), "dt .9995": dict(x0_threshold=.9995),
 "sine": dict(guidance_schedule="sine"), "rising": dict(guidance_schedule="rising"),
 "ens B32+PE": dict(clip_model_name=B32PE), "PE": dict(clip_model_name="PE-Core-B-16"),
 "contrastive": dict(clip_loss="contrastive"),
 "view detached": dict(clip_view="detached"), "view x0": dict(clip_view="x0"),
 "pag .5": dict(pag_scale=.5), "pag 1": dict(pag_scale=1.), "pag 3": dict(pag_scale=3.),
 "contrast .5": dict(contrast_scale=.5), "class 2nd": dict(class_rank=1),
 "ens+contrastive": dict(clip_model_name=B32PE, clip_loss="contrastive"),
 "ens+contrastive+dt.999": dict(clip_model_name=B32PE, clip_loss="contrastive", x0_threshold=.999),
 "er-sde": dict(solver="er-sde"), "secondary": dict(secondary_guidance=True),
 "limit .99": dict(guidance_limit=.99), "limit .999": dict(guidance_limit=.999), "ens+limit .99": dict(clip_model_name=B32PE, guidance_limit=.99), "ens+er-sde": dict(clip_model_name=B32PE, solver="er-sde"),
 "aes .05": dict(aesthetic_scale=.05), "aes .2": dict(aesthetic_scale=.2), "aes .5": dict(aesthetic_scale=.5), "aes 1": dict(aesthetic_scale=1.),
 "ens+aes .2": dict(clip_model_name=B32PE, aesthetic_scale=.2),
 "pag2 .5": dict(pag_scale=.5), "pag2 1": dict(pag_scale=1.), "pag2 3": dict(pag_scale=3.), "contrast2 .5": dict(contrast_scale=.5), "contrast2 .2": dict(contrast_scale=.2),
}
out_json = f"{S}/results/adm{SIZE}_paths.json"
paths = json.load(open(out_json)) if os.path.exists(out_json) else {}
for name, kw in V.items():
    if ONLY and name not in ONLY: continue
    for seed in SEEDS:
        key = f"{name} s{seed}"
        if key in paths: continue
        th.cuda.reset_peak_memory_stats(); t0 = time.perf_counter(); got = []
        for i, p in enumerate(PROMPTS):
            d = f"{S}/adm/out{SIZE}/{key.replace(' ', '_').replace('+', '_')}/{i}"
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

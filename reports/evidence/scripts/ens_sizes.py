"""Ensemble against the default at 256 and 512, alone on the card. argv: <SP> <REPO>."""
import json, sys, time
S, REPO = sys.argv[1], sys.argv[2]
sys.path.insert(0, REPO)
import torch as th
from cgd.cgd import clip_guided_diffusion
exec(open(f"{S}/scripts/prompts.py").read())
out_json = f"{S}/results/ens_sizes.json"; paths = {}
for size, seeds in ((256, (1, 2)), (512, (1,))):
    for name, kw in (("base", {}), ("ens", dict(clip_model_name="ViT-B/32+PE-Core-B-16"))):
        for seed in seeds:
            key = f"{name}{size} s{seed}"; got = []; th.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
            for i, p in enumerate(PROMPTS):
                last = None
                for _, path in clip_guided_diffusion(prompts=[p], image_size=size, seed=seed, prefix_path=f"{S}/adm/ens_sizes/{key.replace(' ', '_')}/{i}", progress=False, save_frequency=1000, **kw):
                    last = path
                got.append(str(last))
            paths[key] = got; json.dump(paths, open(out_json, "w"), indent=0)
            print(key, f"{(time.perf_counter() - t0) / 12:.1f}s/img peak {th.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
print("done", flush=True)

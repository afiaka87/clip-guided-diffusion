"""Secondary model at 256 and 512, and the ensemble's cost at 128, alone on the card. argv: <SP> <REPO>."""
import json, sys, time
S, REPO = sys.argv[1], sys.argv[2]
sys.path.insert(0, REPO)
import torch as th
from cgd.cgd import clip_guided_diffusion
exec(open(f"{S}/scripts/prompts.py").read())
out_json = f"{S}/results/sec_sizes.json"; paths = {}
RUNS = [("sec256", 256, (1, 2), dict(secondary_guidance=True)), ("sec512", 512, (1,), dict(secondary_guidance=True)),
        ("base128t", 128, (1,), {}), ("ens128t", 128, (1,), dict(clip_model_name="ViT-B/32+PE-Core-B-16"))]
for name, size, seeds, kw in RUNS:
    for seed in seeds:
        key = f"{name} s{seed}"; got = []; th.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
        for i, p in enumerate(PROMPTS):
            last = None
            for _, path in clip_guided_diffusion(prompts=[p], image_size=size, seed=seed, prefix_path=f"{S}/adm/sec_sizes/{key.replace(' ', '_')}/{i}", progress=False, save_frequency=1000, **kw):
                last = path
            got.append(str(last))
        paths[key] = got; json.dump(paths, open(out_json, "w"), indent=0)
        print(key, f"{(time.perf_counter() - t0) / 12:.1f}s/img peak {th.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
print("done", flush=True)

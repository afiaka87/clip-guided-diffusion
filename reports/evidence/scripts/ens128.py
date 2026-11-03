"""The ensemble's cost at 128 with the network up. argv: <SP> <REPO>."""
import sys, time
S, REPO = sys.argv[1], sys.argv[2]
sys.path.insert(0, REPO)
import torch as th
from cgd.cgd import clip_guided_diffusion
exec(open(f"{S}/scripts/prompts.py").read())
for name, kw in (("base", {}), ("ens", dict(clip_model_name="ViT-B/32+PE-Core-B-16"))):
    th.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
    for i, p in enumerate(PROMPTS):
        for _ in clip_guided_diffusion(prompts=[p], image_size=128, seed=1, prefix_path=f"{S}/adm/ens128/{name}/{i}", progress=False, save_frequency=1000, **kw):
            pass
    print("RESULT", name, f"{(time.perf_counter() - t0) / 12:.1f}s/img peak {th.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)

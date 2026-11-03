"""Aesthetic per cutout, as the pilot did it, against the repo's whole frame term. argv: <SP> <REPO>; env SEEDS."""
import json, os, sys, time
S, REPO = sys.argv[1], sys.argv[2]
sys.path.insert(0, REPO)
import torch as th, torch.nn.functional as tf
from cgd import losses
from cgd.cgd import clip_guided_diffusion
exec(open(f"{S}/scripts/prompts.py").read())
SEEDS = [int(v) for v in os.environ.get("SEEDS", "1,2,3,4").split(",")]
sac = th.load(f"{S}/weights_sac/sac_b32.pth", map_location="cuda", weights_only=True)
w, b = sac["linear.weight"].float(), sac["linear.bias"].float()
real = losses.spherical_dist_loss
def patch(lam):
    def f(x, y):
        return real(x, y) - lam * ((tf.normalize(x, dim=-1) * 512 ** .5) @ w.T + b).squeeze(-1)
    return f
V = {"aescut .05": (.05, {}), "aescut .1": (.1, {}), "aescut .2": (.2, {}), "aes .1": (0, dict(aesthetic_scale=.1))}
out_json = f"{S}/results/admaescut.json"
paths = json.load(open(out_json)) if os.path.exists(out_json) else {}
for name, (lam, kw) in V.items():
    losses.spherical_dist_loss = patch(lam) if lam else real
    for seed in SEEDS:
        key = f"{name} s{seed}"
        if key in paths: continue
        th.cuda.reset_peak_memory_stats(); t0 = time.perf_counter(); got = []
        for i, p in enumerate(PROMPTS):
            d = f"{S}/adm/aescut/{key.replace(' ', '_')}/{i}"; last = None
            for _, path in clip_guided_diffusion(prompts=[p], image_size=128, seed=seed, prefix_path=d, progress=False, save_frequency=1000, **kw):
                last = path
            got.append(str(last))
        paths[key] = got; json.dump(paths, open(out_json, "w"), indent=0)
        print(key, f"{(time.perf_counter() - t0) / 12:.1f}s/img peak {th.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
print("done", flush=True)

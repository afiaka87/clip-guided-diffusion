"""ADM 128 upscaled by the shipped -up against direct ADM 512, on seed 1. argv: <SP> <REPO>."""
import json, os, sys, time
S, REPO = sys.argv[1], sys.argv[2]
sys.path.insert(0, REPO)
import torch as th
from cgd import upscale, script_util
a = json.load(open(f"{S}/results/adm128_paths.json")); e = json.load(open(f"{S}/results/ens_sizes.json"))
model = upscale.load_upscaler("cuda", script_util.CACHE_PATH)
got = []; th.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
for i, p in enumerate(a["base s1"]):
    got.append(str(upscale.upscale_file(model, p, "cuda")))
print(f"upscale {(time.perf_counter() - t0) / 12:.2f}s/img peak {th.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
d = f"{S}/adm/cascade"; os.makedirs(d, exist_ok=True); moved = []
for i, p in enumerate(got):
    t = f"{d}/{i}.png"; os.replace(p, t); moved.append(t)
json.dump({"base512 s1": e["base512 s1"], "128+up s1": moved}, open(f"{S}/results/cascade.json", "w"), indent=0)
print("done", flush=True)

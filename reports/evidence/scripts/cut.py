"""Cutout schemes for noise optimisation, through cgd.modern. argv: <SP> <REPO> [arms]; env SEEDS, BACKBONE."""
import json, os, sys, time
S, REPO = sys.argv[1], sys.argv[2]; ONLY = sys.argv[3].split(";") if len(sys.argv) > 3 else None
sys.path.insert(0, REPO)
import torch as th, torch.nn.functional as tf
from PIL import Image
from cgd import clip_util, modern, modules
exec(open(f"{S}/scripts/prompts.py").read())
SEEDS = [int(v) for v in os.environ.get("SEEDS", "1").split(",")]
NAME = os.environ.get("BACKBONE", "sd-turbo"); dev = "cuda"; SIZE = modern.BACKBONES[NAME].size; TAG = f"{NAME}{SIZE}"
EMB = th.load(f"{S}/turbo/emb_{NAME}.pt", weights_only=False)
real_load = clip_util.load_clip; loaded = {}
def cached(name="ViT-B/32", device="cpu", fp16=False):
    if (name, fp16) not in loaded:
        real_load.cache_clear(); loaded[(name, fp16)] = real_load(name, device, fp16)
    return loaded[(name, fp16)]
clip_util.load_clip = cached
B = modern.load_backbone(NAME, "", dev, os.environ.get("TINY", "1") == "1")

class Scheme(modules.MakeCutouts):
    """whole: whole-frame views. crops: random crops spanning low..high of the frame. grid: an n x n tile grid at this fraction."""
    def __init__(self, cut_size, whole=0, crops=0, low=.7, high=1., grid=0, frac=.5, jitter=False, power=1.):
        super().__init__(cut_size, whole + crops + grid ** 2)
        self.whole, self.crops, self.low, self.high, self.grid, self.frac, self.jitter, self.power = whole, crops, low, high, grid, frac, jitter, power
    def forward(self, x):
        h, w = x.shape[2:]; side = min(h, w); out = []
        for i in range(self.whole):
            if self.jitter:
                m = int(th.randint(0, side // 16 + 1, ()).item()); a = int(th.randint(0, m + 1, ()).item()); b = int(th.randint(0, m + 1, ()).item())
                c = x[:, :, a:h - (m - a), b:w - (m - b)]
            else:
                c = x
            out.append(modules.adaptive_avg_pool(c, self.cut_size))
        for _ in range(self.crops):
            s = int((th.rand([]) ** self.power * (self.high - self.low) + self.low) * side)
            a, b = th.randint(0, h - s + 1, ()).item(), th.randint(0, w - s + 1, ()).item()
            out.append(modules.adaptive_avg_pool(x[:, :, a:a + s, b:b + s], self.cut_size))
        if self.grid:
            s = int(self.frac * side); span = side - s
            for i in range(self.grid):
                for j in range(self.grid):
                    a0, b0 = span * i // max(self.grid - 1, 1), span * j // max(self.grid - 1, 1)
                    out.append(modules.adaptive_avg_pool(x[:, :, a0:a0 + s, b0:b0 + s], self.cut_size))
        return th.cat(out)

V = {
 "repo default": None,
 "iters 100": dict(iters=100), "iters 200": dict(iters=200),
 "whole": dict(whole=1),
 "whole jitter x2": dict(whole=2, jitter=True),
 "crops .7-1": dict(crops=4),
 "crops .9-1": dict(crops=4, low=.9),
 "crops .44-1": dict(crops=4, low=.44),
 "crops .44-.7": dict(crops=4, low=.44, high=.7),
 "whole+3 .44-.7": dict(whole=1, crops=3, low=.44, high=.7),
 "whole+7 .44-.7": dict(whole=1, crops=7, low=.44, high=.7),
 "whole+3 .3-.5": dict(whole=1, crops=3, low=.3, high=.5),
 "whole+grid2 .6": dict(whole=1, grid=2, frac=.6),
 "whole+grid3 .5": dict(whole=1, grid=3, frac=.5),
}
ENS = "ViT-B/32+PE-Core-B-16"
out_json = f"{S}/results/cut_{TAG}.json"
paths = json.load(open(out_json)) if os.path.exists(out_json) else {}
for arm, cfg in V.items():
    if ONLY and arm not in ONLY: continue
    cfg = dict(cfg) if cfg else None; iters = cfg.pop("iters", 50) if cfg else 50; cfg = cfg or None
    for seed in SEEDS:
        key = f"{arm} s{seed}"
        if key in paths: continue
        d = f"{S}/turbo/cut_{TAG}/{key.replace(' ', '_').replace('+', '_')}"; os.makedirs(d, exist_ok=True); got = []
        th.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
        for i, p in enumerate(PROMPTS):
            B.text, pooled = EMB[p]
            if pooled is not None: B.pooled = pooled
            shape = B.latent_shape(SIZE, SIZE)
            while True:
                try:
                    th.manual_seed(seed)
                    guide = clip_util.ClipGuide(ENS, [p], (), SIZE, 4, dev, min_cut_fraction=.7)
                    if cfg:
                        guide.towers = [t._replace(make_cutouts=Scheme(t.make_cutouts.cut_size, **cfg)) for t in guide.towers]
                    for step in modern.optimise_noise(B, guide, th.randn(shape, device=dev), iterations=iters, progress=False):
                        pass
                    with th.no_grad(): img = modern.render(B, step["best_z"], 1)
                    break
                except th.cuda.OutOfMemoryError:
                    print("OOM, waiting", flush=True); th.cuda.empty_cache(); time.sleep(20)
            Image.fromarray(img[0].add(1).div(2).clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()).save(f"{d}/{i}.png"); got.append(f"{d}/{i}.png")
        paths[key] = got; json.dump(paths, open(out_json, "w"), indent=0)
        print(key, f"{(time.perf_counter() - t0) / 12:.1f}s/img peak {th.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
print("done", flush=True)

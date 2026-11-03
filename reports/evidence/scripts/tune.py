"""Noise optimisation arms through cgd.modern. argv: <SP> <REPO> [arms]; env SEEDS, BACKBONE, SIZE."""
import gc, json, os, sys, time
S, REPO = sys.argv[1], sys.argv[2]; ONLY = sys.argv[3].split(";") if len(sys.argv) > 3 else None
sys.path.insert(0, REPO)
import torch as th
from PIL import Image
from cgd import clip_util, modern
exec(open(f"{S}/scripts/prompts.py").read())
SEEDS = [int(v) for v in os.environ.get("SEEDS", "1").split(",")]
NAME = os.environ.get("BACKBONE", "sd-turbo"); SIZE = int(os.environ.get("SIZE", modern.BACKBONES[NAME].size)); dev = "cuda"
TAG = f"{NAME}{SIZE}"
real_load = clip_util.load_clip; loaded = {}
def cached(name="ViT-B/32", device="cpu", fp16=False):
    if (name, fp16) not in loaded:
        real_load.cache_clear(); loaded[(name, fp16)] = real_load(name, device, fp16)
    return loaded[(name, fp16)]
clip_util.load_clip = cached
B = None
def backbone(tiny):
    global B
    if B is None or (B.guide_decoder is not None) != tiny:
        B = None; gc.collect(); th.cuda.empty_cache()
        B = modern.load_backbone(NAME, "", dev, tiny)
        B.emb = {}
    return B
EMB_PATH = f"{S}/turbo/emb_{NAME}.pt"
def encode_all():
    if os.path.exists(EMB_PATH): return th.load(EMB_PATH, weights_only=False)
    cfg = modern.BACKBONES[NAME]; out = {}
    if cfg.family == "sd":
        from diffusers import StableDiffusionPipeline as P
        pipe = P.from_pretrained(cfg.repo, variant="fp16", torch_dtype=th.float16, unet=None, vae=None, safety_checker=None, feature_extractor=None, requires_safety_checker=False).to(dev)
        for p in PROMPTS: out[p] = (pipe.encode_prompt(p, dev, 1, False)[0].detach(), None)
    else:
        from diffusers import StableDiffusionXLPipeline as P
        pipe = P.from_pretrained(cfg.repo, variant="fp16", torch_dtype=th.float16, unet=None, vae=None).to(dev)
        for p in PROMPTS:
            e, _, q, _ = pipe.encode_prompt(p, device=dev, do_classifier_free_guidance=False); out[p] = (e.detach(), q.detach())
    del pipe; gc.collect(); th.cuda.empty_cache()
    th.save(out, EMB_PATH); return out
EMB = encode_all()
B32, PE, B16 = "ViT-B/32", "PE-Core-B-16", "ViT-B/16"; ENS = f"{B32}+{PE}"
def A(**k): return k
V = {
 "base": A(), "plain": A(iters=0),
 "n1": A(n=1), "n2": A(n=2), "n8": A(n=8), "n16": A(n=16),
 "cut .3": A(frac=.3), "cut 1": A(frac=1.),
 "tiny": A(tiny=True),
 "lr 2.5": A(lr=2.5), "lr 10": A(lr=10.), "iters 25": A(iters=25), "iters 100": A(iters=100),
 "scale 100": A(scale=100.), "penalty 0": A(penalty=0.), "penalty .1": A(penalty=.1),
 "last": A(last=True), "render 2": A(render=2), "render 4": A(render=4),
 "contrastive": A(loss="contrastive", scale=.1), "B32": A(clip=B32), "PE": A(clip=PE),
 "B32+PE+B16": A(clip=f"{B32}+{PE}+{B16}"),
}
out_json = f"{S}/results/tune_{TAG}.json"
paths = json.load(open(out_json)) if os.path.exists(out_json) else {}
def run(arm, seed):
    kw = V[arm]; b = backbone(kw.get("tiny", False))
    key = f"{arm} s{seed}"
    if key in paths: return
    d = f"{S}/turbo/{TAG}/{key.replace(' ', '_').replace('+', '_')}"; os.makedirs(d, exist_ok=True); got = []
    th.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
    for i, p in enumerate(PROMPTS):
        b.text, pooled = EMB[p]
        if pooled is not None: b.pooled = pooled
        shape = b.latent_shape(SIZE, SIZE)
        while True:
            try:
                th.manual_seed(seed)
                iters = kw.get("iters", 50)
                if iters:
                    guide = clip_util.ClipGuide(kw.get("clip", ENS), [p], (), SIZE, kw.get("n", 4), dev, min_cut_fraction=kw.get("frac", .7), loss=kw.get("loss", "spherical"))
                    for step in modern.optimise_noise(b, guide, th.randn(shape, device=dev), iterations=iters, lr=kw.get("lr", 5.), penalty=kw.get("penalty", .01), loss_scale=kw.get("scale", 1.), progress=False):
                        pass
                    z = step["last_z"] if kw.get("last") else step["best_z"]
                else:
                    z = th.randn(shape, device=dev)
                with th.no_grad(): img = modern.render(b, z, kw.get("render", 1))
                break
            except th.cuda.OutOfMemoryError:
                print("OOM, waiting", flush=True); th.cuda.empty_cache(); time.sleep(20)
        x = img[0].add(1).div(2).clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
        Image.fromarray(x).save(f"{d}/{i}.png"); got.append(f"{d}/{i}.png")
    paths[key] = got; json.dump(paths, open(out_json, "w"), indent=0)
    print(key, f"{(time.perf_counter() - t0) / 12:.1f}s/img peak {th.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
for arm in (ONLY or V):
    for seed in SEEDS: run(arm, seed)
print("done", flush=True)

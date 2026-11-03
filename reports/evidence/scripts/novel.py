"""Ideas beyond ReNO, through cgd.modern. argv: <SP> <REPO> [arms]; env SEEDS, BACKBONE."""
import json, math, os, sys, time
S, REPO = sys.argv[1], sys.argv[2]; ONLY = sys.argv[3].split(";") if len(sys.argv) > 3 else None
sys.path.insert(0, REPO)
import torch as th, torch.nn.functional as tf
from PIL import Image
from cgd import clip_util, modern
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
ENS = "ViT-B/32+PE-Core-B-16"
sac = th.load(f"{S}/weights_sac/sac_b32.pth", map_location=dev, weights_only=True)
W, BIAS = sac["linear.weight"].float(), sac["linear.bias"].float()

def preselect(guide, shape, n):
    best, best_score = None, -1e9
    for _ in range(n):
        z = th.randn(shape, device=dev)
        with th.no_grad(): s = guide.score(B.decode(B.denoise(z))).item()
        if s > best_score: best, best_score = z, s
    return best

def pyramid(g, power):
    out = g.clone()
    for s in (2, 4, 8, 16):
        if s < g.shape[-1]:
            out = out + s ** power * tf.interpolate(tf.avg_pool2d(g, s), size=g.shape[-2:], mode="bilinear", align_corners=False)
    return out

def optimise(guide, z0, iters=50, lr=5., clip=.1, penalty=.01, pyr=None, text_lr=0., aes=0., aes_clip=None, steps=1, ema=0.):
    z = z0.clone().requires_grad_()
    params = [z]; delta = None; text = B.text
    if text_lr:
        delta = th.zeros_like(text.float()).requires_grad_(); opt_t = th.optim.SGD([delta], lr=text_lr, momentum=.9, nesterov=True)
    opt = th.optim.SGD([z], lr=lr, momentum=.9, nesterov=True)
    best, best_loss, best_text = z.detach().clone(), math.inf, text
    for it in range(iters):
        B.text = text if delta is None else (text.float() + delta).to(text.dtype)
        latents = B.denoise(z) if steps == 1 else B.sample(z, steps)
        images = B.decode_for_guidance(latents)
        loss = guide.loss(images)
        if aes:
            m, _ = cached("ViT-B/32", dev)
            whole = tf.interpolate(images.add(1).div(2), (224, 224), mode="area")
            e = tf.normalize(m.encode_image(clip_util.preprocess(m, whole)).float(), dim=-1) * 512 ** .5
            loss = loss - aes * (e @ W.T + BIAS).sum()
        v = loss.item()
        if v < best_loss: best, best_loss, best_text = z.detach().clone(), v, B.text.detach()
        if it < iters - 1:
            total = loss + penalty * modern.noise_penalty(z)
            if delta is None:
                g = th.autograd.grad(total, z)[0]
            else:
                g, dg = th.autograd.grad(total, [z, delta])
                delta.grad = th.nan_to_num(dg); th.nn.utils.clip_grad_norm_([delta], clip); opt_t.step()
            g = th.nan_to_num(g, nan=0., posinf=0., neginf=0.)
            if pyr is not None: g = pyramid(g, pyr)
            z.grad = g; th.nn.utils.clip_grad_norm_([z], clip); opt.step()
    B.text = best_text
    return best

V = {
 "base": dict(),
 "preselect 8": dict(pre=8), "preselect 32": dict(pre=32),
 "pyramid p0": dict(pyr=0), "pyramid p1": dict(pyr=1),
 "text 1": dict(text_lr=1.), "text 5": dict(text_lr=5.),
 "aesthetic .05": dict(aes=.05), "aesthetic .2": dict(aes=.2),
 "through 2 steps": dict(steps=2, render=2),
}
out_json = f"{S}/results/novel_{TAG}.json"
paths = json.load(open(out_json)) if os.path.exists(out_json) else {}
for arm, cfg in V.items():
    if ONLY and arm not in ONLY: continue
    cfg = dict(cfg); pre = cfg.pop("pre", 0); render = cfg.pop("render", 1)
    for seed in SEEDS:
        key = f"{arm} s{seed}"
        if key in paths: continue
        d = f"{S}/turbo/novel_{TAG}/{key.replace(' ', '_')}"; os.makedirs(d, exist_ok=True); got = []
        th.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
        for i, p in enumerate(PROMPTS):
            B.text, pooled = EMB[p]
            if pooled is not None: B.pooled = pooled
            shape = B.latent_shape(SIZE, SIZE)
            while True:
                try:
                    th.manual_seed(seed)
                    guide = clip_util.ClipGuide(ENS, [p], (), SIZE, 4, dev, min_cut_fraction=.7)
                    z = preselect(guide, shape, pre) if pre else th.randn(shape, device=dev)
                    best = optimise(guide, z, **cfg)
                    with th.no_grad(): img = modern.render(B, best, render)
                    break
                except th.cuda.OutOfMemoryError:
                    print("OOM, waiting", flush=True); th.cuda.empty_cache(); time.sleep(20)
            Image.fromarray(img[0].add(1).div(2).clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()).save(f"{d}/{i}.png"); got.append(f"{d}/{i}.png")
        paths[key] = got; json.dump(paths, open(out_json, "w"), indent=0)
        print(key, f"{(time.perf_counter() - t0) / 12:.1f}s/img peak {th.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
print("done", flush=True)

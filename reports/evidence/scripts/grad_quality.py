"""Gradient stability and transfer of CLIP-like guides, clean vs robust, single vs ensemble."""
import json, math, os, sys
import clip, open_clip, torch as th, torch.nn.functional as tf
from PIL import Image
import torchvision.transforms.functional as tvf

S, REPO = sys.argv[1], sys.argv[2]
HF = os.path.join(S, "hf")
CACHE = os.path.expanduser("~/.cache/clip-guided-diffusion/clip")
SMOKE = os.environ.get("GQ_SMOKE", "")
CPU = os.environ.get("GQ_CPU", "")  # Full gradient stats on the CPU, and a shorter transfer test, while the GPU is taken.
dev = "cpu" if SMOKE or CPU else "cuda"
OPENAI_MEAN, OPENAI_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)

class Enc:
    def __init__(self, visual, size, mean, std, text_fn):
        self.visual, self.size, self.text_fn = visual, size, text_fn
        self.mean = th.tensor(mean, device=dev)[None, :, None, None]
        self.std = th.tensor(std, device=dev)[None, :, None, None]
    def image(self, x):
        y = self.visual((x - self.mean) / self.std)
        y = y[0] if isinstance(y, tuple) else y
        return y.float()
    def text(self, texts):
        with th.no_grad():
            return tf.normalize(self.text_fn(texts).float(), dim=-1).to(dev)

def load_openai(file):
    m, _ = clip.load(os.path.join(CACHE, file), device="cpu", jit=False)
    m = m.float().eval().requires_grad_(False)
    size = m.visual.input_resolution
    visual = m.visual.to(dev)
    return Enc(visual, size, OPENAI_MEAN, OPENAI_STD, lambda t: m.encode_text(clip.tokenize(t, truncate=True)))

def load_oc(name, pretrained=None, tok=None, cache=HF):
    m = open_clip.create_model(name, pretrained=pretrained, cache_dir=cache).eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(tok or name, cache_dir=cache)
    visual = m.visual.to(dev)
    for mod in visual.modules():
        if hasattr(mod, "fused_attn"):
            mod.fused_attn = False
    return Enc(visual, visual.image_size[0], visual.image_mean or OPENAI_MEAN, visual.image_std or OPENAI_STD,
               lambda t: m.encode_text(tokenizer(t)))

SPECS = {
    "openai/ViT-B-32": lambda: load_openai("ViT-B-32.pt"),
    "openai/ViT-B-16": lambda: load_openai("ViT-B-16.pt"),
    "openai/RN50": lambda: load_openai("RN50.pt"),
    "laion/ViT-B-32": lambda: load_oc("ViT-B-32", "laion2b_s34b_b79k"),
    "FARE4/ViT-B-32": lambda: load_oc("hf-hub:chs20/FARE4-ViT-B-32-laion2B-s34B-b79K"),
    "laion/ViT-B-16": lambda: load_oc("ViT-B-16", "laion2b_s34b_b88k"),
    "FARE4/ViT-B-16": lambda: load_oc("hf-hub:chs20/FARE4-ViT-B-16-laion2B-s34B-b88K"),
    "laion/ConvNeXt-B-w": lambda: load_oc("convnext_base_w", "laion2b_s13b_b82k_augreg"),
    "FARE4/ConvNeXt-B-w": lambda: load_oc("hf-hub:chs20/FARE4-convnext_base_w-laion2B-s13B-b82K-augreg"),
    "PE-Core-B-16": lambda: load_oc("PE-Core-B-16", "meta", cache=CACHE),
    "SigLIP2-L-16-256": lambda: load_oc("ViT-L-16-SigLIP2-256", "webli", cache=CACHE),
}
print("loading", flush=True)
if SMOKE:
    SPECS = {k: SPECS[k] for k in ("openai/ViT-B-32", "openai/RN50", "FARE4/ViT-B-32", "laion/ConvNeXt-B-w", "FARE4/ConvNeXt-B-w", "PE-Core-B-16", "SigLIP2-L-16-256", "openai/ViT-B-16", "laion/ViT-B-32", "laion/ViT-B-16", "FARE4/ViT-B-16")}
M = {k: f() for k, f in SPECS.items()}
print({k: v.size for k, v in M.items()}, flush=True)

JUDGES = ["openai/ViT-B-16", "PE-Core-B-16", "SigLIP2-L-16-256", "laion/ConvNeXt-B-w"]
IMAGES = ["dog_looking_at_the_camera.png", "green-hills.png"]
PROMPTS = ["a red fox in the snow", "a pencil sketch of an old man with a beard", "a plate of sushi on a black slate"]
NEG = "text, words, letters, watermark"
TPL = ["{}", "a photo of {}", "a picture of {}", "an image of {}", "art of {}", "a rendering of {}"]
RES = 256; STEPS = 2 if SMOKE else 20 if CPU else 40; STEP_L2 = 0.5; N = 4
if SMOKE:
    IMAGES, PROMPTS = IMAGES[:1], PROMPTS[:1]

def load_img(n):
    i = Image.open(os.path.join(REPO, "images", n)).convert("RGB")
    i = tvf.center_crop(tvf.resize(i, RES), RES)
    return tvf.to_tensor(i).unsqueeze(0).to(dev)

def cuts(x, n, size, seed):
    g = th.Generator().manual_seed(seed)
    H = x.shape[-1]; out = []
    for _ in range(n):
        s = int(H * (0.7 + 0.3 * th.rand(1, generator=g).item()))
        oy = th.randint(0, H - s + 1, (1,), generator=g).item(); ox = th.randint(0, H - s + 1, (1,), generator=g).item()
        out.append(tf.interpolate(x[..., oy:oy + s, ox:ox + s], (size, size), mode="bicubic", antialias=True, align_corners=False))
    return th.cat(out)

def spherical(a, b):
    a, b = tf.normalize(a, dim=-1), tf.normalize(b, dim=-1)
    return (a - b).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

def grad(key, x, t, seed, loss="sph"):
    m = M[key]
    x = x.detach().requires_grad_(True)
    e = m.image(cuts(x, N, m.size, seed))
    if loss == "sph":
        l = spherical(e, t).mean()
    elif loss == "neg":   # t = (positive, negative)
        l = spherical(e, t[0]).mean() - 0.3 * spherical(e, t[1]).mean()
    elif loss == "ce":    # t = (positive, distractors [K, d])
        logits = 100 * tf.normalize(e, dim=-1) @ th.cat([t[0], t[1]]).T
        l = tf.cross_entropy(logits, th.zeros(len(logits), dtype=th.long, device=dev))
    return th.autograd.grad(l, x)[0]

def cos(a, b):
    return tf.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()

def hf_share(g):
    F = th.fft.fft2(g).abs().pow(2).sum(1)[0]
    H = g.shape[-1]; f = th.fft.fftfreq(H, device=dev)
    r = (f[None, :] ** 2 + f[:, None] ** 2).sqrt()
    return (F[r > 0.25].sum() / F.sum()).item()

def score(key, x, text):
    m = M[key]
    with th.no_grad():
        i = tf.interpolate(x, (m.size, m.size), mode="bicubic", antialias=True, align_corners=False)
        return (tf.normalize(m.image(i), dim=-1) @ m.text([text]).T).item()

TX = {k: {p: m.text([p]) for p in PROMPTS} for k, m in M.items()}
TN = {k: m.text([NEG]) for k, m in M.items()}
TT = {k: {p: tf.normalize(m.text([s.format(p) for s in TPL]).mean(0, keepdim=True), dim=-1) for p in PROMPTS} for k, m in M.items()}
imnet = th.load(os.path.expanduser("~/.cache/clip-guided-diffusion/imagenet_text_features_ViT-B-32.pt"), map_location=dev, weights_only=True)
imnet = tf.normalize(imnet.float(), dim=-1)
print("text ok", imnet.shape, flush=True)

results = {}
# 1. Per-model gradient properties.
for k in M:
    rows = []
    for i_n in IMAGES:
        x = load_img(i_n)
        for p in PROMPTS:
            t = TX[k][p]
            g = grad(k, x, t, 1)
            noise = (x + th.randn_like(x) / 2048).clamp(0, 1)
            rows.append(dict(noise=cos(g, grad(k, noise, t, 1)), cut=cos(g, grad(k, x, t, 2)), hf=hf_share(g)))
    results[k] = {f: sum(r[f] for r in rows) / len(rows) for f in rows[0]}
    print(k, results[k], flush=True)

# 2. Pairwise cosine between models' gradients (averaged over 8 cutouts draws to reduce crop noise).
keys = [k for k in M]
pair = {}
x = load_img(IMAGES[0]); p = PROMPTS[0]
G = {k: sum(grad(k, x, TX[k][p], s) for s in range(8)) for k in keys}
for a in keys:
    pair[a] = {b: round(cos(G[a], G[b]), 3) for b in keys}
results["pairwise_grad_cos"] = pair
print(json.dumps(pair), flush=True)

# 3. Transfer: gradient steps with one guide, judged by other models.
def unit(g):
    return g / g.norm().clamp(min=1e-12)

def cone(w, v, alpha=math.radians(30)):
    w_, v_ = w.flatten(), v.flatten()
    c = (w_ @ v_) / (w_.norm() * v_.norm())
    if c.acos() <= alpha:
        return w
    perp = w_ - (w_ @ v_) / (v_ @ v_) * v_
    u = math.sin(alpha) * perp / perp.norm() + math.cos(alpha) * v_ / v_.norm()
    return ((u @ w_) * u).view_as(w)

GUIDES = {
    "openai/ViT-B-32": lambda x, p, s: unit(grad("openai/ViT-B-32", x, TX["openai/ViT-B-32"][p], s)),
    "openai/RN50": lambda x, p, s: unit(grad("openai/RN50", x, TX["openai/RN50"][p], s)),
    "openai/ViT-B-16": lambda x, p, s: unit(grad("openai/ViT-B-16", x, TX["openai/ViT-B-16"][p], s)),
    "laion/ViT-B-32": lambda x, p, s: unit(grad("laion/ViT-B-32", x, TX["laion/ViT-B-32"][p], s)),
    "laion/ViT-B-16": lambda x, p, s: unit(grad("laion/ViT-B-16", x, TX["laion/ViT-B-16"][p], s)),
    "laion/ConvNeXt-B-w": lambda x, p, s: unit(grad("laion/ConvNeXt-B-w", x, TX["laion/ConvNeXt-B-w"][p], s)),
    "FARE4/ViT-B-32": lambda x, p, s: unit(grad("FARE4/ViT-B-32", x, TX["FARE4/ViT-B-32"][p], s)),
    "FARE4/ViT-B-16": lambda x, p, s: unit(grad("FARE4/ViT-B-16", x, TX["FARE4/ViT-B-16"][p], s)),
    "FARE4/ConvNeXt-B-w": lambda x, p, s: unit(grad("FARE4/ConvNeXt-B-w", x, TX["FARE4/ConvNeXt-B-w"][p], s)),
    "PE-Core-B-16": lambda x, p, s: unit(grad("PE-Core-B-16", x, TX["PE-Core-B-16"][p], s)),
    "SigLIP2-L-16-256": lambda x, p, s: unit(grad("SigLIP2-L-16-256", x, TX["SigLIP2-L-16-256"][p], s)),
}
def ens(keys_):
    return lambda x, p, s: unit(sum(unit(grad(k, x, TX[k][p], s)) for k in keys_))
GUIDES["ens DD (B32+B16+RN50)"] = ens(["openai/ViT-B-32", "openai/ViT-B-16", "openai/RN50"])
GUIDES["ens DAS-like (oai+laion B32)"] = ens(["openai/ViT-B-32", "laion/ViT-B-32"])
GUIDES["ens B32+FARE ConvNeXt"] = ens(["openai/ViT-B-32", "FARE4/ConvNeXt-B-w"])
GUIDES["ens FARE (B32+B16+ConvNeXt)"] = ens(["FARE4/ViT-B-32", "FARE4/ViT-B-16", "FARE4/ConvNeXt-B-w"])
GUIDES["cone FARE-ConvNeXt around B32"] = lambda x, p, s: unit(cone(unit(grad("FARE4/ConvNeXt-B-w", x, TX["FARE4/ConvNeXt-B-w"][p], s)), unit(grad("openai/ViT-B-32", x, TX["openai/ViT-B-32"][p], s))))
def and_mask(keys_):
    def f(x, p, s):
        gs = [unit(grad(k, x, TX[k][p], s)) for k in keys_]
        agree = th.stack([g_.sign() for g_ in gs]).sum(0).abs() == len(gs)
        return unit(sum(gs) * agree)
    return f
def median(keys_):
    return lambda x, p, s: unit(th.stack([unit(grad(k, x, TX[k][p], s)) for k in keys_]).median(0).values)
GUIDES["AND-mask B32+B16+FARE ConvNeXt"] = and_mask(["openai/ViT-B-32", "openai/ViT-B-16", "FARE4/ConvNeXt-B-w"])
GUIDES["median B32+B16+FARE ConvNeXt"] = median(["openai/ViT-B-32", "openai/ViT-B-16", "FARE4/ConvNeXt-B-w"])
GUIDES["mean B32+B16+FARE ConvNeXt"] = ens(["openai/ViT-B-32", "openai/ViT-B-16", "FARE4/ConvNeXt-B-w"])
K = "openai/ViT-B-32"
GUIDES["B32 neg-prompt -0.3"] = lambda x, p, s: unit(grad(K, x, (TX[K][p], TN[K]), s, "neg"))
GUIDES["B32 CE vs ImageNet texts"] = lambda x, p, s: unit(grad(K, x, (TX[K][p], imnet), s, "ce"))
GUIDES["B32 template mean"] = lambda x, p, s: unit(grad(K, x, TT[K][p], s))
GUIDES["B32 grayscale+shift only cuts=16x"] = lambda x, p, s: unit(sum(grad(K, x, TX[K][p], s * 100 + j) for j in range(4)))
K2 = "FARE4/ConvNeXt-B-w"
GUIDES["FARE ConvNeXt CE-free 16 cuts"] = lambda x, p, s: unit(sum(grad(K2, x, TX[K2][p], s * 100 + j) for j in range(4)))

if CPU:
    KEEP = ("openai/ViT-B-32", "openai/RN50", "laion/ConvNeXt-B-w", "FARE4/ViT-B-32", "FARE4/ConvNeXt-B-w", "PE-Core-B-16",
            "ens DD (B32+B16+RN50)", "ens B32+FARE ConvNeXt", "ens FARE (B32+B16+ConvNeXt)", "cone FARE-ConvNeXt around B32",
            "median B32+B16+FARE ConvNeXt", "mean B32+B16+FARE ConvNeXt", "FARE ConvNeXt CE-free 16 cuts",
            "B32 CE vs ImageNet texts", "B32 neg-prompt -0.3", "B32 template mean")
    GUIDES = {k: GUIDES[k] for k in KEEP}
    PROMPTS_T = PROMPTS[:2]
else:
    PROMPTS_T = PROMPTS
transfer = {}
if SMOKE:
    GUIDES = {k: v for k, v in GUIDES.items() if "AND" in k}
for name, fn in GUIDES.items():
    d = {j: [] for j in JUDGES}
    for i_n in IMAGES:
        x0 = load_img(i_n)
        for p in PROMPTS_T:
            x = x0.clone()
            for s in range(STEPS):
                x = (x - STEP_L2 * fn(x, p, s)).clamp(0, 1)
            for j in JUDGES:
                d[j].append(score(j, x, p) - score(j, x0, p))
    transfer[name] = {j: dict(mean=sum(v) / len(v), n=len(v), wins=sum(a > 0 for a in v)) for j, v in d.items()}
    transfer[name]["all_judges_mean"] = sum(sum(v) / len(v) for j, v in d.items() if j != name) / sum(1 for j in d if j != name)
    print(name, json.dumps({k: (round(v["mean"], 4) if isinstance(v, dict) else round(v, 4)) for k, v in transfer[name].items()}), flush=True)
# 4. Does half precision change a robust tower's gradient as it does a clean one's?
fp = {}
for key in () if CPU or SMOKE else ("FARE4/ViT-B-32", "FARE4/ViT-B-16", "FARE4/ConvNeXt-B-w", "laion/ConvNeXt-B-w", "PE-Core-B-16"):
    m = M[key]
    rows = []
    for i_n in IMAGES:
        x = load_img(i_n)
        for p in PROMPTS:
            a = grad(key, x, TX[key][p], 1)
            m.visual.half()
            real = m.image
            m.image = lambda v, r=real: r(v.half()).float()
            b = grad(key, x, TX[key][p], 1)
            m.image = real
            m.visual.float()
            rows.append(cos(a, b))
    fp[key] = sum(rows) / len(rows)
    print("fp16 vs fp32 grad cos", key, round(fp[key], 4), flush=True)
results["fp16_vs_fp32"] = fp
results["transfer"] = transfer
json.dump(results, open(os.path.join(S, "results", "grad_quality_cpu.json" if CPU else "smoke_gq.json" if SMOKE else "grad_quality.json"), "w"), indent=1)
print("done", flush=True)

"""Real-pipeline pilot at 128px: robust models, ensembles, TV, negative, templates, CE loss. Repo untouched."""
import json, os, sys, time
S, REPO = sys.argv[1], sys.argv[2]
sys.path.insert(0, REPO)
import numpy as np, torch as th, torch.nn.functional as tf
from PIL import Image
from cgd import clip_util, script_util, losses, sampling
from cgd.cgd import clip_guided_diffusion

CKPT = os.path.expanduser("~/.cache/clip-guided-diffusion")
SMOKE = os.environ.get("PILOT_SMOKE", "")  # Comma separated variant names, run on the CPU with two steps.
DEV = "cpu" if SMOKE else "cuda"
HF = os.path.join(S, "hf"); IM = os.path.join(S, "imnet"); os.makedirs(IM, exist_ok=True)
clip_util.CLIP_CACHE_PATH = HF
script_util.CACHE_PATH = IM
FARE_B32 = "hf-hub:chs20/FARE4-ViT-B-32-laion2B-s34B-b79K"
FARE_B16 = "hf-hub:chs20/FARE4-ViT-B-16-laion2B-s34B-b88K"
FARE_CNX = "hf-hub:chs20/FARE4-convnext_base_w-laion2B-s13B-b82K-augreg"
clip_util.OPEN_CLIP_MODELS.update({FARE_B32: None, FARE_B16: None, FARE_CNX: None,
                                   "convnext_base_w": "laion2b_s13b_b82k_augreg", "ViT-B-32": "laion2b_s34b_b79k",
                                   "CLIPAG": None})
CLIPAG_PATH = os.path.join(S, "weights_clipag", "CLIPAG_state_dict.pt")
real_tok = clip_util._open_clip_tokenizer
clip_util._open_clip_tokenizer = lambda n: real_tok("ViT-B-32" if n == "CLIPAG" else n)

real_load = clip_util.load_clip.__wrapped__
real_pre, real_top = clip_util.preprocess, clip_util.imagenet_top_n
real_txt = clip_util.encode_texts
cache = {}

class Ens:
    def __init__(self, names, device):
        self.names = names
        self.members = [get(n, device, False)[0:2] for n in names]
        self.size = max(s for _, s in self.members)
    def encode_image(self, x):
        out = []
        for m, s in self.members:
            xi = x if x.shape[-1] == s else tf.interpolate(x, (s, s), mode="bicubic", antialias=True, align_corners=False)
            out.append(tf.normalize(m.encode_image(real_pre(m, xi)).float(), dim=-1))
        return th.cat(out, dim=-1)

def get(name, device="cuda", fp16=False):
    if name not in cache:
        if name.startswith("ens:"):
            e = Ens(name[4:].split("+"), device)
            cache[name] = (e, e.size)
        elif name == "CLIPAG":
            laion = clip_util.OPEN_CLIP_MODELS["ViT-B-32"]
            clip_util.OPEN_CLIP_MODELS["ViT-B-32"] = CLIPAG_PATH
            cache[name] = real_load("ViT-B-32", device, fp16)
            clip_util.OPEN_CLIP_MODELS["ViT-B-32"] = laion
        else:
            cache[name] = real_load(name, device, fp16)
    return cache[name]

def load(model_name="ViT-B/32", device="cpu", fp16=False):
    return get(model_name, device, fp16)
clip_util.load_clip = load
clip_util.check_clip_model = lambda name: None
clip_util.preprocess = lambda m, x: x if isinstance(m, Ens) else real_pre(m, x)

def texts(t, name, device, truncate=False, fp16=False):
    if name.startswith("ens:"):
        return th.cat([tf.normalize(real_txt(t, n, device, truncate, fp16), dim=-1) for n in name[4:].split("+")], dim=-1)
    return real_txt(t, name, device, truncate, fp16)
clip_util.encode_texts = texts

def top(q, device="cuda", n=1000, clip_model_name="ViT-B/32", fp16=False):
    if clip_model_name.startswith("ens:"):
        first = clip_model_name[4:].split("+")[0]
        d = clip_util.imagenet_text_features(first, device, fp16).shape[-1]
        return real_top(q[..., :d], device, n, first, fp16)
    return real_top(q, device, n, clip_model_name, fp16)
clip_util.imagenet_top_n = top

PROMPTS = ["a red fox in the snow", "a pencil sketch of an old man with a beard", "a plate of sushi on a black slate",
           "a lighthouse on a cliff at sunset", "a watercolor painting of a sailboat", "a bowl of ramen with a soft boiled egg",
           "a vintage red bicycle leaning against a brick wall", "an oil painting of a mountain lake at dawn",
           "a close-up photograph of a bumblebee on a sunflower", "a steam locomotive crossing a stone bridge",
           "a cat wearing a tiny wizard hat", "a neon-lit city street in the rain"]
TPL = ["{}", "a photo of {}", "a picture of {}", "an image of {}", "art of {}", "a rendering of {}"]

real_sph = losses.spherical_dist_loss
real_tv = losses.tv_loss
def mean_embed(x, y):
    return real_sph(x.mean(0, keepdim=True), y)
def aes_factory(lam):
    sac = th.load(os.path.join(S, "weights_sac", "sac_b32.pth"), map_location=DEV, weights_only=True)
    w, b = sac["linear.weight"].float(), sac["linear.bias"].float()
    def f(x, y):
        score = (tf.normalize(x, dim=-1) * 512 ** 0.5) @ w.T + b
        return real_sph(x, y) - lam * score.squeeze(-1)
    return f
real_lg = sampling.loss_gradient
def precond(power):
    """DAS's multi-resolution parameterisation as a preconditioner on the pixel gradient: no extra VRAM."""
    def f(loss, x):
        g = real_lg(loss, x)
        out = g.clone()
        for s in (2, 4, 8, 16, 32):
            out = out + s ** power * tf.interpolate(tf.avg_pool2d(g, s), size=g.shape[-2:], mode="bilinear", align_corners=False)
        return out
    return f
def ms_tv(x):
    return real_tv(x) + 0.5 * sum(real_tv(tf.avg_pool2d(x, k)) for k in (2, 4, 8))
def ce_loss_factory(dist):
    def ce(x, y):  # x [c,b,1,d], y [n,d]; one positive at a time
        x_ = tf.normalize(x, dim=-1).squeeze(2)
        y_ = tf.normalize(y, dim=-1)
        out = []
        for i in range(y_.shape[0]):
            keep = (dist @ y_[i]).argsort()[:-10]  # Drop the ten classes closest to the prompt.
            cand = th.cat([y_[i:i + 1], dist[keep]])
            logits = 100 * x_ @ cand.T
            out.append(-tf.log_softmax(logits, dim=-1)[..., 0])
        return th.stack(out, dim=2)
    return ce

V = {  # name: (clip model, kwargs, prompt fn, patch)
    "base ViT-B/32 tv150": ("ViT-B/32", {}, None, None),
    "ViT-B/32 tv0": ("ViT-B/32", dict(tv_scale=0), None, None),
    "ViT-B/32 tv50": ("ViT-B/32", dict(tv_scale=50), None, None),
    "ViT-B/32 neg text -0.3": ("ViT-B/32", {}, lambda p: [p, "text, words, letters, watermark:-0.3"], None),
    "ViT-B/32 6 templates": ("ViT-B/32", {}, lambda p: [t.format(p) for t in TPL], None),
    "ViT-B/32 CE vs ImageNet": ("ViT-B/32", {}, None, "ce"),
    "ViT-B/32 mean embedding": ("ViT-B/32", {}, None, "mean"),
    "ViT-B/32 SAC aesthetic 0.1": ("ViT-B/32", {}, None, "aes"),
    "ViT-B/32 multiscale TV": ("ViT-B/32", {}, None, "mstv"),
    "ViT-B/32 multires grad p1": ("ViT-B/32", {}, None, "pg1"),
    "ViT-B/32 multires grad p2": ("ViT-B/32", {}, None, "pg2"),
    "FARE ConvNeXt multires p1": (FARE_CNX, {}, None, "pg1"),
    "PE-Core-B-16": ("PE-Core-B-16", {}, None, None),
    "laion ViT-B-32": ("ViT-B-32", {}, None, None),
    "CLIPAG ViT-B-32": ("CLIPAG", {}, None, None),
    "FARE4 ViT-B-32": (FARE_B32, {}, None, None),
    "FARE4 ViT-B-16": (FARE_B16, {}, None, None),
    "laion ConvNeXt-B": ("convnext_base_w", {}, None, None),
    "FARE4 ConvNeXt-B": (FARE_CNX, {}, None, None),
    "ens Disco B32+B16+RN50": ("ens:ViT-B/32+ViT-B/16+RN50", {}, None, None),
    "ens B32+PE-Core-B-16": ("ens:ViT-B/32+PE-Core-B-16", {}, None, None),
    "ens B32+FARE ConvNeXt": ("ens:ViT-B/32+" + FARE_CNX, {}, None, None),
    "ens FARE B32+B16+ConvNeXt": ("ens:" + "+".join((FARE_B32, FARE_B16, FARE_CNX)), {}, None, None),
}

if SMOKE:
    V = {k: v for k, v in V.items() if k in SMOKE.split(",")}
    PROMPTS = PROMPTS[:1]
paths = {}
for name, (model, kw, pfn, patch) in V.items():
    safe = name.replace("/", "-").replace(" ", "_")
    paths[name] = []
    t0 = time.perf_counter()
    if patch == "ce":
        dist = clip_util.imagenet_text_features("ViT-B/32", DEV, False).float()
        losses.spherical_dist_loss = ce_loss_factory(tf.normalize(dist, dim=-1))
    if patch == "mean":
        losses.spherical_dist_loss = mean_embed
    if patch == "aes":
        losses.spherical_dist_loss = aes_factory(0.1)
    if patch == "mstv":
        losses.tv_loss = ms_tv
    if patch in ("pg1", "pg2"):
        sampling.loss_gradient = precond(int(patch[-1]))
    if not SMOKE:
        th.cuda.reset_peak_memory_stats()
    for i, p in enumerate(PROMPTS):
        last = None
        try:
            for _, path in clip_guided_diffusion(prompts=pfn(p) if pfn else [p], image_size=128, seed=0, clip_model_name=model,
                                                 prefix_path=os.path.join(S, "pilot", safe, str(i)), progress=False,
                                                 checkpoints_dir=CKPT, device=DEV, **(dict(sample_steps=4) if SMOKE else {}), **kw):
                last = path
        except Exception as e:
            print("FAIL", name, i, repr(e)[:300], flush=True); raise
        paths[name].append(str(last))
    losses.spherical_dist_loss = real_sph
    losses.tv_loss = real_tv
    sampling.loss_gradient = real_lg
    peak = 0 if SMOKE else th.cuda.max_memory_allocated() / 2**30
    print(name, f"{(time.perf_counter() - t0) / len(PROMPTS):.1f}s/img, peak {peak:.2f} GiB", flush=True)
if not SMOKE:
    json.dump(paths, open(os.path.join(S, "results", "pilot_paths.json"), "w"), indent=0)
print("sampling done", flush=True)

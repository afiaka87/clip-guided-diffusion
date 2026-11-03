import json, os, sys
import clip, numpy as np, open_clip, torch as th, torch.nn.functional as tf
import torchvision.transforms.functional as tvf
from PIL import Image
S = sys.argv[1]
HF = os.path.join(S, "hf"); CACHE = os.path.expanduser("~/.cache/clip-guided-diffusion/clip")
paths = json.load(open(os.path.join(S, "results", "pilot_paths.json")))
extra = os.path.join(S, "results", "pilot_paths2.json")
if os.path.exists(extra):
    paths.update(json.load(open(extra)))
exec(open(os.path.join(S, "scripts", "prompts.py")).read())
dev = "cuda"
OM, OS = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
J = {}
m, _ = clip.load(os.path.join(CACHE, "ViT-B-16.pt"), device=dev, jit=False); m = m.float().eval()
J["ViT-B/16"] = (m.encode_image, 224, OM, OS, lambda t: m.encode_text(clip.tokenize(t, truncate=True).to(dev)))
b32, _ = clip.load(os.path.join(CACHE, "ViT-B-32.pt"), device=dev, jit=False); b32 = b32.float().eval()
for key, nm, pre in (("ViT-H-14-quickgelu", "ViT-H-14-quickgelu", "dfn5b"), ("SigLIP2-L-16-256", "ViT-L-16-SigLIP2-256", "webli")):
    o = open_clip.create_model(nm, pretrained=pre, cache_dir=HF).eval().to(dev)
    tk = open_clip.get_tokenizer(nm, cache_dir=HF)
    J[key] = ((lambda o: o.encode_image)(o), o.visual.image_size[0], o.visual.image_mean or OM, o.visual.image_std or OS,
              (lambda o, tk: (lambda t: o.encode_text(tk(t).to(dev))))(o, tk))
sac = th.load(os.path.join(S, "weights_sac", "sac_b32.pth"), map_location=dev, weights_only=True)
sac16 = th.load(os.path.join(S, "weights_sac", "sac_b16.pth"), map_location=dev, weights_only=True)
print({k: v.shape for k, v in sac.items()}, flush=True)
def img(p): return tvf.to_tensor(Image.open(p).convert("RGB")).unsqueeze(0).to(dev)
def judge(k, x, text):
    enc, s, mean, std, txt = J[k]
    i = tf.interpolate(x, (s, s), mode="bicubic", antialias=True, align_corners=False).clamp(0, 1)
    return (tf.normalize(enc(tvf.normalize(i, mean, std)).float(), dim=-1) @ tf.normalize(txt([text]).float(), dim=-1).T).item()
def aes(x):
    i = tf.interpolate(x, (224, 224), mode="bicubic", antialias=True, align_corners=False).clamp(0, 1)
    e = tf.normalize(b32.encode_image(tvf.normalize(i, OM, OS)).float(), dim=-1) * 512 ** 0.5
    return (sac["linear.weight"] @ e.T).item() + sac["linear.bias"].item()
def aes16(x):
    i = tf.interpolate(x, (224, 224), mode="bicubic", antialias=True, align_corners=False).clamp(0, 1)
    e = tf.normalize(m.encode_image(tvf.normalize(i, OM, OS)).float(), dim=-1) * 512 ** 0.5
    return (sac16["linear.weight"] @ e.T).item() + sac16["linear.bias"].item()
def detail(x):
    g = tvf.rgb_to_grayscale(x)[0, 0]
    F = th.fft.fft2(g - g.mean()).abs().pow(2)
    f = th.fft.fftfreq(g.shape[0], device=dev); r = (f[None] ** 2 + f[:, None] ** 2).sqrt()
    lap = (4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])
    return (F[r > 0.25].sum() / F.sum()).item(), lap.pow(2).mean().item()
R = {}
with th.no_grad():
    for name, ps in paths.items():
        rows = []
        for p, text in zip(ps, PROMPTS):
            x = img(p); h, l = detail(x)
            rows.append({**{k: judge(k, x, text) for k in J}, "aes": aes(x), "aes_b16": aes16(x), "hf_share": h, "lap": l})
        R[name] = rows
base = R["base ViT-B/32 tv150"]
out = {}
for name, rows in R.items():
    d = {}
    for k in rows[0]:
        a = np.array([r[k] for r in rows]); b = np.array([r[k] for r in base]); diff = a - b
        d[k] = dict(mean=float(a.mean()), delta=float(diff.mean()), wins=int((diff > 0).sum()), ci95=float(1.96 * diff.std(ddof=1) / len(diff) ** 0.5))
    out[name] = d
    print(name.ljust(30), " ".join(f"{k}:{d[k]['delta']:+.3f}({d[k]['wins']})" if k not in ("hf_share", "lap") else f"{k}:{d[k]['mean']:.4f}" for k in d), flush=True)
json.dump(out, open(os.path.join(S, "results", "pilot_scores.json"), "w"), indent=1)

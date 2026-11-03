"""Paired scoring of galleries. argv: <baseline arm> <paths.json> [...]; arms named 'name' or 'name s<seed>', one path per prompt.
Pairs on (prompt, seed) where the baseline has that seed, and gives a 95% CI across prompts of the mean over 3 held-out judges."""
import gc, json, os, re, sys
import clip, numpy as np, open_clip, torch as th, torch.nn.functional as tf
import torchvision.transforms.functional as tvf
from PIL import Image
S = os.environ["SP"]
exec(open(f"{S}/scripts/prompts.py").read())
base_arm, files = sys.argv[1], sys.argv[2:]
paths = {}
for f in files: paths.update(json.load(open(f)))
HF, CACHE, dev = f"{S}/hf", os.path.expanduser("~/.cache/clip-guided-diffusion/clip"), os.environ.get("DEV", "cuda")
OM, OS = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
db_path = f"{S}/eval/db.json"
db = json.load(open(db_path)) if os.path.exists(db_path) else {}
def key(p): return f"{p}|{int(os.path.getmtime(p))}"
def resize(x, s): return tf.interpolate(x, (s, s), mode="bicubic", antialias=True, align_corners=False).clamp(0, 1)
todo = sorted({(p, i) for ps in paths.values() for i, p in enumerate(ps) if key(p) not in db or len(db[key(p)]) < 8})
print(len(todo), "images to score", flush=True)
if todo:
    for p, i in todo: db.setdefault(key(p), {})
    imgs = {p: tvf.to_tensor(Image.open(p).convert("RGB")).unsqueeze(0) for p, _ in todo}
    def sweep(k, enc, s, mean, std, txt):
        t = {q: tf.normalize(txt([q]).float(), dim=-1) for q in PROMPTS}
        for p, i in todo:
            e = tf.normalize(enc(tvf.normalize(resize(imgs[p].to(dev), s), mean, std)).float(), dim=-1)
            db[key(p)][k] = (e @ t[PROMPTS[i]].T).item()
    with th.no_grad():
        for file, k, sac in (("ViT-B-16.pt", "b16", "sac_b16.pth"), ("ViT-B-32.pt", "b32", "sac_b32.pth")):
            m, _ = clip.load(f"{CACHE}/{file}", device=dev, jit=False); m = m.float().eval()
            sweep(k, m.encode_image, 224, OM, OS, lambda t: m.encode_text(clip.tokenize(t, truncate=True).to(dev)))
            w = th.load(f"{S}/weights_sac/{sac}", map_location=dev, weights_only=True)
            for p, i in todo:
                e = tf.normalize(m.encode_image(tvf.normalize(resize(imgs[p].to(dev), 224), OM, OS)).float(), dim=-1) * 512 ** .5
                db[key(p)]["aes" + k] = (w["linear.weight"] @ e.T).item() + w["linear.bias"].item()
            del m; gc.collect(); th.cuda.empty_cache()
        for nm, pre, k in (("ViT-H-14-quickgelu", "dfn5b", "vith"), ("ViT-L-16-SigLIP2-256", "webli", "sig2")):
            o = open_clip.create_model(nm, pretrained=pre, cache_dir=HF).eval(); o = (o.half() if dev == "cuda" else o).to(dev); tk = open_clip.get_tokenizer(nm, cache_dir=HF)
            sweep(k, lambda x: o.encode_image(x.half() if dev == "cuda" else x), o.visual.image_size[0], o.visual.image_mean or OM, o.visual.image_std or OS, lambda t: o.encode_text(tk(t).to(dev)))
            del o; gc.collect(); th.cuda.empty_cache()
        for p, i in todo:
            x = imgs[p]; mx, mn = x.max(1).values, x.min(1).values
            db[key(p)]["sat"] = ((mx - mn) / mx.clamp(min=1e-3)).mean().item()
            db[key(p)]["clip_px"] = ((x <= 0) | (x >= 1)).float().mean().item()
    json.dump(db, open(db_path, "w"))
def split(a):
    m = re.fullmatch(r"(.*) s(\d+)", a)
    return (m[1], int(m[2])) if m else (a, 0)
arms = {}
for a, ps in paths.items():
    n, s = split(a)
    arms.setdefault(n, {})[s] = [db[key(p)] for p in ps]
held = ["b16", "vith", "sig2"]
base = arms[base_arm]
rows = []
for n, by_seed in arms.items():
    shared = sorted(set(by_seed) & set(base))
    if not shared: continue
    d = np.array([[np.mean([by_seed[s][i][k] - base[s][i][k] for k in held]) for s in shared] for i in range(12)])
    per_prompt = d.mean(1)
    ci = 2.2 * per_prompt.std(ddof=1) / 12 ** .5 if n != base_arm else 0
    raw = {k: np.mean([by_seed[s][i][k] for s in shared for i in range(12)]) for k in held + ["aesb16", "aesb32", "sat", "clip_px"]}
    rows.append((per_prompt.mean(), n, ci, int((per_prompt > 0).sum()), len(shared), raw))
print(f"baseline {base_arm}; pooled change on 3 held-out judges, CI over 12 prompts")
print("arm".ljust(36), "pooled   +-ci  wins seeds |   b16   vith   sig2 | aes16  aes32   sat  clip%")
for m, n, ci, w, ns, r in sorted(rows, key=lambda r: -r[0]):
    print(n.ljust(36), f"{m:+.4f} {ci:.3f} {w:2d}/12 {ns:2d}    | {r['b16']:.3f}  {r['vith']:.3f}  {r['sig2']:.3f} | {r['aesb16']:5.2f}  {r['aesb32']:5.2f}  {r['sat']:.2f}  {100 * r['clip_px']:.1f}", flush=True)

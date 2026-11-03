"""Scratch scoring: three independent judges on the CPU, one at a time so the host keeps its memory, plus simple pixel statistics."""
import gc, json, os, sys, time
REPO = "/home/sam/GitHub/clip-guided-diffusion"
sys.path.insert(0, REPO)
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
import numpy as np
import torch as th
import torch.nn.functional as tf
from PIL import Image
import torchvision.transforms.functional as tvf


def wait_for_memory(gib):
    """Other agents run on this host, and a judge that pushes it into the OOM killer can take their work down with it."""
    while True:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                available = int(line.split()[1]) / 2**20
        if available >= gib:
            return
        time.sleep(5)


rows = [r for r in map(json.loads, open("results.jsonl")) if "pipe" in r]
prompts = sorted({r["prompt"] for r in rows})
previous = json.load(open("scores.json")) if os.path.exists("scores.json") else []
scored = {(r["pipe"], r["prompt"]) for r in previous}
rows = [r for r in rows if (r["pipe"], r["prompt"]) not in scored]
scores = [dict(pipe=r["pipe"], prompt=r["prompt"], sec=r["sec"], peak_alloc_gib=r["peak_alloc_gib"], peak_reserved_gib=r["peak_reserved_gib"],
               others=max(r["others_before"], r["others_after"])) for r in rows]
for out, r in zip(scores, rows):
    arr = np.asarray(Image.open(r["path"]).convert("RGB")).astype(np.float32)
    out["clipped_pct"] = float(((arr == 0) | (arr == 255)).mean() * 100)
    hsv = np.asarray(Image.open(r["path"]).convert("HSV")).astype(np.float32)
    out["saturation"] = float(hsv[..., 1].mean() / 255)
    gray = arr.mean(-1)
    lap = gray[1:-1, 2:] + gray[1:-1, :-2] + gray[2:, 1:-1] + gray[:-2, 1:-1] - 4 * gray[1:-1, 1:-1]
    out["laplacian_var"] = float(lap.var())


def judge(name, model, size, preprocess, text):
    text = tf.normalize(text, dim=-1)
    for out, r in zip(scores, rows):
        img = tvf.to_tensor(Image.open(r["path"]).convert("RGB")).unsqueeze(0)
        with th.no_grad():
            x = tf.interpolate(img, size=(size, size), mode="bicubic", antialias=True).clamp(0, 1)
            feat = tf.normalize(model.encode_image(preprocess(x)).float(), dim=-1)
        out[name] = float(feat[0] @ text[prompts.index(r["prompt"])])
    print(name, "scored", len(rows), flush=True)


from cgd import clip_util
import open_clip
wait_for_memory(6)
m, s = clip_util.load_clip("ViT-B/16", "cpu", False)
judge("ViT-B/16", m, s, lambda x: clip_util.preprocess(m, x), clip_util.encode_texts(prompts, "ViT-B/16", "cpu"))
del m; clip_util.load_clip.cache_clear(); gc.collect()
wait_for_memory(8)
m, s = clip_util.load_clip("ViT-L-16-SigLIP2-256", "cpu", False)
judge("SigLIP2-L", m, s, lambda x: clip_util.preprocess(m, x), clip_util.encode_texts(prompts, "ViT-L-16-SigLIP2-256", "cpu"))
del m; clip_util.load_clip.cache_clear(); gc.collect()
wait_for_memory(10)
m = open_clip.create_model_from_pretrained("ViT-H-14-quickgelu", pretrained="dfn5b", cache_dir=clip_util.CLIP_CACHE_PATH, return_transform=False).eval()
tokens = open_clip.get_tokenizer("ViT-H-14-quickgelu")(prompts)
with th.no_grad():
    text = m.encode_text(tokens).float()
judge("ViT-H", m, m.visual.image_size[0], lambda x: clip_util.preprocess(m, x), text)
json.dump(previous + scores, open("scores.json", "w"))
print(len(scores), "newly scored of", len(previous) + len(scores))

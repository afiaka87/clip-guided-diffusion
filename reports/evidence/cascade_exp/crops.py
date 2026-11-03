"""Scratch: judges at 224 cannot see detail in a 512 image, so also score five crops near native scale."""
import json, os, sys
REPO = "/home/sam/GitHub/clip-guided-diffusion"
sys.path.insert(0, REPO)
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
import torch as th
import torch.nn.functional as tf
from PIL import Image
import torchvision.transforms.functional as tvf
from cgd import clip_util

rows = [r for r in map(json.loads, open("results.jsonl")) if "pipe" in r]
keep = {"d128", "d256", "d512", "sr256_u", "sr256_g", "i2i256_s500", "i2i256_s700", "i2i512_s700", "sr512_u", "sr512_g", "esrgan256", "esrgan512",
        "ultrasharp256", "ultrasharp512", "swinirL256", "swinirL512", "hat256", "hat512", "compact256", "compact512", "v_base", "v_secondary", "v512_base", "v512_secondary"}
rows = [r for r in rows if r["pipe"] in keep]
prompts = sorted({r["prompt"] for r in rows})
model, size = clip_util.load_clip("ViT-B/16", "cpu", False)
text = tf.normalize(clip_util.encode_texts(prompts, "ViT-B/16", "cpu"), dim=-1)
out = []
for r in rows:
    image = tvf.to_tensor(Image.open(r["path"]).convert("RGB"))
    h = image.shape[1]
    side = max(size, h // 2) if h >= size else h
    crops = th.stack([tvf.resize(c, [size, size], antialias=True) for c in tvf.five_crop(image, [side, side])]).clamp(0, 1)
    with th.no_grad():
        features = tf.normalize(model.encode_image(clip_util.preprocess(model, crops)).float(), dim=-1)
    out.append(dict(pipe=r["pipe"], prompt=r["prompt"], crop_b16=float((features @ text[prompts.index(r["prompt"])]).mean())))
json.dump(out, open("crops.json", "w"))
print(len(out), "crop scored")

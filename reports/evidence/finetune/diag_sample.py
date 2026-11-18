"""Does the condition steer samples? Draw held-out conditions with and without it, no CLIP guidance, and score."""

import sys
import time

import numpy as np
import pandas as pd
import torch as th
from PIL import Image

from cgd import sampler
from cgd.adm import load_model
from finetune import data
from finetune.embed import DEFAULT_MODELS, encode_images, encode_texts, load_encoder
from cgd.conditioned import load_adapter

DEV, PER, STEPS, BATCH = "cuda", int(sys.argv[1]), int(sys.argv[2]), 12
CKPT, OUT = sys.argv[3], sys.argv[4]
rng = np.random.default_rng(0)
items, captions, sources = [], [], []
for number, (name, root) in enumerate((("laion", data.LAION_ROOT), ("imagenet", data.IMAGENET_ROOT))):
    source = data.open_source(name, root, DEFAULT_MODELS, split="holdout")
    for i in rng.choice(len(source.dataset), PER, replace=False):
        items.append(source.dataset[int(i)])
        captions.append(data.texts(source.index.iloc[[int(i)]])["caption"][0])
        sources.append(number)
real = th.stack([item["image"] for item in items])
embeds = {
    kind: {n: th.stack([item["embeds"][n][kind] for item in items]).to(DEV) for n in DEFAULT_MODELS}
    for kind in ("image", "caption", "short")
}
count = len(items)

index = pd.read_parquet(data.IMAGENET_ROOT / data.INDEX_NAME, columns=["label"])
first = index.reset_index().groupby("label")["index"].first().sort_index().to_numpy()
class_text = th.from_numpy(
    np.load(data.embed_path(data.IMAGENET_ROOT, "PE-Core-B-16", "short"), mmap_mode="r")[np.sort(first)].astype(
        np.float32
    )
)[np.argsort(np.argsort(first))].to(DEV)
closest = (embeds["caption"]["PE-Core-B-16"] @ class_text.T).argmax(dim=1)

unet, lab = load_model(256, DEV, False, False, True)
ts = sampler.timesteps(lab, STEPS)


def draw(eps_fn):
    """Sample every condition, in batches, from the same noise each time."""
    out = []
    for start in range(0, count, BATCH):
        rows = slice(start, start + BATCH)
        th.manual_seed(1000 + start)
        x = th.randn(len(range(count)[rows]), 3, 256, 256, device=DEV)

        def denoise(x, t, rows=rows):
            with th.autocast("cuda"), th.no_grad():
                return eps_fn(x, t, rows).float()

        for x0 in sampler.sample(denoise, lab, ts, x, None, progress=False):
            last = x0
        out.append(((last.clamp(-1, 1) + 1) * 127.5).round().byte().cpu())
    return th.cat(out)


def pick(kind, rows):
    return {n: embeds[kind][n][rows] for n in DEFAULT_MODELS}


began = time.monotonic()
drawn = {"unconditional model": draw(lambda x, t, rows: unet(x, t)[:, :3])}
print("unconditional done", round(time.monotonic() - began), "s", flush=True)
ckpt = th.load(CKPT, map_location="cpu", weights_only=True)
model = load_adapter(unet, ckpt | {"weights": ckpt["average"]}, fused=True).to(DEV).eval()


def guided(kind, scale):
    def eps_fn(x, t, rows):
        keep = th.ones(len(x), len(DEFAULT_MODELS), dtype=th.bool, device=DEV)
        conditioned = model(x, t, pick(kind, rows), keep)[:, :3]
        if scale == 1:
            return conditioned
        without = model(x, t)[:, :3]
        return without + scale * (conditioned - without)

    return eps_fn


for kind, scale in (("image", 1), ("image", 3), ("caption", 3), ("short", 3)):
    drawn[f"adapter, {kind} embedding, scale {scale}"] = draw(guided(kind, scale))
    print(kind, scale, "done", round(time.monotonic() - began), "s", flush=True)
del model, unet
th.cuda.empty_cache()
cond, _ = load_model(256, DEV, False, False, False)
drawn["class model, class closest to the caption"] = draw(lambda x, t, rows: cond(x, t, closest[rows])[:, :3])
del cond
th.cuda.empty_cache()
print("sampling took", round(time.monotonic() - began), "s", flush=True)

rows_of_images = [real] + list(drawn.values())
for number, name in enumerate(("laion", "imagenet")):
    which = [i for i, s in enumerate(sources) if s == number]
    grid = th.cat([th.cat([images[i] for i in which], dim=2) for images in rows_of_images], dim=1)
    Image.fromarray(grid.permute(1, 2, 0).numpy()).save(f"{OUT}_{name}.jpg", quality=92)
th.save({"drawn": drawn, "real": real, "captions": captions, "sources": sources}, OUT + ".pt")

source_of = th.tensor(sources)
print("rows of each grid, top to bottom: the real image,", "; ".join(drawn))
for judge_name in ("ViT-L-16-SigLIP2-256", "ViT-B/32"):
    judge = load_encoder(judge_name, DEV)
    with th.no_grad(), th.autocast("cuda"):
        target_image = encode_images(judge, real).float()
        target_text = encode_texts(judge, captions).float()
        print(f"--- judge {judge_name}: similarity to the real image / to the caption, LAION then ImageNet")
        for name, images in {"the real image": real, **drawn}.items():
            e = encode_images(judge, images).float()
            to_image, to_text = (e * target_image).sum(1).cpu(), (e * target_text).sum(1).cpu()
            parts = [f"{to_image[source_of == s].mean():.3f} / {to_text[source_of == s].mean():.3f}" for s in (0, 1)]
            print(f"{name:44s} {parts[0]}    {parts[1]}")
    del judge
    th.cuda.empty_cache()

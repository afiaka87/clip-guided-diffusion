"""The conditioning check by noise level, on the CPU: the run's checkpoint, and the class-conditional model for scale."""

import sys
from itertools import islice

import numpy as np
import pandas as pd
import torch as th

from cgd import sampler
from cgd.adm import load_model
from finetune import data, noise, train
from finetune.embed import DEFAULT_MODELS
from cgd.conditioned import load_adapter

BATCHES = int(sys.argv[1]) if len(sys.argv) > 1 else 48
th.manual_seed(0)
holdouts = [
    data.open_source("laion", data.LAION_ROOT, DEFAULT_MODELS, split="holdout"),
    data.open_source("imagenet", data.IMAGENET_ROOT, DEFAULT_MODELS, split="holdout"),
]
DEV = "cuda"
unet, lab = load_model(256, DEV, False, False, True)
table = sampler.log_snr(lab)
generator = th.Generator().manual_seed(0)
probes = []
for batch in islice(data.batches(holdouts, batch_size=2, seed=0, workers=0), BATCHES):
    fraction = th.rand(2, generator=generator, dtype=th.float64) * train.PROBE_NOISIEST
    log_snr = table[-1] + (table[0] - table[-1]) * fraction
    gaussian = th.randn(batch["image"].shape, generator=generator)
    probes.append((batch, log_snr, gaussian))

# The text embedding of each class, and the class each probe's own image embedding is closest to.
index = pd.read_parquet(data.IMAGENET_ROOT / data.INDEX_NAME, columns=["label"])
first = index.reset_index().groupby("label")["index"].first().sort_index().to_numpy()
class_text = th.from_numpy(
    np.load(data.embed_path(data.IMAGENET_ROOT, "PE-Core-B-16", "short"), mmap_mode="r")[np.sort(first)].astype(
        np.float32
    )
)
class_text = class_text[np.argsort(np.argsort(first))].to(DEV)


def errors(predict):
    out = []
    with th.no_grad():
        for batch, log_snr, gaussian in probes:
            alpha, sigma = noise.level_of(log_snr)
            images = batch["image"].float().div(127.5).sub(1)
            t = noise.timestep_of(lab, log_snr).float()
            noisy = alpha.float()[:, None, None, None] * images + sigma.float()[:, None, None, None] * gaussian
            moved = {k: (v.to(DEV) if th.is_tensor(v) else v) for k, v in batch.items()}
            moved["embeds"] = {n: {k: v.to(DEV) for k, v in kinds.items()} for n, kinds in batch["embeds"].items()}
            with th.autocast("cuda"):
                predicted = predict(noisy.to(DEV), t.to(DEV), moved)[:, :3]
            out.append((predicted.float().cpu() - gaussian).square().mean(dim=(1, 2, 3)))
    return th.cat(out)


results = {"base": errors(lambda x, t, b: unet(x, t))}
print("base done", flush=True)
ckpt = th.load("D:/cgd-finetune/uncond256-v2/adapter.pt", map_location="cpu", weights_only=True)
model = load_adapter(unet, ckpt, fused=True).to(DEV).eval()
ones = th.ones(2, len(DEFAULT_MODELS), dtype=th.bool, device=DEV)
results["adapter, image embedding"] = errors(
    lambda x, t, b: model(x, t, {n: b["embeds"][n]["image"] for n in DEFAULT_MODELS}, ones)
)
results["adapter, caption embedding"] = errors(
    lambda x, t, b: model(x, t, {n: b["embeds"][n]["caption"] for n in DEFAULT_MODELS}, ones)
)
results["adapter, no embedding"] = errors(lambda x, t, b: model(x, t))
print("adapter done, step", ckpt["steps"], flush=True)
del model, unet
cond, _ = load_model(256, DEV, False, False, False)


def closest(batch):
    return (batch["embeds"]["PE-Core-B-16"]["image"].to(DEV) @ class_text.T).argmax(dim=1)


results["class model, closest class"] = errors(lambda x, t, b: cond(x, t, closest(b)))
results["class model, true class"] = errors(lambda x, t, b: cond(x, t, b["label"].clamp(min=0)))
table_embedding = cond.label_emb.weight.clone()


class Soft(th.nn.Module):
    value = None

    def forward(self, _labels):
        return self.value


soft = Soft()
cond.label_emb = soft


def mixture(x, t, batch):
    weights = th.softmax(100 * batch["embeds"]["PE-Core-B-16"]["image"] @ class_text.T, dim=1)
    soft.value = weights @ table_embedding
    return cond(x, t, th.zeros(len(x), dtype=th.long, device=DEV))


results["class model, mixture of classes"] = errors(mixture)
source = th.cat([b["source"] for b, _, _ in probes])
labels = th.cat([b["label"] for b, _, _ in probes])
log_snr = th.cat([l for _, l, _ in probes]).float()
agree = (th.cat([closest(b).cpu() for b, _, _ in probes]) == labels)[source == 1].float().mean().item()
print(f"probes {len(source)}; closest class is the true class for {agree:.0%} of the ImageNet probes")
th.save({"results": results, "source": source, "log_snr": log_snr, "labels": labels}, sys.argv[2])
per_image = th.exp(-2 * log_snr)
edges = th.quantile(log_snr, th.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
for name, which in (("LAION", source == 0), ("ImageNet", source == 1), ("both", source >= 0)):
    print(f"--- {name}: how much lower the image error is than the base unconditional model")
    for key, value in results.items():
        if key == "base" or (name != "ImageNet" and "true class" in key):
            continue
        row = []
        for low, high in zip(edges[:-1], edges[1:]):
            m = which & (log_snr >= low) & (log_snr <= high)
            row.append(1 - ((value[m] * per_image[m]).sum() / (results["base"][m] * per_image[m]).sum()).item())
        m = which
        total = 1 - ((value[m] * per_image[m]).sum() / (results["base"][m] * per_image[m]).sum()).item()
        print(
            f"{key:34s} noisiest {row[0]:+7.1%} {row[1]:+7.1%} {row[2]:+7.1%} {row[3]:+7.1%} least noisy | all {total:+7.1%}"
        )
print("quartile edges of log(alpha/sigma):", [round(v, 2) for v in edges.tolist()])

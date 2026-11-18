"""How good the class anchor of `finetune.cond` is: the ImageNet class closest to an image's text, by ViT-B/32.

    python reports/evidence/finetune/diag_anchor.py

For ImageNet it prints how often that class is the image's own, and one of its five closest. For LAION, which
has no classes, how many different classes the texts land on and how much of the data the commonest ones take.
"""

import numpy as np
import pandas as pd

from cgd import clip_models
from cgd.imagenet_classes import IMAGENET_CLASSES
from finetune import data

MODEL = "ViT-B/32"
ROWS = 50_000


def main() -> None:
    classes = clip_models.imagenet_text_features(MODEL, "cpu").numpy().astype(np.float32)
    rng = np.random.default_rng(0)
    for root in (data.IMAGENET_ROOT, data.LAION_ROOT):
        index = pd.read_parquet(root / data.INDEX_NAME)
        rows = np.sort(rng.choice(len(index), ROWS, replace=False))
        for kind in data.EMBED_KINDS:
            embeds = np.load(data.embed_path(root, MODEL, kind), mmap_mode="r")[rows].astype(np.float32)
            scores = embeds @ classes.T
            anchor = scores.argmax(1)
            if "label" in index:
                labels = index["label"].to_numpy()[rows]
                top5 = (np.argsort(-scores, axis=1)[:, :5] == labels[:, None]).any(1).mean()
                print(f"{root.name}  {kind}: the anchor is the label {np.mean(anchor == labels):.3f}, in its top five {top5:.3f}")
            else:
                counts = np.bincount(anchor, minlength=1000)
                order = np.argsort(-counts)
                top = ", ".join(f"{IMAGENET_CLASSES[i]} {counts[i] / ROWS:.3f}" for i in order[:8])
                print(
                    f"{root.name}  {kind}: {int((counts > 0).sum())} classes used, the 10 commonest take "
                    f"{counts[order[:10]].sum() / ROWS:.3f}, the 100 commonest {counts[order[:100]].sum() / ROWS:.3f}. {top}"
                )


if __name__ == "__main__":
    main()

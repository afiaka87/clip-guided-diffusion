r"""A linear map from a text's CLIP embedding to its image's, for sampling `finetune.cond`'s model from a prompt.

    uv run --no-sync python -m finetune.prior

`finetune.cond`'s model takes two conditions per CLIP model: an image's embedding and a text's. A prompt only
gives the text's, and CLIP's text and image embeddings sit apart. This predicts the image's from the text's by
ridge regression on the training pairs of both datasets, every kind of text together: unCLIP's prior at its
simplest. It prints the cosine between the predicted and the real image embedding on held-out pairs, which is
how much noise `finetune.cond --embed-noise` has to cover, and saves the maps.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th

from cgd.finetuned import predict_image
from finetune import data
from finetune.embed import DEFAULT_MODELS

PRIOR_PATH = Path("D:/cgd-finetune/cond/prior.pt")


def fit(texts: np.ndarray, images: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    """The weights and bias of the least squares map from texts [N, dim] to images [N, dim].

    `ridge` is the penalty on the weights, per pair. `texts @ weight + bias` is the prediction.
    """
    text_mean, image_mean = texts.mean(0), images.mean(0)
    centred = texts - text_mean
    gram = centred.T @ centred + ridge * len(texts) * np.eye(texts.shape[1], dtype=texts.dtype)
    weight = np.linalg.solve(gram, centred.T @ (images - image_mean))
    return weight, image_mean - text_mean @ weight


def pairs(root: Path, model_name: str, split: str, count: int, seed: int) -> dict[str, np.ndarray]:
    """`count` rows of one split of a dataset: each kind of embedding of the same images, in float64."""
    index = data.select(pd.read_parquet(root / data.INDEX_NAME), split=split, keep_watermarked=True)
    rows = index["row"].to_numpy()
    rows = np.sort(np.random.default_rng(seed).choice(rows, min(count, len(rows)), replace=False))
    return {
        kind: np.load(data.embed_path(root, model_name, kind), mmap_mode="r")[rows].astype(np.float64)
        for kind in data.EMBED_KINDS
    }


def fit_model(roots: Sequence[Path], model_name: str, count: int, ridge: float) -> dict[str, th.Tensor]:
    """One CLIP model's map, fitted on every kind of text of `count` training images of each dataset."""
    sets = [pairs(root, model_name, "train", count, seed=0) for root in roots]
    texts = np.concatenate([rows[kind] for rows in sets for kind in data.TEXT_KINDS])
    images = np.concatenate([rows["image"] for rows in sets for _ in data.TEXT_KINDS])
    weight, bias = fit(texts, images, ridge)
    return {"weight": th.from_numpy(weight).float(), "bias": th.from_numpy(bias).float()}


def main() -> None:
    """Fit each CLIP model's map, print how close it gets on held-out pairs, and save them."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--laion", type=Path, default=data.LAION_ROOT)
    parser.add_argument("--imagenet", type=Path, default=data.IMAGENET_ROOT)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--count", type=int, default=60_000, help="training images of each dataset")
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--out", type=Path, default=PRIOR_PATH)
    args = parser.parse_args()
    priors = {}
    for name in args.models:
        priors[name] = fit_model([args.laion, args.imagenet], name, args.count, args.ridge)
        for root in (args.laion, args.imagenet):
            held = {kind: th.from_numpy(rows) for kind, rows in pairs(root, name, "holdout", 5000, seed=1).items()}
            for kind in data.TEXT_KINDS:
                predicted = (predict_image(priors[name], held[kind]) * held["image"]).sum(1).mean().item()
                as_it_is = (held[kind] * held["image"]).sum(1).mean().item()
                print(
                    f"{name}  {root.name}  {kind}: cosine to the image's {predicted:.3f}, the text's own {as_it_is:.3f}"
                )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    th.save(priors, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

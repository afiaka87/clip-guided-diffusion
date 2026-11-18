r"""Step 2: the CLIP embeddings of every image and text, for the models the fine-tune conditions on.

    uv run python -m finetune.embed                                   # Both datasets, ViT-B/32 and PE-Core-B-16
    uv run python -m finetune.embed F:/some/dataset --models ViT-B/32

`finetune.index` must have run first. For each dataset and model it writes four arrays into the dataset's
`finetune` folder, one row per row of the index, unit length, at half precision:

    ViT-B_32.image.npy    the image, the whole 256 crop resized to the model's input
    ViT-B_32.caption.npy  the generated caption without its "The image shows"
    ViT-B_32.first.npy    that caption's first sentence
    ViT-B_32.short.npy    LAION's alt text, or "a photo of a <class>" for ImageNet

A text longer than a model reads is cut off: 77 tokens for OpenAI's models, 32 for PE-Core's, which is why the
first sentence is embedded on its own. `finetune.data.texts` makes the texts, so the loader returns the same
strings these came from.

Images are decoded once and go through every model. Run it again to resume: `finetune/progress.json` records how
many rows of each array are written, and a stop loses at most the last minute.
"""

import argparse
import json
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch as th
import torchvision.transforms.functional as tvf
from torch.nn import functional as tf
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

from cgd import clip_models
from cgd.device import default_device
from finetune import data

DEFAULT_MODELS = ("ViT-B/32", "PE-Core-B-16")
TEXT_BATCH = 1024
TEXT_CHUNK = 100_000  # Rows whose texts are encoded and recorded together.
SAVE_EVERY = 50  # Batches of images between records of progress.


class Encoder(NamedTuple):
    """A CLIP model with both towers on one device."""

    name: str
    model: th.nn.Module
    size: int
    tokenize: Callable[[list[str]], th.Tensor]
    device: str


def load_encoder(name: str, device: str) -> Encoder:
    """A CLIP model as `cgd` loads it, at half precision on a GPU. An open_clip text tower moves there too."""
    model, size = clip_models.load_clip(name, device, device != "cpu")
    if name in clip_models.OPEN_CLIP_MODELS:
        import open_clip

        model.to(device)  # cgd keeps this text tower on the CPU, which is too slow for millions of captions.
        tokenize = open_clip.get_tokenizer(name, cache_dir=str(clip_models.CLIP_CACHE_PATH))
    else:
        tokenize = partial(clip_models.clip.tokenize, truncate=True)
    return Encoder(name, model, size, tokenize, device)


def encode_images(encoder: Encoder, images: th.Tensor) -> th.Tensor:
    """Unit embeddings of uint8 images [N, 3, H, W], each resized whole to the model's input."""
    images = images.to(encoder.device).float().div(255)
    size = [encoder.size, encoder.size]
    images = tvf.resize(images, size, interpolation=InterpolationMode.BICUBIC, antialias=True).clamp(0, 1)
    features = encoder.model.encode_image(clip_models.preprocess(encoder.model, images))
    return tf.normalize(features.float(), dim=-1)


def encode_texts(encoder: Encoder, texts: Sequence[str]) -> th.Tensor:
    """Unit embeddings of texts, each cut off at the length the model reads."""
    tokens = encoder.tokenize(list(texts)).to(encoder.device)
    features = th.cat([encoder.model.encode_text(chunk).float() for chunk in tokens.split(TEXT_BATCH)])
    return tf.normalize(features, dim=-1)


def open_store(root: Path, model_name: str, kind: str, rows: int, dim: int) -> np.memmap:
    """The array of one model and kind, created empty or reopened to continue."""
    path = data.embed_path(root, model_name, kind)
    if not path.is_file():
        path.parent.mkdir(exist_ok=True)
        return np.lib.format.open_memmap(path, "w+", np.float16, (rows, dim))
    store = np.load(path, mmap_mode="r+")
    if store.shape != (rows, dim):
        raise SystemExit(f"{path} is {store.shape}, not {(rows, dim)}: the index changed. Remove it to start again.")
    return store


class Progress:
    """How many rows of each array are written, kept in a file beside them."""

    def __init__(self, root: Path):
        self.path = root / data.EMBED_DIR / "progress.json"
        self.rows: dict[str, int] = json.loads(self.path.read_text()) if self.path.is_file() else {}

    def done(self, model_name: str, kind: str) -> int:
        """How many rows of one array are written."""
        return self.rows.get(f"{data.slug(model_name)}.{kind}", 0)

    def record(self, stores: Sequence[np.memmap], model_names: Sequence[str], kind: str, rows: int) -> None:
        """Flush the arrays, then record that their first `rows` rows are written."""
        for store in stores:
            store.flush()
        self.rows |= {f"{data.slug(name)}.{kind}": rows for name in model_names}
        partial_path = self.path.with_suffix(".tmp")
        partial_path.write_text(json.dumps(self.rows, indent=1))
        partial_path.replace(self.path)


def embed_images(  # pylint: disable=too-many-locals  # One pass over the images feeds every model.
    root: Path, index: pd.DataFrame, encoders: Sequence[Encoder], progress: Progress, *, batch_size: int, workers: int
) -> None:
    """Write every model's image embeddings, from the first row any of them lacks."""
    names = [encoder.name for encoder in encoders]
    done = min(progress.done(name, "image") for name in names)
    dims = [encode_texts(encoder, ["a"]).shape[1] for encoder in encoders]
    stores = [open_store(root, name, "image", len(index), dim) for name, dim in zip(names, dims, strict=True)]
    if done == len(index):
        return
    rest = index.assign(row=np.arange(len(index))).iloc[done:]
    loader = DataLoader(data.ShardDataset(root, rest, ()), batch_size=batch_size, num_workers=workers)
    for number, batch in enumerate(loader):
        count = len(batch["image"])
        for encoder, store in zip(encoders, stores, strict=True):
            store[done : done + count] = encode_images(encoder, batch["image"]).cpu().numpy()
        done += count
        if number % SAVE_EVERY == SAVE_EVERY - 1 or done == len(index):
            progress.record(stores, names, "image", done)
            print(f"{root.name}: images {done}/{len(index)}", flush=True)


def embed_texts(root: Path, index: pd.DataFrame, encoder: Encoder, progress: Progress) -> None:
    """Write one model's embeddings of each kind of text. A text that repeats within a chunk is encoded once."""
    dim = encode_texts(encoder, ["a"]).shape[1]
    for kind in data.TEXT_KINDS:
        store = open_store(root, encoder.name, kind, len(index), dim)
        for start in range(progress.done(encoder.name, kind), len(index), TEXT_CHUNK):
            chunk = data.texts(index.iloc[start : start + TEXT_CHUNK])[kind]
            codes, unique = pd.factorize(pd.Series(chunk, dtype=object))
            features = encode_texts(encoder, list(unique)).cpu().numpy()
            store[start : start + len(chunk)] = features[codes]
            progress.record([store], [encoder.name], kind, start + len(chunk))
            print(f"{root.name}: {encoder.name} {kind} {start + len(chunk)}/{len(index)}", flush=True)


def check(root: Path, model_names: Sequence[str], rows: int) -> None:
    """Stop if any embedding is missing: every row of every array must be finite and unit length."""
    for name in model_names:
        for kind in data.EMBED_KINDS:
            store = np.load(data.embed_path(root, name, kind), mmap_mode="r")
            norms = np.concatenate(
                [np.linalg.norm(store[i : i + 100_000].astype(np.float32), axis=1) for i in range(0, rows, 100_000)]
            )
            bad = int((~np.isfinite(norms) | (np.abs(norms - 1) > 0.01)).sum())
            if len(store) != rows or bad:
                raise SystemExit(f"{root}: {name} {kind} has {bad} rows that are not unit length.")
    print(f"{root}: {rows} rows of {len(model_names)} models, all unit length.")


@th.inference_mode()
def embed_dataset(root: Path, encoders: Sequence[Encoder], batch_size: int, workers: int) -> None:
    """Every embedding of one dataset."""
    path = root / data.INDEX_NAME
    if not path.is_file():
        raise SystemExit(f"{path} is missing. Run: uv run python -m finetune.index {root}")
    index = pd.read_parquet(path)
    (root / data.EMBED_DIR).mkdir(exist_ok=True)
    progress = Progress(root)
    embed_images(root, index, encoders, progress, batch_size=batch_size, workers=workers)
    for encoder in encoders:
        embed_texts(root, index, encoder, progress)
    check(root, [encoder.name for encoder in encoders], len(index))


def main() -> None:
    """Embed the datasets named on the command line."""
    parser = argparse.ArgumentParser(description="Precompute CLIP embeddings of indexed datasets.")
    parser.add_argument("roots", nargs="*", type=Path, default=[data.LAION_ROOT, data.IMAGENET_ROOT])
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    for name in args.models:
        clip_models.check_model(name)
    encoders = [load_encoder(name, args.device) for name in args.models]
    for root in args.roots:
        embed_dataset(root, encoders, args.batch, args.workers)


if __name__ == "__main__":
    main()

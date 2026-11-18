r"""The loader: images, captions and CLIP embeddings from several datasets, mixed evenly and filtered on metadata.

    from finetune import data
    sources = [
        data.open_source("laion", data.LAION_ROOT, MODELS, where="aesthetic >= 6.5 and pwatermark < 0.3"),
        data.open_source("imagenet", data.IMAGENET_ROOT, MODELS, where="label in [281, 282, 283]"),
    ]
    for batch in data.batches(sources, weights=[1, 1], batch_size=8, seed=0):
        batch["image"]                        # uint8 [8, 3, 256, 256]
        batch["texts"]["caption"]             # 8 strings. Also "first" and "short", see texts().
        batch["embeds"]["ViT-B/32"]["image"]  # float32 [8, 512], unit length. Also "caption", "first", "short".
        batch["source"], batch["label"]       # 0 or 1 per image, and the ImageNet class or -1

`where` is a pandas query over the columns `finetune.index` wrote, so anything in a sample's JSON can select it.
Images whose generated caption mentions a watermark are left out unless `keep_watermarked` is set.
The mix is exact, not random: with equal weights every batch is half one dataset and half the other, and each
dataset is drawn in its own shuffled order, reshuffled when it runs out. The same seed gives the same batches,
and `start` resumes part way through.

Images are read by seeking into the tars, so the shuffle is over the whole dataset and not within a buffer.
"""

import io
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler

from cgd.conditioned import slug

LAION_ROOT = Path("F:/laion-aesthetic-best1m-256")
IMAGENET_ROOT = Path("F:/imagenet-1k-clean-256/train")
# What these scripts add to a dataset goes in a folder of its own. The script that wrote the shards resumes by
# counting the parquet files beside them, so nothing else may be put there.
EMBED_DIR = "finetune"
INDEX_NAME = f"{EMBED_DIR}/index.parquet"
TEXT_KINDS = ("caption", "first", "short")
EMBED_KINDS = ("image", *TEXT_KINDS)
CAPTION_OPENING = re.compile(r"^The image shows\s+")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def embed_path(root: Path, model_name: str, kind: str) -> Path:
    """Where the embeddings of one model and kind are, one row per row of the index."""
    return root / EMBED_DIR / f"{slug(model_name)}.{kind}.npy"


def clean_caption(caption: str) -> str:
    """A generated caption without the opening nearly all of them share."""
    stripped = CAPTION_OPENING.sub("", caption.strip())
    return stripped[:1].upper() + stripped[1:]


def first_sentence(caption: str) -> str:
    """The part of a caption a text tower that reads 32 tokens gets to see whole."""
    return SENTENCE_END.split(caption, maxsplit=1)[0]


def texts(index: pd.DataFrame) -> dict[str, list[str]]:
    """The three texts of each row, the ones the text embeddings were made from.

    - `caption`: the generated caption, without its "The image shows".
    - `first`: its first sentence.
    - `short`: what a person might type. LAION's alt text, or "a photo of a <class>" for ImageNet.

    A dataset without generated captions uses its short text for all three, and an empty alt text falls back
    to the caption.
    """
    if "class_name" in index:
        short = [f"a photo of a {name}" for name in index["class_name"]]
    else:
        short = [text if isinstance(text, str) else "" for text in index["alt_text"]]
    generated = "generated_caption" in index
    caption = [clean_caption(text) for text in index["generated_caption"]] if generated else short
    short = [text.strip() or fallback for text, fallback in zip(short, caption, strict=True)]
    return {"caption": caption, "first": [first_sentence(text) for text in caption], "short": short}


class ShardDataset(Dataset):
    """The images of selected index rows, each with its embeddings. Files are opened in the worker that reads them."""

    def __init__(self, root: Path, index: pd.DataFrame, models: Sequence[str], source: int = 0):
        self.root, self.models, self.source = root, tuple(models), source
        self.rows = index["row"].to_numpy()
        self.shards = index["shard"].to_numpy()
        self.spans = index[["offset", "size"]].to_numpy()  # Where each JPEG is in its tar.
        self.labels = index["label"].to_numpy() if "label" in index else np.full(len(index), -1)
        self._tars: dict[int, io.BufferedReader] = {}
        self._embeds: dict[tuple[str, str], np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _embedding(self, model_name: str, kind: str, row: int) -> th.Tensor:
        key = (model_name, kind)
        if key not in self._embeds:
            self._embeds[key] = np.load(embed_path(self.root, model_name, kind), mmap_mode="r")
        return th.from_numpy(self._embeds[key][row].astype(np.float32))

    def __getitem__(self, position: int) -> dict:
        shard = int(self.shards[position])
        if shard not in self._tars:
            self._tars[shard] = (self.root / f"{shard:05d}.tar").open("rb")
        tar = self._tars[shard]
        offset, size = self.spans[position].tolist()
        tar.seek(offset)
        image = Image.open(io.BytesIO(tar.read(size))).convert("RGB")
        row = int(self.rows[position])
        return {
            "image": th.from_numpy(np.asarray(image).copy()).permute(2, 0, 1),
            "embeds": {name: {kind: self._embedding(name, kind, row) for kind in EMBED_KINDS} for name in self.models},
            "label": int(self.labels[position]),
            "source": self.source,
            "position": position,
        }

    def __getstate__(self) -> dict:
        return {**self.__dict__, "_tars": {}, "_embeds": {}}  # A worker opens its own.


@dataclass(frozen=True)
class Source:
    """One dataset as the loader sees it: its name, the rows selected from it, and how to read them."""

    name: str
    index: pd.DataFrame
    dataset: ShardDataset


def captioned_as_watermarked(index: pd.DataFrame) -> pd.Series:
    """Which rows have a generated caption that mentions a watermark.

    The captioner reports marks the watermark detector missed. Of 32 such ImageNet images looked at about 13
    had a visible watermark, logo or date stamp, and of 32 from LAION about 5. It is 1.6% and 1.9% of them.
    """
    if "generated_caption" not in index:
        return pd.Series(False, index=index.index)
    return index["generated_caption"].str.contains("watermark", case=False, na=False)


def select(
    index: pd.DataFrame, where: str | None = None, split: str = "train", keep_watermarked: bool = False
) -> pd.DataFrame:
    """The rows of an index to load: a split, without those captioned as watermarked, matching the query `where`.

    The last shard of a dataset is its `holdout` split and the rest its `train` split. `all` is both.
    """
    index = index.assign(row=np.arange(len(index)))
    if split != "all":
        held_out = index["shard"] == index["shard"].max()
        index = index[held_out if split == "holdout" else ~held_out]
    if not keep_watermarked:
        index = index[~captioned_as_watermarked(index)]
    if where:
        index = index.query(where, engine="python")
    return index.reset_index(drop=True)


def open_source(
    name: str,
    root: Path,
    models: Sequence[str] = (),
    *,
    where: str | None = None,
    split: str = "train",
    keep_watermarked: bool = False,
) -> Source:
    """A dataset's selected rows, ready to load. `finetune.index` and `finetune.embed` must have run on `root`."""
    path = root / INDEX_NAME
    if not path.is_file():
        raise SystemExit(f"{path} is missing. Run: uv run python -m finetune.index {root}")
    index = pd.read_parquet(path)
    for model_name in models:
        for kind in EMBED_KINDS:
            file = embed_path(root, model_name, kind)
            if not file.is_file() or len(np.load(file, mmap_mode="r")) != len(index):
                raise SystemExit(f"{file} is missing or not the index's length. Run finetune.embed on {root}.")
    selected = select(index, where, split, keep_watermarked)
    if selected.empty:
        raise SystemExit(f"No rows of {root} match {where!r} in its {split} split.")
    return Source(name, selected, ShardDataset(root, selected, models))


def shuffled_forever(size: int, seed: int, stream: int) -> Iterator[int]:
    """Every index below `size` in a shuffled order, then again in a new one, without end."""
    epoch = 0
    while True:
        yield from np.random.default_rng([seed, stream, epoch]).permutation(size).tolist()
        epoch += 1


def mixed_indices(sizes: Sequence[int], weights: Sequence[float], seed: int) -> Iterator[int]:
    """Indices into the datasets laid end to end, taking each dataset in proportion to its weight.

    The turns are spread as evenly as the weights allow, not drawn at random: each dataset earns its share of
    every draw, and the one owed the most goes next. With weights 1 and 1 the datasets alternate.
    """
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1])).tolist()
    streams = [shuffled_forever(size, seed, stream) for stream, size in enumerate(sizes)]
    shares = [weight / sum(weights) for weight in weights]
    owed = [0.0] * len(sizes)
    while True:
        owed = [debt + share for debt, share in zip(owed, shares, strict=True)]
        turn = max(range(len(sizes)), key=owed.__getitem__)
        owed[turn] -= 1
        yield starts[turn] + next(streams[turn])  # pylint: disable=stop-iteration-return  # They never end.


class MixedSampler(Sampler[int]):  # pylint: disable=too-few-public-methods  # It is PyTorch's interface.
    """`mixed_indices` for a DataLoader, from draw number `start` on.

    With `share`, only one process's share of the draws: they are dealt in blocks of `block` to `share[1]`
    processes in turn, and this is the one numbered `share[0]`. Together the processes draw what one would.
    """

    def __init__(  # pylint: disable=too-many-arguments  # It is one stream's whole description.
        self,
        sizes: Sequence[int],
        weights: Sequence[float],
        seed: int,
        start: int = 0,
        *,
        share: tuple[int, int] = (0, 1),
        block: int = 1,
    ):
        super().__init__()
        self.sizes, self.weights, self.seed, self.start = list(sizes), list(weights), seed, start
        self.share, self.block = share, block

    def __iter__(self) -> Iterator[int]:
        draws = islice(mixed_indices(self.sizes, self.weights, self.seed), self.start, None)
        member, members = self.share
        if members == 1:
            return draws
        return (index for number, index in enumerate(draws) if number // self.block % members == member)


def with_texts(batch: dict, sources: Sequence[Source]) -> dict:
    """Add each image's three texts to a batch. They stay out of the workers, which would each copy them all."""
    rows = [
        texts(sources[source].index.iloc[[position]])
        for source, position in zip(batch["source"].tolist(), batch["position"].tolist(), strict=True)
    ]
    return {**batch, "texts": {kind: [row[kind][0] for row in rows] for kind in TEXT_KINDS}}


def batches(
    sources: Sequence[Source],
    weights: Sequence[float] | None = None,
    *,
    batch_size: int = 8,
    seed: int = 0,
    start: int = 0,
    workers: int = 6,
    share: tuple[int, int] = (0, 1),
) -> Iterator[dict]:
    """Batches without end, mixing `sources` by `weights` (equal by default). `start` is how many images to skip.

    `share` is (this process, how many processes): each takes every batch in turn, so none repeats another's.
    `start` counts the images of all of them.
    """
    datasets = []
    for number, source in enumerate(sources):
        source.dataset.source = number
        datasets.append(source.dataset)
    sizes = [len(dataset) for dataset in datasets]
    sampler = MixedSampler(sizes, weights or [1.0] * len(datasets), seed, start, share=share, block=batch_size)
    loader = DataLoader(
        ConcatDataset(datasets),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        drop_last=True,
        pin_memory=th.cuda.is_available(),
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers else None,
    )
    for batch in loader:
        yield with_texts(batch, sources)

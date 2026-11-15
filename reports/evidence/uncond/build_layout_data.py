"""Turn cache_layouts.py's per-shard thumbnails into what layout_prior.py trains on.

    uv run python reports/evidence/uncond/build_layout_data.py merge D:/laion-cache
    uv run python reports/evidence/uncond/build_layout_data.py embed D:/laion-cache

`merge`, on the CPU, joins the shards in order into images64.u8, raw uint8 [n, 64, 64, 3],
and captions.txt, a caption per line, and writes their count to meta.json. `embed`, on
the GPU, writes each caption's CLIP text embedding, unnormalised float16, to
text_MODEL.f16, [n, features], for the models of the default guide ensemble. Both are
read back with numpy.memmap.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch as th

SIDE = 64
TEXT_MODELS = ("PE-Core-B-16", "ViT-B/32")
BATCH = 1024


def text_file(folder: Path, model_name: str) -> Path:
    return folder / f"text_{model_name.replace('/', '-')}.f16"


def merge(folder: Path) -> None:
    shards = sorted(folder.glob("[0-9]*.npy"))
    counts = [np.load(shard, mmap_mode="r").shape[0] for shard in shards]
    total = sum(counts)
    out = np.memmap(folder / "images64.u8", dtype=np.uint8, mode="w+", shape=(total, SIDE, SIDE, 3))
    start = 0
    with (folder / "captions.txt").open("w", encoding="utf-8", newline="\n") as captions:
        for index, (shard, count) in enumerate(zip(shards, counts, strict=True)):
            lines = shard.with_suffix(".txt").read_text(encoding="utf-8").splitlines()
            assert len(lines) == count, f"{shard} has {count} images and {len(lines)} captions."
            out[start : start + count] = np.load(shard)
            captions.write("\n".join(lines) + "\n")
            start += count
            if (index + 1) % 100 == 0:
                print(f"{index + 1}/{len(shards)} shards, {start} images", flush=True)
    out.flush()
    (folder / "meta.json").write_text(json.dumps({"count": total, "side": SIDE, "shards": len(shards)}))
    print(f"{total} images from {len(shards)} shards", flush=True)


def embed(folder: Path) -> None:
    import clip
    import open_clip

    from cgd.clip_models import CLIP_CACHE_PATH, OPEN_CLIP_MODELS, load_clip

    count = json.loads((folder / "meta.json").read_text())["count"]
    captions = (folder / "captions.txt").read_text(encoding="utf-8").splitlines()
    assert len(captions) == count, f"{len(captions)} captions for {count} images."
    for model_name in TEXT_MODELS:
        path = text_file(folder, model_name)
        if path.exists():
            print(f"{path} exists, skipping", flush=True)
            continue
        model, _ = load_clip(model_name, "cuda", True)
        if model_name in OPEN_CLIP_MODELS:
            model.text.to("cuda", th.float16)  # load_clip() leaves the text tower on the CPU, in float32.
            tokenizer = open_clip.get_tokenizer(model_name, cache_dir=str(CLIP_CACHE_PATH))
        else:
            tokenizer = lambda texts: clip.tokenize(texts, truncate=True)  # noqa: E731
        out = None
        partial = path.with_suffix(".partial")
        for start in range(0, count, BATCH):
            tokens = tokenizer(captions[start : start + BATCH]).to("cuda")
            with th.no_grad(), th.autocast("cuda", dtype=th.float16):
                features = model.encode_text(tokens).float()
            if out is None:
                out = np.memmap(partial, dtype=np.float16, mode="w+", shape=(count, features.shape[1]))
            out[start : start + features.shape[0]] = features.cpu().numpy().astype(np.float16)
            if (start // BATCH) % 200 == 0:
                print(f"{model_name} {start}/{count}", flush=True)
        out.flush()
        out._mmap.close()  # Windows will not rename a file that is still mapped.
        del out
        partial.replace(path)
        load_clip.cache_clear()
        del model
        th.cuda.empty_cache()
        print(f"{model_name} done", flush=True)


if __name__ == "__main__":
    {"merge": merge, "embed": embed}[sys.argv[1]](Path(sys.argv[2]))

"""Train layout_prior.LayoutPrior on build_layout_data.py's thumbnails and caption embeddings.

    uv run python reports/evidence/uncond/train_layout_prior.py --data D:/laion-cache --out runs/layout

Reads images64.u8 and text_MODEL.f16 through numpy.memmap, so nothing is decoded or encoded
while training and the GPU is the only bottleneck. The last --holdout images are never
trained on: their loss is the validation loss. Saves --out/last.pt (weights, EMA, optimiser,
step) every --save-every steps and resumes from it, and writes a grid of the preview
prompts' thumbnails every --preview-every steps.
"""

import argparse
import json
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch as th
import torch.nn.functional as tf
import torchvision.transforms.functional as tvf
from build_layout_data import SIDE, TEXT_MODELS, text_file
from common import HELD_OUT
from layout_prior import LayoutPrior, join_text, sample

from cgd.clip_models import encode_texts, load_clip

PREVIEW_PROMPTS = (*HELD_OUT[:8], "a portrait of a woman with red hair", "a castle on a hill, digital art",
                   "a bowl of fruit on a wooden table", "the Eiffel tower at night")  # fmt: skip


class Batches:
    """Random batches of (uint8 images, text embeddings) read by background threads, forever."""

    def __init__(self, folder: Path, count: int, batch_size: int, seed: int, threads: int = 4):
        self.images = np.memmap(folder / "images64.u8", dtype=np.uint8, mode="r").reshape(-1, SIDE, SIDE, 3)
        self.texts = [
            np.memmap(text_file(folder, name), dtype=np.float16, mode="r").reshape(self.images.shape[0], -1)
            for name in TEXT_MODELS
        ]
        self.count, self.batch_size = count, batch_size
        self.queue: queue.Queue = queue.Queue(maxsize=8)
        for index in range(threads):
            threading.Thread(target=self.fill, args=(seed * 1000 + index,), daemon=True).start()

    def read(self, indices: np.ndarray) -> tuple[th.Tensor, list[th.Tensor]]:
        indices = np.sort(indices)  # In file order, which is kinder to the disk.
        images = th.from_numpy(self.images[indices])
        texts = [th.from_numpy(text[indices].astype(np.float32)) for text in self.texts]
        return images, texts

    def fill(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        while True:
            images, texts = self.read(rng.integers(0, self.count, self.batch_size))
            self.queue.put((images.pin_memory(), [text.pin_memory() for text in texts]))

    def __next__(self) -> tuple[th.Tensor, th.Tensor]:
        images, texts = self.queue.get()
        return prepare(images, texts)


def prepare(images: th.Tensor, texts: list[th.Tensor], flip: bool = True) -> tuple[th.Tensor, th.Tensor]:
    """A batch on the GPU: images in [-1, 1] as [batch, 3, 64, 64], half of them mirrored, and their joined text."""
    x = images.to("cuda", non_blocking=True).permute(0, 3, 1, 2).float().div(127.5).sub(1)
    if flip:
        mirrored = th.rand(x.shape[0], device="cuda") < 0.5
        x = th.where(mirrored[:, None, None, None], x.flip(-1), x)
    return x, join_text([text.to("cuda", non_blocking=True) for text in texts])


def draw_times(count: int, mean: float, std: float) -> th.Tensor:
    """Times in (0, 1): mostly logit-normal, which favours the middle noise levels, and a tenth uniform."""
    logit_normal = th.sigmoid(th.randn(count, device="cuda") * std + mean)
    uniform = th.rand(count, device="cuda")
    return th.where(th.rand(count, device="cuda") < 0.1, uniform, logit_normal).clamp(1e-4, 1 - 1e-4)


def flow_loss(model, x: th.Tensor, text: th.Tensor, t: th.Tensor, present: th.Tensor | None) -> th.Tensor:
    noise = th.randn_like(x)
    tt = t.view(-1, 1, 1, 1)
    with th.autocast("cuda", dtype=th.bfloat16):
        predicted = model((1 - tt) * x + tt * noise, t, text, present)
    return tf.mse_loss(predicted.float(), noise - x)


@th.no_grad()
def validation_loss(model, holdout: tuple[th.Tensor, list[th.Tensor]], batch_size: int = 500) -> float:
    """The loss on the held-out images at fixed times and noise, so it is comparable between steps."""
    images, texts = holdout
    generator = th.Generator("cuda").manual_seed(0)
    total, batches = 0.0, 0
    for start in range(0, images.shape[0], batch_size):
        x, text = prepare(images[start : start + batch_size], [t[start : start + batch_size] for t in texts], False)
        t = th.rand(x.shape[0], device="cuda", generator=generator).clamp(1e-4, 1 - 1e-4)
        noise = th.randn(x.shape, device="cuda", generator=generator)
        tt = t.view(-1, 1, 1, 1)
        with th.autocast("cuda", dtype=th.bfloat16):
            predicted = model((1 - tt) * x + tt * noise, t, text)
        total += tf.mse_loss(predicted.float(), noise - x).item()
        batches += 1
    return total / batches


def preview_text() -> th.Tensor:
    """The preview prompts' joined text embeddings, made once before training takes the GPU's memory."""
    embeddings = []
    for name in TEXT_MODELS:
        embeddings.append(encode_texts(list(PREVIEW_PROMPTS), name, "cuda", True, truncate=True))
        load_clip.cache_clear()
    th.cuda.empty_cache()
    return join_text(embeddings)


def save_preview(model, text: th.Tensor, path: Path, scale: float) -> None:
    generator = th.Generator("cuda").manual_seed(0)
    rows = []
    for _ in range(2):  # Two thumbnails of each prompt.
        images = sample(model, text, scale=scale, generator=generator).add(1).div(2)
        rows.append(th.cat(list(images), dim=-1))
    grid = th.cat(rows, dim=-2)
    tvf.to_pil_image(grid.cpu()).resize((grid.shape[-1] * 2, grid.shape[-2] * 2)).save(path)


def main() -> None:  # pylint: disable=too-many-locals,too-many-statements
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--patch", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--ema", type=float, default=0.9999)
    parser.add_argument("--p-uncond", type=float, default=0.1)
    parser.add_argument("--time-mean", type=float, default=0.0)
    parser.add_argument("--time-std", type=float, default=1.2)
    parser.add_argument("--holdout", type=int, default=10_000)
    parser.add_argument("--steps", type=int, default=400_000)
    parser.add_argument("--stop-after", type=float, default=0.0, help="Save and exit after this many hours.")
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--preview-every", type=int, default=10_000)
    parser.add_argument("--preview-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    th.manual_seed(args.seed)
    th.backends.cuda.matmul.allow_tf32 = True
    th.backends.cudnn.allow_tf32 = True
    args.out.mkdir(parents=True, exist_ok=True)
    count = json.loads((args.data / "meta.json").read_text())["count"]
    train_count = count - args.holdout
    previews = preview_text()

    text_dims = [np.memmap(text_file(args.data, n), dtype=np.float16, mode="r").size // count for n in TEXT_MODELS]
    model = LayoutPrior(text_dims, width=args.width, depth=args.depth, heads=args.heads, patch=args.patch).to("cuda")
    ema = LayoutPrior(**model.config).to("cuda").requires_grad_(False)
    ema.load_state_dict(model.state_dict())
    optimiser = th.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0, fused=True)
    step = 0
    last = args.out / "last.pt"
    if last.exists():
        state = th.load(last, map_location="cuda", weights_only=False)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimiser.load_state_dict(state["optimiser"])
        step = state["step"]
        print(f"Resumed at step {step}", flush=True)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters, {train_count} images, "
          f"batch {args.batch_size}", flush=True)  # fmt: skip

    batches = Batches(args.data, train_count, args.batch_size, args.seed + step)
    holdout = batches.read(np.arange(train_count, count))
    log = (args.out / "log.jsonl").open("a", buffering=1)
    smoothed, started, first = None, time.perf_counter(), step
    while step < args.steps:
        for group in optimiser.param_groups:
            group["lr"] = args.lr * min(1.0, (step + 1) / args.warmup)
        x, text = next(batches)
        present = th.rand(x.shape[0], device="cuda") >= args.p_uncond
        loss = flow_loss(model, x, text, draw_times(x.shape[0], args.time_mean, args.time_std), present)
        loss.backward()
        th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)
        step += 1
        decay = min(args.ema, (1 + step) / (10 + step))
        with th.no_grad():
            for e, p in zip(ema.parameters(), model.parameters(), strict=True):
                e.lerp_(p, 1 - decay)
        if step % 100 == 0:
            value = loss.item()
            smoothed = value if smoothed is None else 0.9 * smoothed + 0.1 * value
        hours = (time.perf_counter() - started) / 3600
        stopping = bool(args.stop_after) and hours >= args.stop_after
        if step % 1000 == 0 or stopping:
            record = {"step": step, "loss": smoothed, "images_per_s": (step - first) * args.batch_size / (hours * 3600),
                      "peak_gib": th.cuda.max_memory_allocated() / 2**30, "hours": hours}  # fmt: skip
            if step % args.save_every == 0 or stopping:
                record["validation"] = validation_loss(ema, holdout)
            log.write(json.dumps(record) + "\n")
            print(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in record.items()),
                  flush=True)  # fmt: skip
        if step % args.save_every == 0 or step == args.steps or stopping:
            state = {"step": step, "args": vars(args) | {"data": str(args.data), "out": str(args.out)},
                     "config": model.config, "model": model.state_dict(), "ema": ema.state_dict(),
                     "optimiser": optimiser.state_dict()}  # fmt: skip
            partial = last.with_suffix(".tmp")
            th.save(state, partial)
            partial.replace(last)
        if step % args.preview_every == 0 or step == args.steps or stopping:
            save_preview(ema, previews, args.out / f"preview_{step:07}.png", args.preview_scale)
        if stopping:
            print(f"Stopping after {hours:.2f} hours at step {step}", flush=True)
            break


if __name__ == "__main__":
    main()

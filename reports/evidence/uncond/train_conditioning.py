"""Fine-tune the unconditional 256 U-Net to follow a CLIP embedding, without changing what it is at step 0.

uv run python reports/evidence/uncond/train_conditioning.py --data '/path/to/laion/*.tar' --out runs/emb

Data is webdataset tars of 256x256 JPEGs with a caption (.txt) and LAION metadata (.json), as
img2dataset writes them. Each image is conditioned, at random, on nothing (so classifier-free
guidance works), on the CLIP text embedding of its caption (what sampling gives it), or on the
CLIP image embedding of itself with LAFITE's noise (which teaches it to read embeddings that are
near an image's but not one). Only the projection, and with --train emb the timestep MLP and
every ResBlock's embedding layers, learn. The rest of the U-Net stays frozen.

Saves the trainable weights, their EMA, the optimiser and the step to --out/last.pt every
--save-every steps, and resumes from it. Writes a preview grid every --preview-every steps.
"""

import argparse
import glob
import io
import json
import random
import tarfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch as th
import torch.nn.functional as tf
import torchvision.transforms.functional as tvf
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from cgd import adm, sampler
from cgd.clip_models import encode_texts, load_clip, preprocess
from cgd.download import download
from conditioning import ClipConditionedUNet, set_trainable

PREVIEW_PROMPTS = (
    "a red fox in the snow",
    "a bowl of ramen with a soft boiled egg",
    "a lighthouse on a cliff at sunset",
    "a neon-lit city street in the rain",
)


class Shards(IterableDataset):
    """(image in [-1, 1], caption) from webdataset tars, shuffled by shard and in a buffer, forever."""

    def __init__(self, paths: list[str], seed: int, max_unsafe: float, max_watermark: float, buffer: int = 2000):
        self.paths, self.seed, self.buffer = paths, seed, buffer
        self.max_unsafe, self.max_watermark = max_unsafe, max_watermark

    def samples(self, path: str) -> Iterator[tuple[bytes, str]]:
        group: dict[str, bytes] = {}
        key = None
        with tarfile.open(path, "r|") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                stem, _, extension = member.name.rpartition(".")
                if stem != key:
                    yield from self.complete(group)
                    group, key = {}, stem
                group[extension] = tar.extractfile(member).read()
        yield from self.complete(group)

    def complete(self, group: dict[str, bytes]) -> Iterator[tuple[bytes, str]]:
        if "jpg" not in group or "txt" not in group:
            return
        meta = json.loads(group.get("json", b"{}"))
        if (meta.get("punsafe") or 0) > self.max_unsafe or (meta.get("pwatermark") or 0) > self.max_watermark:
            return
        yield group["jpg"], group["txt"].decode("utf-8", "replace").strip()

    def __iter__(self):
        worker = get_worker_info()
        index, count = (worker.id, worker.num_workers) if worker else (0, 1)
        rng = random.Random(self.seed * 1000 + index)
        mine = self.paths[index::count]
        pool: list[tuple[bytes, str]] = []
        while True:
            rng.shuffle(mine)
            for path in mine:
                try:
                    for sample in self.samples(path):
                        pool.append(sample)
                        if len(pool) >= self.buffer:
                            yield self.decode(pool.pop(rng.randrange(len(pool))), rng)
                except (tarfile.TarError, OSError) as error:
                    print(f"Skipping the rest of {path}: {error}", flush=True)

    @staticmethod
    def decode(sample: tuple[bytes, str], rng: random.Random) -> tuple[th.Tensor, str]:
        jpeg, caption = sample
        image = Image.open(io.BytesIO(jpeg)).convert("RGB")
        if image.size != (256, 256):
            side = min(image.size)
            image = tvf.center_crop(image, side).resize((256, 256), Image.BICUBIC)
        x = tvf.to_tensor(image).mul(2).sub(1)
        return (x.flip(-1) if rng.random() < 0.5 else x), caption


def conditions(clip_model: str, images: th.Tensor, captions: list[str], args, generator: th.Generator):
    """Each image's condition and whether it has one: a caption's text embedding, or its own noised image embedding."""
    device = "cuda"  # The name load_clip() is cached under.
    model, size = load_clip(clip_model, device, True)
    with th.no_grad():
        pixels = tf.interpolate(images.add(1).div(2), (size, size), mode="bicubic", antialias=True).clamp(0, 1)
        image_embeds = tf.normalize(model.encode_image(preprocess(model, pixels)).float(), dim=-1)
        text_embeds = tf.normalize(encode_texts(captions, clip_model, device, True, truncate=True), dim=-1)
    draw = th.rand(images.shape[0], device=device, generator=generator)
    present = draw >= args.p_uncond
    use_text = (draw >= args.p_uncond) & (draw < args.p_uncond + args.p_text)
    noise = th.randn(image_embeds.shape, device=device, generator=generator)
    xi = th.rand(images.shape[0], 1, device=device, generator=generator) * args.lafite_noise
    noised = tf.normalize(image_embeds + xi * tf.normalize(noise, dim=-1), dim=-1)
    return th.where(use_text[:, None], text_embeds, noised), present


@contextmanager
def ema_weights(parameters: list[th.nn.Parameter], ema: list[th.Tensor]):
    saved = [p.detach().clone() for p in parameters]
    with th.no_grad():
        for p, e in zip(parameters, ema, strict=True):
            p.copy_(e)
    try:
        yield
    finally:
        with th.no_grad():
            for p, s in zip(parameters, saved, strict=True):
                p.copy_(s)


@th.no_grad()
def preview(model: ClipConditionedUNet, log_alpha_bar, clip_model: str, path: Path, scale: float, steps: int = 30):
    """A row per prompt of classifier-free guided samples, without CLIP guidance, at a fixed seed."""
    device = "cuda"
    model.eval()
    embeds = tf.normalize(encode_texts(list(PREVIEW_PROMPTS), clip_model, device, True, truncate=True), dim=-1)
    generator = th.Generator(device).manual_seed(0)
    x = th.randn((len(PREVIEW_PROMPTS), 3, 256, 256), device=device, generator=generator)
    ts = sampler.timesteps(log_alpha_bar, steps)
    levels = [sampler.level(log_alpha_bar, t) for t in ts]
    previous = None
    autocast = th.autocast("cuda", dtype=th.bfloat16 if th.cuda.get_device_capability() >= (8, 0) else th.float16)
    for i in range(len(ts) - 1):
        t = th.full((x.shape[0],), ts[i], device=device)
        with autocast:
            eps_c = model(x, t, embeds)[:, :3].float()
            eps_u = model(x, t)[:, :3].float()
        eps = eps_u + scale * (eps_c - eps_u)
        x0 = ((x - levels[i].sigma * eps) / levels[i].alpha).clamp(-1, 1)
        if i == len(ts) - 2:
            break
        step = levels[i], levels[i + 1]
        noise = th.randn(x.shape, device=device, generator=generator)
        x = sampler.dpm_solver_step(x, x0, x0, previous, step, noise=noise)
        previous = sampler.Previous(x0, sampler.step_size(*step))
    grid = th.cat(list(x0.add(1).div(2).clamp(0, 1)), dim=-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    tvf.to_pil_image(grid.cpu()).save(path)
    model.train()


def build(args, device: str) -> tuple[ClipConditionedUNet, np.ndarray]:
    flags = adm.model_and_diffusion_defaults() | adm.CHECKPOINT_FLAGS[256]
    flags |= {"image_size": 256, "class_cond": False, "use_fp16": False, "use_checkpoint": False}
    with th.device("meta"):
        unet, diffusion = adm.create_model_and_diffusion(**flags)
    url = adm.UNCONDITIONAL_URLS[256]
    path = download(url, url.rpartition("/")[2])
    unet.load_state_dict(th.load(path, map_location="cpu", mmap=True, weights_only=True), assign=True)
    dim = encode_texts(["a photo"], args.clip_model, device, True).shape[-1]
    model = ClipConditionedUNet(unet, dim).to(device)
    model.gradient_checkpointing = True
    return model, sampler.schedule(diffusion.alphas_cumprod)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="A glob of webdataset tars.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--clip-model", default="PE-Core-B-16")
    parser.add_argument("--train", choices=("project", "emb", "emb_attn"), default="emb")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulate", type=int, default=4, help="Micro-batches per optimiser step.")
    parser.add_argument("--lr", type=float, default=1e-4, help="For the new projection.")
    parser.add_argument("--lr-shared", type=float, default=1e-5, help="For the pretrained layers that learn.")
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--ema", type=float, default=0.9995)
    parser.add_argument("--p-uncond", type=float, default=0.1)
    parser.add_argument("--p-text", type=float, default=0.45)
    parser.add_argument("--lafite-noise", type=float, default=0.5, help="The most noise on image embeddings.")
    parser.add_argument("--high-noise", type=float, default=0.0,
                        help="The share of timesteps drawn only from the top 70%% of the schedule, where the "
                             "composition forms and a condition has the most to say. The rest are uniform.")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--max-unsafe", type=float, default=0.1)
    parser.add_argument("--max-watermark", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--preview-every", type=int, default=2000)
    parser.add_argument("--preview-scale", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = "cuda"
    th.manual_seed(args.seed)
    th.backends.cuda.matmul.allow_tf32 = True
    th.backends.cudnn.allow_tf32 = True
    args.out.mkdir(parents=True, exist_ok=True)
    model, log_alpha_bar = build(args, device)
    if args.channels_last:
        model.to(memory_format=th.channels_last)
    model.train()
    trainable = set_trainable(model, args.train)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    new = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("project.")]
    shared = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("project.")]
    groups = [{"params": new, "lr": args.lr}] + ([{"params": shared, "lr": args.lr_shared}] if shared else [])
    optimiser = th.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=0.0)
    base_lrs = [g["lr"] for g in optimiser.param_groups]
    ema = [p.detach().clone().float() for p in trainable]
    bf16 = th.cuda.get_device_capability() >= (8, 0)
    scaler = th.amp.GradScaler("cuda", enabled=not bf16)
    alphas_bar = th.tensor(np.exp(log_alpha_bar), device=device, dtype=th.float32)

    step = 0
    last = args.out / "last.pt"
    if last.exists():
        state = th.load(last, map_location=device, weights_only=False)
        with th.no_grad():
            for p, name in zip(trainable, names, strict=True):
                p.copy_(state["trainable"][name].to(device))
            for e, name in zip(ema, names, strict=True):
                e.copy_(state["ema"][name].to(device))
        optimiser.load_state_dict(state["optimiser"])
        scaler.load_state_dict(state["scaler"])
        step = state["step"]
        print(f"Resumed at step {step}", flush=True)
    print(f"Training {sum(p.numel() for p in trainable) / 1e6:.2f}M parameters ({args.train}), "
          f"{'bf16' if bf16 else 'fp16'}, batch {args.batch_size} x {args.accumulate}", flush=True)

    paths = sorted(glob.glob(args.data))
    loader = DataLoader(
        Shards(paths, args.seed + step, args.max_unsafe, args.max_watermark),
        batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
    )
    batches = iter(loader)
    generator = th.Generator(device).manual_seed(args.seed + step)
    log = (args.out / "log.jsonl").open("a", buffering=1)
    smoothed, started, seen = None, time.perf_counter(), 0
    th.cuda.reset_peak_memory_stats()
    while step < args.steps:
        for group, base in zip(optimiser.param_groups, base_lrs, strict=True):
            group["lr"] = base * min(1.0, (step + 1) / args.warmup)
        total = 0.0
        for _ in range(args.accumulate):
            images, captions = next(batches)
            images = images.to(device, non_blocking=True)
            condition, present = conditions(args.clip_model, images, list(captions), args, generator)
            t = th.randint(0, len(alphas_bar), (images.shape[0],), device=device, generator=generator)
            if args.high_noise:
                floor = int(sampler.RECUR_ABOVE * len(alphas_bar))
                high = th.randint(floor, len(alphas_bar), t.shape, device=device, generator=generator)
                t = th.where(th.rand(t.shape, device=device, generator=generator) < args.high_noise, high, t)
            noise = th.randn(images.shape, device=device, generator=generator)
            a = alphas_bar[t].view(-1, 1, 1, 1)
            noisy = a.sqrt() * images + (1 - a).sqrt() * noise
            with th.autocast("cuda", dtype=th.bfloat16 if bf16 else th.float16):
                predicted = model(noisy, t, condition, present)[:, :3]
            loss = tf.mse_loss(predicted.float(), noise) / args.accumulate
            scaler.scale(loss).backward()
            total += loss.item()
            seen += images.shape[0]
        scaler.unscale_(optimiser)
        th.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimiser)
        scaler.update()
        optimiser.zero_grad(set_to_none=True)
        with th.no_grad():
            decay = min(args.ema, (1 + step) / (10 + step))
            for e, p in zip(ema, trainable, strict=True):
                e.lerp_(p.detach().float(), 1 - decay)
        step += 1
        smoothed = total if smoothed is None else 0.98 * smoothed + 0.02 * total
        if step % 50 == 0:
            elapsed = time.perf_counter() - started
            record = {"step": step, "loss": smoothed, "images_per_s": seen / elapsed,
                      "peak_gib": th.cuda.max_memory_allocated() / 2**30, "lr": optimiser.param_groups[0]["lr"]}
            log.write(json.dumps(record) + "\n")
            print(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in record.items()),
                  flush=True)
        if step % args.save_every == 0 or step == args.steps:
            state = {
                "step": step, "args": vars(args) | {"out": str(args.out)},
                "trainable": {n: p.detach().cpu() for n, p in zip(names, trainable, strict=True)},
                "ema": {n: e.detach().cpu() for n, e in zip(names, ema, strict=True)},
                "optimiser": optimiser.state_dict(), "scaler": scaler.state_dict(),
            }
            partial = last.with_suffix(".tmp")
            th.save(state, partial)
            partial.replace(last)
        if step % args.preview_every == 0 or step == args.steps:
            with ema_weights(trainable, ema):
                preview(model, log_alpha_bar, args.clip_model, args.out / f"preview_{step:07}.png", args.preview_scale)


if __name__ == "__main__":
    main()

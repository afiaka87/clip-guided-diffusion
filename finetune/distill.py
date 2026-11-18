r"""Bake CLIP guidance into the adapters: train them to give `cgd`'s guided prediction in one forward pass.

    uv run --no-sync python -m finetune.distill --hours 0.5 --wandb
    uv run --no-sync python -m finetune.distill --hours 0.2 --overfit --wandb   # Only the texts it samples
    uv run --no-sync python -m finetune.distill --max-steps 0 --wandb           # Only the teacher's own images
    uv run --no-sync python -m finetune.distill --hours 0.5 --direct --wandb    # The CLIP loss itself, no copying

The teacher is `cgd` as it samples. The unconditional 256 model predicts the noise, CLIP scores cutouts of the
denoised image against a text, and the capped gradient of that score moves the prediction: what
`cgd.sampler.predict` does with `cgd.adm.guidance_loss`, by the same functions. The student is the same model
with `cgd.conditioned`'s adapters, told the text's CLIP embeddings. It learns to predict the teacher's moved
noise, so that sampling from it needs no CLIP and no backward pass.

Each example is one noisy image and one text:

- **The image** is a training image at a noise level drawn as the sampler steps, within the part of the
  schedule `cgd` guides.
- **The text** is one of the image's three texts, or with `--p-swap` another image's, because guidance mostly
  has to move an image towards a text it does not match yet.
- **The loss** is the student's squared error from the teacher's noise, divided by the size of the teacher's
  move. `missed` is 1 for a student that ignores the text and 0 for one that reproduces the move.

The teacher's move differs from one draw of cutouts to the next, and only what the draws share can be learnt.
`teacher/agreement` is the cosine between two draws, measured once, and `missed` cannot fall below one minus it.

`--direct` is the other way to the same end. The student is not shown the teacher's move. Its own denoised
image is scored by the same guidance loss, on the same kind of cutouts, and that loss is backpropagated into the
adapters. Its move from the base model's noise is held to `cgd`'s cap as the teacher's is, so what it learns is
the move that lowers the loss most within the cap, averaged over cutouts and images. `gain` is how much lower
its loss is than the base model's on the same cutouts, and `size` its move as a share of the cap. `--tv` is how
much of `cgd`'s smoothness term that loss keeps. It is 0, because a student trained on the whole term learns to
blur: for the teacher it damps one gradient, and here it is a direction every example agrees on.

Every `--eval-minutes` the run draws its sample texts from the student alone, without CLIP, and logs them under
the unconditional model's and the teacher's images of the same noise. `text/...` is each row's similarity to the
texts by a CLIP model that neither the teacher nor the student has seen, and `image/...` its similarity to the
held-out image each text describes. Those, and the images, say whether the student steers. `missed` only says
how well it copies.
"""

import argparse
import copy
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch as th
import wandb
from PIL import Image
from torch import nn
from torch.nn import functional as tf

from cgd import losses, sampler
from cgd.adm import CLIP_SCALE, RANGE_SCALE, TV_SCALE, denoise, guidance_loss, load_model
from cgd.clip_models import ensemble_members, load_clip
from cgd.conditioned import AdapterConfig, ConditionedUNet, load_adapter
from cgd.device import default_device
from cgd.guide import MIN_CUT_FRACTION, Guide, Tower, guide_loss
from cgd.losses import finite_gradient
from finetune import data, noise
from finetune.embed import DEFAULT_MODELS, Encoder, encode_images, encode_texts, load_encoder
from finetune.train import CHECKPOINT_NAME, IMAGE_SIZE, Run, as_input, log, step_generator, track, trim_log

WINDOW = sampler.UNCONDITIONAL_GUIDANCE_WINDOW  # The part of the schedule `cgd` guides this model in.
KIND_WEIGHTS = {"caption": 0.2, "first": 0.4, "short": 0.4}  # A person types something short.
SAMPLE_KIND = "first"
SAMPLES_NAME = "samples.pt"
AGREEMENT_LEVELS = (0.3, 0.6, 0.9)  # Fractions of the schedule the teacher's two draws are compared at.
GAIN_EVERY = 10  # `--direct` compares its loss with the base model's on one example in this many.


class Teacher(NamedTuple):
    """`cgd` under CLIP guidance: its U-Net as it loads it, and the CLIP models that score the denoised image."""

    unet: nn.Module
    towers: tuple[tuple[str, nn.Module, int], ...]  # Each CLIP model's name, the model, and its input size.
    cutouts: int
    rate: float


class Samples(NamedTuple):
    """The texts a run draws to show its progress, and the images to compare the student's with."""

    texts: list[str]
    targets: dict[str, th.Tensor]  # Each CLIP model's [N, dim] embedding of the texts.
    judged_text: th.Tensor  # The judge's embedding of the texts.
    rows: dict[str, th.Tensor]  # uint8 [N, 3, H, W] images by row: the real ones, and what each model drew.


def load_teacher(args: argparse.Namespace) -> tuple[Teacher, np.ndarray]:
    """The teacher and the U-Net's schedule. Load it before the student, which adapts its own U-Net in place."""
    half = th.device(args.device).type == "cuda"
    unet, log_alpha_bar = load_model(IMAGE_SIZE, args.device, half, False, True)
    if not half:
        unet = copy.deepcopy(unet)  # On a CPU the student would be handed this same cached model.
    towers = tuple((name, *load_clip(name, args.device, half)) for name in ensemble_members(args.teacher_clip))
    return Teacher(unet, towers, args.cutouts, args.rate), log_alpha_bar


def guide_for(teacher: Teacher, targets: Mapping[str, th.Tensor]) -> Guide:
    """`cgd`'s guide for one text, from each CLIP model's [1, dim] embedding of it."""
    towers = tuple(
        Tower(model, size, targets[name], th.ones(1, device=targets[name].device))
        for name, model, size in teacher.towers
    )
    return Guide(towers, teacher.cutouts, MIN_CUT_FRACTION)


def guided_shift(
    teacher: Teacher, log_alpha_bar: np.ndarray, x: th.Tensor, t: float, targets: Mapping[str, th.Tensor]
) -> tuple[th.Tensor, th.Tensor]:
    """The teacher's noise prediction for one noisy image, and how far CLIP guidance moves it.

    This is `cgd.sampler.predict` up to the cap, as a change to the noise: moving the denoised image along the
    capped gradient by Tweedie's formula subtracts sigma times that gradient from the predicted noise.
    """
    now = sampler.level(log_alpha_bar, t)
    x = x.detach().requires_grad_()
    eps = denoise(teacher.unet, None, x, th.full((len(x),), t, device=x.device, dtype=th.float32))
    x0 = (x - now.sigma * eps) / now.alpha
    loss = guidance_loss(guide_for(teacher, targets), x, x0, now.sigma)
    gradient = sampler.cap_guidance(-finite_gradient(loss, x), eps.detach(), now.sigma, teacher.rate)
    return eps.detach().float(), (-now.sigma * gradient).float()


def how_much_missed(predicted: th.Tensor, base: th.Tensor, shift: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
    """The share of the teacher's move each prediction misses, and the cosine between its own move and that.

    The first is 1 for a prediction that is the base model's and 0 for one that is the teacher's.
    """
    missed = (predicted - base - shift).square().mean(dim=(1, 2, 3)) / shift.square().mean(dim=(1, 2, 3))
    return missed, tf.cosine_similarity((predicted - base).flatten(1), shift.flatten(1), dim=1)


def capped_move(predicted: th.Tensor, base: th.Tensor, rate: float) -> tuple[th.Tensor, th.Tensor]:
    """The student's move from the base model's noise, scaled down to `cgd`'s cap, and its size as a share of it."""
    move = predicted - base
    size = (move.square().mean(dim=(1, 2, 3), keepdim=True) + 1e-12).sqrt() / (rate * sampler.rms(base))
    return move / size.clamp(min=1), size


def guided_loss(guide: Guide, x: th.Tensor, eps: th.Tensor, now: sampler.Level, tv: float = 1.0) -> th.Tensor:
    """`cgd`'s guidance loss of the image a noise prediction denoises x to, over its CLIP scale.

    It is `cgd.adm.guidance_loss` with `tv` times its smoothness term, so the whole of it at 1.
    """
    x0 = (x - now.sigma * eps) / now.alpha
    if x0.requires_grad:  # A CLIP gradient that overflowed at half precision teaches nothing, as in `cgd`.
        x0.register_hook(partial(th.nan_to_num, nan=0.0, posinf=0.0, neginf=0.0))
    clip_in = x0 * now.sigma + x * (1 - now.sigma)
    smooth = losses.tv_loss(clip_in).sum() * TV_SCALE * tv
    return (guide_loss(guide, clip_in, CLIP_SCALE) + losses.range_loss(x0).sum() * RANGE_SCALE + smooth) / CLIP_SCALE


def averages(window: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """The mean of each quantity over the examples that measured it."""
    keys = dict.fromkeys(key for entry in window for key in entry)
    return {key: float(np.mean([entry[key] for entry in window if key in entry])) for key in keys}


def guided_timestep(log_alpha_bar: np.ndarray, generator: th.Generator) -> float:
    """A timestep in the part of the schedule `cgd` guides, uniform in the noise level as its sampler steps."""
    last = len(log_alpha_bar) - 1
    table = sampler.log_snr(log_alpha_bar)
    cleanest, noisiest = (float(np.interp(bound * last, np.arange(len(table)), table)) for bound in WINDOW)
    level = noisiest + (cleanest - noisiest) * th.rand(1, generator=generator, dtype=th.float64)
    return float(noise.timestep_of(log_alpha_bar, level)[0])


def noisy(image: th.Tensor, log_alpha_bar: np.ndarray, t: float, generator: th.Generator, device: str) -> th.Tensor:
    """One uint8 image as the U-Net takes it, [1, 3, H, W], noised to timestep t."""
    now = sampler.level(log_alpha_bar, t)
    clean = as_input(image[None], device)
    return now.alpha * clean + now.sigma * th.randn(clean.shape, generator=generator).to(device)


def text_statistics(
    roots: Sequence[Path], models: Sequence[str], rows: int = 20_000, seed: int = 0
) -> dict[str, tuple[th.Tensor, th.Tensor]]:
    """Each CLIP model's mean and standard deviation over a sample of every kind of text of every dataset."""
    rng = np.random.default_rng(seed)
    return {name: embedding_statistics(roots, name, data.TEXT_KINDS, rows, rng) for name in models}


def embedding_statistics(
    roots: Sequence[Path], model_name: str, kinds: Sequence[str], rows: int, rng: np.random.Generator
) -> tuple[th.Tensor, th.Tensor]:
    """One CLIP model's mean and standard deviation over `rows` embeddings of each kind from each dataset."""
    parts = []
    for root in roots:
        for kind in kinds:
            store = np.load(data.embed_path(root, model_name, kind), mmap_mode="r")
            picks = np.sort(rng.choice(len(store), min(rows, len(store)), replace=False))
            parts.append(store[picks].astype(np.float32))
    sample = th.from_numpy(np.concatenate(parts))
    return sample.mean(0), sample.std(0)


def student_noise(run: Run, x: th.Tensor, timesteps: th.Tensor, embeds: Mapping[str, th.Tensor]) -> th.Tensor:
    """The noise the student predicts in x, told the text's embeddings."""
    with th.autocast(x.device.type, enabled=run.amp):
        return run.model(x.contiguous(memory_format=th.channels_last), timesteps, embeds)[:, :3].float()


def student_denoiser(
    run: Run, teacher: Teacher, log_alpha_bar: np.ndarray, embeds: Mapping[str, th.Tensor], scale: float
) -> sampler.Denoiser:
    """The student for the sampler: `scale` times its move from the base model's noise, where `cgd` would guide.

    Below the guided part of the schedule it was never trained, and the base model draws alone, as in `cgd`.
    """
    first_guided = WINDOW[0] * (len(log_alpha_bar) - 1)

    def predict(x: th.Tensor, timesteps: th.Tensor) -> th.Tensor:
        guided = float(timesteps[0]) >= first_guided
        base = denoise(teacher.unet, None, x, timesteps).float() if scale != 1 or not guided else None
        if not guided:
            return base
        told = student_noise(run, x, timesteps, embeds)
        return told if base is None else base + scale * (told - base)

    return predict


def draw(
    args: argparse.Namespace,
    log_alpha_bar: np.ndarray,
    denoiser_for: Callable[[int], sampler.Denoiser],
    guidance_for: Callable[[int], sampler.Guidance] | None = None,
) -> th.Tensor:
    """One image per sample text, each from its own seed, so every row of a grid starts from the same noise."""
    ts = sampler.timesteps(log_alpha_bar, args.sample_steps)
    images = []
    for i in range(args.samples):
        th.manual_seed(args.seed + 1000 + i)
        x = th.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=args.device)
        guidance = guidance_for(i) if guidance_for is not None else None
        last = list(sampler.sample(denoiser_for(i), log_alpha_bar, ts, x, guidance, progress=False))[-1]
        images.append(((last.clamp(-1, 1) + 1) * 127.5).round().byte().cpu())
    return th.cat(images)


def on_device(targets: Mapping[str, th.Tensor], rows: slice, device: str) -> dict[str, th.Tensor]:
    """Some rows of each CLIP model's embeddings, moved to where the models are."""
    return {name: embedding[rows].to(device) for name, embedding in targets.items()}


def held_out(args: argparse.Namespace) -> tuple[list[dict], list[str]]:
    """Held-out images with their embeddings, half from each dataset, and the text of each that is drawn."""
    rng = np.random.default_rng(args.seed)
    items, texts = [], []
    for name, root in (("laion", args.laion), ("imagenet", args.imagenet)):
        source = data.open_source(name, root, args.models, split="holdout")
        for position in rng.choice(len(source.dataset), args.samples // 2, replace=False).tolist():
            items.append(source.dataset[position])
            texts.append(data.texts(source.index.iloc[[position]])[SAMPLE_KIND][0])
    return items, texts


def make_samples(args: argparse.Namespace, teacher: Teacher, log_alpha_bar: np.ndarray, judge: Encoder) -> Samples:
    """Held-out images and their texts, with what the unconditional model and the teacher draw for each text."""
    items, texts = held_out(args)
    targets = {name: th.stack([item["embeds"][name][SAMPLE_KIND] for item in items]) for name in args.models}
    plain = partial(denoise, teacher.unet, None)

    def unguided(_: int) -> sampler.Denoiser:
        return plain

    def guidance(i: int) -> sampler.Guidance:
        guide = guide_for(teacher, on_device(targets, slice(i, i + 1), args.device))
        return sampler.Guidance(partial(guidance_loss, guide), teacher.rate, WINDOW)

    rows = {
        "real": th.stack([item["image"] for item in items]),
        "unconditional": draw(args, log_alpha_bar, unguided),
        "teacher": draw(args, log_alpha_bar, unguided, guidance),
    }
    with th.no_grad():
        return Samples(texts, targets, encode_texts(judge, texts), rows)


def load_samples(args: argparse.Namespace, teacher: Teacher, log_alpha_bar: np.ndarray, judge: Encoder) -> Samples:
    """The run's samples, made on its first start and kept with it, so a resumed run compares with the same ones."""
    path = args.out / SAMPLES_NAME
    if args.resume and path.is_file():
        return Samples(**th.load(path, map_location="cpu", weights_only=True))
    samples = make_samples(args, teacher, log_alpha_bar, judge)
    th.save(samples._asdict(), path)
    return samples


@th.no_grad()
def judged(judge: Encoder, samples: Samples, rows: Mapping[str, th.Tensor]) -> dict[str, float]:
    """Each row's mean similarity to the texts and to the real images, by the judge."""
    real = encode_images(judge, samples.rows["real"])
    record = {}
    for name, images in rows.items():
        features = encode_images(judge, images)
        record[f"text/{name}"] = (features * samples.judged_text).sum(1).mean().item()
        record[f"image/{name}"] = (features * real).sum(1).mean().item()
    return record


def teacher_agreement(args: argparse.Namespace, teacher: Teacher, log_alpha_bar: np.ndarray, samples: Samples) -> float:
    """The mean cosine between the teacher's moves from two draws of cutouts, over the samples at a few noise levels.

    A student can only learn what the draws share, so its `missed` cannot fall below one minus this.
    """
    generator = th.Generator().manual_seed(args.seed)
    last, cosines = len(log_alpha_bar) - 1, []
    for i, image in enumerate(samples.rows["real"]):
        t = AGREEMENT_LEVELS[i % len(AGREEMENT_LEVELS)] * last
        x = noisy(image, log_alpha_bar, t, generator, args.device)
        targets = on_device(samples.targets, slice(i, i + 1), args.device)
        first, second = (guided_shift(teacher, log_alpha_bar, x, t, targets)[1] for _ in range(2))
        cosines.append(tf.cosine_similarity(first.flatten(1), second.flatten(1), dim=1).item())
    return float(np.mean(cosines))


@contextmanager
def averaged(run: Run) -> Iterator[None]:
    """The model with the running average of its trained weights, which are put back afterwards."""
    trained = {name: weight.detach().cpu().clone() for name, weight in run.weights.items()}
    with th.no_grad():
        for name, weight in run.weights.items():
            weight.copy_(run.average[name])
    try:
        yield
    finally:
        with th.no_grad():
            for name, weight in run.weights.items():
                weight.copy_(trained[name])


def student_rows(run: Run, teacher: Teacher, log_alpha_bar: np.ndarray, samples: Samples) -> dict[str, th.Tensor]:
    """The samples as the student's averaged weights draw them alone: its move as it is, and times `--sample-scale`."""
    args, rows = run.args, {}
    with averaged(run):
        for scale in dict.fromkeys((1.0, args.sample_scale)):

            def denoiser(i: int, scale: float = scale) -> sampler.Denoiser:
                embeds = on_device(samples.targets, slice(i, i + 1), args.device)
                return student_denoiser(run, teacher, log_alpha_bar, embeds, scale)

            rows[f"student x{scale:g}"] = draw(args, log_alpha_bar, denoiser)
    return rows


def evaluate(
    run: Run, teacher: Teacher, log_alpha_bar: np.ndarray, samples: Samples, judge: Encoder, **extra: float
) -> None:
    """Draw the samples from the student, judge every row, and log the scores and the grid of images."""
    rows = dict(samples.rows)
    if run.steps:  # Before its first step the student is the unconditional model.
        rows |= student_rows(run, teacher, log_alpha_bar, samples)
    log(run.args, step=run.steps, hours=run.seconds / 3600, **judged(judge, samples, rows), **extra)
    grid = th.cat([th.cat(list(images), dim=2) for images in rows.values()], dim=1).permute(1, 2, 0).numpy()
    Image.fromarray(grid).save(run.args.out / f"samples_{run.steps:07d}.jpg", quality=92)
    if wandb.run is not None:
        caption = "Rows: " + ", ".join(rows) + ". Columns: " + " | ".join(samples.texts)
        wandb.log({"samples": wandb.Image(grid, caption=caption)}, step=run.steps)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """The command line's options, in the groups `-h` shows."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    run = parser.add_argument_group("the run")
    run.add_argument("--out", type=Path, default=Path("D:/cgd-finetune/distill"))
    run.add_argument("--resume", action="store_true", help="continue from --out's checkpoint")
    run.add_argument("--hours", type=float, default=0.5, help="stop after this long, sampling included")
    run.add_argument("--max-steps", type=int, default=None, help="stop after this many examples. 0 only samples")
    run.add_argument("--device", default=default_device())
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--no-amp", action="store_true", help="full precision. Half precision is the default on a GPU")
    run.add_argument("--log-every", type=int, default=50)
    run.add_argument("--eval-minutes", type=float, default=10.0, help="how often to draw and judge the samples")
    run.add_argument("--save-minutes", type=float, default=10.0)
    run.add_argument("--wandb", nargs="?", const="cgd-finetune", default=None, metavar="PROJECT")
    run.add_argument("--wandb-name", default=None)
    dataset = parser.add_argument_group("data")
    dataset.add_argument("--laion", type=Path, default=data.LAION_ROOT)
    dataset.add_argument("--imagenet", type=Path, default=data.IMAGENET_ROOT)
    dataset.add_argument(
        "--models", nargs="+", default=list(DEFAULT_MODELS), help="the CLIP models the student is told"
    )
    dataset.add_argument("--workers", type=int, default=4)
    dataset.add_argument("--p-swap", type=float, default=0.5, help="how often an image gets another image's text")
    dataset.add_argument("--overfit", action="store_true", help="train on the sample texts only, to see it can learn")
    teacher = parser.add_argument_group("the teacher")
    teacher.add_argument("--teacher-clip", default="+".join(DEFAULT_MODELS), help="the CLIP models that guide it")
    teacher.add_argument("--cutouts", type=int, default=16, help="cutouts each CLIP model sees")
    teacher.add_argument("--rate", type=float, default=sampler.DEFAULT_GUIDANCE_RATE, help="cgd's guidance rate")
    teacher.add_argument("--direct", action="store_true", help="train on the guidance loss itself, not its move")
    teacher.add_argument("--stay", type=float, default=1.0, help="with --direct, the weight on exceeding the cap")
    teacher.add_argument("--tv", type=float, default=0.0, help="with --direct, how much of the smoothness term")
    adapters = parser.add_argument_group("the student")
    adapters.add_argument("--rank", type=int, default=16, help="of the LoRA updates")
    adapters.add_argument("--tokens", type=int, default=4, help="condition tokens the attention layers see")
    adapters.add_argument("--conv-rank", type=int, default=8, help="of the update on each block's last convolution")
    adapters.add_argument("--layout", type=int, default=8, help="side of the layout's feature map. 0 for none")
    adapters.add_argument("--plain", action="store_true", help="the first runs' adapters: no statistics, layout, ...")
    optimisation = parser.add_argument_group("optimisation")
    optimisation.add_argument("--accumulate", type=int, default=1, help="examples per optimizer step")
    optimisation.add_argument("--lr", type=float, default=1e-4)
    optimisation.add_argument(
        "--widest", type=int, default=64, help="layers with more inputs than this get a lower learning rate"
    )
    optimisation.add_argument("--warmup", type=int, default=50, help="examples over which the learning rate rises")
    optimisation.add_argument("--clip-norm", type=float, default=1.0)
    optimisation.add_argument("--ema", type=float, default=0.995, help="decay of the weights' running average")
    sampling = parser.add_argument_group("the samples")
    sampling.add_argument("--samples", type=int, default=8, help="texts to draw, half from each dataset")
    sampling.add_argument("--sample-steps", type=int, default=30)
    sampling.add_argument(
        "--sample-scale", type=float, default=3.0, help="also draw with the student's move times this"
    )
    sampling.add_argument("--judge", default="ViT-L-16-SigLIP2-256", help="a CLIP model nothing here was guided by")
    return parser.parse_args(argv)


def build(args: argparse.Namespace) -> Run:
    """The student on its own copy of the pretrained U-Net, new or from the run's checkpoint."""
    unet, _ = load_model(IMAGE_SIZE, args.device, False, False, True)
    path = args.out / CHECKPOINT_NAME
    checkpoint = th.load(path, map_location="cpu", weights_only=True) if args.resume and path.is_file() else None
    if checkpoint is not None:
        model = load_adapter(unet, checkpoint)
        print(f"resuming with the checkpoint's adapters, whatever the options say: {model.config}", flush=True)
    else:
        statistics = text_statistics([args.laion, args.imagenet], args.models)
        config = AdapterConfig(
            {name: len(mean) for name, (mean, _) in statistics.items()},
            rank=args.rank,
            tokens=args.tokens,
            standardised=not args.plain,
            film=not args.plain,
            layout=0 if args.plain else args.layout,
            conv_rank=0 if args.plain else args.conv_rank,
        )
        model = ConditionedUNet(unet, config)
        if config.standardised:
            model.conditioner.set_statistics(statistics)
    model.to(args.device).to(memory_format=th.channels_last)
    return Run(args, model, checkpoint)


def examples(args: argparse.Namespace, samples: Samples, start: int) -> Callable[[th.Generator], tuple]:
    """A function giving the next training image and the embeddings of the text it is paired with."""
    sources = [
        data.open_source("laion", args.laion, args.models),
        data.open_source("imagenet", args.imagenet, args.models),
    ]
    stream = data.batches(sources, batch_size=2, seed=args.seed, start=start - start % 2, workers=args.workers)
    kinds, weights = list(KIND_WEIGHTS), th.tensor(list(KIND_WEIGHTS.values()))
    pending: list[tuple[th.Tensor, dict]] = []

    def following(generator: th.Generator) -> tuple[th.Tensor, dict[str, th.Tensor]]:
        if not pending:
            batch = next(stream)
            pending.extend((batch["image"][i], {"batch": batch, "row": i}) for i in (1, 0))
        image, origin = pending.pop()
        if args.overfit:
            text = int(th.randint(len(samples.texts), (), generator=generator))
            return image, {name: embedding[text : text + 1] for name, embedding in samples.targets.items()}
        owner = origin["row"] if th.rand((), generator=generator) >= args.p_swap else 1 - origin["row"]
        kind = kinds[int(th.multinomial(weights, 1, generator=generator))]
        embeds = origin["batch"]["embeds"]
        return image, {name: embeds[name][kind][owner : owner + 1] for name in args.models}

    return following


def seed_cutouts(run: Run) -> None:
    """Seed the global stream, which the cutouts draw from, by the step, so that a resumed run draws the same ones."""
    th.manual_seed(run.args.seed * 1_000_003 + run.steps)


def prepared(
    run: Run, log_alpha_bar: np.ndarray, example: tuple, generator: th.Generator
) -> tuple[th.Tensor, float, dict[str, th.Tensor]]:
    """An example for a training step: its image noised to a drawn timestep, the timestep, and its text's embeddings."""
    t = guided_timestep(log_alpha_bar, generator)
    x = noisy(example[0], log_alpha_bar, t, generator, run.args.device)
    return x, t, {name: embedding.to(run.args.device) for name, embedding in example[1].items()}


def distil_example(
    run: Run, teacher: Teacher, log_alpha_bar: np.ndarray, example: tuple, generator: th.Generator
) -> dict[str, float] | None:
    """One training step on one noisy image and one text: the share missed and the cosine, or None if skipped."""
    x, t, targets = prepared(run, log_alpha_bar, example, generator)
    base, shift = guided_shift(teacher, log_alpha_bar, x, t, targets)
    if not shift.any():  # A gradient that overflowed everywhere leaves nothing to learn from.
        return None
    predicted = student_noise(run, x, th.full((1,), t, device=x.device, dtype=th.float32), targets)
    missed, cosine = how_much_missed(predicted, base, shift)
    if not th.isfinite(missed).all():
        raise SystemExit(f"The loss is {missed.item()} at step {run.steps}. The last checkpoint is intact.")
    run.step(missed.mean())
    return {"missed": missed.item(), "cosine": cosine.item()}


def student_move(
    run: Run, teacher: Teacher, x: th.Tensor, t: float, targets: Mapping[str, th.Tensor]
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """The base model's noise for x, the student's move from it scaled down to the cap, and the move's own size."""
    timesteps = th.full((len(x),), t, device=x.device, dtype=th.float32)
    with th.no_grad():
        base = denoise(teacher.unet, None, x, timesteps).float()
    move, size = capped_move(student_noise(run, x, timesteps, targets), base, teacher.rate)
    return base, move, size


def reward_example(
    run: Run, teacher: Teacher, log_alpha_bar: np.ndarray, example: tuple, generator: th.Generator
) -> dict[str, float]:
    """One training step of `--direct`: lower the guidance loss of the student's own denoised image.

    The move is scaled down to the cap before the image is denoised, and what exceeds the cap is penalised, so
    that the student alone, which nothing caps when it samples, stays within it.
    """
    x, t, targets = prepared(run, log_alpha_bar, example, generator)
    now = sampler.level(log_alpha_bar, t)
    base, move, size = student_move(run, teacher, x, t, targets)
    guide = guide_for(teacher, targets)
    seed_cutouts(run)
    loss = guided_loss(guide, x, base + move, now, run.args.tv)
    if not th.isfinite(loss):
        raise SystemExit(f"The loss is {loss.item()} at step {run.steps}. The last checkpoint is intact.")
    result = {"guided_loss": loss.item(), "size": size.item()}
    if run.steps % GAIN_EVERY == 0:
        seed_cutouts(run)  # The same cutouts again.
        result["gain"] = guided_loss(guide, x, base, now, run.args.tv).item() - result["guided_loss"]
    run.step(loss + run.args.stay * (size - 1).clamp(min=0).square().sum())
    return result


def distil(args: argparse.Namespace) -> Run:  # pylint: disable=too-many-locals  # It is the whole loop.
    """Train until the time or the steps run out. Returns the run as it ended."""
    args.out.mkdir(parents=True, exist_ok=True)
    th.manual_seed(args.seed)
    teacher, log_alpha_bar = load_teacher(args)
    run = build(args)
    trim_log(args.out / "log.jsonl", run.steps)
    track(run)
    judge = load_encoder(args.judge, "cpu")
    samples = load_samples(args, teacher, log_alpha_bar, judge)
    count = sum(weight.numel() for weight in run.weights.values())
    print(f"training {count / 1e6:.1f}M added weights from step {run.steps}, {run.seconds / 3600:.2f} h in", flush=True)
    if not run.steps:
        agreement = teacher_agreement(args, teacher, log_alpha_bar, samples)
        evaluate(run, teacher, log_alpha_bar, samples, judge, **{"teacher/agreement": agreement})
    following = examples(args, samples, run.steps)
    started = time.monotonic() - run.seconds
    saved = judged_at = logged = time.monotonic()
    window: list[dict[str, float]] = []
    train_on = reward_example if args.direct else distil_example
    while run.seconds < args.hours * 3600 and (args.max_steps is None or run.steps < args.max_steps):
        generator = step_generator(args.seed, run.steps)
        seed_cutouts(run)
        result = train_on(run, teacher, log_alpha_bar, following(generator), generator)
        if result is None:
            run.steps += 1  # So a resumed run skips it too.
        else:
            window.append(result)
        run.seconds = time.monotonic() - started
        if run.steps % args.log_every or not window:
            continue
        now = time.monotonic()
        record = averages(window) | {"examples_per_second": len(window) / (now - logged)}
        if th.device(args.device).type == "cuda":
            record["gpu_gb"] = th.cuda.max_memory_allocated() / 2**30
        log(args, step=run.steps, hours=run.seconds / 3600, **record)
        window, logged = [], time.monotonic()
        if run.steps % args.accumulate == 0 and now - saved > args.save_minutes * 60:
            run.save()
            saved = time.monotonic()
        if now - judged_at > args.eval_minutes * 60:
            evaluate(run, teacher, log_alpha_bar, samples, judge)
            judged_at = logged = time.monotonic()
    run.save()
    if run.steps:
        evaluate(run, teacher, log_alpha_bar, samples, judge)
    print(f"stopped at step {run.steps} after {run.seconds / 3600:.2f} h. Saved {args.out / CHECKPOINT_NAME}")
    wandb.finish()  # Does nothing when it was never started.
    return run


if __name__ == "__main__":
    distil(parse_args())

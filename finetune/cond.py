r"""Class-anchored CLIP conditioning: teach the class-conditional ImageNet 256 model to take CLIP embeddings too.

    uv run --no-sync python -m finetune.cond --hours 12 --wandb
    uv run --no-sync python -m finetune.cond --hours 12 --resume --wandb   # Continue the run in --out
    uv run --no-sync python -m finetune.cond --hours 0.4 --align 0.01      # With the patch feature loss
    uv run --no-sync python -m finetune.cond --max-steps 0                 # Only the images to compare with
    uv run --no-sync python -m finetune.parallel --devices 0,1,2,3 finetune.cond --hours 12 --wandb  # Four GPUs

`cgd`'s default pipeline draws the ImageNet class closest to the prompt and steers it with CLIP's gradient.
That gradient approximates, one step at a time, what a model conditioned on a CLIP embedding predicts by
itself. This trains that model directly, with the denoising loss on real images, and keeps what `cgd` already
does well:

- **It starts as `cgd`'s model.** The class stays, chosen as `cgd` chooses it: the class closest to the text by
  the first CLIP model (`anchor_classes`). What the embeddings add starts at zero, so at step 0 this is the
  pretrained model told that class.
- **The encoder half is frozen and the rest is trained whole.** The middle and decoder's own weights, 390M of
  554M, at `--pretrained-rate` of the learning rate. The frozen half is told the timestep and the class as
  before, so the pass for the class alone and the pass with the embeddings share it.
- **Two conditions per CLIP model.** The image's own embedding, with noise on it, and the embedding of one of
  its texts. Per image both, either or neither is kept (`--p-image-only`, `--p-text-only`, `--p-uncond`), so
  the model can be sampled from a text alone, from an image embedding, and with classifier-free guidance
  against the class alone.
- **The loss** is the noise prediction's squared error with every noise level at its whole weight, as the
  model was pretrained, and the noisiest levels counted for more (`--snr-floor`), at levels that lean a little
  to high noise (`--noise-bias`). `finetune.train`'s cap on the cleaner levels' weight (`--min-snr`) is off:
  with it, a stronger lean and the patch feature loss, the pilot's model had lost its detail in 2,100 images.
- **`--align`** adds a second loss, and is off by default. A small head on the middle block predicts the clean
  image's PE-Core patch features, pooled to the middle block's 8 by 8. At high noise the embeddings are all
  that says what those are, so the middle block has to use them. It is REPA's idea (Yu et al. 2024) at the
  place U-REPA puts it (Tian et al. 2025), and unproven for fine-tuning. A weight of 0.01 with each image
  counted by its share of noise keeps its gradient on the middle block about the size of the denoising
  loss's. At REPA's 0.5 it is hundreds of times larger. Even at 0.01 the pilot's samples washed out to pale
  images within 1,654 images, so it stays off.

`gap/image`, `gap/text` and `gap/both` are how much lower the loss at high noise is on held-out images with
those conditions than with the class alone, as a fraction. `probe/class` is the loss with the class alone,
which is the pretrained model's at step 0 and should not rise, and `probe/class_clean` the same at the cleaner
levels, where a fine-tune loses the pretrained model's detail first. Every `--eval-minutes` the run draws its
sample texts with classifier-free guidance from each condition, beside the pretrained model told the class
(`class only`), the same under CLIP guidance (`class guided`) and this model told the class alone
(`class now`). A CLIP model that nothing here was trained or guided with scores every row against the texts
and the real images.

Under `finetune.parallel` one process runs per GPU. A step is then `--batch` images on each, with the gradient
averaged over all of them, and the first process logs, checks, draws the samples and saves while the rest wait.
A run resumes with the number of GPUs it was started with: its place in the data counts the images of all.
"""

import argparse
import time
from collections.abc import Mapping, Sequence
from functools import partial
from itertools import islice
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch as th
import wandb
from PIL import Image
from torch import nn
from torch.nn import functional as tf
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf

from cgd import clip_models, sampler
from cgd.adm import guidance_loss, load_model
from cgd.conditioned import AdapterConfig, ConditionedUNet, load_adapter
from cgd.device import default_device
from cgd.finetuned import Condition, cfg_denoiser, noise_given, slot, slots
from cgd.guide import MIN_CUT_FRACTION, Guide, Tower
from finetune import data, distill, noise, parallel
from finetune.embed import Encoder, encode_texts, load_encoder
from finetune.train import (
    CHECKPOINT_NAME,
    IMAGE_SIZE,
    PROBE_NOISIEST,
    WANDB_PROJECT,
    Run,
    add_data_options,
    as_input,
    finished,
    log,
    open_sources,
    step_generator,
    track,
    trim_log,
)

PATCH_MODEL = "PE-Core-B-16"  # Whose patch features `--align` predicts.
SAMPLE_KIND = distill.SAMPLE_KIND
SAMPLES_NAME = "samples.pt"
# The conditions the check and the samples compare: whether the image's and the text's embeddings are kept.
ROWS = {"image": (True, False), "text": (False, True), "both": (True, True)}
CLEAN_PROBES = 12  # More pairs of held-out images, at the cleaner noise levels, where a fine-tune forgets first.


def kept(models: Sequence[str], image: bool, text: bool) -> th.Tensor:
    """A row of `keep`, [1, conditions], with every model's image embedding kept or not, and its text's."""
    return th.tensor([[image, text] * len(models)])


def slot_statistics(
    roots: Sequence[Path], models: Sequence[str], rows: int = 20_000, seed: int = 0
) -> dict[str, tuple[th.Tensor, th.Tensor]]:
    """Each condition's mean and standard deviation over a sample of every dataset."""
    texts = distill.text_statistics(roots, models, rows, seed)
    rng = np.random.default_rng(seed)
    statistics = {}
    for name in models:
        statistics[slot(name, "image")] = distill.embedding_statistics(roots, name, ["image"], rows, rng)
        statistics[slot(name, "text")] = texts[name]
    return statistics


def anchor_classes(texts: th.Tensor, class_features: th.Tensor) -> th.Tensor:
    """The ImageNet class `cgd` draws for each text's embedding: the closest of the classes' own embeddings."""
    return (tf.normalize(texts.float(), dim=-1) @ class_features.float().T).argmax(dim=1)


def choose_condition(  # pylint: disable=too-many-locals  # One draw per decision, in a fixed order.
    embeds: Mapping[str, Mapping[str, th.Tensor]],
    class_features: th.Tensor,
    generator: th.Generator,
    args: argparse.Namespace,
) -> Condition:
    """Draw each image's condition from a batch's embeddings: `embeds[model][kind]` is [N, dim].

    One of the image's three texts is drawn, the same for every model. Its class is the anchor, and its
    embedding the text condition. The image condition is the image's own embedding with noise of a level up to
    `--embed-noise`. An image keeps only the text's with `--p-text-only`, only the image's with
    `--p-image-only` and both otherwise, loses one model's with `--p-drop-model`, and all with `--p-uncond`.
    """
    models = list(embeds)
    count = len(embeds[models[0]]["image"])
    weights = th.tensor([distill.KIND_WEIGHTS[kind] for kind in data.TEXT_KINDS])
    kind = th.multinomial(weights.expand(count, -1), 1, generator=generator)[:, 0]
    level = th.rand(count, 1, generator=generator) * args.embed_noise
    mode = th.rand(count, 1, generator=generator)
    told = th.rand(count, 1, generator=generator) >= args.p_uncond
    with_image, with_text = mode >= args.p_text_only, mode < 1 - args.p_image_only
    chosen, keep = {}, []
    for name in models:
        kinds = embeds[name]
        chosen[slot(name, "image")] = noise.noisy_embedding(kinds["image"], level, generator)
        chosen[slot(name, "text")] = th.stack([kinds[k] for k in data.TEXT_KINDS], dim=1)[th.arange(count), kind]
        has = told & (th.rand(count, 1, generator=generator) >= args.p_drop_model)
        keep += [with_image & has, with_text & has]
    labels = anchor_classes(chosen[slot(models[0], "text")], class_features)
    return Condition(chosen, th.cat(keep, dim=1), labels)


class PatchTower(NamedTuple):
    """The CLIP model whose vision tower gives the patch features, and its input size."""

    model: nn.Module
    size: int


def load_patch_tower(device: str) -> PatchTower:
    """PE-Core as `cgd` loads it, at half precision on a GPU."""
    return PatchTower(*clip_models.load_clip(PATCH_MODEL, device, th.device(device).type == "cuda"))


def patch_grid(tokens: th.Tensor, prefix: int, side: int) -> th.Tensor:
    """A vision tower's tokens, [N, prefix + patches, dim], as unit features pooled to [N, dim, side, side]."""
    patches = tokens[:, prefix:].float()
    across = round(patches.shape[1] ** 0.5)
    grid = patches.transpose(1, 2).reshape(len(patches), -1, across, across)
    return tf.normalize(tf.adaptive_avg_pool2d(grid, side), dim=1)


@th.no_grad()
def patch_features(tower: PatchTower, images: th.Tensor, side: int) -> th.Tensor:
    """The tower's unit patch features of uint8 images [N, 3, H, W], pooled to [N, dim, side, side]."""
    trunk = tower.model.visual.trunk
    weight = next(trunk.parameters())
    pixels = images.to(weight.device).float().div(255)
    size = [tower.size, tower.size]
    pixels = tvf.resize(pixels, size, interpolation=InterpolationMode.BICUBIC, antialias=True).clamp(0, 1)
    tokens = trunk.forward_features(clip_models.preprocess(tower.model, pixels).to(weight.dtype))
    return patch_grid(tokens, trunk.num_prefix_tokens, side)


def align_miss(aligned: th.Tensor, target: th.Tensor) -> th.Tensor:
    """How far each image's predicted patch features are from the tower's: one minus their mean cosine."""
    return 1 - tf.cosine_similarity(aligned, target, dim=1).mean(dim=(1, 2))


def counted_miss(miss: th.Tensor, log_snr: th.Tensor) -> th.Tensor:
    """A batch's patch feature loss, with each image counted by the share of noise in it.

    Measured on the middle block's weights, this loss's gradient is then 45 to 140 times the denoising loss's
    at every noise level. Counting every image alike it is 30 to 700,000 times, most at the cleanest levels.
    """
    return (th.sigmoid(-2 * log_snr).to(miss) * miss).mean()


def prediction(
    model: ConditionedUNet,
    log_alpha_bar: np.ndarray,
    images: th.Tensor,
    condition: Condition,
    *,
    log_snr: th.Tensor,
    gaussian: th.Tensor,
    amp: bool,
) -> tuple[th.Tensor, th.Tensor | None]:
    """Each image's squared noise error at its own level, and the patch features the middle block predicts, if any.

    The images are in [-1, 1], and `condition` is on their device.
    """
    alpha, sigma = (level.to(images.device, th.float32)[:, None, None, None] for level in noise.level_of(log_snr))
    timesteps = noise.timestep_of(log_alpha_bar, log_snr).to(images.device, th.float32)
    aligns = bool(parallel.unshared(model).config.align_dim)
    with th.autocast(images.device.type, enabled=amp):
        output = model(
            alpha * images + sigma * gaussian,
            timesteps,
            condition.embeds,
            condition.keep,
            labels=condition.labels,
            middle=aligns,
        )
    predicted, aligned = output if aligns else (output, None)
    error = (predicted[:, :3].float() - gaussian).square().mean(dim=(1, 2, 3))  # The rest is the learned variance.
    return error, None if aligned is None else aligned.float()


def class_denoiser(model: ConditionedUNet, amp: bool, condition: Condition) -> sampler.Denoiser:
    """The model for the sampler, told the class alone, as the pretrained one is."""
    return partial(noise_given, model, amp, condition=condition.alone())


class Probe(NamedTuple):
    """Held-out images with a fixed noise level and noise each, and every condition, for the check."""

    images: th.Tensor
    condition: Condition
    log_snr: th.Tensor
    gaussian: th.Tensor


def make_probes(args: argparse.Namespace, class_features: th.Tensor, log_alpha_bar: np.ndarray) -> list[Probe]:
    """`--probe-batches` pairs of held-out images, one from each dataset, at high noise, then some at the rest."""
    generator = th.Generator().manual_seed(args.seed)
    table = sampler.log_snr(log_alpha_bar)
    stream = data.batches(open_sources(args, "holdout"), batch_size=2, seed=args.seed, workers=0)
    probes = []
    for number, batch in enumerate(islice(stream, args.probe_batches + CLEAN_PROBES)):
        fraction = th.rand(2, generator=generator, dtype=th.float64) * PROBE_NOISIEST
        if number >= args.probe_batches:
            fraction = PROBE_NOISIEST + fraction * (1 - PROBE_NOISIEST) / PROBE_NOISIEST
        embeds = {}
        for name in args.models:
            embeds[slot(name, "image")] = batch["embeds"][name]["image"]
            embeds[slot(name, "text")] = batch["embeds"][name][SAMPLE_KIND]
        labels = anchor_classes(embeds[slot(args.models[0], "text")], class_features)
        condition = Condition(embeds, kept(args.models, True, True).expand(2, -1), labels)
        log_snr = table[-1] + (table[0] - table[-1]) * fraction
        probes.append(Probe(batch["image"], condition, log_snr, th.randn(batch["image"].shape, generator=generator)))
    return probes


@th.no_grad()
def check_probes(  # pylint: disable=too-many-locals  # Every row of the check is measured on the same probes.
    run: Run, log_alpha_bar: np.ndarray, probes: Sequence[Probe], tower: PatchTower | None
) -> dict[str, float]:
    """The loss at high noise told the class alone, and how much lower each condition makes it.

    `gap/...` is in terms of the noise, as `finetune.train`'s `gap`. `gap_image/...` is in terms of the image,
    which counts the noisiest probes as much as the rest. `aligned/...` is the mean cosine between the patch
    features the middle block predicts and the real ones. `probe/class_clean` is the loss with the class
    alone at the cleaner levels, where the samples' detail is decided.
    """
    args = run.args
    rows = {"class": (False, False)} | ROWS
    errors: dict[str, list[th.Tensor]] = {row: [] for row in rows}
    cosines: dict[str, list[th.Tensor]] = {row: [] for row in rows}
    probes, clean = probes[: args.probe_batches], probes[args.probe_batches :]
    for probe in probes:
        images, target = as_input(probe.images, args.device), None
        at = {"log_snr": probe.log_snr, "gaussian": probe.gaussian.to(args.device), "amp": run.amp}
        for row, (image, text) in rows.items():
            keep = kept(args.models, image, text).expand(len(images), -1)
            condition = probe.condition._replace(keep=keep).to(args.device)
            error, aligned = prediction(run.model, log_alpha_bar, images, condition, **at)
            errors[row].append(error.cpu())
            if aligned is not None:
                target = patch_features(tower, probe.images, aligned.shape[-1]) if target is None else target
                cosines[row].append(1 - align_miss(aligned, target).cpu())
    per_image = th.exp(-2 * th.cat([probe.log_snr for probe in probes])).float()
    base = th.cat(errors["class"])
    record = {"probe/class": base.mean().item(), "probe/class_image": (base * per_image).mean().item()}
    if clean:
        record["probe/class_clean"] = clean_error(run, log_alpha_bar, clean)
    for row in ROWS:
        told = th.cat(errors[row])
        record[f"gap/{row}"] = 1 - (told.mean() / base.mean()).item()
        record[f"gap_image/{row}"] = 1 - ((told * per_image).mean() / (base * per_image).mean()).item()
    return record | {f"aligned/{row}": th.cat(values).mean().item() for row, values in cosines.items() if values}


def clean_error(run: Run, log_alpha_bar: np.ndarray, probes: Sequence[Probe]) -> float:
    """The mean noise error on probes, told the class alone."""
    device, errors = run.args.device, []
    for probe in probes:
        at = {"log_snr": probe.log_snr, "gaussian": probe.gaussian.to(device), "amp": run.amp}
        condition = probe.condition.alone().to(device)
        errors.append(prediction(run.model, log_alpha_bar, as_input(probe.images, device), condition, **at)[0])
    return th.cat(errors).mean().item()


class Samples(NamedTuple):
    """The texts a run draws to show its progress, and the images to compare its own with."""

    texts: list[str]
    embeds: dict[str, th.Tensor]  # Each condition's [N, dim]: the real image's embedding, or the text's.
    labels: th.Tensor  # The class `cgd` draws for each text.
    judged_text: th.Tensor  # The judge's embedding of the texts.
    rows: dict[str, th.Tensor]  # uint8 [N, 3, H, W] images by row: the real ones, and what each model drew.


def sample_condition(
    samples: Samples, i: int, args: argparse.Namespace, image: bool = False, text: bool = False
) -> Condition:
    """One sample text's condition on the device, with the image's embeddings kept or not, and the text's."""
    embeds = {name: value[i : i + 1] for name, value in samples.embeds.items()}
    return Condition(embeds, kept(args.models, image, text), samples.labels[i : i + 1]).to(args.device)


def load_guide_towers(args: argparse.Namespace) -> dict[str, tuple[nn.Module, int]]:
    """The CLIP models of `--guide-clip` as `cgd` loads them, each with its input size."""
    half = th.device(args.device).type == "cuda"
    # Kept here, because the loader only remembers the last model it loaded.
    return {
        name: clip_models.load_clip(name, args.device, half) for name in clip_models.ensemble_members(args.guide_clip)
    }


def clip_guidance(
    args: argparse.Namespace, towers: Mapping[str, tuple[nn.Module, int]], embeds: Mapping[str, th.Tensor], i: int
) -> sampler.Guidance:
    """`cgd`'s CLIP guidance towards text i of every condition's [N, dim], as it guides the class-conditional model."""
    weight = th.ones(1, device=args.device)
    guided = tuple(
        Tower(model, size, embeds[slot(name, "text")][i : i + 1].to(args.device), weight)
        for name, (model, size) in towers.items()
    )
    guide = Guide(guided, args.cutouts, MIN_CUT_FRACTION)
    return sampler.Guidance(partial(guidance_loss, guide), args.rate, sampler.GUIDANCE_WINDOW)


def make_samples(run: Run, log_alpha_bar: np.ndarray, judge: Encoder, class_features: th.Tensor) -> Samples:
    """Held-out images and their texts, with what the pretrained model draws for each: alone, and under CLIP."""
    args = run.args
    items, texts = distill.held_out(args)
    embeds = {}
    for name in args.models:
        embeds[slot(name, "image")] = th.stack([item["embeds"][name]["image"] for item in items])
        embeds[slot(name, "text")] = th.stack([item["embeds"][name][SAMPLE_KIND] for item in items])
    labels = anchor_classes(embeds[slot(args.models[0], "text")], class_features)
    with th.no_grad():
        judged_text = encode_texts(judge, texts)
    samples = Samples(texts, embeds, labels, judged_text, {"real": th.stack([item["image"] for item in items])})

    def alone(i: int) -> sampler.Denoiser:
        return class_denoiser(run.model, run.amp, sample_condition(samples, i, args))

    samples.rows["class only"] = distill.draw(args, log_alpha_bar, alone)
    guidance = partial(clip_guidance, args, load_guide_towers(args), embeds)
    samples.rows["class guided"] = distill.draw(args, log_alpha_bar, alone, guidance)
    return samples


def load_samples(run: Run, log_alpha_bar: np.ndarray, judge: Encoder, class_features: th.Tensor) -> Samples:
    """The run's samples, made on its first start and kept with it, so a resumed run compares with the same ones."""
    path = run.args.out / SAMPLES_NAME
    if run.args.resume and path.is_file():
        return Samples(**th.load(path, map_location="cpu", weights_only=True))
    if run.steps:
        raise SystemExit(f"{path} is missing, and only the pretrained model can draw the images to compare with.")
    samples = make_samples(run, log_alpha_bar, judge, class_features)
    th.save(samples._asdict(), path)
    return samples


def evaluate(run: Run, log_alpha_bar: np.ndarray, samples: Samples, judge: Encoder) -> None:
    """Draw the samples from each condition, judge every row, and log the scores and the grid of images."""
    args = run.args
    rows = dict(samples.rows)
    if run.steps:  # Before its first step the model is the pretrained one told the class.
        with distill.averaged(run) if run.average else th.no_grad():
            # The class alone again, by the model as it is now: what the fine-tune has cost the pretrained model.
            rows["class now"] = distill.draw(
                args, log_alpha_bar, lambda i: class_denoiser(run.model, run.amp, sample_condition(samples, i, args))
            )
            for name, (image, text) in ROWS.items():

                def told(i: int, image: bool = image, text: bool = text) -> sampler.Denoiser:
                    condition = sample_condition(samples, i, args, image, text)
                    return cfg_denoiser(run.model, run.amp, condition, args.sample_scale)

                rows[f"{name} x{args.sample_scale:g}"] = distill.draw(args, log_alpha_bar, told)
    log(args, step=run.steps, hours=run.seconds / 3600, **distill.judged(judge, samples, rows))
    grid = th.cat([th.cat(list(images), dim=2) for images in rows.values()], dim=1).permute(1, 2, 0).numpy()
    Image.fromarray(grid).save(args.out / f"samples_{run.steps:07d}.jpg", quality=92)
    if wandb.run is not None:
        classes = [clip_models.IMAGENET_CLASSES[label] for label in samples.labels.tolist()]
        columns = " | ".join(f"{text} [{name}]" for text, name in zip(samples.texts, classes, strict=True))
        caption = "Rows: " + ", ".join(rows) + ". Columns: " + columns
        wandb.log({"samples": wandb.Image(grid, caption=caption)}, step=run.steps)


def add_run_options(parser: argparse.ArgumentParser) -> None:
    """The options about the run, its data and its samples."""
    run = parser.add_argument_group("the run")
    run.add_argument("--out", type=Path, default=Path("D:/cgd-finetune/cond256"))
    run.add_argument("--resume", action="store_true", help="continue from --out's checkpoint, with its samples")
    run.add_argument("--hours", type=float, default=12.0, help="stop after this much training, resumed runs included")
    run.add_argument("--max-steps", type=int, default=None, help="stop after this many batches, for a trial run")
    run.add_argument("--device", default=default_device())
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--no-amp", action="store_true", help="full precision. Half precision is the default on a GPU")
    run.add_argument("--memory-fraction", type=float, default=None, help="fail above this share of the GPU's memory")
    run.add_argument("--log-every", type=int, default=100)
    run.add_argument("--eval-every", type=int, default=1000, help="batches between checks on the held-out probes")
    run.add_argument("--eval-minutes", type=float, default=30.0, help="minutes of training between samples")
    run.add_argument("--save-minutes", type=float, default=30.0)
    run.add_argument("--probe-batches", type=int, default=24, help="held-out pairs of images the check uses")
    run.add_argument("--wandb", nargs="?", const=WANDB_PROJECT, default=None, metavar="PROJECT")
    run.add_argument("--wandb-name", default=None, help="the run's name there. One is made up by default")
    add_data_options(parser, workers=4)
    samples = parser.add_argument_group("the samples")
    samples.add_argument("--samples", type=int, default=8, help="held-out texts drawn, half from each dataset")
    samples.add_argument("--sample-steps", type=int, default=30)
    samples.add_argument("--sample-scale", type=float, default=3.0, help="of the classifier-free guidance")
    samples.add_argument("--judge", default="ViT-L-16-SigLIP2-256", help="the CLIP model that scores the samples")
    samples.add_argument("--guide-clip", default="ViT-B/32+PE-Core-B-16", help="what guides the `class guided` row")
    samples.add_argument("--cutouts", type=int, default=16)
    samples.add_argument("--rate", type=float, default=0.08, help="the guidance rate of that row")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """The command line's options, in the groups `-h` shows."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    add_run_options(parser)
    added = parser.add_argument_group("what is added and trained")
    added.add_argument("--tokens", type=int, default=4, help="condition tokens the attention layers see. 0 for none")
    added.add_argument("--align", type=float, default=0.0, help="weight of the patch feature loss, 0.01 if any")
    added.add_argument("--align-warmup", type=int, default=500, help="batches over which that weight rises")
    optimisation = parser.add_argument_group("optimisation")
    optimisation.add_argument(
        "--batch", type=int, default=1, help="images a step on each GPU. 1 fits 12 GB with the optimizer's state"
    )
    optimisation.add_argument("--accumulate", type=int, default=1, help="batches per optimizer step")
    optimisation.add_argument("--lr", type=float, default=3e-4, help="of what is added")
    optimisation.add_argument("--pretrained-rate", type=float, default=1 / 30, help="the U-Net's own share of --lr")
    optimisation.add_argument("--widest", type=int, default=256, help="added layers with more inputs get less of --lr")
    optimisation.add_argument("--warmup", type=int, default=500, help="batches over which the learning rate rises")
    optimisation.add_argument(
        "--decay", type=int, default=0, help="batches before --max-steps over which it falls to zero. 0: none"
    )
    optimisation.add_argument("--clip-norm", type=float, default=1.0, help="the gradient's largest norm")
    optimisation.add_argument(
        "--ema", type=float, default=0.9999, help="decay of the weights' running average. 0: none"
    )
    optimisation.add_argument("--min-snr", type=float, default=1e9, help="cap on the cleaner levels' weight. None")
    optimisation.add_argument("--snr-floor", type=float, default=0.1, help="count no level's signal to noise as less")
    optimisation.add_argument("--noise-bias", type=float, default=1.5, help="1 is uniform in logSNR. More is noisier")
    condition = parser.add_argument_group("the condition")
    condition.add_argument("--embed-noise", type=float, default=1.0, help="the most noise put on an image embedding")
    condition.add_argument("--p-image-only", type=float, default=0.3, help="how often only the image's is kept")
    condition.add_argument("--p-text-only", type=float, default=0.3, help="how often only the text's is kept")
    condition.add_argument("--p-drop-model", type=float, default=0.1, help="how often one model's are dropped")
    condition.add_argument("--p-uncond", type=float, default=0.1, help="how often every embedding is dropped")
    args = parser.parse_args(argv)
    if args.decay and args.max_steps is None:
        parser.error("--decay counts back from --max-steps, so it needs one")
    return args


def build(args: argparse.Namespace, team: parallel.Team) -> tuple[Run, np.ndarray]:
    """The model on the pretrained class-conditional U-Net, new or from the run's checkpoint, and its schedule.

    On several GPUs every process builds the same run: the first one's weights are copied to the rest.
    """
    unet, log_alpha_bar = load_model(IMAGE_SIZE, args.device, False, False, False)
    path = args.out / CHECKPOINT_NAME
    checkpoint = th.load(path, map_location="cpu", weights_only=True) if args.resume and path.is_file() else None
    if checkpoint is not None:
        if checkpoint.get("processes", 1) != team.size:
            raise SystemExit(f"The run in {args.out} was on {checkpoint.get('processes', 1)} GPUs, not {team.size}.")
        model = load_adapter(unet, checkpoint)
        print(f"resuming with the checkpoint's model, whatever the options say: {model.config}", flush=True)
    else:
        statistics = slot_statistics([args.laion, args.imagenet], args.models)
        width = load_patch_tower(args.device).model.visual.trunk.num_features if args.align else 0
        dims = {name: len(statistics[name][0]) for name in slots(args.models)}
        config = AdapterConfig(dims, tokens=args.tokens, standardised=True, full=True, align_dim=width)
        model = ConditionedUNet(unet, config)
        model.conditioner.set_statistics(statistics)
    model.to(args.device).to(memory_format=th.channels_last)
    net = parallel.shared(team, model)  # Before the run, whose running average starts as a copy.
    run = Run(args, model, checkpoint)
    run.net, run.team = net, team
    return run, log_alpha_bar


def train_step(
    run: Run, log_alpha_bar: np.ndarray, batch: Mapping, class_features: th.Tensor, tower: PatchTower | None
) -> dict[str, float]:
    """One batch: draw its conditions, noise levels and noise from the step's own numbers, and step on the loss.

    On several GPUs each process draws its own, and the backward pass that completes a step leaves every one the
    mean gradient.
    """
    args = run.args
    generator = step_generator(args.seed, run.steps * run.team.size + run.team.rank)
    condition = choose_condition(batch["embeds"], class_features, generator, args).to(args.device)
    log_snr = noise.sample_log_snr(log_alpha_bar, len(batch["image"]), args.noise_bias, generator)
    at = {"log_snr": log_snr, "amp": run.amp}
    at["gaussian"] = th.randn(batch["image"].shape, generator=generator).to(args.device)
    # With `--accumulate`, the processes average their gradients only on the batch that completes a step.
    with parallel.quiet(run.net, (run.steps + 1) % args.accumulate == 0):
        error, aligned = prediction(run.net, log_alpha_bar, as_input(batch["image"], args.device), condition, **at)
        loss = (error * noise.min_snr_weight(log_snr, args.min_snr, args.snr_floor).to(error)).mean()
        result = {"loss": loss.item()}
        if aligned is not None:
            miss = align_miss(aligned, patch_features(tower, batch["image"], aligned.shape[-1]))
            result["align"] = miss.mean().item()
            loss = loss + args.align * min(1.0, run.steps / max(args.align_warmup, 1)) * counted_miss(miss, log_snr)
        if not th.isfinite(loss):
            raise SystemExit(f"The loss is {loss.item()} at step {run.steps}. The last checkpoint is intact.")
        run.step(loss)
    return result


class Checks(NamedTuple):
    """What the process that logs measures a run with, and when it last drew samples and saved."""

    judge: Encoder
    samples: Samples
    probes: list[Probe]
    tower: PatchTower | None
    last: dict[str, float]


def start_checks(run: Run, log_alpha_bar: np.ndarray, class_features: th.Tensor, tower: PatchTower | None) -> Checks:
    """Open the run's log, and measure the run as it starts: the samples of a new one, the probes of any."""
    args = run.args
    trim_log(args.out / "log.jsonl", run.steps)
    track(run)
    judge = load_encoder(args.judge, "cpu")
    samples = load_samples(run, log_alpha_bar, judge, class_features)
    probes = make_probes(args, class_features, log_alpha_bar)
    count = sum(weight.numel() for weight in run.weights.values())
    print(f"training {count / 1e6:.1f}M weights from step {run.steps}, {run.seconds / 3600:.2f} h in", flush=True)
    if not run.steps:
        evaluate(run, log_alpha_bar, samples, judge)
    log(args, step=run.steps, hours=run.seconds / 3600, **check_probes(run, log_alpha_bar, probes, tower))
    now = time.monotonic()
    return Checks(judge, samples, probes, tower, {"judged": now, "saved": now})


def report(run: Run, log_alpha_bar: np.ndarray, checks: Checks, record: dict[str, float]) -> None:
    """Log a stretch of training, and check the probes, draw the samples and save when each is due."""
    args, now = run.args, time.monotonic()
    if th.device(args.device).type == "cuda":
        record["gpu_gb"] = th.cuda.max_memory_allocated() / 2**30
    log(args, step=run.steps, hours=run.seconds / 3600, **record)
    if run.steps % args.eval_every == 0:
        probed = check_probes(run, log_alpha_bar, checks.probes, checks.tower)
        log(args, step=run.steps, hours=run.seconds / 3600, **probed)
    if now - checks.last["judged"] > args.eval_minutes * 60:
        evaluate(run, log_alpha_bar, checks.samples, checks.judge)
        checks.last["judged"] = time.monotonic()
    # Only here, with the averages just logged, and with no gradient waiting, so a resume loses nothing.
    if run.steps % args.accumulate == 0 and now - checks.last["saved"] > args.save_minutes * 60:
        run.save()
        checks.last["saved"] = time.monotonic()


def conclude(run: Run, log_alpha_bar: np.ndarray, checks: Checks) -> None:
    """Save the run as it ended, with its last probes and samples."""
    args = run.args
    run.save()
    if run.steps % args.eval_every or run.steps % args.log_every:  # Or the loop has just checked them.
        probed = check_probes(run, log_alpha_bar, checks.probes, checks.tower)
        log(args, step=run.steps, hours=run.seconds / 3600, **probed)
    evaluate(run, log_alpha_bar, checks.samples, checks.judge)


def train(args: argparse.Namespace) -> Run:  # pylint: disable=too-many-locals  # It is the whole loop.
    """Train until the time or the steps run out. Returns the run as it ended.

    Under `finetune.parallel` this is one of several processes with the same weights. Each draws its own share
    of every step's images, and the first one logs, checks, saves and says when the time is up.
    """
    args.out.mkdir(parents=True, exist_ok=True)
    th.manual_seed(args.seed)
    if args.memory_fraction and th.device(args.device).type == "cuda":
        th.cuda.set_per_process_memory_fraction(args.memory_fraction)  # Fail instead of paging to main memory.
    team = parallel.join(str(args.device))
    # The first process downloads and caches what the rest then find on disk. Nothing in here may talk to the
    # other processes, which are waiting for it: sharing the model does, so the run is built after.
    with parallel.lead_first(team):
        load_model(IMAGE_SIZE, args.device, False, False, False)
        class_features = clip_models.imagenet_text_features(args.models[0], "cpu")
    run, log_alpha_bar = build(args, team)
    tower = load_patch_tower(args.device) if run.model.config.align_dim else None
    checks = start_checks(run, log_alpha_bar, class_features, tower) if team.lead else None
    sources, weights = open_sources(args, "train"), [args.laion_weight, args.imagenet_weight]
    stream = data.batches(
        sources,
        weights,
        batch_size=args.batch,
        seed=args.seed,
        start=run.steps * args.batch * team.size,
        workers=args.workers,
        share=(team.rank, team.size),
    )
    started = time.monotonic() - run.seconds
    logged = time.monotonic()
    window: list[dict[str, float]] = []
    # Whether the time is up is the first process's call, made where every process asks: here and at each log.
    going = parallel.agreed(team, run.seconds < args.hours * 3600)
    while going and (args.max_steps is None or run.steps < args.max_steps):
        window.append(train_step(run, log_alpha_bar, next(stream), class_features, tower))
        run.seconds = time.monotonic() - started
        if run.steps % args.log_every:
            continue
        paused = time.monotonic()
        record = parallel.averaged(team, distill.averages(window))
        record["images_per_second"] = len(window) * args.batch * team.size / (paused - logged)
        window = []
        if checks is not None:
            report(run, log_alpha_bar, checks, record)
        logged = time.monotonic()
        started += logged - paused  # The checks and the samples are not training time.
        going = parallel.agreed(team, run.seconds < args.hours * 3600)
    if checks is not None and run.steps:
        conclude(run, log_alpha_bar, checks)
    parallel.agreed(team, True)  # The others wait here until the first has saved, so none leaves it alone.
    parallel.leave(team)
    return finished(run) if team.lead else run


if __name__ == "__main__":
    train(parse_args())

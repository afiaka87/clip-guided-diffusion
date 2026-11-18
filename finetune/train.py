r"""Step 3: teach the unconditional ImageNet 256 model to take CLIP embeddings as its condition.

    uv run python -m finetune.train --hours 12
    uv run python -m finetune.train --hours 12 --resume           # Continue the run in --out
    uv run python -m finetune.train --laion-where "aesthetic >= 6.5" --imagenet-weight 0.5
    uv run python -m finetune.train --hours 12 --wandb            # Also log to Weights & Biases

`finetune.index` and `finetune.embed` must have run on both datasets. The U-Net stays frozen and
`cgd.conditioned` adds what is trained. Each step:

- **The condition.** Per image, either the image's own embedding with noise on it, or the embedding of one of
  its three texts (`--p-text`), so the model learns to work from the text embeddings it is sampled with. Each
  CLIP model's embedding is dropped on its own (`--p-drop-model`), and all of them together (`--p-uncond`),
  which is what classifier-free guidance and conditioning on one model alone need.
- **The noise level.** Uniform in logSNR over the schedule as the sampler steps, leaning to high noise
  (`--noise-bias`), at fractional timesteps.
- **The loss.** The noise prediction's squared error with Min-SNR weights (`--min-snr`), and with the
  noisiest levels counted for more (`--snr-floor`), since that is where the condition matters most. The learned
  variance is left alone: `cgd`'s sampler does not use it.

It stops after `--hours` of training, and saves the added weights, their running average and the optimizer to
`adapter.pt` in `--out` every `--save-minutes`. `log.jsonl` beside it has the loss and images per second,
and `--wandb` sends the same records to Weights & Biases, where a resumed run continues the same charts.

**Resuming.** `--resume` continues as the run would have without stopping: the same images in the same order,
and the same noise, noise levels and conditions at each step, because each step draws its random numbers from
the seed and its own number. A checkpoint is only written on a step that was logged, so nothing is left half
counted. A run that was killed goes back to its last checkpoint, and the records it had logged past that are
taken out of `log.jsonl` and logged again. Weights & Biases keeps the first copy of those steps.

**The check to watch.** Every `--eval-every` steps it measures the loss at high noise on held-out images
twice with the same noise, with their embeddings and without. `gap` is how much lower the loss is with them,
as a fraction. It is zero at step 0 by construction and should grow. If it has not after an hour, the condition
is not being used, and the run prints a warning.
"""

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from itertools import islice
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch as th
import wandb

from cgd import sampler
from cgd.adm import load_model
from cgd.conditioned import AdapterConfig, ConditionedUNet, adapter_checkpoint, load_adapter, trainable
from cgd.device import default_device
from finetune import data, noise, parallel
from finetune.embed import DEFAULT_MODELS

IMAGE_SIZE = 256
PROBE_NOISIEST = 0.4  # The check uses the noisiest 40% of the schedule's logSNR range.
GAP_WARNING = 0.005
GAP_WARNING_AFTER = 3600.0
CHECKPOINT_NAME = "adapter.pt"
WANDB_PROJECT = "cgd-finetune"


class Condition(NamedTuple):
    """What a batch is conditioned on: an embedding per CLIP model, and which of them each image keeps."""

    embeds: dict[str, th.Tensor]
    keep: th.Tensor

    def to(self, device: str) -> "Condition":
        """The condition on `device`."""
        return Condition({name: value.to(device) for name, value in self.embeds.items()}, self.keep.to(device))


def choose_condition(
    embeds: Mapping[str, Mapping[str, th.Tensor]],
    generator: th.Generator,
    *,
    p_text: float,
    embed_noise: float,
    p_drop_model: float,
    p_uncond: float,
) -> Condition:
    """Draw each image's condition from a batch's embeddings: `embeds[model][kind]` is [N, dim].

    An image uses a text's embedding with probability `p_text`, one of the three kinds at random and the same
    kind for every model. Otherwise it uses its image embedding with noise of a level up to `embed_noise`.
    """
    count = len(next(iter(embeds.values()))["image"])
    use_text = th.rand(count, 1, generator=generator) < p_text
    kind = th.randint(len(data.TEXT_KINDS), (count,), generator=generator)
    level = th.rand(count, 1, generator=generator) * embed_noise
    chosen = {}
    for name, kinds in embeds.items():
        text = th.stack([kinds[k] for k in data.TEXT_KINDS], dim=1)[th.arange(count), kind]
        chosen[name] = th.where(use_text, text, noise.noisy_embedding(kinds["image"], level, generator))
    keep = th.rand(count, len(chosen), generator=generator) >= p_drop_model
    return Condition(chosen, keep & (th.rand(count, 1, generator=generator) >= p_uncond))


def prediction_error(
    model: th.nn.Module,
    log_alpha_bar: np.ndarray,
    images: th.Tensor,
    condition: Condition | None,
    *,
    log_snr: th.Tensor,
    gaussian: th.Tensor,
    amp: bool,
) -> th.Tensor:
    """The squared error of the model's noise prediction for each image in [-1, 1], at its own noise level."""

    def per_image(values: th.Tensor) -> th.Tensor:
        return values.to(images.device, th.float32)[:, None, None, None]

    alpha, sigma = noise.level_of(log_snr)
    timesteps = noise.timestep_of(log_alpha_bar, log_snr).to(images.device, th.float32)
    noisy = per_image(alpha) * images + per_image(sigma) * gaussian
    with th.autocast(images.device.type, enabled=amp):
        embeds, keep = condition if condition is not None else (None, None)
        predicted = model(noisy, timesteps, embeds, keep)[:, :3]  # The rest is the learned variance.
    return (predicted.float() - gaussian).square().mean(dim=(1, 2, 3))


def as_input(images: th.Tensor, device: str) -> th.Tensor:
    """A batch of uint8 images as the U-Net takes them."""
    return images.to(device).float().div(127.5).sub(1).contiguous(memory_format=th.channels_last)


class Probe(NamedTuple):
    """Held-out images with a fixed noise level and noise each, for the conditioning check."""

    images: th.Tensor
    embeds: dict[str, th.Tensor]
    log_snr: th.Tensor
    gaussian: th.Tensor


def make_probes(
    holdouts: Sequence[data.Source], log_alpha_bar: np.ndarray, batch_size: int, count: int, seed: int
) -> list[Probe]:
    """`count` batches of held-out images, an equal share from each dataset, at high noise."""
    generator = th.Generator().manual_seed(seed)
    table = sampler.log_snr(log_alpha_bar)
    probes = []
    for batch in islice(data.batches(holdouts, batch_size=batch_size, seed=seed, workers=0), count):
        fraction = th.rand(batch_size, generator=generator, dtype=th.float64) * PROBE_NOISIEST
        log_snr = table[-1] + (table[0] - table[-1]) * fraction
        embeds = {name: kinds["image"] for name, kinds in batch["embeds"].items()}
        probes.append(Probe(batch["image"], embeds, log_snr, th.randn(batch["image"].shape, generator=generator)))
    return probes


@th.no_grad()
def conditioning_gap(
    model: th.nn.Module, log_alpha_bar: np.ndarray, probes: Sequence[Probe], device: str, amp: bool
) -> tuple[th.Tensor, th.Tensor]:
    """Each probe image's error with its image embedding, and without any."""
    with_condition, without = [], []
    for probe in probes:
        images, gaussian = as_input(probe.images, device), probe.gaussian.to(device)
        keep = th.ones(len(images), len(probe.embeds), dtype=th.bool)
        condition = Condition(probe.embeds, keep).to(device)
        at = {"log_snr": probe.log_snr, "gaussian": gaussian, "amp": amp}
        with_condition.append(prediction_error(model, log_alpha_bar, images, condition, **at))
        without.append(prediction_error(model, log_alpha_bar, images, None, **at))
    return th.cat(with_condition).cpu(), th.cat(without).cpu()


def add_data_options(parser: argparse.ArgumentParser, workers: int) -> None:
    """The options about the datasets, which every script that trains on them shares."""
    dataset = parser.add_argument_group("data")
    dataset.add_argument("--laion", type=Path, default=data.LAION_ROOT)
    dataset.add_argument("--imagenet", type=Path, default=data.IMAGENET_ROOT)
    dataset.add_argument("--laion-where", default=None, help='a pandas query on the index, e.g. "aesthetic >= 6.5"')
    dataset.add_argument("--imagenet-where", default=None, help='e.g. "label in [281, 282]"')
    dataset.add_argument("--laion-weight", type=float, default=1.0, help="its share of each batch, against ImageNet's")
    dataset.add_argument("--imagenet-weight", type=float, default=1.0)
    dataset.add_argument("--keep-watermarked", action="store_true", help="keep images captioned as watermarked")
    dataset.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), help="the CLIP models to condition on")
    dataset.add_argument("--workers", type=int, default=workers)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """The command line's options, in the groups `-h` shows."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    run = parser.add_argument_group("the run")
    run.add_argument("--out", type=Path, default=Path("D:/cgd-finetune/uncond256"))
    run.add_argument("--resume", action="store_true", help="continue from --out's checkpoint")
    run.add_argument("--hours", type=float, default=12.0, help="stop after this much training, resumed runs included")
    run.add_argument("--max-steps", type=int, default=None, help="stop after this many steps, for a trial run")
    run.add_argument("--device", default=default_device())
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--compile", action="store_true", help="torch.compile the model. Needs a C++ compiler on Windows")
    run.add_argument("--no-amp", action="store_true", help="full precision. Half precision is the default on a GPU")
    run.add_argument("--log-every", type=int, default=200)
    run.add_argument("--eval-every", type=int, default=2000)
    run.add_argument("--save-minutes", type=float, default=20.0)
    run.add_argument("--probe-batches", type=int, default=32, help="held-out batches the check uses")
    run.add_argument(
        "--wandb",
        nargs="?",
        const=WANDB_PROJECT,
        default=None,
        metavar="PROJECT",
        help=f"also log to Weights & Biases, in this project ({WANDB_PROJECT} if not named)",
    )
    run.add_argument("--wandb-name", default=None, help="the run's name there. One is made up by default")
    add_data_options(parser, workers=6)
    adapters = parser.add_argument_group("what is added to the U-Net")
    adapters.add_argument("--rank", type=int, default=16, help="of the LoRA updates")
    adapters.add_argument("--tokens", type=int, default=4, help="condition tokens the attention layers see. 0 for none")
    adapters.add_argument("--train-encoder", action="store_true", help="adapt the encoder half too. About 1.5x slower")
    adapters.add_argument("--no-fused", action="store_true", help="keep the U-Net's own attention")
    optimisation = parser.add_argument_group("optimisation")
    optimisation.add_argument("--batch", type=int, default=2, help="2 is what fits 12 GB: about 3 GB an image")
    optimisation.add_argument("--accumulate", type=int, default=4, help="batches per optimizer step")
    optimisation.add_argument(
        "--widest", type=int, default=None, help="layers with more inputs than this get a lower learning rate"
    )
    optimisation.add_argument("--lr", type=float, default=1e-4)
    optimisation.add_argument("--warmup", type=int, default=2000, help="steps over which the learning rate rises")
    optimisation.add_argument("--clip-norm", type=float, default=1.0, help="the gradient's largest norm")
    optimisation.add_argument("--ema", type=float, default=0.999, help="decay of the weights' running average")
    optimisation.add_argument("--min-snr", type=float, default=5.0, help="gamma of the loss weights")
    optimisation.add_argument(
        "--snr-floor", type=float, default=0.1, help="count no level's signal to noise ratio as less. 0 for none"
    )
    optimisation.add_argument("--noise-bias", type=float, default=2.0, help="1 is uniform in logSNR. More is noisier")
    condition = parser.add_argument_group("the condition")
    condition.add_argument("--p-text", type=float, default=0.4, help="how often a text's embedding is the condition")
    condition.add_argument("--embed-noise", type=float, default=0.5, help="the most noise put on an image embedding")
    condition.add_argument(
        "--p-drop-model", type=float, default=0.15, help="how often one model's embedding is dropped"
    )
    condition.add_argument("--p-uncond", type=float, default=0.1, help="how often every embedding is dropped")
    return parser.parse_args(argv)


def rate_groups(weights: Mapping[str, th.Tensor], widest: int | None, pretrained: float | None = None) -> list[dict]:
    """The weights as optimizer groups, each with the share of the learning rate it gets.

    Adam moves every weight by about the learning rate, so a layer's output moves by that times its number of
    inputs. A layer with more than `widest` inputs gets proportionally less, so that none moves far faster than
    the rest. None gives every weight the whole rate. With `pretrained`, the U-Net's own weights get that share
    whatever their width: they are already where training put them.
    """
    groups: dict[float, list[th.Tensor]] = {}
    for name, weight in weights.items():
        inputs = weight[0].numel() if weight.ndim > 1 else 1
        rate = 1.0 if widest is None else min(1.0, widest / inputs)
        if pretrained is not None and name.startswith("unet."):
            rate = pretrained
        groups.setdefault(rate, []).append(weight)
    return [{"params": params, "rate": rate} for rate, params in groups.items()]


def rate_share(args: argparse.Namespace, steps: int) -> float:
    """The share of the learning rate at batch `steps`: rising over `--warmup` batches, and with `--decay`, falling
    in a straight line to zero over that many batches before `--max-steps`.

    A constant rate keeps the weights moving by about as much at the end as in the middle, so the measures
    plateau with a jitter. The decay lets them settle.
    """
    share = min(1.0, steps / max(args.warmup, 1))
    decay = getattr(args, "decay", 0)
    if decay:
        share *= min(1.0, max(0.0, (args.max_steps - steps) / decay))
    return share


class Run:  # pylint: disable=too-many-instance-attributes  # It is everything a checkpoint restores.
    """A run's state: the model, what optimises it, the running average of its weights, and how far it has got."""

    def __init__(self, args: argparse.Namespace, model: ConditionedUNet, checkpoint: Mapping | None):
        self.args, self.model = args, model
        # What the loss is taken through, and this process's place: `finetune.parallel`'s on several GPUs.
        self.net: th.nn.Module = model
        self.team = parallel.Team(device=str(args.device))
        self.weights = trainable(model)
        groups = rate_groups(self.weights, args.widest, getattr(args, "pretrained_rate", None))
        on_gpu = th.device(args.device).type == "cuda"
        # The fused update makes no temporary copies, which with every weight of half a U-Net trained is gigabytes.
        self.optimizer = th.optim.AdamW(groups, lr=args.lr, weight_decay=0.0, fused=on_gpu)
        self.amp = not args.no_amp and on_gpu
        self.scaler = th.amp.GradScaler(th.device(args.device).type, enabled=self.amp)
        # No running average with `--ema 0`, which saves a copy of every trained weight on the device.
        self.average = {name: weight.detach().clone() for name, weight in self.weights.items()} if args.ema else {}
        self.steps, self.seconds = 0, 0.0
        self.wandb_id: str | None = None  # Kept in the checkpoint, so a resumed run continues the same charts.
        if checkpoint is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scaler.load_state_dict(checkpoint["scaler"])
            self.average = {name: value.to(args.device) for name, value in checkpoint["average"].items()}
            self.steps, self.seconds = checkpoint["steps"], checkpoint["seconds"]
            self.wandb_id = checkpoint.get("wandb_id")

    def step(self, loss: th.Tensor) -> None:
        """Backpropagate one batch's loss, and update the weights when enough batches have accumulated."""
        args = self.args
        self.scaler.scale(loss / args.accumulate).backward()
        self.steps += 1
        if self.steps % args.accumulate:
            return
        for group in self.optimizer.param_groups:
            group["lr"] = args.lr * group.get("rate", 1.0) * rate_share(args, self.steps)
        self.scaler.unscale_(self.optimizer)
        th.nn.utils.clip_grad_norm_(self.weights.values(), args.clip_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        with th.no_grad():
            for name, average in self.average.items():
                average.lerp_(self.weights[name], 1 - args.ema)

    def save(self) -> None:
        """Write the checkpoint. `weights` are the trained ones, `average` their running average for sampling."""
        checkpoint = adapter_checkpoint(self.model) | {
            "average": {name: value.cpu() for name, value in self.average.items()},
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "steps": self.steps,
            "seconds": self.seconds,
            "processes": self.team.size,
            "wandb_id": self.wandb_id,
            "args": json.loads(json.dumps(vars(self.args), default=str)),
        }
        partial = self.args.out / (CHECKPOINT_NAME + ".tmp")
        th.save(checkpoint, partial)
        partial.replace(self.args.out / CHECKPOINT_NAME)


def open_sources(args: argparse.Namespace, split: str) -> list[data.Source]:
    """One split of both datasets, filtered as the options ask."""
    common = {"split": split, "keep_watermarked": args.keep_watermarked}
    return [
        data.open_source("laion", args.laion, args.models, where=args.laion_where, **common),
        data.open_source("imagenet", args.imagenet, args.models, where=args.imagenet_where, **common),
    ]


def build(args: argparse.Namespace) -> tuple[Run, np.ndarray]:
    """The model on the pretrained U-Net, new or from the run's checkpoint, and the U-Net's schedule."""
    unet, log_alpha_bar = load_model(IMAGE_SIZE, args.device, False, False, True)
    path = args.out / CHECKPOINT_NAME
    checkpoint = th.load(path, map_location="cpu", weights_only=True) if args.resume and path.is_file() else None
    if checkpoint is not None:
        model = load_adapter(unet, checkpoint, not args.no_fused)
        print(f"resuming with the checkpoint's adapters, whatever the options say: {model.config}", flush=True)
    else:
        dims = {
            name: int(np.load(data.embed_path(args.laion, name, "image"), mmap_mode="r").shape[1])
            for name in args.models
        }
        config = AdapterConfig(dims, rank=args.rank, tokens=args.tokens, train_encoder=args.train_encoder)
        model = ConditionedUNet(unet, config, not args.no_fused)
    model.to(args.device).to(memory_format=th.channels_last)
    return Run(args, model, checkpoint), log_alpha_bar


def step_generator(seed: int, step: int) -> th.Generator:
    """The random numbers of one step: the same whether the run reached it in one go or was resumed before it."""
    return th.Generator().manual_seed(seed * 1_000_003 + step + 1)


def trim_log(path: Path, steps: int) -> None:
    """Drop the records past `steps` from a log, which a killed run wrote after its last checkpoint."""
    if not path.is_file():
        return
    lines = path.read_text().splitlines()
    # The check at the checkpoint's own step goes too: the resumed run measures it again.
    kept = [line for line in lines if (record := json.loads(line))["step"] < steps + ("loss" in record)]
    if len(kept) < len(lines):
        path.write_text("".join(line + chr(10) for line in kept))


def track(run: Run) -> None:
    """Start logging to Weights & Biases if `--wandb` asks, as the same run there when this one is resumed."""
    args = run.args
    if args.wandb is None:
        return
    tracked = wandb.init(
        project=args.wandb,
        name=args.wandb_name,
        id=run.wandb_id,
        resume="allow",
        dir=args.out,
        config=json.loads(json.dumps(vars(args), default=str)),
    )
    run.wandb_id = tracked.id


def log(args: argparse.Namespace, **record: float) -> None:
    """Print a record, append it to the run's log, and send it to Weights & Biases when that is on."""
    shown = (f"{name} {value if name == 'step' else format(value, '.4g')}" for name, value in record.items())
    print("  ".join(shown), flush=True)
    with (args.out / "log.jsonl").open("a") as file:
        file.write(json.dumps(record) + "\n")
    if wandb.run is not None:
        # A step logged before a resume is ignored there, so a run that went back to its checkpoint adds no repeats.
        wandb.log(record, step=int(record["step"]))


def check_conditioning(run: Run, log_alpha_bar: np.ndarray, probes: Sequence[Probe]) -> None:
    """Log the conditioning check, and warn when an hour in the condition still makes no difference."""
    errors = conditioning_gap(run.model, log_alpha_bar, probes, run.args.device, run.amp)
    with_condition, without = (error.mean().item() for error in errors)
    gap = 1 - with_condition / without
    # The same in terms of the image and not the noise, which counts the noisiest probes as much as the rest.
    per_image = th.exp(-2 * th.cat([probe.log_snr for probe in probes])).float()
    gap_image = 1 - ((errors[0] * per_image).mean() / (errors[1] * per_image).mean()).item()
    record = {"conditioned": with_condition, "unconditioned": without, "gap": gap, "gap_image": gap_image}
    log(run.args, step=run.steps, hours=run.seconds / 3600, **record)
    if run.seconds > GAP_WARNING_AFTER and gap < GAP_WARNING:
        print(
            f"WARNING: after {run.seconds / 3600:.1f} h the embeddings lower the high-noise loss by {gap:.2%}. "
            "The condition is not being used. Consider stopping to try --train-encoder or more --tokens.",
            flush=True,
        )


def finished(run: Run) -> Run:
    """Say where a run stopped and close its Weights & Biases run. Returns the run."""
    where = run.args.out / CHECKPOINT_NAME
    print(f"stopped at step {run.steps} after {run.seconds / 3600:.2f} h. Saved {where}", flush=True)
    wandb.finish()  # Does nothing when it was never started.
    return run


def train(args: argparse.Namespace) -> Run:  # pylint: disable=too-many-locals  # It is the whole loop.
    """Train until the time or the steps run out. Returns the run as it ended."""
    args.out.mkdir(parents=True, exist_ok=True)
    th.manual_seed(args.seed)
    run, log_alpha_bar = build(args)
    trim_log(args.out / "log.jsonl", run.steps)
    track(run)
    forward = th.compile(run.model) if args.compile else run.model
    probes = make_probes(open_sources(args, "holdout"), log_alpha_bar, args.batch, args.probe_batches, args.seed)
    count = sum(weight.numel() for weight in run.weights.values())
    print(f"training {count / 1e6:.1f}M added weights from step {run.steps}, {run.seconds / 3600:.2f} h in", flush=True)
    check_conditioning(run, log_alpha_bar, probes)
    weights = [args.laion_weight, args.imagenet_weight]
    stream = data.batches(
        open_sources(args, "train"),
        weights,
        batch_size=args.batch,
        seed=args.seed,
        start=run.steps * args.batch,
        workers=args.workers,
    )
    started, saved, losses = time.monotonic() - run.seconds, time.monotonic(), []
    for batch in stream:
        if run.seconds >= args.hours * 3600 or (args.max_steps is not None and run.steps >= args.max_steps):
            break
        generator = step_generator(args.seed, run.steps)
        condition = choose_condition(
            batch["embeds"],
            generator,
            p_text=args.p_text,
            embed_noise=args.embed_noise,
            p_drop_model=args.p_drop_model,
            p_uncond=args.p_uncond,
        ).to(args.device)
        log_snr = noise.sample_log_snr(log_alpha_bar, args.batch, args.noise_bias, generator)
        gaussian = th.randn(batch["image"].shape, generator=generator).to(args.device)
        images = as_input(batch["image"], args.device)
        error = prediction_error(
            forward, log_alpha_bar, images, condition, log_snr=log_snr, gaussian=gaussian, amp=run.amp
        )
        loss = (error * noise.min_snr_weight(log_snr, args.min_snr, args.snr_floor).to(error)).mean()
        if not th.isfinite(loss):
            raise SystemExit(f"The loss is {loss.item()} at step {run.steps}. The last checkpoint is intact.")
        run.step(loss)
        losses.append(loss.item())
        run.seconds = time.monotonic() - started
        if run.steps % args.log_every == 0:
            rate = run.steps * args.batch / run.seconds
            record = {"loss": float(np.mean(losses)), "images_per_second": rate}
            if th.device(args.device).type == "cuda":
                # More than the card has means Windows is paging it to main memory, a hundred times slower.
                record["gpu_gb"] = th.cuda.max_memory_allocated() / 2**30
            log(args, step=run.steps, hours=run.seconds / 3600, **record)
            losses = []
            # Only here, with the loss's average just logged and no gradient waiting, so a resume loses nothing.
            if run.steps % args.accumulate == 0 and time.monotonic() - saved > args.save_minutes * 60:
                run.save()
                saved = time.monotonic()
        if run.steps % args.eval_every == 0:
            check_conditioning(run, log_alpha_bar, probes)
    run.save()
    check_conditioning(run, log_alpha_bar, probes)
    return finished(run)


if __name__ == "__main__":
    train(parse_args())

"""`-ft`: the class-conditional 256 U-Net, fine-tuned by `finetune.cond` to also take the prompts' CLIP embeddings.

It keeps its ImageNet class, the one closest to the text by the first CLIP model it was trained with, as `cgd`
picks one. Each of its CLIP models then tells it two embeddings, an image's and a text's:

- the text's is the mean of the text prompts of positive weight, each weighted by it;
- the image's is the same of the image prompts, each seen whole, or without any, the one the linear prior it
  was exported with predicts from the text's. Its condition is an image's embedding and a text's, and a prompt
  only gives the second.

A negative prompt is left out of the condition, which has no way to say "not", and steers only through CLIP's
guidance. The model runs with classifier-free guidance at CFG_SCALE against the class alone, and CLIP's guidance
on top, as `cgd` guides the pretrained model. `finetune/README.md` has how it was trained and how it scored.
"""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import torch as th
from torch.nn import functional as tf
from torch.nn.attention import SDPBackend, sdpa_kernel

from cgd import sampler
from cgd.conditioned import AdapterConfig, ConditionedUNet, load_adapter
from cgd.prompts import Scene, parse_prompt

SIZE = 256  # The checkpoint it is a fine-tune of.
CFG_SCALE = 3.0  # What it was trained for and scored at.
SLOT_KINDS = ("image", "text")

Encode = Callable[[str, str], th.Tensor]
"""(CLIP model's name, prompt) -> the prompt's [1, dim] embedding by that model."""


class Condition(NamedTuple):
    """What a batch is told: each condition's embedding, which of them each image keeps, and its class."""

    embeds: dict[str, th.Tensor]
    keep: th.Tensor
    labels: th.Tensor

    def to(self, device: str) -> "Condition":
        """The condition on `device`."""
        embeds = {name: value.to(device) for name, value in self.embeds.items()}
        return Condition(embeds, self.keep.to(device), self.labels.to(device))

    def alone(self) -> "Condition":
        """The same with every embedding dropped: the class alone."""
        return Condition(self.embeds, th.zeros_like(self.keep), self.labels)


class Finetuned(NamedTuple):
    """The fine-tuned model, and each of its CLIP models' map from a text's embedding to an image's."""

    model: ConditionedUNet
    priors: dict[str, dict[str, th.Tensor]]


def slot(model_name: str, kind: str) -> str:
    """The name of a condition: a CLIP model's embedding of the image, or of a text."""
    return f"{model_name} {kind}"


def slots(models: Sequence[str]) -> list[str]:
    """Every condition, in the order the model takes them."""
    return [slot(name, kind) for name in models for kind in SLOT_KINDS]


def model_names(config: AdapterConfig) -> list[str]:
    """The CLIP models whose embeddings a model takes, in order. The first one chooses the class."""
    return list(dict.fromkeys(name.rpartition(" ")[0] for name in config.embed_dims))


def predict_image(prior: Mapping[str, th.Tensor], texts: th.Tensor) -> th.Tensor:
    """The unit image embeddings one CLIP model's map predicts for text embeddings [N, dim]."""
    return tf.normalize(texts.float() @ prior["weight"] + prior["bias"], dim=-1)


def export(
    checkpoint: Mapping, priors: Mapping[str, Mapping[str, th.Tensor]], path: Path, which: str = "weights"
) -> None:
    """Write what `load` reads: a run's configuration and its `which` weights, `weights` or `average`, with the priors.

    It leaves out the optimizer, which is most of a run's checkpoint.
    """
    missing = set(model_names(AdapterConfig(**checkpoint["config"]))) - set(priors)
    if missing:
        raise ValueError(f"No prior for {sorted(missing)}, whose embeddings the model takes.")
    exported = {
        "config": checkpoint["config"],
        "weights": checkpoint[which],
        "statistics": checkpoint.get("statistics", {}),
        "priors": {name: {key: value.float().cpu() for key, value in prior.items()} for name, prior in priors.items()},
        "steps": checkpoint.get("steps"),
    }
    partial = path.with_name(path.name + ".tmp")
    th.save(exported, partial)
    partial.replace(path)


def load(path: str | Path, unet: th.nn.Module, device: str) -> Finetuned:
    """The fine-tune `export` wrote, on the pretrained `unet`, frozen, on `device`.

    It keeps the U-Net's own attention, whose backward pass is deterministic, so a seed gives the same image. On a
    GPU the U-Net's convolutions are kept at half precision, which they run at under autocast anyway: the same
    image for a gigabyte less.
    """
    exported = th.load(path, map_location="cpu", mmap=True, weights_only=True)
    # The added layers start random before the checkpoint's weights replace them. A seed's image must not depend on it.
    with th.random.fork_rng(devices=[]):
        model = load_adapter(unet, exported, fused=False)
    if th.device(device).type == "cuda":
        model.unet.convert_to_fp16()  # Before moving, so the 32-bit weights never occupy VRAM.
    model = model.requires_grad_(False).eval().to(device, memory_format=th.channels_last)
    priors = {
        name: {key: value.to(device) for key, value in prior.items()} for name, prior in exported["priors"].items()
    }
    return Finetuned(model, priors)


def weighted(encode: Encode, name: str, prompts: Sequence[str]) -> th.Tensor | None:
    """The direction of the weighted sum of the prompts' unit embeddings by `name`, of those of positive weight.

    None without any.
    """
    parsed = [(prompt, weight) for prompt, weight in map(parse_prompt, prompts) if weight > 0]
    if not parsed:
        return None
    total = sum(weight * tf.normalize(encode(name, prompt).float(), dim=-1) for prompt, weight in parsed)
    return tf.normalize(total, dim=-1)


def prompt_embeddings(
    finetuned: Finetuned, images: Sequence[Scene], encode_text: Encode, encode_image: Encode
) -> list[dict[str, th.Tensor | None]]:
    """Each condition's [1, dim] embedding for each image of a run, or None for one it is not told.

    Each CLIP model encodes every image's prompts before the next model, which it loads in its place.
    """
    embeds: list[dict[str, th.Tensor | None]] = [{} for _ in images]
    for name in model_names(finetuned.model.config):
        for told, image in zip(embeds, images, strict=True):
            text = weighted(encode_text, name, image.prompts)
            picture = weighted(encode_image, name, image.image_prompts)
            if picture is None and text is not None:
                picture = predict_image(finetuned.priors[name], text)
            told[slot(name, "image")], told[slot(name, "text")] = picture, text
    return embeds


def anchor(finetuned: Finetuned, embeds: Mapping[str, th.Tensor | None]) -> tuple[str, th.Tensor]:
    """The CLIP model that chooses the class, and the embedding it chooses it by: the text's, or the image's."""
    name = model_names(finetuned.model.config)[0]
    text = embeds[slot(name, "text")]
    chosen = text if text is not None else embeds[slot(name, "image")]
    if chosen is None:
        raise ValueError("An image of the run has no prompt of positive weight to be conditioned on.")
    return name, chosen


def batch_condition(
    finetuned: Finetuned, embeds: Mapping[str, th.Tensor | None], label: int, *, batch_size: int, device: str
) -> Condition:
    """A batch's condition: `embeds` by `prompt_embeddings` for one image, with zeros where one is dropped."""
    config = finetuned.model.config
    names = list(config.embed_dims)
    told = {
        name: th.zeros(1, config.embed_dims[name]) if embeds[name] is None else embeds[name].float() for name in names
    }
    keep = th.tensor([[embeds[name] is not None for name in names]])
    labels = th.full((batch_size,), label, dtype=th.long)
    return Condition(
        {name: value.to(device).expand(batch_size, -1) for name, value in told.items()},
        keep.to(device).expand(batch_size, -1),
        labels.to(device),
    )


def noise_given(
    model: ConditionedUNet, amp: bool, x: th.Tensor, timesteps: th.Tensor, condition: Condition
) -> th.Tensor:
    """The noise the model predicts in x, told the classes and whichever embeddings are kept.

    The token attention runs on PyTorch's plain attention, whose backward pass sums in a fixed order. The fused
    kernels' may not, as open_clip's did not, which would make a seed give a different image every run. On an
    RTX 4070 they happened to reproduce too, and the plain one took no longer.
    """
    with th.autocast(x.device.type, enabled=amp), sdpa_kernel(SDPBackend.MATH):
        x = x.contiguous(memory_format=th.channels_last)
        output = model(x, timesteps, condition.embeds, condition.keep, labels=condition.labels)
    return output[:, :3].float()  # The rest is the learned variance.


def cfg_denoiser(model: ConditionedUNet, amp: bool, condition: Condition, scale: float) -> sampler.Denoiser:
    """Classifier-free guidance for the sampler: `scale` times as far from the class alone as the embeddings move it.

    Under CLIP guidance, where x carries a gradient, only the pass with the embeddings carries it and the
    rest of the move counts as a constant. That is one backward pass through the U-Net, as `cgd` makes now.
    """
    alone = condition.alone()

    def predict(x: th.Tensor, timesteps: th.Tensor) -> th.Tensor:
        told = noise_given(model, amp, x, timesteps, condition)
        with th.no_grad():
            base = noise_given(model, amp, x.detach(), timesteps, alone)
        return told + (scale - 1) * (told - base).detach()

    return predict

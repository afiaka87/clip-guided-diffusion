"""What CLIP guidance minimises: how far images are from the prompts, by one CLIP model or an ensemble."""

from collections.abc import Callable, Sequence
from functools import partial
from typing import NamedTuple

import torch as th
from torch import nn
from torch.nn import functional as tf

from cgd.checks import require
from cgd.clip_models import encode_image, encode_texts, ensemble_members, load_clip, preprocess
from cgd.cutouts import adaptive_avg_pool, make_cutouts
from cgd.download import download_verified
from cgd.losses import spherical_dist_loss, with_gradient
from cgd.prompts import parse_prompt

# A linear head on ViT-B/32 fit to human ratings of AI art (Simulacra Aesthetic Models, MIT).
AESTHETIC_MODEL = "ViT-B/32"
AESTHETIC_URL = "https://github.com/crowsonkb/simulacra-aesthetic-models/raw/master/models/sac_public_2022_06_29_vit_b_32_linear.pth"
AESTHETIC_FILENAME = "sac_public_2022_06_29_vit_b_32_linear.pth"
AESTHETIC_SHA256 = "64315370e2293a08cc2f4816ae9db715f935791569c1a57d84b943e482ea5e12"
MIN_TOTAL_WEIGHT = 1e-3
NUM_CUTOUTS = 4
MIN_CUT_FRACTION = 0.7  # Cutouts span 70-100% of the frame, or down to CLIP's input size if that is smaller.


class Tower(NamedTuple):
    """One CLIP model of a guide, and the prompts it scores images against."""

    model: nn.Module
    size: int  # The model's input resolution.
    targets: th.Tensor  # [targets, features]: one per text prompt, and one per cutout of each image prompt.
    weights: th.Tensor  # [targets]: each target's share of the prompt weights, which sum to 1 or -1.


class Aesthetic(NamedTuple):
    """A predictor of how people would rate an image, and how much guidance rewards it."""

    model: nn.Module
    weight: th.Tensor
    bias: th.Tensor
    scale: float  # Per point of a rating from 1 to 10.


class Guide(NamedTuple):
    """Everything guidance needs to score images against the prompts."""

    towers: tuple[Tower, ...]
    num_cutouts: int
    min_cut_fraction: float
    aesthetic: Aesthetic | None = None


def load_guide(
    model_name: str,
    prompts: Sequence[str] = (),
    image_prompts: Sequence[str] = (),
    *,
    image_size: int = 128,
    num_cutouts: int = NUM_CUTOUTS,
    device: str = "cpu",
    fp16: bool = False,
    min_cut_fraction: float = MIN_CUT_FRACTION,
    aesthetic_scale: float = 0.0,
) -> Guide:
    """Load each model of `model_name` and encode the prompts with it.

    `image_size` is the size of the image being made, which sets the detail image prompts
    are seen at. Encoding them takes random cutouts, so this draws random numbers.
    """
    aesthetic = load_aesthetic(aesthetic_scale, device, fp16) if aesthetic_scale else None
    towers = tuple(
        load_tower(
            member, prompts, image_prompts, image_size=image_size, num_cutouts=num_cutouts, device=device, fp16=fp16
        )
        for member in ensemble_members(model_name)
    )
    return Guide(towers, num_cutouts, min_cut_fraction, aesthetic)


def load_tower(
    model_name: str,
    prompts: Sequence[str],
    image_prompts: Sequence[str],
    *,
    image_size: int,
    num_cutouts: int,
    device: str,
    fp16: bool,
) -> Tower:
    """One model, with the prompts encoded by it. Each image prompt's weight is shared by its cutouts."""
    model, size = load_clip(model_name, device, fp16)
    encoded = [(encode_texts([text], model_name, device, fp16), weight) for text, weight in map(parse_prompt, prompts)]
    encoded += [
        (encode_image(path, model_name, device, fp16, image_size=image_size, num_cutouts=num_cutouts), weight)
        for path, weight in map(parse_prompt, image_prompts)
    ]
    weights = th.tensor([weight / len(targets) for targets, weight in encoded for _ in targets], device=device)
    total = weights.sum().abs()
    require(float(total) >= MIN_TOTAL_WEIGHT, "The prompt weights must not sum to 0.")
    return Tower(model, size, th.cat([targets for targets, _ in encoded]), weights / total)


def load_aesthetic(scale: float, device: str, fp16: bool = False) -> Aesthetic:
    """The aesthetic predictor, weighted by `scale`."""
    model, _ = load_clip(AESTHETIC_MODEL, device, fp16)
    head = th.load(
        download_verified(AESTHETIC_URL, AESTHETIC_FILENAME, AESTHETIC_SHA256), map_location=device, weights_only=True
    )
    return Aesthetic(model, head["linear.weight"].float(), head["linear.bias"].float(), scale)


def query(guide: Guide) -> th.Tensor:
    """The prompts as one weighted [1, features] feature of the first model, for looking things up."""
    tower = guide.towers[0]
    return (tower.weights[:, None] * tf.normalize(tower.targets, dim=-1)).sum(0, keepdim=True)


def tower_distance(guide: Guide, tower: Tower, images: th.Tensor) -> th.Tensor:
    """Each image's weighted distance from the prompts, for images in [0, 1], averaged over random cutouts."""
    cutouts = make_cutouts(images, tower.size, guide.num_cutouts, guide.min_cut_fraction)
    embeds = tower.model.encode_image(preprocess(tower.model, cutouts)).float()
    embeds = embeds.view(-1, images.shape[0], embeds.shape[-1])  # [cutouts, batch, features]
    return spherical_dist_loss(embeds.unsqueeze(2), tower.targets).mul(tower.weights).sum(2).mean(0)


def aesthetic_rating(aesthetic: Aesthetic, images: th.Tensor) -> th.Tensor:
    """The predicted human rating of each whole image in [0, 1], from 1 to 10."""
    size = aesthetic.model.visual.input_resolution
    embeds = aesthetic.model.encode_image(preprocess(aesthetic.model, adaptive_avg_pool(images, size))).float()
    embeds = tf.normalize(embeds, dim=-1) * embeds.shape[-1] ** 0.5
    return (embeds @ aesthetic.weight.T + aesthetic.bias).squeeze(-1)


def cutout_rating(aesthetic: Aesthetic, images: th.Tensor, num_cutouts: int, min_cut_fraction: float) -> th.Tensor:
    """The rating averaged over random cutouts, which buys more of it per point of CLIP score than the whole frame."""
    cutouts = make_cutouts(images, aesthetic.model.visual.input_resolution, num_cutouts, min_cut_fraction)
    return aesthetic_rating(aesthetic, cutouts).view(-1, images.shape[0]).mean(0)


@th.no_grad()
def score(guide: Guide, images: th.Tensor) -> th.Tensor:
    """How well each whole image in [-1, 1] matches the prompts, higher being better, averaged over the models."""
    return th.stack([_tower_score(tower, images) for tower in guide.towers]).mean(0)


def _tower_score(tower: Tower, images: th.Tensor) -> th.Tensor:
    pixels = images.add(1).div(2).clamp(0, 1)
    whole = tf.interpolate(pixels, (tower.size, tower.size), mode="bicubic", antialias=True, align_corners=False)
    embeds = tf.normalize(tower.model.encode_image(preprocess(tower.model, whole.clamp(0, 1))).float(), dim=-1)
    return embeds @ (tower.weights[:, None] * tf.normalize(tower.targets, dim=-1)).sum(0)


def guide_loss(guide: Guide, images: th.Tensor, scale: float = 1.0) -> th.Tensor:
    """The guide's loss on images in [-1, 1]: each image's mean CLIP loss, summed over the batch, times `scale`.

    With several terms, an ensemble or the aesthetic one, each backpropagates on its own, so
    only one model's activations are ever in memory. The result has the value of the real
    loss and the gradient the terms accumulated, and everything upstream of `images` only
    runs backward once.
    """
    terms: list[Callable[[th.Tensor], th.Tensor]] = [partial(_tower_term, guide, tower) for tower in guide.towers]
    if guide.aesthetic is not None:
        terms.append(partial(_aesthetic_term, guide, guide.aesthetic))
    if len(terms) == 1:
        return terms[0](images.add(1).div(2)).sum() * scale
    detached = images.detach().requires_grad_()
    total, gradient = 0.0, th.zeros_like(detached)
    for term in terms:
        value = term(detached.add(1).div(2)).sum() * scale
        gradient = gradient + th.autograd.grad(value, detached)[0]
        total += value.item()
    return with_gradient(images, total, gradient)


def _tower_term(guide: Guide, tower: Tower, pixels: th.Tensor) -> th.Tensor:
    return tower_distance(guide, tower, pixels) / len(guide.towers)


def _aesthetic_term(guide: Guide, aesthetic: Aesthetic, pixels: th.Tensor) -> th.Tensor:
    return -aesthetic.scale * cutout_rating(aesthetic, pixels, guide.num_cutouts, guide.min_cut_fraction)

"""Thumbnail guidance: a model that reads the noisy image as a 64x64 thumbnail sets the composition.

Averaging a noisy image x = alpha * x0 + sigma * noise over f x f blocks gives a noisy
thumbnail with the same alpha and 1/f of the noise: y = alpha * D(x0) + sigma / f * noise.
A model of noisy 64x64 images can read y at that higher signal to noise ratio, at the
noise levels where the composition forms, the U-Net's own denoised image is a blur and the
usual CLIP sees nothing. What it says the prompts change about the noise in y is a
correction to the U-Net's noise prediction, once it is sent back through the averaging:

    eps = eps_unet(x) + scale / f * enlarged(change(y))

For a text-to-image model the change is eps(y | prompt) - eps(y), its classifier-free
guidance direction. For a classifier it is -sigma * grad log p(prompt | y), the same thing.
It only ever moves f x f blocks together, so it sets the composition and leaves every
detail to the U-Net and to CLIP guidance on cutouts.

The models:

- clip: the CLIP OpenAI trained on noisy 64x64 images and released with GLIDE. A
  classifier, so with it nothing in the pipeline was trained to draw from text.
- glide: GLIDE's 64x64 text-to-image model, trained on data filtered of people.
- laionide: GLIDE fine-tuned on LAION, people included. It scored as GLIDE did.
- karlo: the 64x64 decoder of Karlo, an open unCLIP. Its prior first draws a CLIP image
  embedding for each prompt, which the decoder is conditioned on beside the text.

Each prompt's change is weighted as CLIP guidance weights the prompts, a negative one
pushing away, and each scale is the one its model was measured with.
"""

import math
import warnings
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from functools import lru_cache, partial
from typing import Any, NamedTuple

import numpy as np
import torch as th
import torch.nn.functional as tf
from torch import nn

from cgd import sampler
from cgd.checks import require
from cgd.clip_models import skip_weight_init
from cgd.download import CACHE_PATH, download
from cgd.guide import MIN_TOTAL_WEIGHT
from cgd.prompts import parse_prompt

SIZE = 64  # What every thumbnail model reads.
SCALES = {"clip": 10.0, "glide": 3.0, "laionide": 3.0, "karlo": 3.0}
MODELS = tuple(SCALES)

GLIDE_CACHE_PATH = CACHE_PATH / "glide"
GLIDE_URL = "https://openaipublic.blob.core.windows.net/diffusion/dec-2021/"
GLIDE_WEIGHTS = {
    "glide": f"{GLIDE_URL}base.pt",
    "laionide": "https://github.com/afiaka87/laionide/releases/download/Checkpoints/laionide-v3-base.pt",
}
CLIP_WEIGHTS = {"image_encoder": f"{GLIDE_URL}clip_image_enc.pt", "text_encoder": f"{GLIDE_URL}clip_text_enc.pt"}

KARLO = "kakaobrain/karlo-v1-alpha"
KARLO_PRIOR_SCALE = 4.0  # The classifier-free guidance of its prior and the steps it takes, as its pipeline has them.
KARLO_PRIOR_STEPS = 25
KARLO_SEED_OFFSET = 15838  # Keeps the noise its prior draws apart from the run's own stream.

Change = Callable[[th.Tensor, float, float], th.Tensor]
"""(y, signal, noise) -> what the prompts change about the noise in y = signal * thumbnail + noise * standard normal."""

ModelChange = Callable[[th.Tensor, float], th.Tensor]
"""(y, t) -> that change, for y at timestep t of a thumbnail model's own schedule."""


class Thumbnail(NamedTuple):
    """One run's thumbnail guidance."""

    change: Change
    scale: float


class NoisyClip(NamedTuple):
    """OpenAI's CLIP of noisy 64x64 images."""

    model: Any
    log_alpha_bar: np.ndarray  # Of the schedule it was trained on.


class Glide(NamedTuple):
    """GLIDE's 64x64 text-to-image model."""

    model: nn.Module
    log_alpha_bar: np.ndarray  # Of its schedule.
    text_context: int  # How many tokens it reads.


class Karlo(NamedTuple):
    """The parts of Karlo that thumbnail guidance uses."""

    tokenizer: Any
    text_encoder: nn.Module
    prior: nn.Module
    prior_scheduler: Any
    text_proj: nn.Module
    decoder: nn.Module
    log_alpha_bar: np.ndarray  # Of the decoder's schedule.


class KarloText(NamedTuple):
    """What Karlo's decoder is conditioned on: a row per prompt and image of the batch, then one per image for none."""

    hidden: th.Tensor
    additive: th.Tensor
    mask: th.Tensor


def weighted_texts(prompts: Sequence[str], device: str) -> tuple[list[str], th.Tensor]:
    """The text prompts and their weights, which sum to 1 or -1 as a guide's do."""
    texts, weights = zip(*map(parse_prompt, prompts), strict=True)
    weights = th.tensor(weights, device=device)
    total = weights.sum().abs()
    require(float(total) >= MIN_TOTAL_WEIGHT, "The text prompts' weights must not sum to 0.")
    return list(texts), weights / total


def weighted_change(conditional: th.Tensor, unconditional: th.Tensor, weights: th.Tensor) -> th.Tensor:
    """What the prompts change about a noise prediction, each by its weight.

    `conditional` holds the predictions of the whole batch for the first prompt, then for
    the next, and `unconditional` those for no prompt.
    """
    change = conditional.view(len(weights), *unconditional.shape) - unconditional
    return (weights.view(-1, 1, 1, 1, 1) * change).sum(0)


@th.no_grad()
def matched(change: ModelChange, log_alpha_bar: np.ndarray, y: th.Tensor, signal: float, noise: float) -> th.Tensor:
    """`change` of y, at the timestep of its model's schedule `log_alpha_bar` whose signal to noise ratio is y's."""
    ratio = (signal / noise) ** 2
    alpha_bar = ratio / (1 + ratio)
    steps = np.arange(len(log_alpha_bar), dtype=np.float64)
    t = float(np.interp(math.log(alpha_bar), log_alpha_bar[::-1], steps[::-1]))
    return change(y * (math.sqrt(alpha_bar) / signal), t)


def guided_denoise(
    denoise: sampler.Denoiser, log_alpha_bar: np.ndarray, thumbnail: Thumbnail, x: th.Tensor, t: th.Tensor
) -> th.Tensor:
    """`denoise`'s noise prediction for x on the schedule `log_alpha_bar`, corrected by a thumbnail model."""
    eps = denoise(x, t)
    factor = x.shape[-1] // SIZE
    now = sampler.level(log_alpha_bar, float(t[0]))
    change = thumbnail.change(tf.avg_pool2d(x.detach().float(), factor), now.alpha, now.sigma / factor)
    return eps + (thumbnail.scale / factor) * tf.interpolate(change, scale_factor=factor, mode="nearest").to(eps.dtype)


def load(name: str, prompts: Sequence[str], *, batch_size: int, seed: int, device: str) -> Thumbnail:
    """The guidance of the thumbnail model `name` towards the text `prompts`, for a batch drawn from `seed`.

    Loading draws none of the run's random numbers, so it does not move a seed's image.
    """
    texts, weights = weighted_texts(prompts, device)
    loaded = load_model(name, device)
    if isinstance(loaded, NoisyClip):
        target = (weights[:, None] * loaded.model.text_embeddings(texts)).sum(0, keepdim=True)
        change: ModelChange = partial(clip_change, loaded.model, target, loaded.log_alpha_bar)
    elif isinstance(loaded, Karlo):
        generator = th.Generator(device).manual_seed(seed + KARLO_SEED_OFFSET)
        change = partial(karlo_change, loaded.decoder, karlo_text(loaded, texts, batch_size, generator), weights)
    else:
        tokens = glide_tokens(loaded.model, texts, loaded.text_context, device)
        change = partial(glide_change, loaded.model, tokens, weights)
    return Thumbnail(partial(matched, change, loaded.log_alpha_bar), SCALES[name])


def forked_rng(device: str) -> AbstractContextManager[None]:
    """Fork the CPU's random stream and the device's, so a model built inside draws none of the run's random numbers.

    Both matter: a model built straight on the GPU initialises its weights from the GPU's stream,
    which the seed's initial noise is drawn from.
    """
    place = th.device(device)
    return th.random.fork_rng(devices=[place] if place.type == "cuda" else [])


@lru_cache(maxsize=1)
def load_model(name: str, device: str) -> NoisyClip | Glide | Karlo:
    """The thumbnail model `name`, frozen. Only one stays loaded, so trying another frees the last one's VRAM."""
    if name == "clip":
        return load_clip(device)
    if name == "karlo":
        return load_karlo(device)
    return load_glide(name, device)


# The CLIP of noisy images.


def load_clip(device: str) -> NoisyClip:
    """OpenAI's CLIP of noisy 64x64 images, frozen."""
    from glide_text2im.clip.model_creation import create_clip_model
    from glide_text2im.gaussian_diffusion import get_named_beta_schedule

    # Building it on the device draws random numbers there, which must not move the seed's stream.
    with forked_rng(device), skip_weight_init():
        model = create_clip_model(device=th.device(device))
    for part, url in CLIP_WEIGHTS.items():
        path = download(url, url.rpartition("/")[2], GLIDE_CACHE_PATH)
        encoder = getattr(model, part)
        encoder.load_state_dict(th.load(path, map_location=device, weights_only=True))
        encoder.eval().requires_grad_(False)
    betas = get_named_beta_schedule(model.config["beta_schedule"], model.config["n_timesteps"])
    return NoisyClip(model, np.log(np.cumprod(1 - np.asarray(betas, dtype=np.float64))))


def clip_change(model: Any, target: th.Tensor, log_alpha_bar: np.ndarray, y: th.Tensor, t: float) -> th.Tensor:
    """What the noisy CLIP's guidance towards `target` adds to a noise prediction of y at timestep t.

    A classifier's guidance is -sigma * grad log p(prompt | y), and log p is the model's
    logit, 100 times its cosine similarity, as in GLIDE's own CLIP guidance. `target` is the
    weighted sum of the prompts' embeddings, so the logit is the weighted sum of theirs.
    """
    step = min(max(round(t), 0), len(log_alpha_bar) - 1)  # Its timestep embedding is a table.
    sigma = math.sqrt(1 - math.exp(log_alpha_bar[step]))
    with th.enable_grad():
        noisy = y.detach().float().requires_grad_()
        timesteps = th.full((y.shape[0],), step, device=y.device, dtype=th.long)
        logit = model.logit_scale.exp() * (model.image_embeddings(noisy, timesteps) * target).sum()
        gradient = th.autograd.grad(logit, noisy)[0]
    return -sigma * gradient


# GLIDE, and LAIONIDE, which is the same model with other weights.


def load_glide(name: str, device: str) -> Glide:
    """GLIDE's 64x64 model with the weights of `name`, frozen."""
    from glide_text2im.model_creation import create_model_and_diffusion, model_and_diffusion_defaults

    options = model_and_diffusion_defaults() | {"use_fp16": device != "cpu"}
    # Building it draws random numbers, which must not move the seed's stream.
    with forked_rng(device), skip_weight_init():
        model, diffusion = create_model_and_diffusion(**options)
    url = GLIDE_WEIGHTS[name]
    path = download(url, url.rpartition("/")[2], GLIDE_CACHE_PATH)
    model.load_state_dict(th.load(path, map_location="cpu", weights_only=True))
    model.eval().requires_grad_(False)
    if options["use_fp16"]:
        model.convert_to_fp16()  # Before moving, so the 32-bit weights never occupy VRAM.
    return Glide(model.to(device), sampler.schedule(diffusion.alphas_cumprod), options["text_ctx"])


def glide_tokens(model: nn.Module, texts: Sequence[str], text_context: int, device: str) -> dict[str, th.Tensor]:
    """Each prompt, then the empty prompt, as GLIDE takes them."""
    encoded = [*(model.tokenizer.encode(text) for text in texts), []]
    tokens, masks = zip(*(model.tokenizer.padded_tokens_and_mask(text, text_context) for text in encoded), strict=True)
    return {"tokens": th.tensor(tokens, device=device), "mask": th.tensor(masks, dtype=th.bool, device=device)}


def glide_change(model: nn.Module, tokens: dict[str, th.Tensor], weights: th.Tensor, y: th.Tensor, t: float):
    """What the prompts change about GLIDE's noise prediction for y at its timestep t."""
    count, runs = y.shape[0], len(weights) + 1
    timesteps = th.full((runs * count,), t, device=y.device, dtype=th.float32)
    text = {name: value.repeat_interleave(count, dim=0) for name, value in tokens.items()}
    eps = model(y.repeat(runs, 1, 1, 1), timesteps, **text)[:, :3].float()  # The rest is the learned variance.
    return weighted_change(eps[:-count], eps[-count:], weights)


# Karlo.


def load_karlo(device: str) -> Karlo:
    """Karlo's text encoder, prior and 64x64 decoder, frozen, at half precision on a GPU.

    The prior is only needed once a run, so it waits on the CPU and karlo_text() borrows the device for it.
    """
    from diffusers import PriorTransformer, UnCLIPScheduler, UNet2DConditionModel
    from diffusers.pipelines.deprecated.unclip.text_proj import UnCLIPTextProjModel
    from huggingface_hub.utils import logging as hub_logging
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer

    hub_logging.set_verbosity_error()  # The Hub otherwise asks for a token on every run.
    dtype = th.float32 if device == "cpu" else th.float16

    def part(kind: Any, folder: str) -> nn.Module:
        return (
            kind.from_pretrained(KARLO, subfolder=folder, use_safetensors=True, dtype=dtype)
            .eval()
            .requires_grad_(False)
        )

    # Building them may draw random numbers, which must not move the seed's stream.
    with forked_rng(device), warnings.catch_warnings():
        # diffusers passes the Hub an argument it has deprecated, which is nothing to act on.
        warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*")
        schedule = UnCLIPScheduler.from_pretrained(KARLO, subfolder="decoder_scheduler")
        return Karlo(
            CLIPTokenizer.from_pretrained(KARLO, subfolder="tokenizer"),
            part(CLIPTextModelWithProjection, "text_encoder").to(device),
            part(PriorTransformer, "prior"),
            UnCLIPScheduler.from_pretrained(KARLO, subfolder="prior_scheduler"),
            part(UnCLIPTextProjModel, "text_proj").to(device),
            part(UNet2DConditionModel, "decoder").to(device),
            np.log(schedule.alphas_cumprod.double().numpy()),
        )


@th.no_grad()
def karlo_text(karlo: Karlo, texts: Sequence[str], batch_size: int, generator: th.Generator) -> KarloText:
    """The decoder's conditioning: each prompt's text, and a CLIP image embedding the prior draws for it per image."""
    device = generator.device
    tokens = karlo.tokenizer(
        [*texts, ""],
        padding="max_length",
        max_length=karlo.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    encoded = karlo.text_encoder(tokens.input_ids.to(device))
    # A row per prompt and image, then as many rows of the empty prompt.
    count = len(texts) * batch_size
    rows = [*th.arange(len(texts)).repeat_interleave(batch_size).tolist(), *[len(texts)] * count]
    embeds, hidden = encoded.text_embeds[rows], encoded.last_hidden_state[rows]
    mask = tokens.attention_mask.bool().to(device)[rows]
    image_embeddings = karlo_image_embeddings(karlo, embeds, hidden, mask, generator)
    # The projection puts its learned embedding of no image first, so it takes the empty prompt first.
    hidden, additive = karlo.text_proj(
        image_embeddings=image_embeddings,
        prompt_embeds=embeds.roll(count, 0),
        text_encoder_hidden_states=hidden.roll(count, 0),
        do_classifier_free_guidance=True,
    )
    mask = tf.pad(mask.roll(count, 0), (karlo.text_proj.clip_extra_context_tokens, 0), value=True)
    # Every row of the empty prompt is the same, so one per image is enough.
    kept = [*range(count, 2 * count), *range(batch_size)]
    return KarloText(hidden[kept], additive[kept], mask[kept])


def karlo_image_embeddings(
    karlo: Karlo, embeds: th.Tensor, hidden: th.Tensor, mask: th.Tensor, generator: th.Generator
) -> th.Tensor:
    """A CLIP image embedding the prior draws for each prompt of the first half of the rows.

    The second half holds as many rows of the empty prompt, for its classifier-free guidance.
    """
    count, device = embeds.shape[0] // 2, generator.device
    prior = karlo.prior.to(device)
    karlo.prior_scheduler.set_timesteps(KARLO_PRIOR_STEPS, device=device)
    timesteps = karlo.prior_scheduler.timesteps
    latents = th.randn((count, prior.config.embedding_dim), device=device, generator=generator).to(embeds.dtype)
    for i, t in enumerate(timesteps):
        predicted = prior(
            latents.repeat(2, 1), timestep=t, proj_embedding=embeds, encoder_hidden_states=hidden, attention_mask=mask
        ).predicted_image_embedding
        predicted = predicted[count:] + KARLO_PRIOR_SCALE * (predicted[:count] - predicted[count:])
        following = timesteps[i + 1] if i + 1 < len(timesteps) else None
        latents = karlo.prior_scheduler.step(
            predicted, timestep=t, sample=latents, generator=generator, prev_timestep=following
        ).prev_sample
    image_embeddings = prior.post_process_latents(latents)
    prior.to("cpu")
    return image_embeddings


def karlo_change(decoder: nn.Module, text: KarloText, weights: th.Tensor, y: th.Tensor, t: float) -> th.Tensor:
    """What the prompts change about the decoder's noise prediction for y at its timestep t."""
    count, runs = y.shape[0], len(weights) + 1
    eps = decoder(
        sample=y.repeat(runs, 1, 1, 1).to(text.hidden.dtype),
        timestep=th.full((runs * count,), t, device=y.device, dtype=th.float32),
        encoder_hidden_states=text.hidden,
        class_labels=text.additive,
        attention_mask=text.mask,
    ).sample
    eps = eps[:, :3].float()  # The rest is the learned variance.
    return weighted_change(eps[:-count], eps[-count:], weights)

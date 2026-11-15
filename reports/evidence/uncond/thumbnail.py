"""Thumbnail guidance: a small text-conditioned model of 64x64 images tells the unconditional U-Net where things go.

Averaging a noisy 256x256 image x_t = alpha * x0 + sigma * noise over 4x4 blocks gives a
noisy 64x64 image with the same alpha and a quarter of the noise: y = alpha * D(x0) + (sigma / 4) * noise.
So a text-conditioned diffusion model of thumbnails can be asked, at that higher signal to
noise ratio, what the prompt changes about y. Its classifier-free guidance direction,
eps(y | prompt) - eps(y), is the gradient of log p(prompt | y), and sent back through the
averaging it is a correction to the U-Net's own noise prediction:

    eps = eps_unet(x_t) + scale / 4 * upsample(eps_y(y | prompt) - eps_y(y))

It only ever moves 4x4 blocks together, so it sets the composition and leaves every detail
to the U-Net and CLIP guidance. It costs two calls of a model that is tiny next to the
U-Net, takes no gradient, and is strongest exactly where CLIP guidance is weakest: at high
noise, where the U-Net's denoised image is a blur but the thumbnail is already legible.

The other way to use the same model is to let it draw a whole thumbnail first and start
the U-Net from that, noised partway down the schedule (hand-off, as layout.py does with
another arm's images).

Thumbnail models here: OpenAI's GLIDE base model (64x64, text-conditioned, public weights),
LAIONIDE v3 (the same model fine-tuned on LAION), the 64x64 decoder of Karlo (an open unCLIP,
conditioned on the prompt and on a CLIP image embedding its prior draws for the prompt), and
layout_prior.LayoutPrior, trained in this folder.

A thumbnail model need not be able to draw. OpenAI released the CLIP it trained on noisy
64x64 images with GLIDE ("glide-clip" here). The gradient of its similarity to the prompt is
the classifier form of the same direction, -sigma * grad log p(prompt | y), so it goes through
the same code: CLIP guidance on the thumbnail, from a CLIP that can read an image at the
noise levels where the composition forms and the usual CLIP sees a grey blur.

Karlo's prior also gives CLIP guidance a better target. A text embedding is not where
images are: guidance towards it is paid for anything that reads as the prompt, a collage
included. The prior turns the prompt into the embedding of one plausible picture of it,
composition and style included, and with `target` guidance pulls the image's own embedding
in that CLIP model (ViT-L/14) towards it: the unconditional U-Net as an unCLIP decoder, by guidance.

    uv run python run_arm.py xres_glide_w3_e thumbnail:generate \
        '{"prior": "glide", "scale": 3.0, "clip_model": "ViT-B/32+PE-Core-B-16"}'
"""

import math
from collections.abc import Callable, Iterator, Sequence
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch as th
import torch.nn.functional as tf
import torchvision.transforms.functional as tvf

from cgd import adm, sampler
from cgd.clip_models import encode_texts, load_clip
from cgd.download import CACHE_PATH, download
from cgd.guide import NUM_CUTOUTS, Guide, Tower, load_guide, score
from cgd.outputs import frame_path, run_name, save_frame, save_frames

FACTOR = 4  # 256 / 64.
PRIOR_SEED_OFFSET = 7919  # Keeps a drawn thumbnail's noise apart from the run's own stream.

Predictions = Callable[[th.Tensor, float, float], tuple[th.Tensor, th.Tensor]]
"""(y, signal, noise) -> eps(y | prompt) and eps(y), for y = signal * thumbnail + noise * standard normal."""

Eps = Callable[[th.Tensor, float], tuple[th.Tensor, th.Tensor]]
"""(x, t) -> eps(x | prompt) and eps(x) of a thumbnail model, for x at timestep t of that model's own schedule."""


@th.no_grad()
def matched_predictions(eps: Eps, log_alpha_bar: np.ndarray, y: th.Tensor, signal: float, noise: float):
    """`eps` of y, at the timestep of its model's schedule `log_alpha_bar` whose signal to noise ratio is y's."""
    ratio = (signal / noise) ** 2
    alpha_bar = ratio / (1 + ratio)
    t = float(np.interp(math.log(alpha_bar), log_alpha_bar[::-1], np.arange(len(log_alpha_bar), dtype=np.float64)[::-1]))
    return eps(y * (math.sqrt(alpha_bar) / signal), t)


@th.no_grad()
def drawn_thumbnails(
    eps: Eps, log_alpha_bar: np.ndarray, ts: Sequence[float], scale: float, generator: th.Generator, count: int = 1
) -> th.Tensor:
    """`count` thumbnails in [-1, 1] drawn with classifier-free guidance: DDIM over the timesteps `ts`."""
    levels = [sampler.level(log_alpha_bar, t) for t in ts]
    x = th.randn((count, 3, 64, 64), device=generator.device, generator=generator)
    x0 = x
    for i in range(len(ts) - 1):
        conditional, unconditional = eps(x, ts[i])
        guided = unconditional + scale * (conditional - unconditional)
        x0 = ((x - levels[i].sigma * guided) / levels[i].alpha).clamp(-1, 1)
        if i < len(ts) - 2:
            guided = (x - levels[i].alpha * x0) / levels[i].sigma
            x = levels[i + 1].alpha * x0 + levels[i + 1].sigma * guided
    return x0


# GLIDE.


# GLIDE base fine-tuned on LAION and other captioned photos, people included, by this repository's maintainer.
LAIONIDE_URL = "https://github.com/afiaka87/laionide/releases/download/Checkpoints/laionide-v3-base.pt"


@lru_cache(maxsize=1)
def load_glide(device: str, weights: str = "glide"):
    """GLIDE's 64x64 base model, frozen, and the log(alpha_bar) of its schedule.

    `weights` is "glide" for OpenAI's, trained on data filtered of people, or "laionide" for LAIONIDE v3.
    """
    from glide_text2im.download import load_checkpoint
    from glide_text2im.model_creation import create_model_and_diffusion, model_and_diffusion_defaults

    options = model_and_diffusion_defaults() | {"use_fp16": device != "cpu"}
    with th.random.fork_rng(devices=[]):  # Building it draws random numbers, which must not move the seed's stream.
        model, diffusion = create_model_and_diffusion(**options)
    if weights == "glide":
        state = load_checkpoint("base", th.device("cpu"), cache_dir=str(CACHE_PATH / "glide"))
    else:
        path = download(LAIONIDE_URL, LAIONIDE_URL.rpartition("/")[2], CACHE_PATH / "glide")
        state = th.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval().requires_grad_(False)
    if options["use_fp16"]:
        model.convert_to_fp16()
    return model.to(device), np.log(np.asarray(diffusion.alphas_cumprod, dtype=np.float64)), options


def glide_text(model, options: dict, prompt: str, device: str) -> dict[str, th.Tensor]:
    """The prompt and the empty prompt, as the batch of two that classifier-free guidance needs."""
    tokens, mask = model.tokenizer.padded_tokens_and_mask(model.tokenizer.encode(prompt), options["text_ctx"])
    no_tokens, no_mask = model.tokenizer.padded_tokens_and_mask([], options["text_ctx"])
    return {
        "tokens": th.tensor([tokens, no_tokens], device=device),
        "mask": th.tensor([mask, no_mask], dtype=th.bool, device=device),
    }


def glide_eps(model, text: dict, x: th.Tensor, t: float) -> tuple[th.Tensor, th.Tensor]:
    """GLIDE's noise prediction for each image of x at its timestep t, with the prompt and without."""
    count = x.shape[0]
    timesteps = th.full((2 * count,), t, device=x.device, dtype=th.float32)
    both = {name: value.repeat_interleave(count, dim=0) for name, value in text.items()}
    eps = model(x.repeat(2, 1, 1, 1), timesteps, **both)[:, :3].float()
    return eps[:count], eps[count:]


def glide_predictions(model, log_alpha_bar: np.ndarray, text: dict, y: th.Tensor, signal: float, noise: float):
    """eps(y | prompt) and eps(y) from GLIDE, at the timestep of its schedule whose signal to noise ratio is y's."""
    return matched_predictions(partial(glide_eps, model, text), log_alpha_bar, y, signal, noise)


def glide_thumbnail(
    model, log_alpha_bar: np.ndarray, text: dict, scale: float, steps: int, generator, count: int = 1
) -> th.Tensor:
    """`count` thumbnails in [-1, 1] drawn by GLIDE with classifier-free guidance: DDIM in uniform logSNR steps."""
    ts = sampler.timesteps(log_alpha_bar, steps)
    return drawn_thumbnails(partial(glide_eps, model, text), log_alpha_bar, ts, scale, generator, count)


# The CLIP of noisy images released with GLIDE.


@lru_cache(maxsize=1)
def load_noised_clip(device: str):
    """OpenAI's CLIP of noisy 64x64 images, frozen, and the log(alpha_bar) of the cosine schedule it was trained on."""
    from glide_text2im.clip.model_creation import create_clip_model
    from glide_text2im.download import load_checkpoint
    from glide_text2im.gaussian_diffusion import get_named_beta_schedule

    with th.random.fork_rng(devices=[]):  # Building it draws random numbers, which must not move the seed's stream.
        model = create_clip_model(device=th.device(device))
    for part, name in ((model.image_encoder, "clip/image-enc"), (model.text_encoder, "clip/text-enc")):
        part.load_state_dict(load_checkpoint(name, th.device(device), cache_dir=str(CACHE_PATH / "glide")))
        part.eval().requires_grad_(False)
    betas = get_named_beta_schedule(model.config["beta_schedule"], model.config["n_timesteps"])
    return model, np.log(np.cumprod(1 - np.asarray(betas, dtype=np.float64)))


def noised_clip_eps(model, text: th.Tensor, log_alpha_bar: np.ndarray, x: th.Tensor, t: float):
    """What the noised CLIP's guidance adds to a noise prediction of x at timestep t, and zero.

    A classifier's guidance is -sigma * grad log p(prompt | x), where a generator's is
    eps(x | prompt) - eps(x). Returned as that pair's first member with a second of zero, so
    guided_denoise() scales and enlarges it as it does a generator's. log p is the model's
    logit, its scale of 100 times the cosine similarity, as in GLIDE's own CLIP guidance.
    """
    step = min(max(round(t), 0), len(log_alpha_bar) - 1)  # Its timestep embedding is a table.
    sigma = math.sqrt(1 - math.exp(log_alpha_bar[step]))
    with th.enable_grad():
        noisy = x.detach().float().requires_grad_()
        timesteps = th.full((x.shape[0],), step, device=x.device, dtype=th.long)
        logit = model.logit_scale.exp() * (model.image_embeddings(noisy, timesteps) * text).sum()
        gradient = th.autograd.grad(logit, noisy)[0]
    return -sigma * gradient, th.zeros_like(gradient)


# Karlo.


KARLO = "kakaobrain/karlo-v1-alpha"  # An open reproduction of unCLIP (DALL-E 2), trained on 115M captioned images.
KARLO_PRIOR_SCALE = 4.0  # The classifier-free guidance of its prior and the steps it takes, as its pipeline has them.
KARLO_PRIOR_STEPS = 25
KARLO_CLIP = "ViT-L/14"  # The CLIP model whose image embeddings its prior draws.


class Karlo(NamedTuple):
    tokenizer: Any
    text_encoder: Any
    prior: Any
    prior_scheduler: Any
    text_proj: Any
    decoder: Any
    log_alpha_bar: np.ndarray  # Of the decoder's schedule, a cosine one.


class KarloText(NamedTuple):
    """What the decoder is conditioned on: a row for the prompt, then a row for the empty prompt and no image embedding."""

    hidden: th.Tensor
    additive: th.Tensor
    mask: th.Tensor
    image: th.Tensor  # [1, features]: the CLIP image embedding the prior drew for the prompt.


@lru_cache(maxsize=1)
def load_karlo(device: str) -> Karlo:
    """Karlo's text encoder, prior and 64x64 decoder, frozen, at half precision on a GPU.

    The prior is only needed once an image, so it waits on the CPU and karlo_text() borrows the device for it.
    """
    from diffusers import PriorTransformer, UNet2DConditionModel, UnCLIPScheduler
    from diffusers.pipelines.deprecated.unclip.text_proj import UnCLIPTextProjModel
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer

    dtype = th.float32 if device == "cpu" else th.float16

    def part(kind, folder: str):
        return kind.from_pretrained(KARLO, subfolder=folder, use_safetensors=True).eval().requires_grad_(False).to(dtype)

    with th.random.fork_rng(devices=[]):  # Building them may draw random numbers, which must not move the seed's stream.
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
def karlo_text(karlo: Karlo, prompt: str, device: str, generator: th.Generator) -> KarloText:
    """The decoder's conditioning for a prompt: its text, and a CLIP image embedding the prior draws for it."""
    tokens = karlo.tokenizer(
        [prompt, ""], padding="max_length", max_length=karlo.tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )
    mask = tokens.attention_mask.bool().to(device)
    encoded = karlo.text_encoder(tokens.input_ids.to(device))
    embeds, hidden = encoded.text_embeds, encoded.last_hidden_state
    prior = karlo.prior.to(device)
    karlo.prior_scheduler.set_timesteps(KARLO_PRIOR_STEPS, device=device)
    timesteps = karlo.prior_scheduler.timesteps
    latents = th.randn((1, prior.config.embedding_dim), device=device, generator=generator).to(embeds.dtype)
    for i, t in enumerate(timesteps):
        predicted = prior(
            latents.repeat(2, 1), timestep=t, proj_embedding=embeds, encoder_hidden_states=hidden, attention_mask=mask
        ).predicted_image_embedding
        predicted = predicted[1:] + KARLO_PRIOR_SCALE * (predicted[:1] - predicted[1:])
        following = timesteps[i + 1] if i + 1 < len(timesteps) else None
        latents = karlo.prior_scheduler.step(
            predicted, timestep=t, sample=latents, generator=generator, prev_timestep=following
        ).prev_sample
    image_embedding = prior.post_process_latents(latents)
    prior.to("cpu")
    # The projection puts its learned embedding of no image first, so it takes the empty prompt first.
    hidden, additive = karlo.text_proj(
        image_embeddings=image_embedding,
        prompt_embeds=embeds.flip(0),
        text_encoder_hidden_states=hidden.flip(0),
        do_classifier_free_guidance=True,
    )
    mask = tf.pad(mask.flip(0), (karlo.text_proj.clip_extra_context_tokens, 0), value=True)
    return KarloText(hidden.flip(0), additive.flip(0), mask.flip(0), image_embedding.float())


@lru_cache(maxsize=1)
def load_karlo_clip(device: str):
    """Karlo's CLIP model and its input size, kept apart from load_clip()'s cache of one, which the guide's models take turns in."""
    return load_clip.__wrapped__(KARLO_CLIP, device)


def towards_embedding(guide: Guide, embedding: th.Tensor, device: str, alone: bool) -> Guide:
    """The guide with one more tower, or with only it: Karlo's CLIP model, scoring images against an image embedding."""
    model, size = load_karlo_clip(device)
    tower = Tower(model, size, embedding.to(device), th.ones(1, device=device))
    return guide._replace(towers=(tower,) if alone else (*guide.towers, tower))


def karlo_eps(decoder, text: KarloText, x: th.Tensor, t: float) -> tuple[th.Tensor, th.Tensor]:
    """The decoder's noise prediction for each image of x at its timestep t, with the prompt and without."""
    count = x.shape[0]
    eps = decoder(
        sample=x.repeat(2, 1, 1, 1).to(text.hidden.dtype),
        timestep=th.full((2 * count,), t, device=x.device, dtype=th.float32),
        encoder_hidden_states=text.hidden.repeat_interleave(count, dim=0),
        class_labels=text.additive.repeat_interleave(count, dim=0),
        attention_mask=text.mask.repeat_interleave(count, dim=0),
    ).sample
    eps = eps[:, :3].float()  # The other three channels are the variance it learned.
    return eps[:count], eps[count:]


# layout_prior.LayoutPrior.


@lru_cache(maxsize=1)
def load_layout_prior(checkpoint: str, device: str):
    from layout_prior import load

    return load(checkpoint, device)[0]


def layout_text(prompt: str, device: str) -> th.Tensor:
    """The prompt's text embeddings as LayoutPrior takes them."""
    from build_layout_data import TEXT_MODELS
    from layout_prior import join_text

    return join_text([encode_texts([prompt], name, device, True, truncate=True) for name in TEXT_MODELS])


@th.no_grad()
def layout_predictions(model, text: th.Tensor, y: th.Tensor, signal: float, noise: float):
    """eps(y | prompt) and eps(y) from a LayoutPrior, a rectified flow: x_t = (1 - t) * image + t * eps."""
    t = noise / (signal + noise)
    x = y / (signal + noise)
    times = th.full((1,), t, device=y.device)
    with th.autocast(y.device.type, dtype=th.bfloat16, enabled=y.device.type == "cuda"):
        conditional = model(x, times, text).float()
        unconditional = model(x, times, text, th.zeros(1, dtype=th.bool, device=y.device)).float()
    return x + (1 - t) * conditional, x + (1 - t) * unconditional  # eps = x_t + (1 - t) * velocity.


# The pipeline.


class Limits(NamedTuple):
    """How a thumbnail model's push is held back. The defaults leave it as it is."""

    parallel: float = 1.0
    limit: float | None = None
    cap: float | None = None


def limited_push(push: th.Tensor, x0: th.Tensor, eps: th.Tensor, now: sampler.Level, limits: Limits) -> th.Tensor:
    """What a thumbnail model adds to the noise prediction of a thumbnail, held back in three ways.

    `x0` is the denoised image's thumbnail and `eps` the noise prediction of the whole image,
    both without the push, at noise level `now`.
    The part of the push along x0 only scales the image, its contrast and saturation with
    it, and `parallel` is how much of that part is kept: adaptive projected guidance
    (Sadat et al. 2024, arXiv 2410.02416), which found that the rest is what improves an image.
    `limit` is a sampler.limit_guidance() quantile, `-gl` for the thumbnail: only as much of
    the push as keeps that share of the denoised thumbnail's pixels in range, or no further
    out of it than the U-Net's own already are. At high noise a push that is small next to
    the noise moves the denoised thumbnail by more than its whole range, and the sampler's
    clamp then flattens it into a poster.
    `cap` is then the most its RMS may be, relative to the noise prediction's, as
    sampler.cap_guidance() caps CLIP's.
    """
    along = (push * x0).sum(dim=(1, 2, 3), keepdim=True) / x0.square().sum(dim=(1, 2, 3), keepdim=True)
    push = push - (1 - limits.parallel) * along * x0
    if limits.limit is not None:
        reach = -now.sigma / now.alpha  # What a push on the noise prediction does to the denoised image.
        bound = th.quantile(x0.abs().flatten(1), limits.limit, dim=1).clamp(min=1).view(-1, 1, 1, 1)
        push = (sampler.limit_guidance(x0 / bound, reach * push / bound, limits.limit) * bound - x0) / reach
    if limits.cap is not None:
        push = push * (limits.cap * sampler.rms(eps) / sampler.rms(push)).clamp(max=1)
    return push


def guided_denoise(
    denoise,
    log_alpha_bar: np.ndarray,
    predictions: Predictions | None,
    scale: float,
    above: float,
    x: th.Tensor,
    t: th.Tensor,
    replace: bool = False,
    limits: Limits = Limits(),
) -> th.Tensor:
    """`denoise`'s noise prediction, corrected by the thumbnail model above timestep `above`.

    By default the thumbnail model's guidance direction, times `scale`, is added to it. With
    `replace`, the part of the prediction that the thumbnail holds, each 4x4 block's mean,
    is instead taken from the thumbnail model's own classifier-free guided prediction, so
    the thumbnail of x follows that model alone and the U-Net only draws the detail.
    `limits` are limited_push()'s, for what is added.
    """
    eps = denoise(x, t)[:, :3]
    timestep = float(t[0])
    if predictions is None or not scale or timestep < above:
        return eps
    now = sampler.level(log_alpha_bar, timestep)
    y = tf.avg_pool2d(x.detach().float(), FACTOR)
    conditional, unconditional = predictions(y, now.alpha, now.sigma / FACTOR)

    def enlarged(thumbnail: th.Tensor) -> th.Tensor:
        return tf.interpolate(thumbnail, scale_factor=FACTOR, mode="nearest").to(eps.dtype)

    if replace:
        guided = unconditional + scale * (conditional - unconditional)
        return eps - enlarged(tf.avg_pool2d(eps, FACTOR)) + enlarged(guided) / FACTOR
    if limits == Limits():
        return eps + (scale / FACTOR) * enlarged(conditional - unconditional)
    x0 = (y - now.sigma * tf.avg_pool2d(eps.detach().float(), FACTOR)) / now.alpha  # The denoised image's thumbnail.
    push = (scale / FACTOR) * (conditional - unconditional).float()
    return eps + enlarged(limited_push(push, x0, eps.detach().float(), now, limits))


def consistent_loss(guide, thumbnail: th.Tensor, weight: float, x: th.Tensor, x0: th.Tensor, sigma: float):
    """adm.guidance_loss(), plus `weight` times the squared distance of the denoised image's thumbnail to the one drawn.

    It is what makes the U-Net an upsampler of the thumbnail (as in DPS super-resolution) instead of
    only starting from it: the denoised image, averaged down, has to stay the thumbnail.
    """
    small = tf.adaptive_avg_pool2d(x0.float(), thumbnail.shape[-1])
    return adm.guidance_loss(guide, x, x0, sigma) + weight * (small - thumbnail).square().sum()


def no_pull(_x: th.Tensor, x0: th.Tensor, _sigma: float) -> th.Tensor:
    """A guidance loss with no gradient: its steps still recur as guided ones do, but CLIP has no say in them."""
    return (x0 * 0).sum()


def generate(  # pylint: disable=too-many-locals
    prompts: Sequence[str] = (),
    image_prompts: Sequence[str] = (),
    *,
    prior: str = "glide",
    scale: float = 3.0,
    above: float = 0.0,
    handoff: float | None = None,
    handoff_scale: float = 3.0,
    handoff_side: int = 64,
    replace: bool = False,
    parallel: float = 1.0,
    limit: float | None = None,
    cap: float | None = None,
    target: str = "text",
    clip: bool = True,
    candidates: int = 1,
    consistency: float = 0.0,
    pag_scale: float = 0.0,
    window: Sequence[float] = sampler.UNCONDITIONAL_GUIDANCE_WINDOW,
    aesthetic_scale: float = 0.0,
    steps: int = sampler.UNCONDITIONAL_STEPS,
    recurrence: int = sampler.UNCONDITIONAL_RECURRENCE,
    guidance_rate: float = sampler.DEFAULT_GUIDANCE_RATE,
    clip_model: str = adm.DEFAULT_CLIP_MODEL,
    num_cutouts: int = NUM_CUTOUTS,
    seed: int = 0,
    output_dir: str | Path = "outputs",
    save_frequency: int = 10,
    progress: bool = False,
    device: str = "cuda",
) -> Iterator[tuple[int, Path]]:
    """adm.generate() for the unconditional 256 checkpoint, with a thumbnail model setting the composition.

    `prior` is "glide", "laionide", "karlo", "glide-clip" (the CLIP of noisy images, which guides but cannot
    draw a thumbnail to hand off) or the path of a LayoutPrior checkpoint. `scale` is the strength of
    thumbnail guidance at each call above timestep `above` (0 for none), and `replace` is
    guided_denoise()'s. `parallel`, `limit` and `cap` are the Limits on what it adds. With `handoff`,
    the thumbnail model first draws a whole thumbnail with classifier-free guidance of
    `handoff_scale`, and sampling starts from it at that timestep instead of from noise.
    With `candidates` it draws that many and starts from the one the guide's CLIP models
    score closest to the prompts, which costs little at 64x64. `consistency` then also
    holds the denoised image's own thumbnail to it, with that weight. `pag_scale` adds
    perturbed-attention guidance (self_guidance.py), `window` is where CLIP guides, as
    fractions of the schedule, and `aesthetic_scale` is adm.generate()'s. `target` is what CLIP
    guidance pulls towards: "text", the prompts as always, "embedding", only the image embedding
    Karlo's prior drew for the prompt, or "both", that embedding as one more model of the ensemble.
    Without `clip`, CLIP guidance has no say but its steps still recur, which a `guidance_rate` of 0 would also stop.
    """
    output_dir = Path(output_dir)
    eps = embedding = None
    if prior in ("glide", "laionide"):
        glide, prior_schedule, options = load_glide(device, prior)
        eps = partial(glide_eps, glide, glide_text(glide, options, prompts[0], device))
        prior_ts = sampler.timesteps(prior_schedule, 50)
    elif prior == "glide-clip":
        assert handoff is None, "A CLIP cannot draw the thumbnail a hand-off starts from."
        noised_clip, prior_schedule = load_noised_clip(device)
        with th.no_grad():
            eps = partial(noised_clip_eps, noised_clip, noised_clip.text_embeddings([prompts[0]]), prior_schedule)
    elif prior == "karlo":
        karlo = load_karlo(device)
        embedding_noise = th.Generator(device).manual_seed(seed + 2 * PRIOR_SEED_OFFSET)
        karlo_conditioning = karlo_text(karlo, prompts[0], device, embedding_noise)
        eps = partial(karlo_eps, karlo.decoder, karlo_conditioning)
        embedding = karlo_conditioning.image
        prior_schedule = karlo.log_alpha_bar
        prior_ts = list(np.linspace(len(prior_schedule) - 1, 0, 51))  # Its cosine schedule is made for uniform timesteps.
    else:
        layout_model = load_layout_prior(prior, device)
        text = layout_text(prompts[0], device)
        predictions = partial(layout_predictions, layout_model, text)
    if eps is not None:
        predictions = partial(matched_predictions, eps, prior_schedule)
    assert target == "text" or embedding is not None, "Only Karlo's prior draws an image embedding."
    # From here on in adm.generate()'s order. Drawing a thumbnail uses its own generator, so the seed's stream is untouched.
    th.manual_seed(seed)
    guide = load_guide(
        clip_model, prompts, image_prompts, image_size=256, num_cutouts=num_cutouts, device=device,
        aesthetic_scale=aesthetic_scale,
    )  # fmt: skip
    if target != "text":
        guide = towards_embedding(guide, embedding, device, alone=target == "embedding")
    thumbnail = None
    if handoff is not None:
        generator = th.Generator(device).manual_seed(seed + PRIOR_SEED_OFFSET)
        if eps is not None:
            drawn = drawn_thumbnails(eps, prior_schedule, prior_ts, handoff_scale, generator, candidates)
        else:
            from layout_prior import sample

            drawn = sample(layout_model, text.expand(candidates, -1), scale=handoff_scale, generator=generator)
        enlarged = tf.interpolate(drawn.float(), (256, 256), mode="bicubic", align_corners=False).clamp(-1, 1)
        best = int(score(guide, enlarged).argmax()) if candidates > 1 else 0
        thumbnail = drawn[best : best + 1]
        if handoff_side != thumbnail.shape[-1]:
            thumbnail = tf.adaptive_avg_pool2d(thumbnail, handoff_side)
        output_dir.mkdir(parents=True, exist_ok=True)
        tvf.to_pil_image(thumbnail[0].add(1).div(2).clamp(0, 1).cpu()).save(output_dir / "thumbnail.png")
    model, log_alpha_bar = adm.load_model(256, device, fp16=True, use_checkpoint=False, unconditional=True)
    ts = sampler.timesteps(log_alpha_bar, steps, handoff)
    x = th.randn((1, 3, 256, 256), device=device)
    if thumbnail is not None:
        init = tf.interpolate(thumbnail.float(), (256, 256), mode="bicubic", align_corners=False).clamp(-1, 1)
        x = sampler.noised(log_alpha_bar, ts[0], init, x)
    loss_fn = partial(adm.guidance_loss, guide)
    if thumbnail is not None and consistency:
        loss_fn = partial(consistent_loss, guide, thumbnail.float(), consistency)
    if not clip:
        loss_fn = no_pull
    guidance = sampler.Guidance(loss_fn, guidance_rate, tuple(window), None, recurrence)
    unet = partial(adm.denoise, model, None)
    if pag_scale:
        from self_guidance import PAG_LAYERS, pag_denoise

        unet = partial(pag_denoise, model, pag_scale, PAG_LAYERS)
    limits = Limits(parallel, limit, cap)
    denoise = partial(guided_denoise, unet, log_alpha_bar, predictions, scale, above, replace=replace, limits=limits)
    samples = sampler.sample(denoise, log_alpha_bar, ts, x, guidance, progress=progress)
    yield from save_frames(samples, output_dir, run_name(prompts, image_prompts), save_frequency, len(ts) - 2)


# GLIDE on its own, as a reference.


@lru_cache(maxsize=1)
def load_glide_upsampler(device: str):
    """GLIDE's own 64 to 256 upsampler, frozen, and its 27-step diffusion."""
    from glide_text2im.download import load_checkpoint
    from glide_text2im.model_creation import create_model_and_diffusion, model_and_diffusion_defaults_upsampler

    options = model_and_diffusion_defaults_upsampler() | {"use_fp16": device != "cpu", "timestep_respacing": "fast27"}
    with th.random.fork_rng(devices=[]):
        model, diffusion = create_model_and_diffusion(**options)
    model.load_state_dict(load_checkpoint("upsample", th.device("cpu"), cache_dir=str(CACHE_PATH / "glide")))
    model.eval().requires_grad_(False)
    if options["use_fp16"]:
        model.convert_to_fp16()
    return model.to(device), diffusion


@th.no_grad()
def glide_alone(
    prompts: Sequence[str] = (),
    image_prompts: Sequence[str] = (),
    *,
    weights: str = "glide",
    scale: float = 3.0,
    upsample: bool = True,
    seed: int = 0,
    output_dir: str | Path = "outputs",
    save_frequency: int = 10,  # pylint: disable=unused-argument
    progress: bool = False,
    device: str = "cuda",
) -> Iterator[tuple[int, Path]]:
    """What GLIDE makes without the unconditional U-Net: the thumbnail generate() hands off, enlarged
    by GLIDE's own upsampler as its paper does, or only bicubically, which is the bare thumbnail."""
    glide, schedule, options = load_glide(device, weights)
    text = glide_text(glide, options, prompts[0], device)
    generator = th.Generator(device).manual_seed(seed + PRIOR_SEED_OFFSET)
    thumbnail = glide_thumbnail(glide, schedule, text, scale, 50, generator)
    if upsample:
        upsampler, diffusion = load_glide_upsampler(device)
        th.manual_seed(seed)
        image = diffusion.ddim_sample_loop(
            upsampler,
            (1, 3, 256, 256),
            noise=th.randn((1, 3, 256, 256), device=device) * 0.997,  # The temperature of GLIDE's own notebook.
            clip_denoised=True,
            model_kwargs={
                "low_res": ((thumbnail + 1) * 127.5).round() / 127.5 - 1,
                "tokens": text["tokens"][:1],
                "mask": text["mask"][:1],
            },
            device=device,
            progress=progress,
        )
    else:
        image = tf.interpolate(thumbnail.float(), (256, 256), mode="bicubic", align_corners=False).clamp(-1, 1)
    yield 0, save_frame(image[0].float(), frame_path(output_dir, run_name(prompts, image_prompts), 0, 0))

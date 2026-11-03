"""CLIP guidance on modern one-step models.

CLIP guidance can steer a model but cannot supply what the model never learned, and the
ADM checkpoints only learned ImageNet. Everything about cutouts, losses and solvers
together moves their images less than swapping the model. So here the model is the
source of the image, and CLIP corrects it by optimising the initial noise through one
step of the model and a tiny decoder (ReNO, Eyring et al. 2024). There is no solver or
schedule to retune, and the noise stays on the prior, so every candidate along the way
is a normal output of the model. This is a better fit for CLIP than per-step guidance:
a one-step model has no steps to guide, and the gradient cannot leave its manifold.
"""

import gc
import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import torch as th
from torch import nn
from tqdm.auto import tqdm

from cgd.checks import check_common, require
from cgd.clip_models import check_model
from cgd.device import default_device
from cgd.guide import Guide, guide_loss, load_guide
from cgd.losses import finite_gradient
from cgd.outputs import frame_path, run_name, save_frame
from cgd.prompts import parse_prompt

DEFAULT_CLIP_MODEL = "ViT-B/32+PE-Core-B-16"
DEFAULT_ITERATIONS = 50
DEFAULT_AESTHETIC_SCALE = 0.05
# ReNO's optimiser settings.
LEARNING_RATE = 5.0
MOMENTUM = 0.9
MAX_GRAD_NORM = 0.1
NOISE_PENALTY = 0.01  # How hard to keep the noise on the prior.
LATENT_FACTOR = 8  # Pixels per latent, along each side.
LATENT_CHANNELS = 4
FINAL_TIMESTEP = 999  # One step from here, pure noise, gives the image.


@dataclass(frozen=True)
class Spec:
    """Where a backbone's parts come from."""

    repo: str
    family: str  # sd or sdxl, which sets the pipeline that loads it.
    size: int  # What the checkpoint was trained for.
    prediction: str = "epsilon"  # What the U-Net outputs: epsilon, or sample for x0.
    tiny_decoder: str = "madebyollin/taesd"
    vae_repo: str | None = None  # Replaces the checkpoint's own decoder.
    lora: str | None = None
    unet_weights: tuple[str, str] | None = None  # (repo, filename) of a U-Net to load over the checkpoint's own.


BACKBONES = {
    "sd-turbo": Spec("stabilityai/sd-turbo", "sd", 512),
    "vega": Spec(
        "segmind/Segmind-Vega",
        "sdxl",
        1024,
        tiny_decoder="madebyollin/taesdxl",
        vae_repo="madebyollin/sdxl-vae-fp16-fix",
        lora="segmind/Segmind-VegaRT",
    ),
    "sdxl-lightning": Spec(
        "stabilityai/stable-diffusion-xl-base-1.0",
        "sdxl",
        1024,
        prediction="sample",
        tiny_decoder="madebyollin/taesdxl",
        vae_repo="madebyollin/sdxl-vae-fp16-fix",
        unet_weights=("ByteDance/SDXL-Lightning", "sdxl_lightning_1step_unet_x0.safetensors"),
    ),
}


class Backbone(NamedTuple):
    """A loaded one-step model, with the prompt it draws already encoded."""

    spec: Spec
    unet: nn.Module
    vae: nn.Module
    tiny_decoder: nn.Module  # What guidance backpropagates through: as good, in half the time and 1 GiB less at 512.
    text: th.Tensor  # The prompt's embedding, as the U-Net takes it.
    pooled: th.Tensor | None  # SDXL's pooled text embedding.
    alpha_bar: th.Tensor
    dtype: th.dtype
    device: str


class Iteration(NamedTuple):
    """One step of optimise_noise()."""

    images: th.Tensor  # What the noise renders to, before this step changed it.
    loss: float
    best_z: th.Tensor  # The noise that scored best so far.


def load_backbone(name: str, text: str, device: str) -> Backbone:
    """The backbone called `name`, drawing `text`."""
    # Slow to import, and only --backbone needs them.
    from diffusers import AutoencoderKL, AutoencoderTiny, EulerDiscreteScheduler

    spec = BACKBONES[name]
    dtype = th.float16 if device != "cpu" else th.float32
    unet, vae, embeds, pooled = _LOADERS[spec.family](spec, text, device, dtype)
    gc.collect()  # The text encoders have done their job, and the graph needs that VRAM.
    if device != "cpu":
        th.cuda.empty_cache()
    if spec.vae_repo:
        vae = AutoencoderKL.from_pretrained(spec.vae_repo, dtype=dtype)
    unet, vae = _freeze(unet, device, dtype), _freeze(vae, device, dtype)
    unet.enable_gradient_checkpointing()
    tiny_decoder = AutoencoderTiny.from_pretrained(spec.tiny_decoder, dtype=dtype)
    tiny_decoder = tiny_decoder.to(device).eval().requires_grad_(False)
    scheduler = EulerDiscreteScheduler.from_pretrained(spec.repo, subfolder="scheduler")
    alpha_bar = scheduler.alphas_cumprod.to(device, th.float32)
    return Backbone(spec, unet, vae, tiny_decoder, embeds, pooled, alpha_bar, dtype, device)


def _freeze(module: nn.Module, device: str, dtype: th.dtype) -> nn.Module:
    """A diffusers model on `device`, frozen, with plain attention.

    The fused attention kernels have a nondeterministic backward pass, like open_clip's.
    """
    from diffusers.models.attention_processor import AttnProcessor

    module = module.to(device, dtype).eval().requires_grad_(False)
    module.set_attn_processor(AttnProcessor())
    return module


def _load_sd(spec: Spec, text: str, device: str, dtype: th.dtype) -> tuple[nn.Module, nn.Module, th.Tensor, None]:
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        spec.repo,
        variant="fp16",
        dtype=dtype,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to(device)
    embeds = pipe.encode_prompt(text, device, 1, False)[0]
    return pipe.unet, pipe.vae, embeds.detach(), None


def _load_sdxl(spec: Spec, text: str, device: str, dtype: th.dtype) -> tuple[nn.Module, None, th.Tensor, th.Tensor]:
    from diffusers import StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_pretrained(spec.repo, variant="fp16", dtype=dtype, vae=None)
    if spec.lora:
        pipe.load_lora_weights(spec.lora)
        pipe.fuse_lora()
        pipe.unload_lora_weights()
    if spec.unet_weights:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        pipe.unet.load_state_dict(load_file(hf_hub_download(*spec.unet_weights)))
    pipe.text_encoder.to(device)
    pipe.text_encoder_2.to(device)
    embeds, _, pooled, _ = pipe.encode_prompt(text, device=device, do_classifier_free_guidance=False)
    return pipe.unet, None, embeds.detach(), pooled.detach()


_LOADERS = {"sd": _load_sd, "sdxl": _load_sdxl}


def latent_shape(height: int, width: int) -> tuple[int, int, int, int]:
    """The shape of the noise for one image of this size."""
    return 1, LATENT_CHANNELS, height // LATENT_FACTOR, width // LATENT_FACTOR


def unet_conditioning(backbone: Backbone, height: int, width: int) -> dict[str, Any]:
    """What the U-Net takes besides its input and timestep, for images of this size."""
    conditioning: dict[str, Any] = {"encoder_hidden_states": backbone.text}
    if backbone.pooled is not None:  # SDXL is also told the size, and that the image is uncropped.
        time_ids = th.tensor([[height, width, 0, 0, height, width]], device=backbone.device, dtype=backbone.dtype)
        conditioning["added_cond_kwargs"] = {"text_embeds": backbone.pooled, "time_ids": time_ids}
    return conditioning


def predicted_x0(z: th.Tensor, output: th.Tensor, prediction: str, alpha_bar: th.Tensor) -> th.Tensor:
    """The clean latents a U-Net's output stands for, from the noisy latents z."""
    if prediction == "sample":
        return output
    return (z - (1 - alpha_bar).sqrt() * output) / alpha_bar.sqrt()


def denoise(backbone: Backbone, conditioning: dict[str, Any], z: th.Tensor) -> th.Tensor:
    """The clean latents for pure noise z, in one model call. Gradients flow through this."""
    timestep = th.tensor(FINAL_TIMESTEP, device=backbone.device)
    output = backbone.unet(z.to(backbone.dtype), timestep, **conditioning).sample.float()
    return predicted_x0(z, output, backbone.spec.prediction, backbone.alpha_bar[FINAL_TIMESTEP])


def decode(backbone: Backbone, latents: th.Tensor) -> th.Tensor:
    """Images in [-1, 1], from the full decoder."""
    latents = (latents / backbone.vae.config.scaling_factor).to(backbone.dtype)
    return backbone.vae.decode(latents).sample.float().clamp(-1, 1)


def decode_preview(backbone: Backbone, latents: th.Tensor) -> th.Tensor:
    """Images in [-1, 1], from the tiny decoder."""
    return backbone.tiny_decoder.decode(latents.to(backbone.dtype)).sample.float().clamp(-1, 1)


def preview(backbone: Backbone, conditioning: dict[str, Any], z: th.Tensor) -> th.Tensor:
    """What guidance sees of the noise z: the image through the tiny decoder."""
    return decode_preview(backbone, denoise(backbone, conditioning, z))


@th.no_grad()
def render(backbone: Backbone, conditioning: dict[str, Any], z: th.Tensor) -> th.Tensor:
    """The finished image of the noise z, through the full decoder."""
    return decode(backbone, denoise(backbone, conditioning, z))


def noise_penalty(z: th.Tensor) -> th.Tensor:
    """Negative log density of each noise's norm under the Gaussian prior, a chi distribution, up to a constant."""
    norm = z.flatten(1).norm(dim=1)
    return (0.5 * norm**2 - (z[0].numel() - 1) * norm.log()).sum()


def optimise_noise(
    draw: Callable[[th.Tensor], th.Tensor],
    loss_fn: Callable[[th.Tensor], th.Tensor],
    z: th.Tensor,
    iterations: int = DEFAULT_ITERATIONS,
    progress: bool = True,
) -> Iterator[Iteration]:
    """Nesterov SGD on the noise z, to minimise the loss of the images `draw` makes of it.

    The noise penalty keeps z on the prior, so it stays a normal input of the model.
    The last iteration only scores, since its step would never be seen.
    """
    z = z.clone().requires_grad_()
    optimiser = th.optim.SGD([z], lr=LEARNING_RATE, momentum=MOMENTUM, nesterov=True)
    best_z, best_loss = z.detach().clone(), math.inf
    progress_bar = tqdm(range(iterations), disable=not progress)
    for iteration in progress_bar:
        images = draw(z)
        loss = loss_fn(images)
        value = loss.item()
        progress_bar.set_postfix(loss=f"{value:.3f}", refresh=False)
        if value < best_loss:
            best_z, best_loss = z.detach().clone(), value
        if iteration < iterations - 1:
            z.grad = finite_gradient(loss + NOISE_PENALTY * noise_penalty(z), z)
            th.nn.utils.clip_grad_norm_([z], MAX_GRAD_NORM)
            optimiser.step()
        yield Iteration(images.detach(), value, best_z)


def conditioning_text(prompts: Sequence[str]) -> str:
    """What the model is told to draw: every prompt being rewarded. Only CLIP sees the rest."""
    return ", ".join(text for text, weight in map(parse_prompt, prompts) if weight > 0)


def check_parameters(
    prompts: Sequence[str],
    image_prompts: Sequence[str],
    *,
    height: int,
    width: int,
    batch_size: int,
    iterations: int,
    aesthetic_scale: float,
    save_frequency: int,
) -> None:
    """Reject a configuration that cannot work, before loading anything."""
    check_common(
        prompts, image_prompts, batch_size=batch_size, save_frequency=save_frequency, aesthetic_scale=aesthetic_scale
    )
    require(
        height > 0 and width > 0 and not height % LATENT_FACTOR and not width % LATENT_FACTOR,
        f"Height and width must be positive multiples of {LATENT_FACTOR}. Got {height}x{width}.",
    )
    require(iterations >= 0, "--iterations must not be negative.")


def generate(  # pylint: disable=too-many-locals  # Its keyword arguments are the command's options.
    prompts: Sequence[str] = (),
    image_prompts: Sequence[str] = (),
    *,
    backbone: str = "sd-turbo",
    image_size: int | None = None,
    width_offset: int = 0,
    height_offset: int = 0,
    batch_size: int = 1,
    iterations: int = DEFAULT_ITERATIONS,
    clip_model: str = DEFAULT_CLIP_MODEL,
    clip_fp16: bool = False,
    aesthetic_scale: float = DEFAULT_AESTHETIC_SCALE,
    seed: int = 0,
    output_dir: str | Path = "outputs",
    save_frequency: int = 10,
    device: str | None = None,
    progress: bool = True,
) -> Iterator[tuple[int, Path]]:
    """Make images of the prompts with `backbone`, and yield (batch index, path) of each one saved, like adm.generate().

    `image_size` defaults to what the backbone was trained for. An image is saved every
    `save_frequency` iterations, and the result always last. 0 iterations is the plain model.
    """
    require(backbone in BACKBONES, f"Unknown backbone {backbone!r}. Choose one of {list(BACKBONES)}.")
    size = BACKBONES[backbone].size if image_size is None else image_size
    height, width = size + height_offset, size + width_offset
    check_parameters(
        prompts,
        image_prompts,
        height=height,
        width=width,
        batch_size=batch_size,
        iterations=iterations,
        aesthetic_scale=aesthetic_scale,
        save_frequency=save_frequency,
    )
    check_model(clip_model)
    device = default_device(device)

    # Everything from here on draws random numbers in this order, so a seed always makes the same image.
    th.manual_seed(seed)
    model = load_backbone(backbone, conditioning_text(prompts), device)
    guide = None  # The plain model never uses it, and at 1024 it needs the room.
    if iterations:
        size = min(height, width)
        guide = load_guide(
            clip_model,
            prompts,
            image_prompts,
            image_size=size,
            device=device,
            fp16=clip_fp16,
            aesthetic_scale=aesthetic_scale,
        )
    conditioning = unet_conditioning(model, height, width)
    name = run_name(prompts, image_prompts)
    # The backward pass of the U-Net's widest convolutions is nondeterministic below about
    # 448px unless cuDNN is told otherwise, which makes a seed produce a different image
    # every run. Its algorithm of choice for those shapes is no faster.
    with th.backends.cudnn.flags(enabled=True, deterministic=True):
        for index in range(batch_size):
            z = th.randn(latent_shape(height, width), device=device)
            frames = corrected_frames(
                model, conditioning, guide, z, iterations=iterations, save_frequency=save_frequency, progress=progress
            )
            for step, image in frames:
                yield index, save_frame(image, frame_path(output_dir, name, step, index))


def corrected_frames(
    backbone: Backbone,
    conditioning: dict[str, Any],
    guide: Guide | None,
    z: th.Tensor,
    *,
    iterations: int,
    save_frequency: int,
    progress: bool,
) -> Iterator[tuple[int, th.Tensor]]:
    """(iteration, image) of every `save_frequency`th iteration of correcting the noise z, then of the result.

    The result is the best noise found, through the full decoder, as iteration `iterations`.
    """
    draw, loss_fn = partial(preview, backbone, conditioning), partial(guide_loss, guide)
    best_z = z
    for iteration, step in enumerate(optimise_noise(draw, loss_fn, z, iterations, progress)):
        best_z = step.best_z
        if iteration % save_frequency == 0:
            yield iteration, step.images[0]
    yield iterations, render(backbone, conditioning, best_z)[0]

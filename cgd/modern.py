"""CLIP guidance on modern one-step models.

CLIP guidance can steer a model but cannot supply what the model never learned, and the
ADM checkpoints only learned ImageNet. Everything about cutouts, losses and solvers
together moves their images less than swapping the model. So here the model is the
source of the image, and CLIP corrects it by optimising the initial noise through one
step of the model and its decoder (ReNO, Eyring et al. 2024). There is no solver or
schedule to retune, and the noise stays on the prior, so every candidate along the way
is a normal output of the model. This is a better fit for CLIP than per-step guidance:
a one-step model has no steps to guide, and the gradient cannot leave its manifold.
"""
import gc
import math
from dataclasses import dataclass
from pathlib import Path

import torch as th
from tqdm.auto import tqdm

from cgd import clip_util, script_util

DEFAULT_CLIP_MODEL = "ViT-B/32+PE-Core-B-16"
DEFAULT_ITERATIONS = 50
DEFAULT_LR = 5.0
DEFAULT_GRAD_NORM = 0.1
DEFAULT_NOISE_PENALTY = 0.01
DEFAULT_NUM_CUTOUTS = 4
MIN_CUT_FRACTION = 0.7
DEFAULT_AESTHETIC_SCALE = 0.05


@dataclass(frozen=True)
class BackboneConfig:
    repo: str
    family: str
    size: int  # What the checkpoint is trained for.
    prediction: str = "epsilon"  # What the U-Net outputs: epsilon, or sample for x0.
    tiny_decoder: str = "madebyollin/taesd"
    vae_repo: str = None  # Replaces the checkpoint's own decoder.
    lora: str = None
    unet_weights: tuple = None  # (repo, filename) of a U-Net to load over the checkpoint's own.


BACKBONES = {
    "sd-turbo": BackboneConfig("stabilityai/sd-turbo", "sd", 512),
    "vega": BackboneConfig("segmind/Segmind-Vega", "sdxl", 1024, tiny_decoder="madebyollin/taesdxl",
                           vae_repo="madebyollin/sdxl-vae-fp16-fix", lora="segmind/Segmind-VegaRT"),
    "sdxl-lightning": BackboneConfig("stabilityai/stable-diffusion-xl-base-1.0", "sdxl", 1024, prediction="sample", tiny_decoder="madebyollin/taesdxl",
                                     vae_repo="madebyollin/sdxl-vae-fp16-fix",
                                     unet_weights=("ByteDance/SDXL-Lightning", "sdxl_lightning_1step_unet_x0.safetensors")),
}
LATENT_FACTOR = 8
LATENT_CHANNELS = 4


def _plain_attention(*modules):
    """Fused attention kernels have a nondeterministic backward pass, like open_clip's."""
    from diffusers.models.attention_processor import AttnProcessor
    for module in modules:
        module.set_attn_processor(AttnProcessor())


class LatentBackbone:
    """A distilled Stable Diffusion: one step from t=999 gives x0."""

    def __init__(self, config: BackboneConfig, text: str, device: str, tiny_decoder: bool = True):
        from diffusers import AutoencoderKL, AutoencoderTiny, EulerDiscreteScheduler

        self.config, self.device = config, device
        self.dtype = th.float16 if device != "cpu" else th.float32
        self.unet_kwargs = {}
        self._load(text)
        gc.collect()  # The text encoders have done their job, and the graph needs that VRAM.
        if device != "cpu":
            th.cuda.empty_cache()
        if config.vae_repo:
            self.vae = AutoencoderKL.from_pretrained(config.vae_repo, torch_dtype=self.dtype)
        self.unet = self.unet.to(device, self.dtype).eval().requires_grad_(False)
        self.vae = self.vae.to(device, self.dtype).eval().requires_grad_(False)
        _plain_attention(self.unet, self.vae)
        self.unet.enable_gradient_checkpointing()
        self.vae.enable_gradient_checkpointing()
        self.guide_decoder = None
        if tiny_decoder:
            self.guide_decoder = AutoencoderTiny.from_pretrained(config.tiny_decoder, torch_dtype=self.dtype).to(device).eval().requires_grad_(False)
        alphas_cumprod = EulerDiscreteScheduler.from_pretrained(config.repo, subfolder="scheduler").alphas_cumprod
        self.alpha_bar = alphas_cumprod.to(device, th.float32)
        self.scaling_factor = self.vae.config.scaling_factor

    def _load(self, text: str):
        """Sets self.unet, self.vae and self.text, and whatever else the U-Net takes."""
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(self.config.repo, variant="fp16", torch_dtype=self.dtype, safety_checker=None,
                                                       feature_extractor=None, requires_safety_checker=False).to(self.device)
        self.text = pipe.encode_prompt(text, self.device, 1, False)[0].detach()
        self.unet, self.vae = pipe.unet, pipe.vae

    def latent_shape(self, height: int, width: int, batch: int = 1):
        return batch, LATENT_CHANNELS, height // LATENT_FACTOR, width // LATENT_FACTOR

    def denoise(self, z: th.Tensor, t: int = 999) -> th.Tensor:
        """The clean latents for z, a sample at timestep t. Gradients flow through this."""
        timestep = th.tensor(t, device=self.device)
        output = self.unet(z.to(self.dtype), timestep, encoder_hidden_states=self.text, **self.unet_kwargs).sample.float()
        if self.config.prediction == "sample":
            return output
        alpha_bar = self.alpha_bar[t]
        return (z - (1 - alpha_bar).sqrt() * output) / alpha_bar.sqrt()

    def decode(self, latents: th.Tensor) -> th.Tensor:
        """The full decoder, as images in [-1, 1]."""
        return self.vae.decode((latents / self.scaling_factor).to(self.dtype)).sample.float().clamp(-1, 1)

    def decode_for_guidance(self, latents: th.Tensor) -> th.Tensor:
        if self.guide_decoder is None:
            return self.decode(latents)
        return self.guide_decoder.decode(latents.to(self.dtype)).sample.float().clamp(-1, 1)

    def sample(self, z: th.Tensor, steps: int = 1) -> th.Tensor:
        """The latents after `steps` model calls from z, re-noising between them."""
        ts = [999 - i * 1000 // steps for i in range(steps)]
        latents = self.denoise(z, ts[0])
        for t in ts[1:]:
            alpha_bar = self.alpha_bar[t]
            latents = self.denoise(alpha_bar.sqrt() * latents + (1 - alpha_bar).sqrt() * th.randn_like(z), t)
        return latents


class SDXLBackbone(LatentBackbone):
    def _load(self, text: str):
        from diffusers import StableDiffusionXLPipeline
        config = self.config
        pipe = StableDiffusionXLPipeline.from_pretrained(config.repo, variant="fp16", torch_dtype=self.dtype, vae=None)
        pipe.vae = None
        if config.lora:
            pipe.load_lora_weights(config.lora)
            pipe.fuse_lora()
            pipe.unload_lora_weights()
        if config.unet_weights:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            pipe.unet.load_state_dict(load_file(hf_hub_download(*config.unet_weights)))
        pipe.text_encoder.to(self.device), pipe.text_encoder_2.to(self.device)
        embeds, _, pooled, _ = pipe.encode_prompt(text, device=self.device, do_classifier_free_guidance=False)
        self.text = embeds.detach()
        self.pooled = pooled.detach()
        self.unet = pipe.unet
        self.unet_kwargs = {"added_cond_kwargs": None}

    def latent_shape(self, height: int, width: int, batch: int = 1):
        time_ids = th.tensor([[height, width, 0, 0, height, width]], device=self.device, dtype=self.dtype)
        self.unet_kwargs["added_cond_kwargs"] = {"text_embeds": self.pooled, "time_ids": time_ids}
        return super().latent_shape(height, width, batch)


def load_backbone(name: str, text: str, device: str, tiny_decoder: bool = True):
    config = BACKBONES[name]
    return {"sd": LatentBackbone, "sdxl": SDXLBackbone}[config.family](config, text, device, tiny_decoder)


def noise_penalty(z: th.Tensor) -> th.Tensor:
    """Negative log density of z's norm under the Gaussian prior: chi, up to a constant."""
    norm = z.flatten(1).norm(dim=1)
    return (0.5 * norm ** 2 - (z[0].numel() - 1) * norm.log()).sum()


def optimise_noise(backbone, guide, z: th.Tensor, iterations: int = DEFAULT_ITERATIONS, lr: float = DEFAULT_LR, grad_norm: float = DEFAULT_GRAD_NORM,
                   penalty: float = DEFAULT_NOISE_PENALTY, loss_scale: float = 1.0, progress: bool = True):
    """Yield {"images", "loss", "best_z", "last_z"} per iteration of Nesterov SGD on the noise z.

    Settings are ReNO's. The gradient is capped, so `loss_scale` only sets how much the
    penalty that keeps z on the prior counts. `best_z` is the noise that scored best.
    """
    z = z.clone().requires_grad_()
    optimiser = th.optim.SGD([z], lr=lr, momentum=0.9, nesterov=True)
    best_z, best_loss = z.detach().clone(), math.inf
    bar = tqdm(range(iterations), disable=not progress)
    for iteration in bar:
        images = backbone.decode_for_guidance(backbone.denoise(z))
        loss = guide.loss(images, loss_scale)
        value = loss.item()
        bar.set_postfix(loss=f"{value:.3f}", refresh=False)
        if value < best_loss:
            best_z, best_loss = z.detach().clone(), value
        if iteration < iterations - 1:
            z.grad = th.nan_to_num(th.autograd.grad(loss + penalty * noise_penalty(z), z)[0], nan=0.0, posinf=0.0, neginf=0.0)
            th.nn.utils.clip_grad_norm_([z], grad_norm)
            optimiser.step()
        yield {"images": images.detach(), "loss": value, "best_z": best_z, "last_z": z.detach()}


@th.no_grad()
def render(backbone, z: th.Tensor, steps: int = 1) -> th.Tensor:
    return backbone.decode(backbone.sample(z, steps))


def conditioning_text(prompts) -> str:
    """What the model is told to draw: every prompt being rewarded. Only CLIP sees the rest."""
    parsed = [script_util.parse_prompt(prompt) for prompt in prompts]
    return ", ".join(text for text, weight in parsed if weight > 0)


def check_parameters(prompts, image_prompts, backbone: str, height: int, width: int, iterations: int, render_steps: int, save_frequency: int,
                     aesthetic_scale: float = 0.0):
    """Reject a bad configuration before spending time loading anything."""
    if not (prompts or image_prompts):
        raise script_util.ConfigError("Provide at least one prompt, text or image.")
    if backbone not in BACKBONES:
        raise script_util.ConfigError(f"Unknown backbone {backbone!r}. Choose one of {list(BACKBONES)}.")
    if height <= 0 or width <= 0 or height % LATENT_FACTOR or width % LATENT_FACTOR:
        raise script_util.ConfigError(f"Height and width must be positive multiples of {LATENT_FACTOR}. Got {height}x{width}.")
    if iterations < 0:
        raise script_util.ConfigError("--iterations must not be negative.")
    if render_steps < 1:
        raise script_util.ConfigError("--render_steps must be at least 1.")
    if save_frequency < 1:
        raise script_util.ConfigError("--save_frequency must be at least 1.")
    if aesthetic_scale < 0:
        raise script_util.ConfigError("--aesthetic_scale must not be negative.")


def generate(
    prompts: "list[str]" = (),
    image_prompts: "list[str]" = (),
    backbone: str = "sd-turbo",
    image_size: int = None,  # Defaults to what the backbone is trained for.
    height_offset: int = 0,
    width_offset: int = 0,
    clip_model_name: str = DEFAULT_CLIP_MODEL,
    clip_fp16: bool = False,
    clip_loss: str = "spherical",
    aesthetic_scale: float = DEFAULT_AESTHETIC_SCALE,  # Also reward what people rate highly, per point of a 1 to 10 rating.
    num_cutouts: int = DEFAULT_NUM_CUTOUTS,
    cutout_power: float = 1.0,
    use_augs: bool = False,
    iterations: int = DEFAULT_ITERATIONS,  # 0 gives the plain model.
    lr: float = DEFAULT_LR,
    noise_penalty_scale: float = DEFAULT_NOISE_PENALTY,
    render_steps: int = 1,
    tiny_decoder: bool = True,  # Backpropagate through a small decoder. As good, in half the time and 1 GiB less at 512.
    batch_size: int = 1,
    seed: int = 0,
    prefix_path: Path = Path("./outputs"),
    save_frequency: int = 10,
    device: str = "",
    progress: bool = True,
):
    """Yield (batch_idx, image_path) like clip_guided_diffusion()."""
    if image_size is None and backbone in BACKBONES:
        image_size = BACKBONES[backbone].size
    height, width = image_size + height_offset, image_size + width_offset
    check_parameters(prompts, image_prompts, backbone, height, width, iterations, render_steps, save_frequency, aesthetic_scale)
    clip_util.check_clip_model(clip_model_name)
    device = device or ("cuda" if th.cuda.is_available() else "cpu")

    th.manual_seed(seed)
    model = load_backbone(backbone, conditioning_text(prompts), device, tiny_decoder)
    guide = None  # The plain model never uses it, and at 1024 it needs the room.
    if iterations:
        guide = clip_util.ClipGuide(clip_model_name, prompts, image_prompts, min(height, width), num_cutouts, device, clip_fp16,
                                    cutout_power, use_augs, MIN_CUT_FRACTION, clip_loss, aesthetic_scale)
    name = script_util.run_name(prompts, image_prompts)
    shape = model.latent_shape(height, width)
    # The backward pass of the U-Net's widest convolutions is nondeterministic below about
    # 448px unless cuDNN is told otherwise, which makes a seed produce a different image
    # every run. Its algorithm of choice for those shapes is no faster.
    with th.backends.cudnn.flags(enabled=True, deterministic=True):
        for batch_idx in range(batch_size):
            best_z = th.randn(shape, device=device)
            steps = optimise_noise(model, guide, best_z, iterations, lr, penalty=noise_penalty_scale, progress=progress)
            for iteration, step in enumerate(steps):
                best_z = step["best_z"]
                if iteration % save_frequency == 0:
                    yield batch_idx, script_util.log_image(step["images"][0], prefix_path, name, iteration, batch_idx)
            yield batch_idx, script_util.log_image(render(model, best_z, render_steps)[0], prefix_path, name, iterations, batch_idx)

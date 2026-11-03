"""CLIP guidance of OpenAI's class-conditional ImageNet diffusion models (ADM, Dhariwal & Nichol 2021).

CLIP picks the ImageNet class closest to the prompts for the U-Net to draw, and its
gradient steers each denoised image towards them, through sampler.sample().
"""

import warnings
from collections.abc import Callable, Iterator, Sequence
from functools import lru_cache, partial
from pathlib import Path

import numpy as np
import torch as th
import torchvision.transforms.functional as tvf
from PIL import Image
from torch import nn
from tqdm.auto import tqdm

from cgd import losses, sampler
from cgd.checks import check_common, require
from cgd.clip_models import check_model, closest_imagenet_class, ensemble_members
from cgd.device import default_device
from cgd.download import download, fetch
from cgd.guide import Guide, guide_loss, load_guide, query
from cgd.imagenet_classes import IMAGENET_CLASSES
from cgd.outputs import run_name, save_frames

with warnings.catch_warnings():  # It uses a deprecated spelling of torch.amp, which is nothing to act on.
    warnings.simplefilter("ignore", FutureWarning)
    from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

DEFAULT_IMAGE_SIZE = 128
DEFAULT_CLIP_MODEL = "ViT-B/32"
CLIP_SCALE = 1000
TV_SCALE = 150  # Smoothness.
RANGE_SCALE = 50  # How far out of RGB range the denoised image may get.
NUM_TIMESTEPS = 1000  # Every checkpoint was trained on this schedule length.
# Guidance backpropagates through the U-Net. Above this many pixels per batch, recompute
# its activations instead of storing them: 1.5x the time per guided step for under half
# the memory, which is what lets 512px run in 8 GB.
GRADIENT_CHECKPOINT_PIXELS = 256**2
CHECKPOINT_URL = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/{filename}"
VGG16_URL = "https://download.pytorch.org/models/vgg16-397923af.pth"

_FLAGS = {
    "attention_resolutions": "32,16,8",
    "class_cond": True,
    "diffusion_steps": NUM_TIMESTEPS,
    "learn_sigma": True,
    "noise_schedule": "linear",
    "num_channels": 256,
    "num_head_channels": 64,
    "num_res_blocks": 2,
    "resblock_updown": True,
    "use_scale_shift_norm": True,
}
# Each checkpoint's architecture and schedule. Sampling a model on a schedule it was not
# trained on produces garbage.
CHECKPOINT_FLAGS = {
    64: {
        **_FLAGS,
        "dropout": 0.1,
        "noise_schedule": "cosine",
        "num_channels": 192,
        "num_res_blocks": 3,
        "use_new_attention_order": True,
    },
    128: {**_FLAGS, "num_head_channels": -1, "num_heads": 4},
    256: _FLAGS,
    512: _FLAGS,
}

InitLoss = Callable[[th.Tensor], th.Tensor]


def check_parameters(
    prompts: Sequence[str],
    image_prompts: Sequence[str],
    *,
    image_size: int,
    height: int,
    width: int,
    batch_size: int,
    steps: int,
    guidance_rate: float,
    guidance_limit: float | None,
    aesthetic_scale: float,
    init_image: str | None,
    init_scale: float,
    skip_timesteps: int,
    save_frequency: int,
) -> None:
    """Reject a configuration that cannot work, before loading anything."""
    check_common(
        prompts,
        image_prompts,
        batch_size=batch_size,
        save_frequency=save_frequency,
        aesthetic_scale=aesthetic_scale,
        other_images=[init_image] if init_image else [],
    )
    require(
        image_size in CHECKPOINT_FLAGS, f"No checkpoint for size {image_size}. Choose one of {list(CHECKPOINT_FLAGS)}."
    )
    stride = image_size // 8  # The U-Net halves the resolution down to 8x8.
    require(
        height > 0 and width > 0 and not height % stride and not width % stride,
        f"At size {image_size}, height and width must be positive multiples of {stride}. Got {height}x{width}.",
    )
    require(steps >= 1, "--steps must be at least 1.")
    require(guidance_rate >= 0, "--guidance-rate must not be negative.")
    require(
        guidance_limit is None or 0 < guidance_limit <= 1, "--guidance-limit must be a quantile, above 0 and at most 1."
    )
    require(0 <= skip_timesteps < NUM_TIMESTEPS, f"--skip-timesteps must be between 0 and {NUM_TIMESTEPS - 1}.")
    require(
        bool(init_image) or not (skip_timesteps or init_scale),
        "--skip-timesteps and --init-scale need an --init-image.",
    )


@lru_cache(maxsize=1)
def load_model(image_size: int, device: str, fp16: bool, use_checkpoint: bool = False) -> tuple[nn.Module, np.ndarray]:
    """The U-Net for `image_size`, frozen, and the log(alpha_bar) of its schedule.

    `use_checkpoint` recomputes activations during the backward pass instead of storing them.
    """
    filename = f"{image_size}x{image_size}_diffusion.pt"
    path = download(CHECKPOINT_URL.format(filename=filename), filename)
    flags = model_and_diffusion_defaults() | CHECKPOINT_FLAGS[image_size]
    flags |= {"image_size": image_size, "use_fp16": fp16, "use_checkpoint": use_checkpoint}
    # Build on the meta device: random-initialising 400M+ parameters only to overwrite
    # them is the slowest part of startup. mmap and assign then adopt the checkpoint's
    # tensors without a copy.
    with th.device("meta"):
        model, diffusion = create_model_and_diffusion(**flags)
    model.load_state_dict(th.load(path, map_location="cpu", mmap=True, weights_only=True), assign=True)
    model.requires_grad_(False).eval()
    if fp16:
        model.convert_to_fp16()  # Before moving, so the 32-bit weights never occupy VRAM.
    return model.to(device), sampler.schedule(diffusion.alphas_cumprod)


def load_init_image(path: str, height: int, width: int, device: str) -> th.Tensor:
    """An init image in [-1, 1], stretched to the size being made."""
    image = Image.open(fetch(path)).convert("RGB").resize((width, height))
    return tvf.to_tensor(image).to(device).unsqueeze(0).mul(2).sub(1)


def load_lpips(device: str) -> nn.Module:
    """LPIPS on VGG16, the perceptual distance an init scale keeps the image within.

    LPIPS would fetch VGG16 itself, but torch hub reads without a timeout, so a
    connection that stalls near the end of the file hangs the run forever. This puts
    the weights where torch hub looks for them first.
    """
    import lpips  # Slow to import, and only an init scale needs it.

    download(VGG16_URL, VGG16_URL.rpartition("/")[2], Path(th.hub.get_dir()) / "checkpoints")
    return lpips.LPIPS(net="vgg").to(device)


def guidance_loss(
    guide: Guide, x: th.Tensor, x0: th.Tensor, sigma: float, init_loss: InitLoss | None = None
) -> th.Tensor:
    """What guidance minimises: the CLIP loss and regularisers of the denoised image x0.

    CLIP sees x0 blended with the noisy x by the noise level, which keeps its gradient
    from fighting the noise early on. The range loss keeps x0 itself in RGB range.
    """
    clip_in = x0 * sigma + x * (1 - sigma)
    clip_term = guide_loss(guide, clip_in, CLIP_SCALE)
    range_term, tv_term = losses.range_loss(x0), losses.tv_loss(clip_in)
    range_term, tv_term = range_term.sum() * RANGE_SCALE, tv_term.sum() * TV_SCALE
    loss = clip_term + tv_term + range_term
    if init_loss is not None:
        loss = loss + init_loss(clip_in)
    return loss


def perceptual_init_loss(lpips: nn.Module, init: th.Tensor, scale: float, images: th.Tensor) -> th.Tensor:
    """How far images are from the init image to LPIPS, times `scale`.

    cuDNN's choice of backward pass through VGG16 sums in no fixed order, which made a
    seed give a different image every run. So the gradient is taken here, with cuDNN told
    to be deterministic, instead of when guidance backpropagates through everything.
    """
    detached = images.detach().requires_grad_()
    with th.backends.cudnn.flags(enabled=True, deterministic=True):
        value = lpips(detached, init).sum() * scale
        gradient = th.autograd.grad(value, detached)[0]
    return losses.with_gradient(images, value.item(), gradient)


def generate(  # pylint: disable=too-many-locals  # Its keyword arguments are the command's options.
    prompts: Sequence[str] = (),
    image_prompts: Sequence[str] = (),
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
    width_offset: int = 0,
    height_offset: int = 0,
    batch_size: int = 1,
    steps: int = sampler.DEFAULT_STEPS,
    guidance_rate: float = sampler.DEFAULT_GUIDANCE_RATE,
    guidance_limit: float | None = None,
    clip_model: str = DEFAULT_CLIP_MODEL,
    clip_fp16: bool = False,
    aesthetic_scale: float = 0.0,
    init_image: str | None = None,
    init_scale: float = 0.0,
    skip_timesteps: int = 0,
    seed: int = 0,
    output_dir: str | Path = "outputs",
    save_frequency: int = 10,
    device: str | None = None,
    progress: bool = True,
) -> Iterator[tuple[int, Path]]:
    """Make images of the prompts, and yield (batch index, path) of each one saved.

    An image is saved every `save_frequency` model calls, and the result always last. The
    prompts take weights, as in "a fox:2", and image prompts are paths or URLs. Without
    `skip_timesteps`, an init image skips half the schedule.
    """
    height, width = image_size + height_offset, image_size + width_offset
    check_parameters(
        prompts,
        image_prompts,
        image_size=image_size,
        height=height,
        width=width,
        batch_size=batch_size,
        steps=steps,
        guidance_rate=guidance_rate,
        guidance_limit=guidance_limit,
        aesthetic_scale=aesthetic_scale,
        init_image=init_image,
        init_scale=init_scale,
        skip_timesteps=skip_timesteps,
        save_frequency=save_frequency,
    )
    check_model(clip_model)
    device = default_device(device)
    _status(f"Using device {device}", progress)

    # Everything from here on draws random numbers in this order, so a seed always makes the same image.
    th.manual_seed(seed)
    guide = load_guide(
        clip_model,
        prompts,
        image_prompts,
        image_size=image_size,
        device=device,
        fp16=clip_fp16,
        aesthetic_scale=aesthetic_scale,
    )
    init = load_init_image(init_image, height, width, device) if init_image else None
    # The U-Net does the heavy lifting when it draws the closest real class.
    class_index = closest_imagenet_class(query(guide), ensemble_members(clip_model)[0], device, clip_fp16)
    _status(f"Conditioning on ImageNet class: {IMAGENET_CLASSES[class_index]}", progress)
    use_checkpoint = batch_size * height * width > GRADIENT_CHECKPOINT_PIXELS
    model, log_alpha_bar = load_model(image_size, device, fp16=device != "cpu", use_checkpoint=use_checkpoint)
    labels = th.full((batch_size,), class_index, device=device, dtype=th.long)

    start, init_loss = None, None
    if init is not None:
        if not skip_timesteps:  # An init image only has an effect partway down the schedule.
            skip_timesteps = NUM_TIMESTEPS // 2
            _status(
                f"Skipping {skip_timesteps} timesteps to blend in the init image. Set --skip-timesteps to change this.",
                progress,
            )
        start = NUM_TIMESTEPS - 1 - skip_timesteps
        if init_scale:
            init_loss = partial(perceptual_init_loss, load_lpips(device), init, init_scale)

    ts = sampler.timesteps(log_alpha_bar, steps, start)
    x = th.randn((batch_size, 3, height, width), device=device)
    if init is not None:
        x = sampler.noised(log_alpha_bar, ts[0], init, x)
    guidance = sampler.Guidance(partial(guidance_loss, guide, init_loss=init_loss), guidance_rate, limit=guidance_limit)
    samples = sampler.sample(partial(denoise, model, labels), log_alpha_bar, ts, x, guidance, progress=progress)
    yield from save_frames(samples, output_dir, run_name(prompts, image_prompts), save_frequency, len(ts) - 2)


def denoise(model: nn.Module, labels: th.Tensor, x: th.Tensor, t: th.Tensor) -> th.Tensor:
    """The noise the U-Net predicts in x at timesteps t, drawing the classes `labels`."""
    return model(x, t, y=labels)[:, :3]  # The rest is the learned variance.


def _status(message: str, progress: bool) -> None:
    if progress:
        tqdm.write(message)

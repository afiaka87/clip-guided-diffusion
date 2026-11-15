"""CLIP guidance of OpenAI's ImageNet diffusion models (ADM, Dhariwal & Nichol 2021).

CLIP picks the ImageNet class closest to the prompts for the U-Net to draw, and its
gradient steers each denoised image towards them, through sampler.sample(). The
unconditional checkpoints draw no class, so guidance alone brings the subject: CLIP's,
and that of a model which reads the noisy image as a thumbnail, see thumbnails.py.

A sequence of images, "a castle|>a fox", is each image as its prompts make it alone, and
frames that slide from each one to the next, see slide.py and between().
"""

import warnings
from collections.abc import Callable, Iterator, Sequence
from functools import lru_cache, partial
from pathlib import Path
from typing import NamedTuple, TypeVar

import numpy as np
import torch as th
import torchvision.transforms.functional as tvf
from PIL import Image
from torch import nn
from tqdm.auto import tqdm

from cgd import losses, pag, sampler, slide, thumbnails
from cgd.checks import check_common, require
from cgd.clip_models import check_model, closest_imagenet_class, ensemble_members
from cgd.device import default_device, limit_to_vram, random_state, replayed_rng
from cgd.download import download, fetch
from cgd.guide import NUM_CUTOUTS, Guide, guide_loss, load_guides, query
from cgd.imagenet_classes import IMAGENET_CLASSES
from cgd.outputs import run_name, save_batch
from cgd.prompts import by_image, loops, scenes

with warnings.catch_warnings():  # It uses a deprecated spelling of torch.amp, which is nothing to act on.
    warnings.simplefilter("ignore", FutureWarning)
    from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

NO_THUMBNAIL = "none"
CLIP_SCALE = 1000
TV_SCALE = 150  # Smoothness.
RANGE_SCALE = 50  # How far out of RGB range the denoised image may get.
NUM_TIMESTEPS = 1000  # Every checkpoint was trained on this schedule length.
PAG_SIZE = 256  # The unconditional checkpoint that pag.LAYERS names the attention blocks of.
TRANSITION_FRAMES = 30  # From one image of a sequence to the next: 3 seconds of the animation.
# Guidance backpropagates through the U-Net. Above this many pixels per batch, recompute
# its activations instead of storing them: 1.5x the time per guided step for under half
# the memory, which is what lets 512px run in 8 GB.
GRADIENT_CHECKPOINT_PIXELS = 256**2
OPENAI_URL = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/"
CHECKPOINT_URLS = {size: f"{OPENAI_URL}{size}x{size}_diffusion.pt" for size in (64, 128, 256, 512)}
# OpenAI's at 256, and at 512 Katherine Crowson's fine-tune of OpenAI's class-conditional model.
UNCONDITIONAL_URLS = {
    256: f"{OPENAI_URL}256x256_diffusion_uncond.pt",
    512: "https://huggingface.co/lowlevelware/512x512_diffusion_unconditional_ImageNet/resolve/main/"
    "512x512_diffusion_uncond_finetune_008100.pt",
}
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
# Each class-conditional checkpoint's architecture and schedule. An unconditional one is the
# same without class_cond. Sampling a model on a schedule it was not trained on produces garbage.
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
T = TypeVar("T")


class Defaults(NamedTuple):
    """What the options a run leaves unset default to."""

    image_size: int
    steps: int
    recurrence: int
    clip_model: str
    num_cutouts: int
    thumbnail: str  # One of thumbnails.MODELS, or NO_THUMBNAIL.


class Drawing(NamedTuple):
    """What draws one image of a sequence."""

    unet: sampler.Denoiser  # The U-Net drawing the image's class, with whatever else shapes its noise prediction.
    plain: sampler.Denoiser  # The U-Net drawing that class and nothing more, which the frames between images use.
    guidance: sampler.Guidance


DEFAULTS = Defaults(128, sampler.DEFAULT_STEPS, 1, "ViT-B/32", NUM_CUTOUTS, NO_THUMBNAIL)
# What made the best images without a class to bring the subject, for the time it takes. The README has the numbers.
UNCONDITIONAL_DEFAULTS = Defaults(
    256, sampler.UNCONDITIONAL_STEPS, sampler.UNCONDITIONAL_RECURRENCE, "ViT-B/32+PE-Core-B-16", 4, "clip"
)


def defaults(unconditional: bool) -> Defaults:
    """The defaults of the class-conditional checkpoints, or of the unconditional ones."""
    return UNCONDITIONAL_DEFAULTS if unconditional else DEFAULTS


def chosen(given: T | None, default: T) -> T:
    """An option's value: the one given, or its default when none was."""
    return default if given is None else given


def default_thumbnail(unconditional: bool, prompts: Sequence[str], square: bool) -> str:
    """The thumbnail model of a run that names none: its checkpoint's, if the run has what such a model reads.

    That is a text prompt for every image of a sequence, and a square image to average down to a thumbnail.
    """
    return defaults(unconditional).thumbnail if all(by_image(prompts)) and square else NO_THUMBNAIL


def check_parameters(  # pylint: disable=too-many-locals  # Its keyword arguments are the command's options.
    prompts: Sequence[str],
    image_prompts: Sequence[str],
    *,
    image_size: int,
    unconditional: bool,
    height: int,
    width: int,
    batch_size: int,
    num_cutouts: int,
    steps: int,
    recurrence: int,
    guidance_rate: float,
    guidance_limit: float | None,
    thumbnail: str,
    perturbed_attention: bool,
    aesthetic_scale: float,
    init_image: str | None,
    init_scale: float,
    skip_timesteps: int,
    save_frequency: int,
    transition_frames: int | None = None,
) -> None:
    """Reject a configuration that cannot work, before loading anything."""
    check_common(
        prompts,
        image_prompts,
        batch_size=batch_size,
        num_cutouts=num_cutouts,
        save_frequency=save_frequency,
        aesthetic_scale=aesthetic_scale,
        transition_frames=transition_frames,
        other_images=[init_image] if init_image else [],
    )
    sizes = list(UNCONDITIONAL_URLS if unconditional else CHECKPOINT_URLS)
    kind = "unconditional checkpoint" if unconditional else "checkpoint"
    require(image_size in sizes, f"No {kind} for size {image_size}. Choose one of {sizes}.")
    stride = image_size // 8  # The U-Net halves the resolution down to 8x8.
    require(
        height > 0 and width > 0 and not height % stride and not width % stride,
        f"At size {image_size}, height and width must be positive multiples of {stride}. Got {height}x{width}.",
    )
    require(steps >= 1, "--steps must be at least 1.")
    require(recurrence >= 1, "--recurrence must be at least 1.")
    require(guidance_rate >= 0, "--guidance-rate must not be negative.")
    require(
        guidance_limit is None or 0 < guidance_limit <= 1, "--guidance-limit must be a quantile, above 0 and at most 1."
    )
    names = [*thumbnails.MODELS, NO_THUMBNAIL]
    require(thumbnail in names, f"Unknown thumbnail model {thumbnail!r}. Choose one of {names}.")
    if thumbnail != NO_THUMBNAIL:
        require(unconditional, "--thumbnail only applies with --unconditional.")
        require(
            all(by_image(prompts)),
            "--thumbnail reads the text prompts, and there are none.",
        )
        require(
            height == width == image_size,
            f"--thumbnail needs a square image: its models read {thumbnails.SIZE}x{thumbnails.SIZE} thumbnails.",
        )
    require(
        not perturbed_attention or (unconditional and image_size == PAG_SIZE),
        f"--perturbed-attention only applies with --unconditional at {PAG_SIZE}, the checkpoint its layers are from.",
    )
    require(0 <= skip_timesteps < NUM_TIMESTEPS, f"--skip-timesteps must be between 0 and {NUM_TIMESTEPS - 1}.")
    require(
        bool(init_image) or not (skip_timesteps or init_scale),
        "--skip-timesteps and --init-scale need an --init-image.",
    )


@lru_cache(maxsize=1)
def load_model(
    image_size: int, device: str, fp16: bool, use_checkpoint: bool = False, unconditional: bool = False
) -> tuple[nn.Module, np.ndarray]:
    """The U-Net for `image_size`, frozen, and the log(alpha_bar) of its schedule.

    `use_checkpoint` recomputes activations during the backward pass instead of storing them.
    """
    url = (UNCONDITIONAL_URLS if unconditional else CHECKPOINT_URLS)[image_size]
    path = download(url, url.rpartition("/")[2])
    flags = model_and_diffusion_defaults() | CHECKPOINT_FLAGS[image_size]
    flags |= {"image_size": image_size, "class_cond": not unconditional}
    flags |= {"use_fp16": fp16, "use_checkpoint": use_checkpoint}
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
    image_size: int | None = None,
    unconditional: bool = False,
    width_offset: int = 0,
    height_offset: int = 0,
    batch_size: int = 1,
    num_cutouts: int | None = None,
    steps: int | None = None,
    recurrence: int | None = None,
    guidance_rate: float = sampler.DEFAULT_GUIDANCE_RATE,
    guidance_limit: float | None = None,
    thumbnail: str | None = None,
    perturbed_attention: bool = False,
    clip_model: str | None = None,
    clip_fp16: bool = False,
    aesthetic_scale: float = 0.0,
    init_image: str | None = None,
    init_scale: float = 0.0,
    skip_timesteps: int = 0,
    seed: int = 0,
    output_dir: str | Path = "outputs",
    save_frequency: int = 10,
    transition_frames: int | None = None,
    device: str | None = None,
    progress: bool = True,
) -> Iterator[tuple[int, Path]]:
    """Make images of the prompts, and yield (batch index, path) of each one saved.

    An image is saved every `save_frequency` model calls, and the result always last. The
    prompts take weights, as in "a fox:2", and image prompts are paths or URLs. Without
    `skip_timesteps`, an init image skips half the schedule.

    A prompt that starts with > begins the next image of a sequence. Each image is the one
    its prompts make alone, with these settings and this seed. After the first, the frames
    saved are the `transition_frames` that slide to each next image, the last of them the
    image itself. The default is TRANSITION_FRAMES. A > alone at the end slides back to the
    first image, and the frames then begin at that image instead of at noise, so they loop.

    `image_size`, `steps`, `recurrence`, `clip_model`, `num_cutouts` and `thumbnail` default to
    DEFAULTS, or to UNCONDITIONAL_DEFAULTS for an unconditional checkpoint. `thumbnail` is
    one of thumbnails.MODELS, whose guidance sets the composition, or "none". It reads the
    text prompts off a square image, so by default a run without either does without it.
    `perturbed_attention` adds pag.py's guidance.
    """
    default = defaults(unconditional)
    image_size = chosen(image_size, default.image_size)
    steps, recurrence = chosen(steps, default.steps), chosen(recurrence, default.recurrence)
    clip_model, num_cutouts = chosen(clip_model, default.clip_model), chosen(num_cutouts, default.num_cutouts)
    height, width = image_size + height_offset, image_size + width_offset
    if thumbnail is None:
        thumbnail = default_thumbnail(unconditional, prompts, height == width == image_size)
        if thumbnail != default.thumbnail:
            _status("No thumbnail guidance: it reads a text prompt off a square image.", progress)
    check_parameters(
        prompts,
        image_prompts,
        image_size=image_size,
        unconditional=unconditional,
        height=height,
        width=width,
        batch_size=batch_size,
        num_cutouts=num_cutouts,
        steps=steps,
        recurrence=recurrence,
        guidance_rate=guidance_rate,
        guidance_limit=guidance_limit,
        thumbnail=thumbnail,
        perturbed_attention=perturbed_attention,
        aesthetic_scale=aesthetic_scale,
        init_image=init_image,
        init_scale=init_scale,
        skip_timesteps=skip_timesteps,
        save_frequency=save_frequency,
        transition_frames=transition_frames,
    )
    check_model(clip_model)
    images = scenes(prompts, image_prompts)
    device = default_device(device)
    limit_to_vram(device)
    _status(f"Using device {device}", progress)

    # Everything from here on draws random numbers in this order, so a seed always makes the same image.
    th.manual_seed(seed)
    guides = load_guides(
        clip_model,
        images,
        image_size=image_size,
        num_cutouts=num_cutouts,
        device=device,
        fp16=clip_fp16,
        aesthetic_scale=aesthetic_scale,
    )
    init = load_init_image(init_image, height, width, device) if init_image else None
    labels: list[th.Tensor | None] = [None] * len(images)
    if not unconditional:
        labels = [
            closest_class(guide, clip_model, batch_size=batch_size, device=device, fp16=clip_fp16, progress=progress)
            for guide in guides
        ]
    use_checkpoint = batch_size * height * width > GRADIENT_CHECKPOINT_PIXELS
    model, log_alpha_bar = load_model(
        image_size, device, fp16=device != "cpu", use_checkpoint=use_checkpoint, unconditional=unconditional
    )

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

    window = sampler.UNCONDITIONAL_GUIDANCE_WINDOW if unconditional else sampler.GUIDANCE_WINDOW
    drawings = []
    for image, guide, plain in zip(images, guides, class_denoisers(model, labels), strict=True):
        unet = plain
        if perturbed_attention:
            unet = partial(pag.denoise, unet, model, pag.LAYERS)
        if thumbnail != NO_THUMBNAIL:
            guidance_of = thumbnails.load(thumbnail, image.prompts, batch_size=batch_size, seed=seed, device=device)
            unet = partial(thumbnails.guided_denoise, unet, log_alpha_bar, guidance_of)
        loss_fn = partial(guidance_loss, guide, init_loss=init_loss)
        guidance = sampler.Guidance(loss_fn, guidance_rate, window, guidance_limit, recurrence)
        drawings.append(Drawing(unet, plain, guidance))

    ts = sampler.timesteps(log_alpha_bar, steps, start)
    noise = th.randn((batch_size, 3, height, width), device=device)
    x = noise if init is None else sampler.noised(log_alpha_bar, ts[0], init, noise)
    frames = sampled_frames(
        drawings,
        log_alpha_bar,
        ts,
        x,
        noise,
        steps=steps,
        save_frequency=save_frequency,
        transition_frames=chosen(transition_frames, TRANSITION_FRAMES),
        loop=loops(prompts, image_prompts),
        progress=progress,
    )
    name = run_name(prompts, image_prompts)
    # The backward pass of the U-Net's convolutions is nondeterministic at 128 and 256 unless cuDNN is told
    # otherwise, since PyTorch 2.15, which makes a seed produce a different image every run. It is no slower.
    with th.backends.cudnn.flags(enabled=True, deterministic=True):
        for step, batch in frames:
            yield from save_batch(batch, output_dir, name, step)


def class_denoisers(model: nn.Module, labels: Sequence[th.Tensor | None]) -> list[sampler.Denoiser]:
    """The U-Net drawing each of `labels`, one batch of them per image of a sequence, or no class for None.

    Images of the same class share a denoiser, so sampler.blend() between them calls the U-Net once.
    """
    shared: dict[int | None, sampler.Denoiser] = {}
    return [
        shared.setdefault(None if label is None else int(label[0]), partial(denoise, model, label)) for label in labels
    ]


def sampled_frames(  # pylint: disable=too-many-locals  # It takes all a run samples with, to make every frame of it.
    drawings: Sequence[Drawing],
    log_alpha_bar: np.ndarray,
    ts: Sequence[float],
    x: th.Tensor,
    noise: th.Tensor,
    *,
    steps: int,
    save_frequency: int,
    transition_frames: int,
    progress: bool,
    loop: bool = False,
) -> Iterator[tuple[int, th.Tensor]]:
    """(step, batch of images) of each frame to save, for one image per drawing, each sampled from x down `ts`.

    First the denoised image of every `save_frequency`th model call of the first image,
    and its result. Then, for each next image, the `transition_frames` that slide to it,
    see between(). `noise` is the standard normal noise x was made from, and `steps` how
    many the whole schedule takes.

    With `loop`, the last frames slide back to the first image, and the first frame is
    that image instead of the model calls that made it, so the frames go round without a cut.
    """
    device = str(x.device)
    drawn_from = random_state(device)  # Every image draws what the first does from here on.
    before, result, step = drawings[0], x, len(ts) - 2
    samples = sampler.sample(before.unet, log_alpha_bar, ts, x, before.guidance, progress=progress)
    for model_call, result in enumerate(samples):
        if model_call == step or (model_call % save_frequency == 0 and not loop):
            yield model_call, result
    first = result

    def later_images() -> Iterator[tuple[Drawing, th.Tensor]]:
        """What draws each image after the first, and the image. A loop ends on the first again, as it was made."""
        for number, drawing in enumerate(drawings[1:], start=2):
            _status(
                f"Image {number} of {len(drawings)}, then the {transition_frames} frames that slide to it.", progress
            )
            with replayed_rng(device, drawn_from):
                image = sampler.final(
                    sampler.sample(drawing.unet, log_alpha_bar, ts, x, drawing.guidance, progress=progress)
                )
            yield drawing, image
        if loop:
            _status(f"The {transition_frames} frames that slide back to the first image.", progress)
            yield drawings[0], first

    for after, following in later_images():
        ends = result.clamp(-1, 1), following.clamp(-1, 1)
        draw = partial(between, (before.plain, after.plain), log_alpha_bar, noise, ends, steps=steps)
        for frame in slide.place(draw, ends, transition_frames, progress):
            step += 1
            yield step, frame
        before, result = after, following


def between(
    denoisers: tuple[sampler.Denoiser, sampler.Denoiser],
    log_alpha_bar: np.ndarray,
    noise: th.Tensor,
    ends: tuple[th.Tensor, th.Tensor],
    position: float,
    *,
    steps: int,
) -> th.Tensor:
    """The frame at `position` between two finished images, the `ends`: at 0 the first, at 1 the second.

    The two are faded into each other and redrawn by the U-Net: noised with `noise`, most
    halfway, and denoised between the two classes, without CLIP. slide.py has why.
    """
    share = slide.fade(position)
    unet = partial(sampler.blend, *denoisers, share)
    faded = th.lerp(*ends, share)
    return sampler.redraw(unet, log_alpha_bar, faded, noise, t=slide.noise_timestep(position), steps=steps).clamp(-1, 1)


def closest_class(
    guide: Guide, clip_model: str, *, batch_size: int, device: str, fp16: bool, progress: bool
) -> th.Tensor:
    """The ImageNet class closest to the prompts, as labels for the batch.

    The U-Net does the heavy lifting when it draws a real class.
    """
    class_index = closest_imagenet_class(query(guide), ensemble_members(clip_model)[0], device, fp16)
    _status(f"Conditioning on ImageNet class: {IMAGENET_CLASSES[class_index]}", progress)
    return th.full((batch_size,), class_index, device=device, dtype=th.long)


def denoise(model: nn.Module, labels: th.Tensor | None, x: th.Tensor, t: th.Tensor) -> th.Tensor:
    """The noise the U-Net predicts in x at timesteps t, drawing the classes `labels`, or none without them."""
    return model(x, t, y=labels)[:, :3]  # The rest is the learned variance.


def _status(message: str, progress: bool) -> None:
    if progress:
        tqdm.write(message)

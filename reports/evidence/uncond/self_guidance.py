"""Perturbed-attention guidance on the unconditional pipeline, in the configuration its paper used for this checkpoint.

PAG (Ahn et al. 2024) runs the U-Net a second time with some self-attention maps replaced
by the identity, and moves the prediction away from that structureless one. On this exact
checkpoint, unguided, it took FID from 26.21 to 16.23 at scale 1 with four layers
perturbed: input_blocks.14.1, 16.1, 17.1 and middle_block.1. The repo's earlier test, which
lost, perturbed one layer of the class-conditional 128 model.

generate() is adm.generate() for the unconditional 256 checkpoint with that added. The
second run takes no gradient, so CLIP guidance still backpropagates through one U-Net.
With pag_scale 0 it makes exactly adm.generate()'s image.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import torch as th
from guided_diffusion.unet import AttentionBlock
from torch import nn

from cgd import adm, sampler
from cgd.guide import NUM_CUTOUTS, load_guide
from cgd.outputs import run_name, save_frames

PAG_LAYERS = ("input_blocks.14.1", "input_blocks.16.1", "input_blocks.17.1", "middle_block.1")


@contextmanager
def perturbed_attention(model: nn.Module, layers: Sequence[str]):
    """Within this, each named attention block attends only to itself: its output is its values."""

    def attend_to_self(attention, inputs, _):
        qkv = inputs[0]
        batch, width, length = qkv.shape
        channels = width // (3 * attention.n_heads)
        values = qkv.reshape(batch * attention.n_heads, channels * 3, length).split(channels, dim=1)[2]
        return values.reshape(batch, -1, length)

    modules = dict(model.named_modules())
    handles = []
    for name in layers:
        assert isinstance(modules[name], AttentionBlock), f"{name} is not an attention block."
        handles.append(modules[name].attention.register_forward_hook(attend_to_self))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def pag_denoise(model: nn.Module, scale: float, layers: Sequence[str], x: th.Tensor, t: th.Tensor) -> th.Tensor:
    """The predicted noise, moved away from what the U-Net predicts with its attention perturbed."""
    eps = model(x, t)[:, :3]
    if not scale:
        return eps
    with th.no_grad(), perturbed_attention(model, layers):
        perturbed = model(x.detach(), t)[:, :3]
    return eps + scale * (eps.detach() - perturbed)


def generate(
    prompts: Sequence[str] = (),
    image_prompts: Sequence[str] = (),
    *,
    pag_scale: float = 1.0,
    pag_layers: Sequence[str] = PAG_LAYERS,
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
    th.manual_seed(seed)
    guide = load_guide(clip_model, prompts, image_prompts, image_size=256, num_cutouts=num_cutouts, device=device)
    model, log_alpha_bar = adm.load_model(256, device, fp16=True, use_checkpoint=False, unconditional=True)
    ts = sampler.timesteps(log_alpha_bar, steps)
    x = th.randn((1, 3, 256, 256), device=device)
    window = sampler.UNCONDITIONAL_GUIDANCE_WINDOW
    guidance = sampler.Guidance(partial(adm.guidance_loss, guide), guidance_rate, window, None, recurrence)
    denoise = partial(pag_denoise, model, pag_scale, tuple(pag_layers))
    samples = sampler.sample(denoise, log_alpha_bar, ts, x, guidance, progress=progress)
    yield from save_frames(samples, output_dir, run_name(prompts, image_prompts), save_frequency, len(ts) - 2)

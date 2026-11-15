"""Perturbed-attention guidance (Ahn et al. 2024, arXiv 2403.17377) for the unconditional 256 checkpoint.

The U-Net runs a second time with the self-attention of its deepest blocks replaced by
the identity, so every position sees only itself and the prediction loses its structure.
Moving the real prediction away from that one makes the image more of one coherent
picture. The layers and the scale are the ones the paper used on this very checkpoint,
where they took its unguided FID from 26.21 to 16.23.

The second run takes no gradient, so CLIP guidance still backpropagates through one U-Net.
"""

import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import torch as th
from torch import nn

from cgd.sampler import Denoiser

with warnings.catch_warnings():  # It uses a deprecated spelling of torch.amp, which is nothing to act on.
    warnings.simplefilter("ignore", FutureWarning)
    from guided_diffusion.unet import AttentionBlock, QKVAttentionLegacy

SCALE = 1.0
# The second 16x16 block on the way down, both 8x8 blocks and the middle one.
LAYERS = ("input_blocks.14.1", "input_blocks.16.1", "input_blocks.17.1", "middle_block.1")


def own_values(attention: QKVAttentionLegacy, inputs: tuple[th.Tensor, ...], _output: th.Tensor) -> th.Tensor:
    """What an attention module returns when every position attends only to itself: its values.

    A forward hook, so what it returns replaces the module's output. The module takes the
    queries, keys and values of each head in turn along the channels, as [batch, heads * 3 * channels, positions].
    """
    qkv = inputs[0]
    batch, width, positions = qkv.shape
    channels = width // (3 * attention.n_heads)
    values = qkv.reshape(batch * attention.n_heads, 3 * channels, positions).split(channels, dim=1)[2]
    return values.reshape(batch, -1, positions)


@contextmanager
def perturbed_attention(model: nn.Module, layers: Sequence[str]) -> Iterator[None]:
    """Within this, each attention block of `model` named in `layers` attends only to itself."""
    blocks = dict(model.named_modules())
    handles = []
    try:
        for name in layers:
            block = blocks[name]
            if not isinstance(block, AttentionBlock) or not isinstance(block.attention, QKVAttentionLegacy):
                raise TypeError(f"{name} is not an attention block that splits its heads before its values.")
            handles.append(block.attention.register_forward_hook(own_values))
        yield
    finally:
        for handle in handles:
            handle.remove()


def denoise(unet: Denoiser, model: nn.Module, layers: Sequence[str], x: th.Tensor, t: th.Tensor) -> th.Tensor:
    """The noise `unet` predicts in x, moved away from what it predicts with `layers` of `model` perturbed.

    `unet` is a denoiser that calls `model`. Only its first, unperturbed call carries a gradient.
    """
    eps = unet(x, t)
    with th.no_grad(), perturbed_attention(model, layers):
        perturbed = unet(x.detach(), t)
    return eps + SCALE * (eps.detach() - perturbed)

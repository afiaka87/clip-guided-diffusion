"""Katherine Crowson's small U-Net for guiding without backpropagating through the big one.

CLIP's gradient has to reach the noisy image through a model that denoises it. This one
denoises nearly as well for that purpose and is 13.8M parameters against the 256px
U-Net's 553M, so guided 256 falls from 5.3 to 1.9 GiB and 512 from 4.1 to 2.6, at no
measurable loss. At 128 it saves 1 GiB and scores lower. The sampler still takes its images
from the big model. Network from v-diffusion-pytorch (MIT), trimmed to what the checkpoint uses. Disco
Diffusion's hosts for the weights are gone, so they come from a mirror and are checked
against the hash Disco pinned.
"""
import math

import torch as th
import torch.nn.functional as tf
from torch import nn

from cgd import script_util

WEIGHTS_URL = "https://huggingface.co/spaces/huggi/secondary_model_imagenet_2.pth/resolve/main/secondary_model_imagenet_2.pth"
WEIGHTS_FILENAME = "secondary_model_imagenet_2.pth"
WEIGHTS_SHA256 = "983e3de6f95c88c81b2ca7ebb2c217933be1973b1ff058776b970f901584613a"
CHANNELS = (64, 128, 128, 256, 256, 512)
TIME_FEATURES = 16


class HalfSize(nn.Module):
    """nn.AvgPool2d(2), with a reproducible gradient. See modules.adaptive_avg_pool()."""

    def forward(self, input: th.Tensor) -> th.Tensor:
        batch, channels, height, width = input.shape
        return tf.pixel_unshuffle(input, 2).view(batch, channels, 4, height // 2, width // 2).mean(2)


def _double_axis(input: th.Tensor, dim: int) -> th.Tensor:
    length = input.shape[dim]
    previous = th.cat([input.narrow(dim, 0, 1), input.narrow(dim, 0, length - 1)], dim)
    following = th.cat([input.narrow(dim, 1, length - 1), input.narrow(dim, length - 1, 1)], dim)
    return th.stack([0.75 * input + 0.25 * previous, 0.75 * input + 0.25 * following], dim + 1).flatten(dim, dim + 1)


class DoubleSize(nn.Module):
    """nn.Upsample(scale_factor=2, mode="bilinear"), with a reproducible gradient."""

    def forward(self, input: th.Tensor) -> th.Tensor:
        return _double_axis(_double_axis(input, 2), 3)


class ConvBlock(nn.Sequential):
    def __init__(self, channels_in: int, channels_out: int):
        super().__init__(nn.Conv2d(channels_in, channels_out, 3, padding=1), nn.ReLU(inplace=True))


class SkipBlock(nn.Module):
    def __init__(self, main):
        super().__init__()
        self.main = nn.Sequential(*main)

    def forward(self, input: th.Tensor) -> th.Tensor:
        return th.cat([self.main(input), input], dim=1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(th.zeros(out_features // 2, in_features))

    def forward(self, input: th.Tensor) -> th.Tensor:
        features = 2 * math.pi * input @ self.weight.T
        return th.cat([features.cos(), features.sin()], dim=-1)


class SecondaryModel(nn.Module):
    """SecondaryDiffusionImageNet2. Module names must match the checkpoint's."""

    def __init__(self):
        super().__init__()
        c = CHANNELS
        self.timestep_embed = FourierFeatures(1, TIME_FEATURES)
        self.down = HalfSize()
        self.up = DoubleSize()
        inner = [self.down, ConvBlock(c[4], c[5]), ConvBlock(c[5], c[5]), ConvBlock(c[5], c[5]), ConvBlock(c[5], c[4]), self.up]
        for level in (4, 3, 2, 1):
            inner = [self.down, ConvBlock(c[level - 1], c[level]), ConvBlock(c[level], c[level]), SkipBlock(inner),
                     ConvBlock(c[level] * 2, c[level]), ConvBlock(c[level], c[level - 1]), self.up]
        self.net = nn.Sequential(ConvBlock(3 + TIME_FEATURES, c[0]), ConvBlock(c[0], c[0]), SkipBlock(inner),
                                 ConvBlock(c[0] * 2, c[0]), nn.Conv2d(c[0], 3, 3, padding=1))

    def forward(self, x: th.Tensor, alpha: float, sigma: float) -> th.Tensor:
        """The denoised image, for x at this noise level. Gradients flow through this."""
        t = th.full((x.shape[0], 1), math.atan2(sigma, alpha) * 2 / math.pi, device=x.device, dtype=x.dtype)
        embedding = self.timestep_embed(t)[:, :, None, None].expand(-1, -1, *x.shape[2:])
        v = self.net(th.cat([x, embedding], dim=1))
        return x * alpha - v * sigma


def load_secondary(device: str, checkpoints_dir: str = script_util.CACHE_PATH) -> SecondaryModel:
    weights = script_util.download_verified(WEIGHTS_URL, WEIGHTS_FILENAME, WEIGHTS_SHA256, root=checkpoints_dir)
    model = SecondaryModel()
    model.load_state_dict(th.load(weights, map_location="cpu", weights_only=True))
    return model.eval().requires_grad_(False).to(device)

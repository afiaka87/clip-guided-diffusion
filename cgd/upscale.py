"""Real-ESRGAN x4plus, for building large images from a small guided one.

Direct 512px spends 18.5 s and 4.1 GiB on an image that scores no better than a 128px
one taken through this network, which adds .06 s and .16 GiB. Network from BasicSR
(Wang et al., BSD-3-Clause), trimmed to what the x4plus checkpoint uses.
"""
from pathlib import Path

import torch as th
import torch.nn.functional as tf
import torchvision.transforms.functional as tvf
from PIL import Image
from torch import nn

from cgd import script_util

WEIGHTS_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
WEIGHTS_FILENAME = "RealESRGAN_x4plus.pth"
SCALE = 4
TILE_SIZE = 256
TILE_MARGIN = 16


class ResidualDenseBlock(nn.Module):
    def __init__(self, features: int = 64, growth: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(features, growth, 3, 1, 1)
        self.conv2 = nn.Conv2d(features + growth, growth, 3, 1, 1)
        self.conv3 = nn.Conv2d(features + 2 * growth, growth, 3, 1, 1)
        self.conv4 = nn.Conv2d(features + 3 * growth, growth, 3, 1, 1)
        self.conv5 = nn.Conv2d(features + 4 * growth, features, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(th.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(th.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(th.cat((x, x1, x2, x3), 1)))
        return self.conv5(th.cat((x, x1, x2, x3, x4), 1)) * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, features: int = 64, growth: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(features, growth)
        self.rdb2 = ResidualDenseBlock(features, growth)
        self.rdb3 = ResidualDenseBlock(features, growth)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, features: int = 64, blocks: int = 23, growth: int = 32):
        super().__init__()
        self.conv_first = nn.Conv2d(3, features, 3, 1, 1)
        self.body = nn.Sequential(*(RRDB(features, growth) for _ in range(blocks)))
        self.conv_body = nn.Conv2d(features, features, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(features, features, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(features, features, 3, 1, 1)
        self.conv_hr = nn.Conv2d(features, features, 3, 1, 1)
        self.conv_last = nn.Conv2d(features, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: th.Tensor) -> th.Tensor:
        features = self.conv_first(x)
        features = features + self.conv_body(self.body(features))
        features = self.lrelu(self.conv_up1(tf.interpolate(features, scale_factor=2, mode="nearest")))
        features = self.lrelu(self.conv_up2(tf.interpolate(features, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(features)))


def load_upscaler(device: str, checkpoints_dir: str = script_util.CACHE_PATH) -> RRDBNet:
    weights = script_util.download(WEIGHTS_URL, WEIGHTS_FILENAME, root=checkpoints_dir)
    model = RRDBNet()
    model.load_state_dict(th.load(weights, map_location="cpu", weights_only=True)["params_ema"])
    model = model.eval().requires_grad_(False).to(device)
    return model.half() if device != "cpu" else model


@th.no_grad()
def upscale(model: RRDBNet, images: th.Tensor) -> th.Tensor:
    """Four times larger, for images in [-1, 1]. Tiled, so a large image fits in 8 GB."""
    pixels = images.add(1).div(2).clamp(0, 1).to(next(model.parameters()).dtype)
    height, width = pixels.shape[-2:]
    output = images.new_zeros(*images.shape[:2], height * SCALE, width * SCALE)
    for top in range(0, height, TILE_SIZE):
        for left in range(0, width, TILE_SIZE):
            bottom, right = min(top + TILE_SIZE, height), min(left + TILE_SIZE, width)
            # Each tile keeps only its middle, as the network needs context to match its neighbours.
            pad_top, pad_left = max(top - TILE_MARGIN, 0), max(left - TILE_MARGIN, 0)
            tile = model(pixels[..., pad_top:min(bottom + TILE_MARGIN, height), pad_left:min(right + TILE_MARGIN, width)]).float()
            output[..., top * SCALE:bottom * SCALE, left * SCALE:right * SCALE] = tile[
                ..., (top - pad_top) * SCALE:(bottom - pad_top) * SCALE, (left - pad_left) * SCALE:(right - pad_left) * SCALE]
    return output.mul(2).sub(1).clamp(-1, 1)


def upscale_file(model: RRDBNet, path: str, device: str) -> str:
    """Write the upscaled copy of an image next to it, and return its path."""
    image = tvf.to_tensor(Image.open(path).convert("RGB")).unsqueeze(0).to(device).mul(2).sub(1)
    output = Path(path).with_name(f"{Path(path).stem}_x{SCALE}.png")
    tvf.to_pil_image(upscale(model, image)[0].add(1).div(2).clamp(0, 1).cpu()).save(output)
    return str(output)

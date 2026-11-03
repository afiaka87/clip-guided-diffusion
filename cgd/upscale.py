"""Real-ESRGAN x4plus, for building large images from a small guided one.

Direct 512px spends 18.5 s and 4.1 GiB on an image that scores no better than a 128px
one taken through this network, which adds .06 s and .16 GiB. Network from BasicSR
(Wang et al., BSD-3-Clause), trimmed to what the x4plus checkpoint uses.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import torch as th
import torchvision.transforms.functional as tvf
from PIL import Image
from torch import nn
from torch.nn import functional as tf

from cgd.download import download

WEIGHTS_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
SCALE = 4
TILE_SIZE = 256
TILE_MARGIN = 16  # Context each tile sees beyond its edges, so it matches its neighbours.


class ResidualDenseBlock(nn.Module):
    """Five convolutions, each seeing the block's input and every earlier output."""

    def __init__(self, features: int = 64, growth: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(features, growth, 3, 1, 1)
        self.conv2 = nn.Conv2d(features + growth, growth, 3, 1, 1)
        self.conv3 = nn.Conv2d(features + 2 * growth, growth, 3, 1, 1)
        self.conv4 = nn.Conv2d(features + 3 * growth, growth, 3, 1, 1)
        self.conv5 = nn.Conv2d(features + 4 * growth, features, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: th.Tensor) -> th.Tensor:  # pylint: disable=missing-function-docstring
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(th.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(th.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(th.cat((x, x1, x2, x3), 1)))
        return self.conv5(th.cat((x, x1, x2, x3, x4), 1)) * 0.2 + x


class RRDB(nn.Module):
    """Three residual dense blocks, with a scaled residual around them."""

    def __init__(self, features: int = 64, growth: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(features, growth)
        self.rdb2 = ResidualDenseBlock(features, growth)
        self.rdb3 = ResidualDenseBlock(features, growth)

    def forward(self, x: th.Tensor) -> th.Tensor:  # pylint: disable=missing-function-docstring
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    """ESRGAN's generator: RRDBs at the input resolution, then two nearest upsamplings."""

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

    def forward(self, x: th.Tensor) -> th.Tensor:  # pylint: disable=missing-function-docstring
        features = self.conv_first(x)
        features = features + self.conv_body(self.body(features))
        features = self.lrelu(self.conv_up1(tf.interpolate(features, scale_factor=2, mode="nearest")))
        features = self.lrelu(self.conv_up2(tf.interpolate(features, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(features)))


class Tile(NamedTuple):
    """One piece of a tiled upscale, as (rows, columns) slices."""

    source: tuple[slice, slice]  # What the network sees: the tile and its margin.
    keep: tuple[slice, slice]  # The tile's own part of the network's output.
    target: tuple[slice, slice]  # Where that goes in the whole output.


def tiles(
    height: int, width: int, size: int = TILE_SIZE, margin: int = TILE_MARGIN, scale: int = SCALE
) -> Iterator[Tile]:
    """Tiles covering an image, each `size` square or cut off by the edge, with up to `margin` of context."""
    for top in range(0, height, size):
        for left in range(0, width, size):
            bottom, right = min(top + size, height), min(left + size, width)
            pad_top, pad_left = max(top - margin, 0), max(left - margin, 0)
            yield Tile(
                source=(slice(pad_top, min(bottom + margin, height)), slice(pad_left, min(right + margin, width))),
                keep=(
                    slice((top - pad_top) * scale, (bottom - pad_top) * scale),
                    slice((left - pad_left) * scale, (right - pad_left) * scale),
                ),
                target=(slice(top * scale, bottom * scale), slice(left * scale, right * scale)),
            )


def load_upscaler(device: str) -> RRDBNet:
    """Real-ESRGAN x4plus, frozen, at half precision on a GPU."""
    weights = download(WEIGHTS_URL, WEIGHTS_URL.rpartition("/")[2])
    model = RRDBNet()
    model.load_state_dict(th.load(weights, map_location="cpu", weights_only=True)["params_ema"])
    model = model.eval().requires_grad_(False).to(device)
    return model.half() if device != "cpu" else model


@th.no_grad()
def upscale(model: nn.Module, images: th.Tensor) -> th.Tensor:
    """Images in [-1, 1], four times larger. Tiled, so a large image fits in 8 GB."""
    pixels = images.add(1).div(2).clamp(0, 1).to(next(model.parameters()).dtype)
    height, width = pixels.shape[-2:]
    output = images.new_zeros(*images.shape[:2], height * SCALE, width * SCALE)
    for tile in tiles(height, width):
        output[(..., *tile.target)] = model(pixels[(..., *tile.source)]).float()[(..., *tile.keep)]
    return output.mul(2).sub(1).clamp(-1, 1)


def upscale_file(model: nn.Module, path: str | Path, device: str) -> Path:
    """Write the upscaled copy of an image next to it, and return its path."""
    path = Path(path)
    image = tvf.to_tensor(Image.open(path).convert("RGB")).unsqueeze(0).to(device).mul(2).sub(1)
    output = path.with_name(f"{path.stem}_x{SCALE}.png")
    tvf.to_pil_image(upscale(model, image)[0].add(1).div(2).clamp(0, 1).cpu()).save(output)
    return output

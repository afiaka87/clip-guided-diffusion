"""Real-ESRGAN x4plus on the GPU."""

from pathlib import Path

import pytest
import torch as th
import torchvision.transforms.functional as tvf
from PIL import Image
from torch.nn import functional as tf

from cgd import upscale

pytestmark = pytest.mark.cuda


def image(path: Path, size: int) -> th.Tensor:
    """The image at `path` as a square of `size` on the GPU, in [-1, 1]."""
    pixels = tvf.to_tensor(Image.open(path).convert("RGB").resize((size, size)))
    return pixels.unsqueeze(0).cuda().mul(2).sub(1)


def test_the_output_is_four_times_larger_and_downsamples_to_the_input(photon: Path) -> None:
    small = image(photon, 128)
    large = upscale.upscale(upscale.load_upscaler("cuda"), small)
    assert large.shape[-2:] == (512, 512)
    assert (tf.avg_pool2d(large, upscale.SCALE) - small).abs().mean().item() < 0.1


def test_tiling_matches_upscaling_the_whole_image(photon: Path) -> None:
    """Close, but not exact: the network sees further than the margin, at half precision."""
    small = image(photon, 300)
    model = upscale.load_upscaler("cuda")
    with th.no_grad():
        whole = model(small.add(1).div(2).half()).float().mul(2).sub(1).clamp(-1, 1)
    tiled = upscale.upscale(model, small)
    assert (whole - tiled).abs().mean().item() < 0.01
    seam = upscale.TILE_SIZE * upscale.SCALE
    assert (whole - tiled)[..., seam - 2 : seam + 2, :].abs().mean().item() < 0.02


def test_upscale_file_writes_beside_the_image(photon: Path, tmp_path: Path) -> None:
    path = tmp_path / "0001.png"
    Image.open(photon).convert("RGB").resize((64, 64)).save(path)
    output = upscale.upscale_file(upscale.load_upscaler("cuda"), path, "cuda")
    assert output == tmp_path / "0001_x4.png"
    assert Image.open(output).size == (256, 256)

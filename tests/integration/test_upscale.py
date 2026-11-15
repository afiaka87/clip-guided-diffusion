"""Real-ESRGAN x4plus on the GPU. tests/unit checks the tiling, and test_cli.py that -up saves the copy."""

from pathlib import Path

import pytest
import torchvision.transforms.functional as tvf
from PIL import Image
from torch.nn import functional as tf

from cgd import upscale

pytestmark = pytest.mark.cuda


def test_the_output_is_four_times_larger_and_downsamples_to_the_input(photon: Path) -> None:
    pixels = tvf.to_tensor(Image.open(photon).convert("RGB").resize((128, 128)))
    small = pixels.unsqueeze(0).cuda().mul(2).sub(1)
    large = upscale.upscale(upscale.load_upscaler("cuda"), small)
    assert large.shape[-2:] == (512, 512)
    assert (tf.avg_pool2d(large, upscale.SCALE) - small).abs().mean().item() < 0.1

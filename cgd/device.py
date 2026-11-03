"""Which device to run on."""

import re

import torch as th

from cgd.checks import require


def default_device(device: str | None = None) -> str:
    """`device`, or CUDA whenever PyTorch can use it, or else the CPU."""
    require(
        device is None or re.fullmatch(r"cpu|cuda(:\d+)?", device) is not None,
        f"Unknown device {device!r}. Use cpu, cuda or cuda:<index>.",
    )
    return device or ("cuda" if th.cuda.is_available() else "cpu")

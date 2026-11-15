"""Which device to run on, and how much of its memory a run may take."""

import re
import sys

import torch as th

from cgd.checks import require


def default_device(device: str | None = None) -> str:
    """`device`, or CUDA whenever PyTorch can use it, or else the CPU."""
    require(
        device is None or re.fullmatch(r"cpu|cuda(:\d+)?", device) is not None,
        f"Unknown device {device!r}. Use cpu, cuda or cuda:<index>.",
    )
    return device or ("cuda" if th.cuda.is_available() else "cpu")


def limit_to_vram(device: str) -> None:
    """On Windows, make running out of VRAM an out of memory error, as it is elsewhere.

    There NVIDIA's driver backs a CUDA allocation that does not fit in VRAM with system
    memory instead of failing it, which is many times slower and can take all of the
    system's. So PyTorch may only take what the GPU has free, plus what it already holds.
    """
    place = th.device(device)
    if sys.platform != "win32" or place.type != "cuda":
        return
    index = th.cuda.current_device() if place.index is None else place.index  # The fraction needs an index.
    free, total = th.cuda.mem_get_info(index)
    th.cuda.set_per_process_memory_fraction(min((free + th.cuda.memory_reserved(index)) / total, 1.0), index)

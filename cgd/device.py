"""Which device to run on, how much of its memory a run may take, and its random streams."""

import re
import sys
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import NamedTuple

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


class RandomState(NamedTuple):
    """Where the random streams a run draws from stand: the CPU's, and its device's if that is a GPU."""

    cpu: th.Tensor
    cuda: th.Tensor | None


def forked_rng(device: str) -> AbstractContextManager[None]:
    """Fork the CPU's random stream and the device's, so what runs inside draws none of the run's random numbers.

    Both matter: a model built straight on the GPU initialises its weights from the GPU's stream,
    which the seed's initial noise is drawn from.
    """
    place = th.device(device)
    return th.random.fork_rng(devices=[place] if place.type == "cuda" else [])


def random_state(device: str) -> RandomState:
    """The state of the random streams, to draw the same numbers again with replayed_rng()."""
    place = th.device(device)
    return RandomState(th.get_rng_state(), th.cuda.get_rng_state(place) if place.type == "cuda" else None)


@contextmanager
def replayed_rng(device: str, state: RandomState) -> Iterator[None]:
    """Within this, random numbers are the ones drawn after `state` was taken. Afterwards the streams go on as before.

    It is how every image of a sequence gets the noise its prompts would get alone.
    """
    with forked_rng(device):
        th.set_rng_state(state.cpu)
        if state.cuda is not None:
            th.cuda.set_rng_state(state.cuda, th.device(device))
        yield

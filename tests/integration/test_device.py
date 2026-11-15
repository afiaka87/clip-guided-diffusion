"""How much of the GPU's memory a run may take."""

import pytest
import torch as th

from cgd.device import limit_to_vram

pytestmark = pytest.mark.cuda


def test_a_run_fails_when_it_outgrows_the_gpu_instead_of_borrowing_system_memory() -> None:
    """Windows' driver would back the allocation with system memory, many times slower, where elsewhere it fails."""
    th.cuda.empty_cache()
    limit_to_vram("cuda")
    free, _ = th.cuda.mem_get_info()
    # What PyTorch holds and does not use is room too: the models earlier tests left loaded have gaps between them.
    unused = th.cuda.memory_reserved() - th.cuda.memory_allocated()
    with pytest.raises(th.cuda.OutOfMemoryError):
        th.empty(free + unused + (1 << 30), dtype=th.uint8, device="cuda")

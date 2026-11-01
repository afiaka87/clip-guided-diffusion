import os

# PyTorch reads this at its first CUDA allocation. Without it, fragmentation alone puts
# configurations out of reach that fit in VRAM. As of 2.9 its CUDA allocator only reads
# the old name, and warns that the old name is deprecated unless the new one is set too.
if not {"PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"} & os.environ.keys():
    os.environ["PYTORCH_ALLOC_CONF"] = os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from . import *

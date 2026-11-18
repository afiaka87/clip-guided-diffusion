#!/bin/bash
# On the instance, once the upload is done: install the environment and check what the run needs is there.
#   bash /workspace/code/finetune/vast/setup.sh
set -euo pipefail
cd /workspace/code
command -v git > /dev/null || (apt-get update -qq && apt-get install -y -qq git)
command -v uv > /dev/null || [ -x ~/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
uv sync --frozen --python 3.12  # 3.12 as tested in WSL. On 3.11 nothing brings setuptools, which `clip` imports.
uv run --no-sync python - <<'PY'
from pathlib import Path

import torch

from finetune import data

count = torch.cuda.device_count()
print(f"PyTorch {torch.__version__}, {count} GPUs: {[torch.cuda.get_device_name(i) for i in range(count)]}")
assert count >= 1, "No GPU is visible."
for i in range(count):
    torch.zeros(1, device=f"cuda:{i}")  # Fails here if this PyTorch cannot run on the card.
for root in [Path("/workspace/data/laion-aesthetic-best1m-256"), Path("/workspace/data/imagenet-1k-clean-256/train")]:
    source = data.open_source(root.parts[3], root, ["ViT-B/32", "PE-Core-B-16"], split="train", keep_watermarked=False, where=None)
    item = source.dataset[len(source.dataset) - 1]
    print(f"{root}: {len(source.dataset):,} images, the last one {tuple(item['image'].shape)}")
PY
echo "setup is done"

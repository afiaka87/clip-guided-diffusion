#!/bin/bash
cd "$(dirname "$0")"
while pgrep -f "python -I run.py" >/dev/null; do sleep 5; done
echo "run.py finished $(date)" >> chain.log
uv run --project /home/sam/GitHub/clip-guided-diffusion --frozen --with spandrel python gan.py >> chain.log 2>&1
echo "gan finished $(date) rc=$?" >> chain.log
/home/sam/GitHub/clip-guided-diffusion/.venv/bin/python -I variants.py >> chain.log 2>&1
echo "variants finished $(date) rc=$?" >> chain.log
CUDA_VISIBLE_DEVICES= /home/sam/GitHub/clip-guided-diffusion/.venv/bin/python -I score.py >> chain.log 2>&1
echo "score finished $(date) rc=$?" >> chain.log

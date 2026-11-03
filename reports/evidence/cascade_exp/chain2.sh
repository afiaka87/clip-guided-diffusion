#!/bin/bash
cd "$(dirname "$0")"
PY=/home/sam/GitHub/clip-guided-diffusion/.venv/bin/python
step() { name=$1; shift; "$@" >> chain.log 2>&1; code=$?; echo "$name finished $(date) rc=$code" >> chain.log; }
step run2 $PY -I run2.py
step gan uv run --project /home/sam/GitHub/clip-guided-diffusion --frozen --with spandrel python -I gan.py
step variants $PY -I variants.py
step score env CUDA_VISIBLE_DEVICES= $PY -I score.py
echo "chain2 complete" >> chain.log

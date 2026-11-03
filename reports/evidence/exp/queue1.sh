#!/bin/bash
# Run on an idle card only: other researchers are measuring on it too.
waited=0
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && [ $waited -lt 1500 ]; do sleep 10; waited=$((waited+10)); done
echo "gpu wait $waited s"; nvidia-smi --query-gpu=memory.used --format=csv,noheader
cd /home/sam/GitHub/clip-guided-diffusion
PYTHONPATH=/tmp/claude-1000/-home-sam-GitHub-clip-guided-diffusion/34d680cf-d9a7-412c-9f94-d042cb1cd642/scratchpad/exp uv run --quiet python /tmp/claude-1000/-home-sam-GitHub-clip-guided-diffusion/34d680cf-d9a7-412c-9f94-d042cb1cd642/scratchpad/exp/e1b_determinism.py 2>&1 | grep -v Warning | tee /tmp/claude-1000/-home-sam-GitHub-clip-guided-diffusion/34d680cf-d9a7-412c-9f94-d042cb1cd642/scratchpad/exp/e1b.log
PYTHONPATH=/tmp/claude-1000/-home-sam-GitHub-clip-guided-diffusion/34d680cf-d9a7-412c-9f94-d042cb1cd642/scratchpad/exp uv run --quiet python /tmp/claude-1000/-home-sam-GitHub-clip-guided-diffusion/34d680cf-d9a7-412c-9f94-d042cb1cd642/scratchpad/exp/e2_cost.py 2>&1 | grep -v Warning | tee /tmp/claude-1000/-home-sam-GitHub-clip-guided-diffusion/34d680cf-d9a7-412c-9f94-d042cb1cd642/scratchpad/exp/e2.log
PYTHONPATH=/tmp/claude-1000/-home-sam-GitHub-clip-guided-diffusion/34d680cf-d9a7-412c-9f94-d042cb1cd642/scratchpad/exp uv run --quiet python /tmp/claude-1000/-home-sam-GitHub-clip-guided-diffusion/34d680cf-d9a7-412c-9f94-d042cb1cd642/scratchpad/exp/e3_variance.py "ViT-B/32" 128 2>&1 | grep -v Warning | tee /tmp/claude-1000/-home-sam-GitHub-clip-guided-diffusion/34d680cf-d9a7-412c-9f94-d042cb1cd642/scratchpad/exp/e3_vitb32_128.log

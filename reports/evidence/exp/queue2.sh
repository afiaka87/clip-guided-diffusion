#!/bin/bash
# After queue1 finishes, then only on an idle card.
while pgrep -f queue1.sh > /dev/null; do sleep 10; done
waited=0
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && [ $waited -lt 900 ]; do sleep 10; waited=$((waited+10)); done
echo "gpu wait $waited s"
cd /home/sam/GitHub/clip-guided-diffusion
PYTHONPATH=/tmp/claude-1000/-home-sam-GitHub-clip-guided-diffusion/34d680cf-d9a7-412c-9f94-d042cb1cd642/scratchpad/exp uv run --quiet python /tmp/claude-1000/-home-sam-GitHub-clip-guided-diffusion/34d680cf-d9a7-412c-9f94-d042cb1cd642/scratchpad/exp/e5_blend_spectrum.py 2>&1 | grep -v Warning | tee /tmp/claude-1000/-home-sam-GitHub-clip-guided-diffusion/34d680cf-d9a7-412c-9f94-d042cb1cd642/scratchpad/exp/e5.log
PYTHONPATH=/tmp/claude-1000/-home-sam-GitHub-clip-guided-diffusion/34d680cf-d9a7-412c-9f94-d042cb1cd642/scratchpad/exp uv run --quiet python /tmp/claude-1000/-home-sam-GitHub-clip-guided-diffusion/34d680cf-d9a7-412c-9f94-d042cb1cd642/scratchpad/exp/e4_end_to_end.py ofat 128 2>&1 | grep -v Warning | tee /tmp/claude-1000/-home-sam-GitHub-clip-guided-diffusion/34d680cf-d9a7-412c-9f94-d042cb1cd642/scratchpad/exp/e4_ofat_128.log

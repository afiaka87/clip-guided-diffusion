#!/bin/bash
# After queue2: rerun the layout comparison with a distinct crop seed per cell, on more than one model and size.
EXP=/tmp/claude-1000/-home-sam-GitHub-clip-guided-diffusion/34d680cf-d9a7-412c-9f94-d042cb1cd642/scratchpad/exp
while pgrep -f queue2.sh >/dev/null; do sleep 20; done
cd /home/sam/GitHub/clip-guided-diffusion
export PYTHONPATH=$EXP
uv run --quiet python $EXP/e3b_variance.py "ViT-B/32" 128 2>&1 | grep -v Warning | tee $EXP/e3b_vitb32_128.log
uv run --quiet python $EXP/e3b_variance.py "PE-Core-B-16" 128 fp16 2>&1 | grep -v Warning | tee $EXP/e3b_pecore_128.log
uv run --quiet python $EXP/e3b_variance.py "ViT-B/32" 256 2>&1 | grep -v Warning | tee $EXP/e3b_vitb32_256.log

#!/bin/bash
EXP=/tmp/claude-1000/-home-sam-GitHub-clip-guided-diffusion/34d680cf-d9a7-412c-9f94-d042cb1cd642/scratchpad/exp
while pgrep -f queue3.sh >/dev/null; do sleep 20; done
cd /home/sam/GitHub/clip-guided-diffusion
export PYTHONPATH=$EXP
uv run --quiet python $EXP/e4_end_to_end.py confirm 128 "ViT-B/32" 2>&1 | grep -v Warning | tee $EXP/e4_confirm_128.log
uv run --quiet python $EXP/e4_end_to_end.py confirm 128 "PE-Core-B-16" fp16 2>&1 | grep -v Warning | tee $EXP/e4_confirm_128_pecore.log
uv run --quiet python $EXP/e4_end_to_end.py confirm 256 "ViT-B/32" 2>&1 | grep -v Warning | tee $EXP/e4_confirm_256.log

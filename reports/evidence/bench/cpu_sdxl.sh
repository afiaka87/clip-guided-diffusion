#!/bin/bash
SP=/tmp/claude-1000/-home-sam-GitHub-clip-guided-diffusion/34d680cf-d9a7-412c-9f94-d042cb1cd642/scratchpad
PY=/home/sam/GitHub/clip-guided-diffusion/.venv/bin/python
export CUDA_VISIBLE_DEVICES="" BENCH_THREADS=6 BENCH_LAZY=1 PYTHONPATH=$SP/pylib:/home/sam/GitHub/clip-guided-diffusion
cd "$SP/bench" || exit 1
ulimit -v 28000000
BENCH_LAZY= BENCH_CAST=1 BENCH_DEPTH=0.2 "$PY" measure_cpu.py klein4b 512 x0_tiny,x0_tiny,tiny_ckpt,x0_tiny 2>>cpu.err | grep --line-buffered '^{' >> cast2.jsonl
"$PY" measure_cpu.py sdxl 384,512 forward,tiny,tiny_ckpt,x0_tiny 2>>cpu.err | grep --line-buffered '^{' >> cpu.jsonl
echo "SDXL_DONE $(date +%T)" >> cpu.jsonl

#!/bin/bash
# A second size for each U-Net, for a fit in pixels. The tiny decoder paths are the ones that matter.
SP=/tmp/claude-1000/-home-sam-GitHub-clip-guided-diffusion/34d680cf-d9a7-412c-9f94-d042cb1cd642/scratchpad
PY=/home/sam/GitHub/clip-guided-diffusion/.venv/bin/python
export PYTHONPATH=$SP/pylib:/home/sam/GitHub/clip-guided-diffusion
export CUDA_VISIBLE_DEVICES="" BENCH_THREADS=12
cd "$SP/bench" || exit 1
OUT=cpu.jsonl
MODES=forward,tiny,tiny_ckpt,x0_tiny
while pgrep -f 'bench/cpu_queue.sh' >/dev/null; do sleep 10; done
run() { (ulimit -v 18000000; "$PY" measure_cpu.py "$@" 2>>cpu.err | grep --line-buffered '^{' >> "$OUT"); }
run vega 384 $MODES
run sd21 384 $MODES
run sdxl 384 $MODES
run pixart_s512 384 $MODES
echo CPU_QUEUE2_DONE >> "$OUT"

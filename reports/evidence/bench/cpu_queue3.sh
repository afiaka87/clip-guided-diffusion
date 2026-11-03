#!/bin/bash
# fp8 storage with bf16 compute on the two big transformers. Runs after the other queues.
SP=/tmp/claude-1000/-home-sam-GitHub-clip-guided-diffusion/34d680cf-d9a7-412c-9f94-d042cb1cd642/scratchpad
PY=/home/sam/GitHub/clip-guided-diffusion/.venv/bin/python
export PYTHONPATH=$SP/pylib:/home/sam/GitHub/clip-guided-diffusion
export CUDA_VISIBLE_DEVICES="" BENCH_THREADS=12 BENCH_CAST=1
cd "$SP/bench" || exit 1
OUT=cast.jsonl
while pgrep -f 'bench/cpu_queue2?.sh' >/dev/null; do sleep 10; done
deep() { local depth=$1; shift; (ulimit -v 18000000; BENCH_DEPTH=$depth "$PY" measure_cpu.py "$@" 2>>cpu.err | grep --line-buffered '^{' >> "$OUT"); }
deep 0.2 klein4b 512 x0_tiny,tiny_ckpt
deep 0.4 klein4b 512 x0_tiny,tiny_ckpt
deep 0.2 zimage 512 x0_tiny,tiny_ckpt
deep 0.4 zimage 512 x0_tiny,tiny_ckpt
echo CPU_QUEUE3_DONE >> "$OUT"

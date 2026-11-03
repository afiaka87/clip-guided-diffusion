#!/bin/bash
# Allocation accounting for the models that matter most, while the GPU is taken by other work.
SP=/tmp/claude-1000/-home-sam-GitHub-clip-guided-diffusion/34d680cf-d9a7-412c-9f94-d042cb1cd642/scratchpad
PY=/home/sam/GitHub/clip-guided-diffusion/.venv/bin/python
export PYTHONPATH=$SP/pylib:/home/sam/GitHub/clip-guided-diffusion
export CUDA_VISIBLE_DEVICES="" BENCH_THREADS=12
cd "$SP/bench" || exit 1
OUT=cpu.jsonl
ulimit -v 18000000
TINY=forward,full,ckpt,tiny,tiny_ckpt,x0_tiny
NOTINY=forward,full,ckpt,x0_vae
run() { "$PY" measure_cpu.py "$@" 2>>cpu.err | grep '^{' >> "$OUT"; }
deep() { local depth=$1; shift; BENCH_DEPTH=$depth "$PY" measure_cpu.py "$@" 2>>cpu.err | grep '^{' >> "$OUT"; }
run sprint600 512 $NOTINY
run sprint1600 512 $NOTINY
run pixart_s512 512,768 $TINY
deep 0.25 pixart_s512 512 $TINY
deep 0.5 pixart_s512 512 $TINY
run vega 512,768 $TINY
run sd21 512,768 $TINY
run sdxl 512,640 $TINY
deep 0.2 klein4b 512 $TINY
deep 0.4 klein4b 512 $TINY
deep 0.2 zimage 512 $TINY
deep 0.4 zimage 512 $TINY
run kandinsky22 512 forward,full,ckpt,x0_vae
deep 0.25 sd35m 512 $TINY
deep 0.5 sd35m 512 $TINY
deep 0.2 klein4b 768 $TINY
deep 0.4 klein4b 768 $TINY
deep 0.2 zimage 768 $TINY
deep 0.4 zimage 768 $TINY
run ssd1b 512,768 $TINY
deep 0.2 flux1 512 $TINY
deep 0.4 flux1 512 $TINY
run sd15 640 $TINY
echo CPU_SWEEP_DONE >> "$OUT"

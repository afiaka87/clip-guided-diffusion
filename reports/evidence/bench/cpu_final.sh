#!/bin/bash
# One memory-hungry job at a time, most decision-relevant first.
SP=/tmp/claude-1000/-home-sam-GitHub-clip-guided-diffusion/34d680cf-d9a7-412c-9f94-d042cb1cd642/scratchpad
PY=/home/sam/GitHub/clip-guided-diffusion/.venv/bin/python
export CUDA_VISIBLE_DEVICES="" BENCH_THREADS=12
cd "$SP/bench" || exit 1
OUT=cpu.jsonl
TINY=forward,full,ckpt,tiny,tiny_ckpt,x0_tiny
SMALL=forward,tiny,tiny_ckpt,x0_tiny
NOTINY=forward,full,ckpt,x0_vae
effect() { (cd "$SP" && ulimit -v 22000000 && "$PY" -I "$SP/bench/grad_effect.py"); }
run() { (ulimit -v 18000000; PYTHONPATH=$SP/pylib:/home/sam/GitHub/clip-guided-diffusion "$PY" measure_cpu.py "$@" 2>>cpu.err | grep --line-buffered '^{' >> "$OUT"); }
deep() { local depth=$1; shift; (ulimit -v 18000000; BENCH_DEPTH=$depth PYTHONPATH=$SP/pylib:/home/sam/GitHub/clip-guided-diffusion "$PY" measure_cpu.py "$@" 2>>cpu.err | grep --line-buffered '^{' >> "$OUT"); }
cast() { (ulimit -v 18000000; BENCH_CAST=1 BENCH_DEPTH=$1 PYTHONPATH=$SP/pylib:/home/sam/GitHub/clip-guided-diffusion "$PY" measure_cpu.py "$2" 512 x0_tiny,tiny_ckpt 2>>cpu.err | grep --line-buffered '^{' >> cast.jsonl); }
BENCH_PARTS=A BENCH_PROMPTS=3 effect > effect_a.log 2> effect_a.err
echo "EFFECT_A_DONE $(date +%T)" >> "$OUT"
BENCH_PARTS=B BENCH_PROMPTS=2 BENCH_SEEDS=0 effect > effect_b.log 2> effect_b.err
echo "EFFECT_B_DONE $(date +%T)" >> "$OUT"
run sprint600 192,256,320 $NOTINY
run sprint1600 192,256,320 $NOTINY
run vega 512 $TINY
run sd21 512 $TINY
run sdxl 512 $SMALL
run vega 384 $SMALL
run sd21 384 $SMALL
run sdxl 384 $SMALL
run pixart_s512 384 $SMALL
cast 0.2 klein4b
cast 0.4 klein4b
cast 0.2 zimage
cast 0.4 zimage
deep 0.2 klein4b 384 $SMALL
deep 0.4 klein4b 384 $SMALL
deep 0.25 sd35m 512 $SMALL
deep 0.5 sd35m 512 $SMALL
run kandinsky22 512 forward,full,ckpt,x0_vae
echo CPU_FINAL_DONE >> "$OUT"

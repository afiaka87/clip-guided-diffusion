#!/bin/bash
# One memory-hungry job at a time: the machine is already into swap from the other work.
SP=/tmp/claude-1000/-home-sam-GitHub-clip-guided-diffusion/34d680cf-d9a7-412c-9f94-d042cb1cd642/scratchpad
PY=/home/sam/GitHub/clip-guided-diffusion/.venv/bin/python
export PYTHONPATH=$SP/pylib:/home/sam/GitHub/clip-guided-diffusion
export CUDA_VISIBLE_DEVICES="" BENCH_THREADS=12
cd "$SP/bench" || exit 1
OUT=cpu.jsonl
TINY=forward,full,ckpt,tiny,tiny_ckpt,x0_tiny
NOTINY=forward,full,ckpt,x0_vae
while pgrep -f 'measure_cpu.py pixart_s512 512,768' >/dev/null; do sleep 5; done
run() { (ulimit -v 18000000; "$PY" measure_cpu.py "$@" 2>>cpu.err | grep --line-buffered '^{' >> "$OUT"); }
deep() { local depth=$1; shift; (ulimit -v 18000000; BENCH_DEPTH=$depth "$PY" measure_cpu.py "$@" 2>>cpu.err | grep --line-buffered '^{' >> "$OUT"); }
deep 0.25 pixart_s512 512 $TINY
deep 0.5 pixart_s512 512 $TINY
deep 0.2 klein4b 512 $TINY
deep 0.4 klein4b 512 $TINY
deep 0.2 zimage 512 $TINY
deep 0.4 zimage 512 $TINY
(cd "$SP" && ulimit -v 20000000 && env -u PYTHONPATH "$PY" -I "$SP/bench/grad_agree_cpu.py" > "$SP/bench/grad_agree_cpu.log" 2> "$SP/bench/grad_agree_cpu.err"; echo "grad_agree_cpu exit $?" >> "$SP/bench/grad_agree_cpu.log")
run sprint600 192,256,320 $NOTINY
run sprint1600 192,256,320 $NOTINY
run vega 512,768 $TINY
run sd21 512,768 $TINY
run sdxl 512 $TINY
deep 0.25 sd35m 512 $TINY
deep 0.5 sd35m 512 $TINY
deep 0.2 klein4b 768 $TINY
deep 0.4 klein4b 768 $TINY
deep 0.2 zimage 768 $TINY
deep 0.4 zimage 768 $TINY
run kandinsky22 512 forward,full,ckpt,x0_vae
echo CPU_QUEUE_DONE >> "$OUT"

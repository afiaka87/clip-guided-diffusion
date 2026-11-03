#!/bin/bash
# DC-AE models, at sizes small enough to finish, for a fit in pixels. Runs after cpu_sweep.sh.
SP=/tmp/claude-1000/-home-sam-GitHub-clip-guided-diffusion/34d680cf-d9a7-412c-9f94-d042cb1cd642/scratchpad
PY=/home/sam/GitHub/clip-guided-diffusion/.venv/bin/python
export PYTHONPATH=$SP/pylib:/home/sam/GitHub/clip-guided-diffusion
export CUDA_VISIBLE_DEVICES="" BENCH_THREADS=12
cd "$SP/bench" || exit 1
ulimit -v 18000000
until ! pgrep -f 'bench/cpu_sweep.sh' >/dev/null; do sleep 10; done
for spec in sprint600 sprint1600; do
    "$PY" measure_cpu.py $spec 192,256,320 forward,full,ckpt,x0_vae 2>>cpu.err | grep '^{' >> cpu.jsonl
done
echo CPU_SWEEP2_DONE >> cpu.jsonl

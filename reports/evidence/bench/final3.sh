#!/bin/bash
# Kandinsky's one row, then the other two prompts of Part B. Runs after cpu_final2.sh is stopped.
# Sibling experiments share this machine and swap is full, so this stops itself before they suffer.
SP=/tmp/claude-1000/-home-sam-GitHub-clip-guided-diffusion/34d680cf-d9a7-412c-9f94-d042cb1cd642/scratchpad
PY=/home/sam/GitHub/clip-guided-diffusion/.venv/bin/python
export CUDA_VISIBLE_DEVICES="" BENCH_THREADS=12
cd "$SP/bench" || exit 1
OUT=cpu.jsonl
FLOOR_MIB=1500

available() { awk '/MemAvailable/ { print int($2 / 1024) }' /proc/meminfo; }

guard() {
    while sleep 5; do
        local free_mib
        free_mib=$(available)
        if [ "$free_mib" -lt "$FLOOR_MIB" ]; then
            echo "MEM_GUARD_STOP $(date +%T) available ${free_mib} MiB" >> "$OUT"
            touch guard_tripped
            pkill -f 'measure_cpu.py kandinsky22'
            pkill -f 'bench/grad_effect.py'
            return
        fi
    done
}

rm -f guard_tripped
guard &
GUARD=$!

(ulimit -v 21000000; PYTHONPATH=$SP/pylib:/home/sam/GitHub/clip-guided-diffusion "$PY" measure_cpu.py kandinsky22 512 x0_vae 2>>cpu.err | grep --line-buffered '^{' >> "$OUT")
echo "KANDINSKY_DONE $(date +%T)" >> "$OUT"

if [ ! -e guard_tripped ]; then
    (cd "$SP" && ulimit -v 22000000 && BENCH_PARTS=B BENCH_START_PROMPT=2 BENCH_PROMPTS=4 BENCH_SEEDS=0 "$PY" -I "$SP/bench/grad_effect.py") > effect_b2.log 2> effect_b2.err
fi
echo "EFFECT_B2_DONE $(date +%T)" >> "$OUT"
kill "$GUARD" 2>/dev/null

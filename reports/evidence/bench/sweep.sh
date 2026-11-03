#!/bin/bash
# Runs each model only when no other process is computing on the GPU, and records any
# process that appears during a run, because that makes an out of memory result suspect.
SP=/tmp/claude-1000/-home-sam-GitHub-clip-guided-diffusion/34d680cf-d9a7-412c-9f94-d042cb1cd642/scratchpad
OUT=$SP/bench/results.jsonl
PY=/home/sam/GitHub/clip-guided-diffusion/.venv/bin/python
export PYTHONPATH=$SP/pylib:/home/sam/GitHub/clip-guided-diffusion
cd "$SP/bench" || exit 1

others() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -; }

wait_free() {
    local quiet=0
    while [ "$quiet" -lt 2 ]; do  # Two clean reads, in case the other job is between steps.
        if [ -z "$(others)" ]; then quiet=$((quiet + 1)); else
            quiet=0
            echo "$(date +%T) waiting for: $(others)" >> "$SP/bench/wait.log"
        fi
        sleep 5
    done
}

run() {
    wait_free
    local start_stamp seen
    start_stamp=$(date +%T)
    "$PY" measure.py "$@" 2>>"$SP/bench/stderr.log" | grep '^{' > "$SP/bench/current.jsonl" &
    local job=$!
    seen=""
    while kill -0 "$job" 2>/dev/null; do
        sleep 2
        for pid in $(others | tr ',' ' '); do
            if ! ps -o cmd= -p "$pid" | grep -q measure.py; then seen="$pid"; fi
        done
    done
    wait "$job"
    cat "$SP/bench/current.jsonl" >> "$OUT"
    echo "{\"spec\": \"$1\", \"start\": \"$start_stamp\", \"end\": \"$(date +%T)\", \"other_gpu_pid\": \"$seen\"}" >> "$OUT"
}

wait_free
"$PY" adm_baseline.py > "$SP/bench/adm_baseline.log" 2>&1

run sd15 --sizes 512 640
run sd21 --sizes 512 768
run sdturbo --sizes 512
run sdxl --sizes 512 768 1024
run ssd1b --sizes 512 768 1024
run vega --sizes 512 768 1024
run kolors --sizes 512 768 1024
run kandinsky22 --sizes 512 768
run pixart_a512 --sizes 512
run pixart_s512 --sizes 512
run pixart_s1024 --sizes 512 768 1024
run sana600_512 --sizes 512 768
run sana1600_512 --sizes 512 768
run sana1600_1024 --sizes 768 1024
run sprint600 --sizes 512 1024
run sprint1600 --sizes 512 1024
run sd35m --sizes 512 768 1024
run flux1 --sizes 384 512 768 1024
run klein4b --sizes 384 512 768 1024
run zimage --sizes 384 512 768 1024
run auraflow --sizes 512 768
run lumina2 --sizes 512 768
run dit256 --sizes 256
run dit512 --sizes 512
run prx512 --sizes 512
run edm2xs_proxy --sizes 512
run edm2s_proxy --sizes 512
echo DONE >> "$OUT"

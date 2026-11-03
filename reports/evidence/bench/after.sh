#!/bin/bash
# After the sweep: gradient agreement on real weights, then the cost of emulated bf16 on this card.
SP=/tmp/claude-1000/-home-sam-GitHub-clip-guided-diffusion/34d680cf-d9a7-412c-9f94-d042cb1cd642/scratchpad
PY=/home/sam/GitHub/clip-guided-diffusion/.venv/bin/python
export PYTHONPATH=$SP/pylib:/home/sam/GitHub/clip-guided-diffusion
cd "$SP/bench" || exit 1
others() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -; }
wait_free() {
    local quiet=0
    while [ "$quiet" -lt 2 ]; do
        if [ -z "$(others)" ]; then quiet=$((quiet + 1)); else quiet=0; fi
        sleep 5
    done
}
until grep -q '^DONE' results.jsonl 2>/dev/null; do
    pgrep -f sweep.sh >/dev/null || exit 3
    sleep 10
done
wait_free
"$PY" grad_agree.py > grad_agree.log 2> grad_agree.err
echo "grad_agree exit $?" >> grad_agree.log
wait_free
for dtype in fp16 bf16; do
    BENCH_DTYPE=$dtype "$PY" measure.py klein4b --sizes 512 --modes forward x0_tiny tiny_ckpt 2>/dev/null | grep '^{' | sed "s/^{/{\"dtype\": \"$dtype\", /" >> dtype.jsonl
done
echo AFTER_DONE

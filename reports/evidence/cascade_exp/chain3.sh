#!/bin/bash
cd "$(dirname "$0")"
PY=/home/sam/GitHub/clip-guided-diffusion/.venv/bin/python
while ! grep -q "chain2 complete" chain.log; do sleep 10; done
$PY -I run3.py >> chain.log 2>&1; echo "run3 finished rc=$? $(date)" >> chain.log
env CUDA_VISIBLE_DEVICES= $PY -I score.py >> chain.log 2>&1; echo "rescore finished rc=$? $(date)" >> chain.log
env CUDA_VISIBLE_DEVICES= $PY -I crops.py >> chain.log 2>&1; echo "crops finished rc=$? $(date)" >> chain.log
$PY -I sheet.py >> chain.log 2>&1
$PY -I analysis.py > analysis.txt 2>&1
echo "all complete" >> chain.log

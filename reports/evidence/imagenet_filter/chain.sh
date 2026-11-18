#!/bin/bash
# Waits for the stage B run in progress, finishes any shard it left, then stage C for train and stage D for both splits.
set -u
HERE=$(cd "$(dirname "$0")" && pwd -W)
BENCH=$HOME/AppData/Local/Temp/claude/47f7e6194d69b690/c4d5a3e0-1b30-45de-b1cc-77fad810a0c7/scratchpad/bench
L=/d/imagenet-filter/logs
while powershell -NoProfile -Command "exit [int]((Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*stage_b_embed*' } | Measure-Object).Count -eq 0)"; do sleep 30; done
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1
(cd "$BENCH" && CGD_SRC="D:/imagenet-filter/train/crops256" CGD_STAGEB="D:/imagenet-filter/train/stageB" CGD_STAGE_A="D:/imagenet-filter/train/selection/stage_a.parquet" uv run python -W ignore "$HERE/../laion_filter/pipeline/stage_b_embed.py" > $L/stage_b_train_rerun.txt 2>&1)
echo "stage B train complete $(date)" | tee -a $L/run.txt
bash "$HERE/run_rest.sh" train c || { echo "stage C train failed $(date)" | tee -a $L/run.txt; exit 1; }
bash "$HERE/run_rest.sh" validation d || { echo "stage D validation failed $(date)" | tee -a $L/run.txt; exit 1; }
bash "$HERE/run_rest.sh" train d || { echo "stage D train failed $(date)" | tee -a $L/run.txt; exit 1; }

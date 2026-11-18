#!/bin/bash
# Stage B (resumes) -> stage C -> stage D for both tiers (best: captioned on the GPU; clean: CPU, in parallel).
set -u
B=$HOME/AppData/Local/Temp/claude/47f7e6194d69b690/c4d5a3e0-1b30-45de-b1cc-77fad810a0c7/scratchpad/bench
P='C:\Users\Clay\GitHub\clip-guided-diffusion\.claude\worktrees\imagenet-dataset-filtering-ce2d4b\reports\evidence\laion_filter\pipeline'
L=/d/laion2b-filter/logs; mkdir -p $L
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONUNBUFFERED=1
cd "$B"
# Wait for the stage-B run already in progress, then rerun it so any incomplete shard is finished.
while powershell -NoProfile -Command "exit [int]((Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*stage_b_embed*' } | Measure-Object).Count -eq 0)"; do sleep 30; done
echo "stage B first pass finished $(date)" | tee -a $L/overnight.txt
uv run python -W ignore "$P\stage_b_embed.py" >> $L/stage_b_rerun.txt 2>&1
echo "stage B complete $(date)" | tee -a $L/overnight.txt
uv run python -W ignore "$P\stage_c_select.py" > $L/stage_c.txt 2>&1 || { echo "stage C failed $(date)" | tee -a $L/overnight.txt; exit 1; }
echo "stage C complete $(date)" | tee -a $L/overnight.txt
uv run python -W ignore "$P\stage_d_write.py" clean > $L/stage_d_clean.txt 2>&1 &
uv run python -W ignore "$P\stage_d_write.py" best > $L/stage_d_best.txt 2>&1
wait
echo "stage D complete $(date)" | tee -a $L/overnight.txt

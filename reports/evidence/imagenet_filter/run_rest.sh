#!/bin/bash
# After prepare.py, select_a.py and stage B for a split: stage C (junk, near-duplicates, scores), then stage D
# (captioned webdataset shards). Reuses laion_filter/pipeline through its CGD_ and STAGEC_ overrides.
#   bash run_rest.sh train c     # only stage C        bash run_rest.sh train d     # only stage D
set -eu
split=$1; steps=${2:-cd}
HERE=$(cd "$(dirname "$0")" && pwd -W)
BENCH=${BENCH:-$HOME/AppData/Local/Temp/claude/47f7e6194d69b690/c4d5a3e0-1b30-45de-b1cc-77fad810a0c7/scratchpad/bench}  # holds laion_filter/bench/pyproject.toml's environment
W="D:/imagenet-filter/$split"; L=/d/imagenet-filter/logs; mkdir -p $L
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1
export CGD_SRC="$W/crops256" CGD_STAGEB="$W/stageB" CGD_STAGE_A="$W/selection/stage_a.parquet" STAGEC_OUT="$W/selection"
cd "$BENCH"
export STAGEC_CFG=$(uv run python -c "import json,sys; c=json.load(open(sys.argv[1])); s=c.pop(sys.argv[2]); [c.pop(k) for k in ('train','validation','_note') if k in c]; c.update(s); print(json.dumps(c))" "$HERE/calibration.json" $split)
if [[ $steps == *c* ]]; then
  uv run python -W ignore "$HERE/../laion_filter/pipeline/stage_c_select.py" > $L/stage_c_$split.txt 2>&1
  echo "stage C $split complete $(date)" | tee -a $L/run.txt
fi
if [[ $steps == *d* ]]; then
  SEL_PATH="$W/selection/clean.parquet" OUT_DIR="F:/imagenet-1k-clean-256/$split" \
    uv run python -W ignore "$HERE/../laion_filter/pipeline/stage_d_write.py" best > $L/stage_d_$split.txt 2>&1
  echo "stage D $split complete $(date)" | tee -a $L/run.txt
fi

#!/bin/bash
# On the instance: the run, on every GPU, started again from its checkpoint if it stops with an error.
#   bash /workspace/code/finetune/vast/train.sh              # The run: HOURS=12 by default
# It uses every GPU the instance has, or those in GPUS (such as GPUS=0,1,2,3). A step is TOTAL images (32 by
# default) whatever the number of GPUs, so that runs on different machines train alike: TOTAL / GPUs on each,
# over as many batches as their memory needs (`--accumulate`). Measured on an RTX 5090 with the running average
# on: 9.3 GB and 3.0 GB an image, so with 5 GB kept for the samples, 5 images fit a 32 GB card. Four of them
# take a step of 32 as two batches of 4 each, at 36.9 images a second. BATCH and ACCUMULATE set them directly.
#   SMOKE=1 bash /workspace/code/finetune/vast/train.sh      # 40 steps and 20 more resumed, in /workspace/smoke
set -uo pipefail
cd /workspace/code
export PATH=$HOME/.local/bin:$PATH
gpus=${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)}
data="--laion /workspace/data/laion-aesthetic-best1m-256 --imagenet /workspace/data/imagenet-1k-clean-256/train"
count=$(echo "$gpus" | tr ',' '
' | wc -l)
memory=$(nvidia-smi --id="$gpus" --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1)
fits=$(( (memory - 14300) / 3100 ))
[ "$fits" -lt 1 ] && echo "A GPU with $memory MB is too small for this model with its running average." && exit 1
TOTAL=${TOTAL:-32}
share=$(( (TOTAL + count - 1) / count ))
ACCUMULATE=${ACCUMULATE:-$(( (share + fits - 1) / fits ))}
BATCH=${BATCH:-$(( (share + ACCUMULATE - 1) / ACCUMULATE ))}
[ "$BATCH" -gt "$fits" ] && echo "WARNING: $BATCH images a GPU is more than the $fits that should fit."
spare=$(( $(nproc) / count - 2 ))
WORKERS=${WORKERS:-$(( spare > 8 ? 8 : (spare < 2 ? 2 : spare) ))}
echo "GPUs $gpus of $memory MB: $BATCH images a batch on each, $ACCUMULATE batches a step, $(( BATCH * count * ACCUMULATE )) images a step, $WORKERS loader processes each"
launch="uv run --no-sync python -m finetune.parallel --devices $gpus finetune.cond $data --batch $BATCH --accumulate $ACCUMULATE --workers $WORKERS"

if [ -n "${SMOKE:-}" ]; then
  small="--out /workspace/smoke --log-every 10 --eval-every 20 --samples 2 --sample-steps 8 --probe-batches 4 --memory-fraction 0.97"
  $launch $small --max-steps 40 && $launch $small --max-steps 60 --resume && echo "the smoke run passed"
  exit
fi

run="--out /workspace/run --hours ${HOURS:-12} --save-minutes 15 --memory-fraction 0.97 --resume ${EXTRA:-}"
[ -n "${WANDB_API_KEY:-}" ] || [ -f ~/.netrc ] && run="$run --wandb --wandb-name ${NAME:-cond256-vast}"
for attempt in $(seq 1 20); do
  $launch $run 2>&1 | tee -a /workspace/run.log
  [ "${PIPESTATUS[0]}" -eq 0 ] && echo "the run finished" && break
  echo "attempt $attempt stopped with an error at $(date). Resuming in 30 s." | tee -a /workspace/run.log
  sleep 30
done

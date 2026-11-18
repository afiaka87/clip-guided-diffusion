#!/bin/bash
# Keep a copy of the run here while it trains: its checkpoint, log and samples, every 20 minutes.
#   bash finetune/vast/pull.sh <host> <port>
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/common.sh"
TO=${TO:-/d/cgd-finetune/cond256-vast}
mkdir -p "$TO"
while true; do
  # The checkpoint is written beside itself and renamed, so a copy is never half of one.
  copy --exclude '*.tmp' --exclude wandb "$REMOTE:/workspace/run/" "$TO/" | tail -2
  date
  sleep "${EVERY:-1200}"
done

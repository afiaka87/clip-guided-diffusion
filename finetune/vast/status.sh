#!/bin/bash
# Look at the run on the instance from this machine, or start it there in tmux.
#   bash finetune/vast/status.sh <host> <port>           # GPUs, disk, and the end of the log
#   bash finetune/vast/status.sh <host> <port> smoke     # Start the smoke run in the tmux session `run`
#   bash finetune/vast/status.sh <host> <port> start     # Start the run there. HOURS and EXTRA pass through.
#   bash finetune/vast/status.sh <host> <port> shell     # A shell on the instance
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/common.sh"
script=/workspace/code/finetune/vast/train.sh
case "${3:-status}" in
  smoke) $SSH "$REMOTE" "tmux new-session -d -s run 'SMOKE=1 bash $script 2>&1 | tee /workspace/smoke.log'" && echo started ;;
  start) $SSH "$REMOTE" "tmux new-session -d -s run 'HOURS=${HOURS:-12} EXTRA=\"${EXTRA:-}\" bash $script'" && echo started ;;
  shell) $SSH -t "$REMOTE" ;;
  status)
    $SSH "$REMOTE" '
      nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
      df -h /workspace | tail -1
      tmux ls 2>&1 | head -3
      for log in /workspace/run/log.jsonl /workspace/smoke/log.jsonl; do [ -f $log ] && echo "== $log" && tail -n 6 $log | cut -c1-400; done
      for log in /workspace/run.log /workspace/smoke.log; do [ -f $log ] && echo "== $log" && grep -v "^\[rank" $log | tail -n 5 | cut -c1-300; done
      true' ;;
  *) echo "status, smoke, start or shell" ;;
esac

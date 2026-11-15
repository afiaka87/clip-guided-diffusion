#!/bin/sh
# Follow-ups to recurrence's aesthetic gain on -uncond: does it stack with 90 steps, does a third
# pass help, and does recurring below t 300 add anything. $1 is the guide and $2 the arm suffix.
set -e
cd "$(dirname "$0")"
G="$1"; S="$2"
run() { uv run python run_arm.py "$@"; }
run "r2n90$S" search:generate "{\"clip_model\": \"$G\", \"recurrence\": 2, \"steps\": 90}"
run "r3$S" search:generate "{\"clip_model\": \"$G\", \"recurrence\": 3}"
run "r2all$S" search:generate "{\"clip_model\": \"$G\", \"recurrence\": 2, \"recur_from\": 0}"

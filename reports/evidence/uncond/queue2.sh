#!/bin/sh
# Search on -uncond: recurrence against more steps at equal time, the high-noise rollout,
# and selection (best-of-4, Feynman-Kac steering, pruning 8 -> 4 -> 2) by PE-Core-L-14-336,
# which reports nothing. $1 is the guide and $2 the arm suffix.
set -e
cd "$(dirname "$0")"
G="$1"; S="$2"
run() { uv run python run_arm.py "$@"; }
run "r2$S" search:generate "{\"clip_model\": \"$G\", \"recurrence\": 2}"
run "n90$S" cgd.adm:generate "{\"unconditional\": true, \"clip_model\": \"$G\", \"steps\": 90}"
run "roll3$S" search:generate "{\"clip_model\": \"$G\", \"rollout_calls\": 3}"
run "bo4$S" search:generate "{\"clip_model\": \"$G\", \"particles\": 4}"
run "fk4$S" search:generate "{\"clip_model\": \"$G\", \"particles\": 4, \"events\": [16, 19, 22, 25, 28], \"keep\": [null, null, null, null, null]}"
run "prune$S" search:generate "{\"clip_model\": \"$G\", \"particles\": 8, \"events\": [15, 30], \"keep\": [4, 2]}"

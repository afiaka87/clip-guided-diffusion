#!/bin/sh
# Recurrence as cgd now has it: -uncond's new default through the command's own code, a rerun that
# keeps the second-order correction, and recurrence on the class-conditional default at 128.
# $1 is the guide for -uncond and $2 the arm suffix.
set -e
cd "$(dirname "$0")"
G="$1"; S="$2"
run() { uv run python run_arm.py "$@"; }
run "cgd_uncond$S" cgd.adm:generate "{\"unconditional\": true, \"clip_model\": \"$G\"}"
run "r2c$S" search:generate "{\"clip_model\": \"$G\", \"recurrence\": 2, \"correct_reruns\": true}"
run cc128 cgd.adm:generate '{"recurrence": 1}'
run cc128_r2 cgd.adm:generate '{"recurrence": 2}'

#!/bin/sh
# The first runs on the RTX 4070: the baselines its arms pair against (the 2070's images are not
# the same images), then queue4, which the 2070 never ran.
set -e
cd "$(dirname "$0")"
G="ViT-B/32+PE-Core-B-16"
run() { uv run python run_arm.py "$@"; }
run uncond_ens4070 cgd.adm:generate "{\"unconditional\": true, \"clip_model\": \"$G\", \"recurrence\": 1}"
run r2_e4070 search:generate "{\"clip_model\": \"$G\", \"recurrence\": 2}"
sh queue4.sh "$G" _e

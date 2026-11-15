#!/bin/sh
# The re-baseline: -uncond, -uncond with the ensemble, SD-Turbo and class-conditional 256, all at the defaults.
set -e
cd "$(dirname "$0")"
run() { uv run python run_arm.py "$@"; }
run uncond cgd.adm:generate '{"unconditional": true}'
run uncond_ens cgd.adm:generate '{"unconditional": true, "clip_model": "ViT-B/32+PE-Core-B-16"}'
run turbo cgd.backbone:generate '{"backbone": "sd-turbo"}'
run cc256 cgd.adm:generate '{"image_size": 256}'

#!/bin/bash

caption='"An armchair with the texture of an avocado. An armchair with the shape of an avocado. An armchair imitating an avocado."'
# or grab the first line from a file
# caption=$(cat input.txt | head -n 1)
# --use_fp16 \

python cgd.py \
  --num_cutouts=8 \
  --prefix='outputs' \
  --batch_size=1 \
  --clip_guidance_scale=5000 \
  --tv_scale=100 \
  --seed=0 \
  --save_frequency=100 \
  --diffusion_steps=1000 \
  --timestep_respacing=1000 \
  --cutout_power=1.0 \
  --clip_model=ViT-B/16 \
  $caption

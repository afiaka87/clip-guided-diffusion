#!/bin/bash
# Copy the code, the two datasets and the pretrained models to a vast.ai instance. Run again to resume.
#   bash finetune/vast/upload.sh <host> <port>
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/common.sh"
LAION=${LAION:-/f/laion-aesthetic-best1m-256}
IMAGENET=${IMAGENET:-/f/imagenet-1k-clean-256/train}
CACHE=${CACHE:-$HOME/.cache/clip-guided-diffusion}

$SSH "$REMOTE" "mkdir -p /workspace/code /workspace/data/imagenet-1k-clean-256 /root/.cache/clip-guided-diffusion/clip"

# The code as it is in this folder, without what is built or made from it.
copy --delete --exclude .venv --exclude .git --exclude .claude --exclude outputs --exclude reports \
  --exclude __pycache__ --exclude '*.pyc' --exclude .ruff_cache --exclude .pytest_cache \
  "$here/../../" "$REMOTE:/workspace/code/"

# What the run loads: the U-Net, the class names' text features, both guiding CLIP models and the judge.
copy "$CACHE"/256x256_diffusion.pt "$CACHE"/imagenet_text_features_ViT-B-32.pt \
  "$CACHE"/imagenet_text_features_PE-Core-B-16.pt "$CACHE"/imagenet_text_features_ViT-L-16-SigLIP2-256.pt \
  "$REMOTE:/root/.cache/clip-guided-diffusion/"
copy "$CACHE"/clip/ViT-B-32.pt "$CACHE"/clip/models--timm--PE-Core-B-16 "$CACHE"/clip/models--timm--ViT-L-16-SigLIP2-256 \
  "$REMOTE:/root/.cache/clip-guided-diffusion/clip/"

# The datasets, both at once. Each is shards beside a `finetune` folder with the index and the embeddings.
copy "$LAION" "$REMOTE:/workspace/data/" > "$here/upload_laion.log" 2>&1 &
laion=$!
copy "$IMAGENET" "$REMOTE:/workspace/data/imagenet-1k-clean-256/" > "$here/upload_imagenet.log" 2>&1 &
imagenet=$!
wait $laion
wait $imagenet

# What arrived, against what is here.
for folder in "$LAION" "$IMAGENET"; do
  echo "here:  $(find "$folder" -type f | wc -l) files, $(du -sb "$folder" | cut -f1) bytes in $folder"
done
$SSH "$REMOTE" 'for folder in /workspace/data/laion-aesthetic-best1m-256 /workspace/data/imagenet-1k-clean-256/train; do
  echo "there: $(find $folder -type f | wc -l) files, $(du -sb --apparent-size $folder | cut -f1) bytes in $folder"; done'

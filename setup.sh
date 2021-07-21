#!/bin/bash

git clone https://github.com/openai/CLIP
git clone https://github.com/openai/guided-diffusion
pip install -e ./CLIP
pip install -e ./guided-diffusion
pip install kornia

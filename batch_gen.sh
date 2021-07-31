#!/bin/bash
# Usage: ./batch_gen.sh <caption file>
# Description: Generate one image per line of caption file in current directory.
# Example: ./batch_gen.sh captions.txt
# Uses xargs to run the python script in parallel for all lines in caption file.
# -n 1 means run one at a time.
# -I {} means replace {} with the line of the caption file.

caption_file=$1

cat $caption_file | xargs -n 1 -I {} ./cgd_venv/bin/python cgd.py '{}';

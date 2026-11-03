#!/bin/bash
SP=/tmp/claude-1000/-home-sam-GitHub-clip-guided-diffusion/34d680cf-d9a7-412c-9f94-d042cb1cd642/scratchpad
R=/home/sam/GitHub/clip-guided-diffusion
until [ -f $SP/after.done ]; do /usr/bin/env sleep 5; done
cd $SP/cli
P="a red fox in the snow"
run() {
  name=$1; shift
  t0=$(date +%s.%N)
  $R/.venv/bin/cgd "$@" -seed 1 -dir $SP/cli/wall/$name "$P" > $SP/cli/wall/$name.log 2>&1
  code=$?
  t1=$(date +%s.%N)
  echo "$name exit $code wall $(echo "$t1 - $t0" | bc) s" >> $SP/cli/wall/results.txt
}
: > $SP/cli/wall/results.txt
run warm -bb sd-turbo -iters 0
run turbo_plain -bb sd-turbo -iters 0
run turbo_clip -bb sd-turbo
run vega768_plain -bb vega -size 768 -iters 0
run vega768_clip -bb vega -size 768
run vega1024_plain -bb vega -iters 0
run vega1024_clip -bb vega
run light768_plain -bb sdxl-lightning -size 768 -iters 0
run light768_clip -bb sdxl-lightning -size 768
run light1024_plain -bb sdxl-lightning -iters 0
run light1024_clip -bb sdxl-lightning
run turbo_up -bb sd-turbo -up
run adm_up -up
echo finished > $SP/wall.done

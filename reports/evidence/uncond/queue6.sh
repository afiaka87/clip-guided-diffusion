#!/bin/sh
# The runs after queue5, as a list that can be reordered while it runs: see queue.sh and todo.txt.
# What ran, in order, is in queue6.log.
exec sh "$(dirname "$0")/queue.sh"

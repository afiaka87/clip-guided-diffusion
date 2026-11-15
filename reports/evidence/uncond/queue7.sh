#!/bin/sh
# The options as built into cgd (2026-11-15), as a list that can be reordered while it runs: see
# queue.sh and todo.txt. What ran, in order, is in queue7.log.
exec sh "$(dirname "$0")/queue.sh"

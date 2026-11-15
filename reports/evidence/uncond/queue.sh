#!/bin/sh
# Runs the commands of todo.txt one at a time, top first, removing each as it starts, until none
# are left. One GPU job at a time, and the list can be edited or reordered while it runs.
cd "$(dirname "$0")"
while [ -s todo.txt ]; do
    line="$(head -n 1 todo.txt)"
    tail -n +2 todo.txt > todo.tmp && mv todo.tmp todo.txt
    case "$line" in ''|'#'*) continue ;; esac
    echo ">>> $line"
    sh -c "$line" || echo "FAILED: $line"
done

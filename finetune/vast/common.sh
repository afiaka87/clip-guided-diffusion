# Sourced by the scripts that run on this machine, from Git Bash. usage: <script> <host> <port>
# The host and port are what `vastai ssh-url <instance>` prints.
HOST=${1:?the address of the instance}
PORT=${2:?its ssh port}
KEY=${KEY:-$HOME/.ssh/vast_ed25519}
# MSYS2's own ssh: rsync from Git Bash fails with Git's ("dup() in/out/err failed").
SSH="/c/msys64/usr/bin/ssh.exe -i $KEY -p $PORT -o IdentitiesOnly=yes -o BatchMode=yes -o ServerAliveInterval=30 \
-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$HOME/.ssh/known_hosts_vast"
REMOTE=root@$HOST
copy() { rsync -rt --partial --info=progress2,stats1 -e "$SSH" "$@"; }

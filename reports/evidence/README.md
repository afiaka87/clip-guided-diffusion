# Evidence

The scripts here ran against the code as it was up to commit `0fb1717`, before the refactor that renamed its modules and removed the options that tied or lost. They import `cgd.cgd`, `cgd.clip_util`, `cgd.script_util`, `cgd.sampling`, `cgd.modern`, `cgd.modules` and `data`, which no longer exist, and many of them use the removed options. To rerun one, check out that commit first:

```sh
git switch --detach 0fb1717
uv sync
```

The logs, scores and contact sheets are what those runs produced.

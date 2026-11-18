"""Which part of the teacher's move a distilled student reproduces: by spatial scale and by noise level.

    uv run --no-sync python reports/evidence/finetune/diag_distill.py D:/cgd-finetune/distill/overfit8b

For each of the run's sample texts and three noise levels, on the held-out image of another text, it takes the
teacher's move and the student's (averaged weights), average-pools both by 1, 2, 4, ... and prints, per pooling:
the share of the teacher's move that survives the pooling (its energy below that frequency), the cosine between
the two pooled moves, and the ratio of their sizes.
"""

import sys
from pathlib import Path

import numpy as np
import torch as th
from torch.nn import functional as tf

from finetune import distill

POOLS = (1, 2, 4, 8, 16, 32, 64)
LEVELS = (0.3, 0.6, 0.9)


def main() -> None:
    out = Path(sys.argv[1])
    args = distill.parse_args(["--out", str(out), "--resume"])
    teacher, log_alpha_bar = distill.load_teacher(args)
    run = distill.build(args)
    samples = distill.Samples(**th.load(out / distill.SAMPLES_NAME, map_location="cpu", weights_only=True))
    last = len(log_alpha_bar) - 1
    generator = th.Generator().manual_seed(0)
    rows = {level: [] for level in LEVELS}
    with distill.averaged(run):
        for i in range(len(samples.texts)):
            for level in LEVELS:
                t = level * last
                image = samples.rows["real"][(i + 3) % len(samples.texts)]  # Another text's image, as a swap.
                x = distill.noisy(image, log_alpha_bar, t, generator, args.device)
                targets = distill.on_device(samples.targets, slice(i, i + 1), args.device)
                th.manual_seed(i)
                base, shift = distill.guided_shift(teacher, log_alpha_bar, x, t, targets)
                with th.no_grad():
                    steps = th.full((1,), t, device=args.device, dtype=th.float32)
                    moved = distill.student_noise(run, x, steps, targets) - base
                row = []
                for pool in POOLS:
                    theirs, ours = tf.avg_pool2d(shift, pool).flatten(), tf.avg_pool2d(moved, pool).flatten()
                    share = float(theirs.square().sum() * pool**2 / shift.square().sum())
                    cosine = float(tf.cosine_similarity(theirs, ours, dim=0))
                    row.append((share, cosine, float(ours.norm() / theirs.norm())))
                rows[level].append(row)
    print("pool      " + "".join(f"{pool:>8d}" for pool in POOLS))
    for level in LEVELS:
        mean = np.mean(rows[level], axis=0)
        print(f"t = {level:.1f} of the schedule")
        for column, name in enumerate(("share", "cosine", "size")):
            print(f"  {name:8s}" + "".join(f"{value:8.3f}" for value in mean[:, column]))


if __name__ == "__main__":
    main()

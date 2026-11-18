r"""Write what `cgd -ft` draws with from a `finetune.cond` run: its weights and `finetune.prior`'s maps.

    uv run python -m finetune.export D:/cgd-finetune/cond256-vast
    uv run cgd -ft D:/cgd-finetune/cond256-vast/cgd.pt "a red fox in the snow"

It leaves out the optimizer's state and the running average, which are most of a run's checkpoint: what is
left is the 420M trained weights, 1.7 GB. `--average` exports the running average in place of the weights.
After the rented run's decay the weights themselves scored better, see the README.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch as th

from cgd.finetuned import export
from finetune.prior import PRIOR_PATH
from finetune.train import CHECKPOINT_NAME

EXPORT_NAME = "cgd.pt"


def main(argv: Sequence[str] | None = None) -> Path:
    """Export a run, and return where it went."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("run", type=Path, help=f"the run's folder, which holds its {CHECKPOINT_NAME}")
    parser.add_argument("--prior", type=Path, default=PRIOR_PATH, help="finetune.prior's maps")
    parser.add_argument("--average", action="store_true", help="the running average of the weights instead")
    parser.add_argument("--out", type=Path, help=f"where to write it. Default {EXPORT_NAME} in the run's folder")
    args = parser.parse_args(argv)
    checkpoint = th.load(args.run / CHECKPOINT_NAME, map_location="cpu", mmap=True, weights_only=True)
    priors = th.load(args.prior, map_location="cpu", weights_only=True)
    out = args.out or args.run / EXPORT_NAME
    export(checkpoint, priors, out, "average" if args.average else "weights")
    print(f"Saved {out}, from step {checkpoint.get('steps')}", flush=True)
    return out


if __name__ == "__main__":
    main()

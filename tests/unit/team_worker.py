"""One process of a small run under `finetune.parallel`, for the tests of it in `test_finetune.py`.

    python -m finetune.parallel --devices cpu,cpu tests.unit.team_worker train <folder>

`train` takes steps on this process's own images until the first process says three are enough, in the order
`finetune.cond` makes its calls, and saves what it ends with. `fail` ends the second
process with exit code 3 while the first would wait a minute.
"""

import os
import sys
import time
from pathlib import Path

STEPS = 3
OPTIONS = ["--device", "cpu", "--ema", "0.5", "--warmup", "1", "--lr", "1e-2", "--accumulate", "1"]


def images(step: int, rank: int):
    """Two noisy images of a step for one process, with their timesteps and conditions."""
    import torch as th  # pylint: disable=import-outside-toplevel  # `fail` starts without it, in a moment.

    from finetune import cond  # pylint: disable=import-outside-toplevel
    from tests.unit.test_finetune import SLOT_DIMS  # pylint: disable=import-outside-toplevel

    generator = th.Generator().manual_seed(100 + 10 * step + rank)
    embeds = {name: th.randn(2, dim, generator=generator) for name, dim in SLOT_DIMS.items()}
    keep = th.rand(2, len(SLOT_DIMS), generator=generator) < 0.7
    condition = cond.Condition(embeds, keep, th.randint(0, 10, (2,), generator=generator))
    return th.randn(2, 3, 16, 16, generator=generator), th.tensor([40.0, 700.0]) + step + rank, condition


def train(folder: Path, device: str = "cpu") -> None:
    """Three steps as `finetune.cond` takes them, each on this process's images. `cuda` is for a check by hand."""
    import torch as th  # pylint: disable=import-outside-toplevel

    from finetune import cond, parallel  # pylint: disable=import-outside-toplevel
    from tests.unit.test_finetune import anchored_model  # pylint: disable=import-outside-toplevel

    team = parallel.join(device)
    with parallel.lead_first(team):
        if team.lead:
            time.sleep(1)
            (folder / "first").write_text("")
        elif not (folder / "first").is_file():
            sys.exit("The first process had not finished.")
    # A different model in each process: sharing it has to replace all but the first one's.
    model = anchored_model(seed=team.rank, align_dim=0).to(device)
    net = parallel.shared(team, model)
    run = cond.Run(cond.parse_args([*OPTIONS, "--device", device, "--no-amp"]), model, None)
    going = parallel.agreed(team, True)
    while going:
        x, t, condition = (part.to(device) for part in images(run.steps, team.rank))
        run.step(net(x, t, condition.embeds, condition.keep, labels=condition.labels).square().mean())
        going = parallel.agreed(team, run.steps < STEPS if team.lead else True)  # Only the first one counts.
    result = {
        "weights": {name: weight.detach().cpu() for name, weight in run.weights.items()},
        "average": {name: weight.cpu() for name, weight in run.average.items()},
        "record": parallel.averaged(team, {"rank": float(team.rank)}),
        "agreed": parallel.agreed(team, team.lead),
    }
    th.save(result, folder / f"{team.rank}.pt")
    parallel.leave(team)


if __name__ == "__main__":
    if sys.argv[1] == "fail":
        if os.environ["CGD_TEAM_RANK"] == "1":
            sys.exit(3)
        time.sleep(60)
    else:
        train(Path(sys.argv[2]), *sys.argv[3:])

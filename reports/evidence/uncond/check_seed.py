"""Does a seed reproduce an arm's image byte for byte, in a new process?

    uv run python check_seed.py ARM MODULE:FUNCTION 'JSON' [PROMPT] [SEED]

Draws one image of the arm again, into outputs/eval/seed_checks/ARM, and compares its pixels
with the one the arm saved. Every image this repository makes is meant to be reproducible.
"""

import importlib
import json
import sys

import numpy as np
from common import IMAGES, PROMPT_SETS, located
from PIL import Image


def main() -> None:
    arm, target, kwargs = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
    prompt = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    module, function = target.split(":")
    generate = getattr(importlib.import_module(module), function)
    runs = map(json.loads, (IMAGES / arm / "runs.jsonl").read_text().splitlines())
    saved = next(r["path"] for r in runs if (r["prompt"], r["seed"]) == (prompt, seed))
    again = list(
        generate(
            [PROMPT_SETS["held_out"][prompt]],
            seed=seed,
            output_dir=IMAGES / "seed_checks" / arm,
            save_frequency=10**6,
            progress=False,
            **kwargs,
        )
    )[-1][1]
    first, second = (np.asarray(Image.open(path).convert("RGB"), dtype=np.int16) for path in (located(saved), again))
    different = int((first != second).any(-1).sum())
    print(f"{arm} p{prompt:02} s{seed}: {different} of {first.shape[0] * first.shape[1]} pixels differ, "
          f"by at most {int(np.abs(first - second).max())}. {'REPRODUCED' if not different else 'NOT REPRODUCED'}")  # fmt: skip


if __name__ == "__main__":
    main()

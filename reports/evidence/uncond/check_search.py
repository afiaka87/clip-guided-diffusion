"""Check that search.generate() with no search makes exactly adm.generate()'s image, and that each option runs.

uv run python reports/evidence/uncond/check_search.py
"""

import tempfile
import time
from pathlib import Path

import torch as th
from PIL import Image

import search
from cgd import adm

PROMPT = ["a red fox in the snow"]


def pixels(path: Path) -> bytes:
    return Image.open(path).tobytes()


def main() -> None:
    with tempfile.TemporaryDirectory() as folder:
        reference = list(adm.generate(PROMPT, unconditional=True, seed=3, output_dir=f"{folder}/a", progress=False))
        mine = list(search.generate(PROMPT, seed=3, output_dir=f"{folder}/b"))
        assert pixels(reference[-1][1]) == pixels(mine[-1][1]), "search.generate() differs from adm.generate()"
        print("No search: identical to adm.generate()")
        for options in (
            {"recurrence": 2},
            {"rollout_calls": 2},
            {"particles": 2, "events": [16, 22], "keep": [None, None]},
            {"particles": 3, "events": [10, 20], "keep": [2, 1]},
        ):
            th.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            saved = list(search.generate(PROMPT, seed=3, output_dir=f"{folder}/c", progress=True, **options))
            print(options, f"{time.perf_counter() - start:.1f} s", f"{th.cuda.max_memory_allocated() / 2**30:.2f} GiB",
                  "differs" if pixels(saved[-1][1]) != pixels(reference[-1][1]) else "SAME")


if __name__ == "__main__":
    main()

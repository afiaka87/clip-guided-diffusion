"""Generate one arm: every held-out prompt at each seed, recording each result's path, time and peak VRAM.

uv run python reports/evidence/uncond/run_arm.py ARM MODULE:FUNCTION '{"keyword": "arguments"}' [SEEDS] [SET]

MODULE:FUNCTION is a generate() with adm.generate's interface, such as cgd.adm:generate,
or one of this folder's experiments. SEEDS defaults to 0,1,2 and SET to held_out, the 12
prompts; confirm is the 66 of confirm_prompts.json. Results already on disk are kept, so a
stopped arm resumes where it stopped. Appends to outputs/eval/ARM/runs.jsonl.
"""

import importlib
import json
import sys
import time

import torch as th

from common import IMAGES, PROMPT_SETS, stored


def main() -> None:
    arm, target, kwargs = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
    seeds = [int(s) for s in (sys.argv[4] if len(sys.argv) > 4 else "0,1,2").split(",")]
    prompt_set = sys.argv[5] if len(sys.argv) > 5 else "held_out"
    module, function = target.split(":")
    generate = getattr(importlib.import_module(module), function)
    folder = IMAGES / arm
    folder.mkdir(parents=True, exist_ok=True)
    log = folder / "runs.jsonl"
    done = {(r["prompt"], r["seed"]) for r in map(json.loads, log.read_text().splitlines())} if log.exists() else set()
    with log.open("a", buffering=1) as out:
        for seed in seeds:
            for index, prompt in enumerate(PROMPT_SETS[prompt_set]):
                if (index, seed) in done:
                    continue
                th.cuda.reset_peak_memory_stats()
                th.cuda.synchronize()
                start = time.perf_counter()
                saved = list(
                    generate(
                        [prompt],
                        seed=seed,
                        output_dir=folder / f"s{seed}" / f"p{index:02}",
                        save_frequency=10**6,
                        progress=False,
                        **kwargs,
                    )
                )
                th.cuda.synchronize()
                seconds = time.perf_counter() - start
                peak = th.cuda.max_memory_allocated() / 2**30
                path = stored(saved[-1][1])
                record = {"prompt": index, "seed": seed, "path": path, "seconds": seconds, "peak": peak}
                if prompt_set != "held_out":
                    record["set"] = prompt_set
                out.write(json.dumps(record) + "\n")
                print(f"{arm} s{seed} p{index:02} {seconds:6.1f} s {peak:5.2f} GiB", flush=True)


if __name__ == "__main__":
    main()

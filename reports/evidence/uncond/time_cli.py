"""Wall clock and peak VRAM of the `cgd` command, as the README's Speed table measures them.

    uv run python time_cli.py [WORD ...]

Each configuration runs three times: twice as `uv run cgd ...` in a new process, the second
timed from start to exit so that startup and model loading count and the weights are already
in the disk cache, and once in this process for torch's peak of allocated VRAM, with no model
left loaded by the configuration before, as in a new process. Appends to
time_cli.jsonl next to this file. With words, only the configurations whose name has one.

The records without `options` are from 2026-11-14, when `-uncond` alone was
`-uncond -clip ViT-B/32 -thumb none` and every command carried `-cutn 4`.
"""

import gc
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch as th

from cgd import adm, clip_models, thumbnails
from cgd.cli import main

REPO = Path(__file__).resolve().parents[3]
PROMPT = "A red fox in the snow"
CONFIGURATIONS = {
    "128 default": ["-cutn", "4"],
    "256 default": ["-cutn", "4", "-size", "256"],
    "256 -uncond": ["-uncond"],
    "256 -uncond -thumb none": ["-uncond", "-thumb", "none"],
    "256 -uncond -thumb none -clip ViT-B/32": ["-uncond", "-thumb", "none", "-clip", "ViT-B/32"],
    "256 -uncond -pag": ["-uncond", "-pag"],
    "256 -uncond -thumb glide": ["-uncond", "-thumb", "glide"],
    "256 -uncond -thumb laionide": ["-uncond", "-thumb", "laionide"],
    "256 -uncond -thumb karlo": ["-uncond", "-thumb", "karlo"],
    "256 -uncond -cutn 16": ["-uncond", "-cutn", "16"],
    "512 -uncond": ["-uncond", "-size", "512"],
    "512 -uncond -thumb none": ["-uncond", "-size", "512", "-thumb", "none"],
}


def run() -> None:
    log = Path(__file__).with_name("time_cli.jsonl")
    gpu = th.cuda.get_device_name(0)
    words = sys.argv[1:]
    with tempfile.TemporaryDirectory() as scratch, log.open("a") as out:
        for name, options in CONFIGURATIONS.items():
            if words and not any(word in name for word in words):
                continue
            arguments = [*options, "-q", "-dir", str(Path(scratch) / name.replace(" ", "_").replace("/", "-")), PROMPT]
            seconds = 0.0
            for _ in range(2):
                start = time.perf_counter()
                command = ["uv", "run", "--project", str(REPO), "cgd", *arguments]
                subprocess.run(command, check=True, cwd=scratch, capture_output=True)
                seconds = time.perf_counter() - start
            for cached in (adm.load_model, clip_models.load_clip, thumbnails.load_model):
                cached.cache_clear()
            gc.collect()
            th.cuda.empty_cache()
            th.cuda.reset_peak_memory_stats()
            main(arguments)
            peak = th.cuda.max_memory_allocated() / 2**30
            th.cuda.empty_cache()
            record = {"configuration": name, "options": options, "seconds": seconds, "peak": peak, "gpu": gpu}
            out.write(json.dumps(record) + "\n")
            print(f"time_cli {name:40} {seconds:6.1f} s {peak:5.2f} GiB  {gpu}", flush=True)


if __name__ == "__main__":
    sys.exit(run())

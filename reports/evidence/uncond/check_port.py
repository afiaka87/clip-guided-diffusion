"""Check that cgd's own options make the images the experiments in this folder made, byte for byte.

    uv run python check_port.py [ARM ...]

Each arm below was made by an experiment (thumbnail.py, self_guidance.py) or by an earlier
cgd. Its settings here are the options of today's `cgd -uncond` that should be the same
computation. One prompt and seed of each is made again with cgd.adm.generate() and compared
with the arm's stored image, so the arm's measurements are measurements of the option.
"""

import json
import sys
import tempfile
from pathlib import Path

from cgd import adm
from common import HELD_OUT, IMAGES, REPO

ARMS = {
    "uncond_e_c4": {"thumbnail": "none"},
    "cgd_uncond_e": {"thumbnail": "none", "num_cutouts": 16},
    "xres_nclip_w10_e_c4": {},
    "xres_nclip_w10_e": {"num_cutouts": 16},
    "xres_nclip_w10_pag1_e_c4": {"perturbed_attention": True},
    "xres_nclip_w10_pag1_e": {"perturbed_attention": True, "num_cutouts": 16},
    "pag4_s1_e": {"thumbnail": "none", "perturbed_attention": True, "num_cutouts": 16},
    "xres_glide_w3_e": {"thumbnail": "glide", "num_cutouts": 16},
    "xres_laionide_w3_e": {"thumbnail": "laionide", "num_cutouts": 16},
    "xres_karlo_w3_e": {"thumbnail": "karlo", "num_cutouts": 16},
}
PROMPT, SEED = 3, 1


def main() -> None:
    different = []
    with tempfile.TemporaryDirectory() as scratch:
        for arm in sys.argv[1:] or ARMS:
            records = map(json.loads, (IMAGES / arm / "runs.jsonl").read_text().splitlines())
            stored = next(REPO / r["path"] for r in records if (r["prompt"], r["seed"]) == (PROMPT, SEED))
            saved = list(
                adm.generate(
                    [HELD_OUT[PROMPT]],
                    unconditional=True,
                    seed=SEED,
                    output_dir=Path(scratch) / arm,
                    save_frequency=10**6,
                    progress=False,
                    **ARMS[arm],
                )
            )
            same = saved[-1][1].read_bytes() == stored.read_bytes()
            different += [] if same else [arm]
            print(f"{'SAME     ' if same else 'DIFFERENT'} {arm}: cgd -uncond with {ARMS[arm] or 'its defaults'}", flush=True)
    sys.exit(f"{len(different)} differ: {different}" if different else 0)


if __name__ == "__main__":
    main()

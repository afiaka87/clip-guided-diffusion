"""A contact sheet of arms: one row per arm, one column per prompt, at one seed.

    uv run python sheet_rows.py OUT.jpg SEED PROMPTS ARM[=LABEL] ...

PROMPTS is a comma-separated list of held-out prompt numbers, and a LABEL replaces the arm's
name above its row. The sheets sent with the 2026-11-14 results are outputs/eval/sheets/compare_seed1_*.jpg.
"""

import json
import sys
from pathlib import Path

from common import IMAGES, located
from PIL import Image, ImageDraw

out, seed = Path(sys.argv[1]), int(sys.argv[2])
prompts = [int(p) for p in sys.argv[3].split(",")]
arms = sys.argv[4:]
SIDE = 192
sheet = Image.new("RGB", (SIDE * len(prompts), (SIDE + 14) * len(arms)), "white")
draw = ImageDraw.Draw(sheet)
for row, arm in enumerate(arms):
    arm, _, label = arm.partition("=")
    runs = {(r["prompt"], r["seed"]): r["path"] for r in map(json.loads, (IMAGES / arm / "runs.jsonl").read_text().splitlines())}
    draw.text((2, row * (SIDE + 14) + 1), label or arm, fill="black")
    for column, prompt in enumerate(prompts):
        if (prompt, seed) in runs:
            image = Image.open(located(runs[(prompt, seed)])).convert("RGB").resize((SIDE, SIDE), Image.LANCZOS)
            sheet.paste(image, (column * SIDE, row * (SIDE + 14) + 14))
sheet.save(out, quality=90)
print(out)

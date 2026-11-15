"""A contact sheet of arms side by side: a row per prompt, a column per arm, at one seed.

uv run python reports/evidence/uncond/sheet.py OUTPUT.jpg SEED ARM [ARM ...]
"""

import json
import sys

from PIL import Image, ImageDraw

from common import HELD_OUT, IMAGES, located

SIDE = 192


def main() -> None:
    output, seed, arms = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    runs = {}
    for arm in arms:
        log = IMAGES / arm / "runs.jsonl"
        runs[arm] = {r["prompt"]: r["path"] for r in map(json.loads, log.read_text().splitlines()) if r["seed"] == seed}
    label = 18
    sheet = Image.new("RGB", (SIDE * len(arms), label + (SIDE + label) * len(HELD_OUT)), "white")
    draw = ImageDraw.Draw(sheet)
    for column, arm in enumerate(arms):
        draw.text((column * SIDE + 4, 3), arm, fill="blue")
    for row, prompt in enumerate(HELD_OUT):
        top = label + row * (SIDE + label)
        draw.text((4, top + 3), prompt, fill="black")
        for column, arm in enumerate(arms):
            if row in runs[arm]:
                image = Image.open(located(runs[arm][row])).convert("RGB").resize((SIDE, SIDE), Image.LANCZOS)
                sheet.paste(image, (column * SIDE, top + label))
    sheet.save(output, quality=90)


if __name__ == "__main__":
    main()

"""Blind pairwise comparison of two arms: which image is the better picture of its prompt?

    uv run python blind.py make NAME ARM_A ARM_B [SEEDS] [SET]   # Writes outputs/eval/blind/NAME_*.jpg and NAME.key.json.
    uv run python blind.py reveal NAME CHOICES [JUDGE]            # CHOICES is a letter per pair: L, R, or T for a tie.

Automatic judges miss what a person sees first: CLIP similarity rewards a collage of the
prompt's textures as much as a coherent picture, and the preference models mostly rate
rendering. So each pair, the two arms' images of one prompt at one seed, is shown side by
side under its prompt, with which arm is on the left drawn at random from the operating
system's entropy. The key is written to another folder, outputs/eval/blind_keys, which the
judge does not open until the choices are made. `reveal` then counts the wins per arm with
a two-sided sign test and appends them, with the judge's name, to NAME.results.jsonl there.
"""

import json
import os
import random
import sys

from common import IMAGES, PROMPT_SETS, located
from PIL import Image, ImageDraw
from scipy import stats

SIDE = 256
PER_SHEET = 6
LABEL = 20
BLIND = IMAGES / "blind"
KEYS = IMAGES / "blind_keys"  # Apart from the sheets, so a judge given the sheets' folder cannot come across them.


def runs_of(arm: str) -> dict[tuple[int, int], str]:
    lines = (IMAGES / arm / "runs.jsonl").read_text().splitlines()
    return {(r["prompt"], r["seed"]): r["path"] for r in map(json.loads, lines)}


def make(name: str, arm_a: str, arm_b: str, seeds: list[int], prompt_set: str) -> None:
    rng = random.Random(os.urandom(16))
    a, b = runs_of(arm_a), runs_of(arm_b)
    pairs = [k for k in sorted(set(a) & set(b)) if k[1] in seeds]
    rng.shuffle(pairs)
    BLIND.mkdir(parents=True, exist_ok=True)
    key = []
    for start in range(0, len(pairs), PER_SHEET):
        chunk = pairs[start : start + PER_SHEET]
        sheet = Image.new("RGB", (2 * SIDE + 12, (SIDE + LABEL + 6) * len(chunk)), "white")
        draw = ImageDraw.Draw(sheet)
        for row, (prompt, seed) in enumerate(chunk):
            a_left = rng.random() < 0.5
            key.append({"pair": start + row + 1, "prompt": prompt, "seed": seed, "left": arm_a if a_left else arm_b})
            top = row * (SIDE + LABEL + 6)
            draw.text((4, top + 4), f"{start + row + 1}. {PROMPT_SETS[prompt_set][prompt]}", fill="black")
            for column, path in enumerate((a[(prompt, seed)], b[(prompt, seed)]) if a_left else (b[(prompt, seed)], a[(prompt, seed)])):
                image = Image.open(located(path)).convert("RGB").resize((SIDE, SIDE), Image.LANCZOS)
                sheet.paste(image, (column * (SIDE + 12), top + LABEL))
        path = BLIND / f"{name}_{start // PER_SHEET + 1:02}.jpg"
        sheet.save(path, quality=92)
        print(path)
    KEYS.mkdir(parents=True, exist_ok=True)
    (KEYS / f"{name}.key.json").write_text(json.dumps({"arms": [arm_a, arm_b], "pairs": key}))
    print(len(pairs), "pairs")


def reveal(name: str, choices: str, judge: str = "") -> None:
    key = json.loads((KEYS / f"{name}.key.json").read_text())
    arm_a, arm_b = key["arms"]
    assert len(choices) == len(key["pairs"]), f"{len(choices)} choices for {len(key['pairs'])} pairs."
    wins = {arm_a: 0, arm_b: 0, "tie": 0}
    for choice, pair in zip(choices.upper(), key["pairs"], strict=True):
        other = arm_b if pair["left"] == arm_a else arm_a
        wins[{"L": pair["left"], "R": other, "T": "tie"}[choice]] += 1
    decided = wins[arm_a] + wins[arm_b]
    p_value = stats.binomtest(wins[arm_a], decided, 0.5).pvalue if decided else 1.0
    print(f"{arm_a} {wins[arm_a]}, {arm_b} {wins[arm_b]}, ties {wins['tie']}; sign test p {p_value:.4f}")
    results = KEYS / f"{name}.results.jsonl"
    with results.open("a") as file:
        file.write(json.dumps({"judge": judge, "choices": choices.upper(), "wins": wins, "p": p_value}) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "make":
        seeds = [int(s) for s in (sys.argv[5] if len(sys.argv) > 5 else "0,1,2").split(",")]
        make(sys.argv[2], sys.argv[3], sys.argv[4], seeds, sys.argv[6] if len(sys.argv) > 6 else "held_out")
    else:
        reveal(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "")

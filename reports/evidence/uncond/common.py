"""What every experiment on the unconditional 256 pipeline shares: the prompts, and where things go."""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
IMAGES = REPO / "outputs" / "eval"  # Ignored by git.


def stored(path: str | Path) -> str:
    """A result's path as the records keep it: relative to the repository, with forward slashes, so they
    work wherever it is checked out, on Linux or Windows."""
    return Path(path).resolve().relative_to(REPO).as_posix()


def located(path: str) -> Path:
    """Where a recorded path is on this machine."""
    return REPO / path

# The 12 prompts held out from tuning, behind every SD-Turbo and class-conditional number
# in the README (reports/evidence/scripts/prompts.py).
HELD_OUT = (
    "a red fox in the snow",
    "a pencil sketch of an old man with a beard",
    "a plate of sushi on a black slate",
    "a lighthouse on a cliff at sunset",
    "a watercolor painting of a sailboat",
    "a bowl of ramen with a soft boiled egg",
    "a vintage red bicycle leaning against a brick wall",
    "an oil painting of a mountain lake at dawn",
    "a close-up photograph of a bumblebee on a sunflower",
    "a steam locomotive crossing a stone bridge",
    "a cat wearing a tiny wizard hat",
    "a neon-lit city street in the rain",
)

# A second, larger set that nothing was tuned on, for confirming what the 12 prompts above suggest:
# 66 PartiPrompts, 6 from each category (confirm_prompts.json says how they were drawn).
CONFIRM = tuple(json.loads((HERE / "confirm_prompts.json").read_text(encoding="utf-8"))["prompts"])
PROMPT_SETS = {"held_out": HELD_OUT, "confirm": CONFIRM}


def prompt_of(record: dict) -> str:
    """The prompt a runs.jsonl record is of. Records of the first set do not name it."""
    return PROMPT_SETS[record.get("set", "held_out")][record["prompt"]]

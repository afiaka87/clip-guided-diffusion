"""Move queued arms to the front of todo.txt, in the order given; the rest keep their order.

    uv run python reorder.py ARM ARM ...
"""

import sys
from pathlib import Path

todo = Path(__file__).with_name("todo.txt")
lines = [line for line in todo.read_text(encoding="utf-8").splitlines() if line.strip()]


def name(line: str) -> str:
    return line.split("run_arm.py ")[-1].split()[0]


by_name = {name(line): line for line in lines}
missing = [arm for arm in sys.argv[1:] if arm not in by_name]
assert not missing, f"Not queued: {missing}"
first = [by_name[arm] for arm in sys.argv[1:]]
rest = [line for line in lines if name(line) not in sys.argv[1:]]
new = todo.with_suffix(".new")
new.write_text("\n".join(first + rest) + "\n", encoding="utf-8", newline="\n")
new.replace(todo)
print(len(first + rest), "queued:", ", ".join(name(line) for line in first + rest))

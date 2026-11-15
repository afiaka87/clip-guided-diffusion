"""Insert an arm into todo.txt, the list queue.sh runs, before another arm (or at the end if that one is not queued).

    uv run python add_arm.py BEFORE NAME 'OPTIONS AS JSON' [MODULE:FUNCTION] [SEEDS] [SET]

MODULE:FUNCTION defaults to thumbnail:generate, and SEEDS and SET to run_arm.py's.
"""

import sys
from pathlib import Path

before, name, options = sys.argv[1], sys.argv[2], sys.argv[3]
module = sys.argv[4] if len(sys.argv) > 4 else "thumbnail:generate"
line = f"uv run python run_arm.py {name} {module} '{options}' " + " ".join(sys.argv[5:])
todo = Path(__file__).with_name("todo.txt")
lines = todo.read_text(encoding="utf-8").splitlines()
index = next((i for i, queued in enumerate(lines) if f" {before} " in queued), len(lines))
lines.insert(index, line.rstrip())
new = todo.with_suffix(".new")
new.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
new.replace(todo)
print(len(lines), "queued:", ", ".join(queued.split("run_arm.py ")[-1].split()[0] for queued in lines[:9]), "...")

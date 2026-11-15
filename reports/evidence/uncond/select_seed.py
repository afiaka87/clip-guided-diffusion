"""What keeping the best of an arm's seeds would buy, from the scores already on disk.

    SCORE_DEVICE=cpu uv run python select_seed.py ARM [ARM ...]

For each prompt, one judge picks the seed it scores highest, and the others say how good that
pick is. The gain is the picked image's score minus the mean over the prompt's seeds, which is
what a random seed gets. A judge never grades its own pick: PickScore's picks are graded by HPS
and the CLIP judges, and HPS's by PickScore. Run score.py on the arm first.
"""

import json
import sys

import numpy as np
from scipy import stats

from score import DB, JUDGES, key, load_runs

SELECTORS = ("pick", "hps", "clip")


def value(scores: dict, metric: str) -> float:
    return float(np.mean([scores[j] for j in JUDGES])) if metric == "clip" else scores[metric]


def main() -> None:
    db = json.loads(DB.read_text())
    for arm in sys.argv[1:]:
        runs = load_runs(arm)
        prompts = sorted({p for p, _ in runs})
        seeds = sorted({s for _, s in runs})
        print(f"{arm}: best of {len(seeds)} seeds on {len(prompts)} prompts. Gain over a random seed, 95% t interval, prompts improved.")
        for selector in SELECTORS:
            gains = {metric: [] for metric in ("pick", "hps", "clip", "aesb16") if metric != selector}
            for prompt in prompts:
                scores = [db[key(runs[(prompt, seed)]["path"])] for seed in seeds if (prompt, seed) in runs]
                best = max(scores, key=lambda s: value(s, selector))
                for metric, column in gains.items():
                    column.append(value(best, metric) - float(np.mean([value(s, metric) for s in scores])))
            cells = []
            for metric, column in gains.items():
                column = np.array(column)
                half = stats.t.ppf(0.975, len(column) - 1) * column.std(ddof=1) / len(column) ** 0.5
                cells.append(f"{metric} {column.mean():+.4f} ±{half:.4f} {int((column > 0).sum())}/{len(column)}")
            print(f"  picked by {selector:5}: " + " | ".join(cells))


if __name__ == "__main__":
    main()

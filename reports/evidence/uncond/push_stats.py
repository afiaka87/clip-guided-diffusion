"""How large a thumbnail model's push on the noise prediction is, and how much of it only scales the image.

    uv run python push_stats.py NAME '{"prior": "glide-clip", "scale": 10.0, ...}' [PROMPTS] [SEED]

Runs thumbnail.generate() with those options on the held-out prompts numbered PROMPTS
(default 0,5,9) at SEED (default 0), and records at every model call:

- ratio: the RMS of what the thumbnail model adds to the noise prediction, over the RMS of
  the U-Net's own. cgd caps CLIP guidance's at sampler.DEFAULT_GUIDANCE_RATE of it.
- cosine: between the change that push makes to the denoised thumbnail and the denoised
  thumbnail itself. Adaptive projected guidance (Sadat et al. 2024, arXiv 2410.02416) finds
  that this parallel part is what oversaturates, and the rest what improves the image.
- share: the part of the push's squared norm that is parallel.
- gain: how much the push scales the denoised thumbnail, as a fraction of it.
- moved: the RMS of the change to the denoised thumbnail, whose pixels are in -1 to 1.
- beyond: the share of the denoised thumbnail's pixels the push takes past sampler.X0_CLAMP,
  where the sampler clamps them.

The images go to outputs/eval/push_stats/NAME and are the arm's own, since the recording
changes nothing. Appends the rows to push_stats.jsonl next to this file and prints their
mean by timestep.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import thumbnail
import torch as th
import torch.nn.functional as tf
from common import HELD_OUT, IMAGES

from cgd import sampler

ROWS: list[dict] = []


def recording_denoise(  # noqa: PLR0913, PLR0917
    denoise, log_alpha_bar, predictions, scale, above, x, t, replace=False, limits=thumbnail.Limits()
):
    """thumbnail.guided_denoise() in its additive form, noting what the push does on the way."""
    assert not replace
    eps = denoise(x, t)[:, :3]
    timestep = float(t[0])
    if predictions is None or not scale or timestep < above:
        return eps
    now = sampler.level(log_alpha_bar, timestep)
    y = tf.avg_pool2d(x.detach().float(), thumbnail.FACTOR)
    conditional, unconditional = predictions(y, now.alpha, now.sigma / thumbnail.FACTOR)

    def enlarged(small: th.Tensor) -> th.Tensor:
        return tf.interpolate(small, scale_factor=thumbnail.FACTOR, mode="nearest").to(eps.dtype)

    own = eps.detach().float()
    x0 = (y - now.sigma * tf.avg_pool2d(own, thumbnail.FACTOR)) / now.alpha  # The denoised image's thumbnail.
    if limits == thumbnail.Limits():
        push = (scale / thumbnail.FACTOR) * enlarged(conditional - unconditional)
    else:
        small = (scale / thumbnail.FACTOR) * (conditional - unconditional).float()
        push = enlarged(thumbnail.limited_push(small, x0, own, now, limits))
    with th.no_grad():
        x0 = x0.flatten(1)
        change = (-now.sigma / now.alpha * tf.avg_pool2d(push.float(), thumbnail.FACTOR)).flatten(1)
        along = (change * x0).sum() / x0.square().sum()  # The change is `along` times x0, plus a part orthogonal to it.
        ROWS.append({
            "t": timestep,
            "ratio": (sampler.rms(push.float()) / sampler.rms(own)).item(),
            "cosine": tf.cosine_similarity(change, x0).item(),
            "share": (along**2 * x0.square().sum() / change.square().sum()).item(),
            "gain": along.item(),
            "x0_rms": x0.square().mean().sqrt().item(),
            "x0_max": x0.abs().max().item(),
            "moved": change.square().mean().sqrt().item(),
            "beyond": ((x0 + change).abs() > sampler.X0_CLAMP).float().mean().item(),
        })  # fmt: skip
    return eps + push


def main() -> None:
    name, options = sys.argv[1], json.loads(sys.argv[2])
    prompts = [int(p) for p in (sys.argv[3] if len(sys.argv) > 3 else "0,5,9").split(",")]
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    thumbnail.guided_denoise = recording_denoise
    log = Path(__file__).with_name("push_stats.jsonl")
    with log.open("a") as out:
        for prompt in prompts:
            ROWS.clear()
            folder = IMAGES / "push_stats" / name / f"s{seed}" / f"p{prompt:02}"
            arguments = {"seed": seed, "output_dir": folder, "save_frequency": 10**6, **options}
            list(thumbnail.generate([HELD_OUT[prompt]], **arguments))
            for call, row in enumerate(ROWS):
                out.write(json.dumps({"name": name, "prompt": prompt, "seed": seed, "call": call, **row}) + "\n")
            by_step = defaultdict(list)
            for row in ROWS:
                by_step[round(row["t"])].append(row)
            print(f"{name} p{prompt:02} s{seed}: {len(ROWS)} calls", flush=True)
            print(f"{'t':>4} {'calls':>5} {'ratio':>6} {'cosine':>7} {'share':>6} {'gain':>7} {'x0 rms':>6} {'x0 max':>6} {'moved':>6} {'beyond':>6}")
            for step in sorted(by_step, reverse=True)[:: max(len(by_step) // 20, 1)]:
                rows = by_step[step]
                mean = {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
                print(
                    f"{step:4} {len(rows):5} {mean['ratio']:6.3f} {mean['cosine']:+7.3f} {mean['share']:6.3f} "
                    f"{mean['gain']:+7.4f} {mean['x0_rms']:6.3f} {mean['x0_max']:6.2f} {mean['moved']:6.3f} {mean['beyond']:6.3f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()

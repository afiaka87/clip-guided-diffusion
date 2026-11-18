"""Whether CLIP guidance on a distilled student beats guidance on the unconditional model at the same steps.

    uv run --no-sync python reports/evidence/finetune/diag_distill_guided.py D:/cgd-finetune/distill/general 15 30

For each number of steps it draws the run's sample texts three ways from the same noise: `cgd`'s guidance on the
unconditional model, the student alone, and the same guidance on the student. It prints the held-out judge's
scores of each row and saves the grid as `guided_<steps>.jpg` in the run's folder, under the real images.
"""

import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch as th
from PIL import Image

from cgd import sampler
from cgd.adm import denoise, guidance_loss, load_model
from finetune import distill
from finetune.embed import load_encoder
from cgd.conditioned import load_adapter
from finetune.train import CHECKPOINT_NAME, IMAGE_SIZE


def main() -> None:
    out = Path(sys.argv[1])
    args = distill.parse_args(["--out", str(out), "--resume"])
    teacher, log_alpha_bar = distill.load_teacher(args)
    # Only the averaged weights, frozen, and no optimizer: a gradient through the whole student needs the memory.
    checkpoint = th.load(out / CHECKPOINT_NAME, map_location="cpu", weights_only=True)
    unet, _ = load_model(IMAGE_SIZE, args.device, False, False, True)
    model = load_adapter(unet, checkpoint | {"weights": checkpoint["average"]}).to(args.device)
    run = SimpleNamespace(model=model.requires_grad_(False).to(memory_format=th.channels_last), amp=True)
    judge = load_encoder(args.judge, "cpu")
    samples = distill.Samples(**th.load(out / distill.SAMPLES_NAME, map_location="cpu", weights_only=True))

    def embeds(i: int) -> dict[str, th.Tensor]:
        return distill.on_device(samples.targets, slice(i, i + 1), args.device)

    def guidance(i: int) -> sampler.Guidance:
        return sampler.Guidance(
            partial(guidance_loss, distill.guide_for(teacher, embeds(i))), teacher.rate, distill.WINDOW
        )

    def base(_: int) -> sampler.Denoiser:
        return partial(denoise, teacher.unet, None)

    def student(i: int) -> sampler.Denoiser:
        return distill.student_denoiser(run, teacher, log_alpha_bar, embeds(i), 1.0)

    for steps in map(int, sys.argv[2:]):
        args.sample_steps = steps
        rows = {"real": samples.rows["real"], "guided": distill.draw(args, log_alpha_bar, base, guidance)}
        rows["student"] = distill.draw(args, log_alpha_bar, student)
        rows["student guided"] = distill.draw(args, log_alpha_bar, student, guidance)
        scores = distill.judged(judge, samples, rows)
        print(f"{steps} steps: " + "  ".join(f"{name} {value:.4f}" for name, value in scores.items()), flush=True)
        grid = th.cat([th.cat(list(images), dim=2) for images in rows.values()], dim=1).permute(1, 2, 0).numpy()
        Image.fromarray(grid).save(out / f"guided_{steps}.jpg", quality=92)


if __name__ == "__main__":
    main()

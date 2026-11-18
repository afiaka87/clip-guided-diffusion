"""`diag_distill_guided.py` on held-out texts the run never drew, with each text's two images compared as a pair.

    python reports/evidence/finetune/diag_distill_guided_more.py D:/cgd-finetune/distill/general 1 16 15 30

The arguments are the run's folder, a seed for which held-out texts and which noise, how many texts, and the
numbers of steps. For each number of steps it draws every text twice from the same noise, with `cgd`'s guidance
on the unconditional model and with the same guidance on the student, and prints the judge's mean score of each,
the mean of the differences with its standard error, and how many texts the student's image won.
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
from finetune.embed import encode_images, encode_texts, load_encoder
from cgd.conditioned import load_adapter
from finetune.train import CHECKPOINT_NAME, IMAGE_SIZE


def main() -> None:
    out = Path(sys.argv[1])
    args = distill.parse_args(["--out", str(out), "--seed", sys.argv[2], "--samples", sys.argv[3]])
    teacher, log_alpha_bar = distill.load_teacher(args)
    checkpoint = th.load(out / CHECKPOINT_NAME, map_location="cpu", weights_only=True)
    unet, _ = load_model(IMAGE_SIZE, args.device, False, False, True)
    model = load_adapter(unet, checkpoint | {"weights": checkpoint["average"]}).to(args.device)
    run = SimpleNamespace(model=model.requires_grad_(False).to(memory_format=th.channels_last), amp=True)
    judge = load_encoder(args.judge, "cpu")
    items, texts = distill.held_out(args)
    targets = {name: th.stack([item["embeds"][name][distill.SAMPLE_KIND] for item in items]) for name in args.models}
    real = th.stack([item["image"] for item in items])
    with th.no_grad():
        judged_text, judged_real = encode_texts(judge, texts), encode_images(judge, real)

    def embeds(i: int) -> dict[str, th.Tensor]:
        return distill.on_device(targets, slice(i, i + 1), args.device)

    def guidance(i: int) -> sampler.Guidance:
        return sampler.Guidance(
            partial(guidance_loss, distill.guide_for(teacher, embeds(i))), teacher.rate, distill.WINDOW
        )

    def base(_: int) -> sampler.Denoiser:
        return partial(denoise, teacher.unet, None)

    def student(i: int) -> sampler.Denoiser:
        return distill.student_denoiser(run, teacher, log_alpha_bar, embeds(i), 1.0)

    for steps in map(int, sys.argv[4:]):
        args.sample_steps = steps
        rows = {
            "guided": distill.draw(args, log_alpha_bar, base, guidance),
            "student guided": distill.draw(args, log_alpha_bar, student, guidance),
        }
        with th.no_grad():
            features = {name: encode_images(judge, images) for name, images in rows.items()}
        for kind, reference in (("text", judged_text), ("image", judged_real)):
            ours, theirs = ((features[name] * reference).sum(1) for name in ("student guided", "guided"))
            gap = ours - theirs
            error = gap.std(unbiased=True) / len(gap) ** 0.5
            print(
                f"{steps} steps, {kind}: guided {theirs.mean():.4f}  student guided {ours.mean():.4f}  "
                f"difference {gap.mean():+.4f} +- {error:.4f}  student wins {int((gap > 0).sum())} of {len(gap)}",
                flush=True,
            )
        grid = th.cat([th.cat(list(images), dim=2) for images in (real, *rows.values())], dim=1)
        Image.fromarray(grid.permute(1, 2, 0).numpy()).save(out / f"guided_more_{steps}.jpg", quality=90)
    (out / "guided_more_texts.txt").write_text("\n".join(texts), encoding="utf-8")


if __name__ == "__main__":
    main()

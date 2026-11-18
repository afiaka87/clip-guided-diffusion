"""CLIP guidance on `finetune.cond`'s model against `cgd` as it is: the pretrained model told the closest class.

    python reports/evidence/finetune/diag_cond_guided.py D:/cgd-finetune/cond/align 1 16 30
    python reports/evidence/finetune/diag_cond_guided.py D:/cgd-finetune/cond256-vast 1 32 30 average

The arguments are the run's folder, a seed for which held-out texts and which noise, how many texts, the
number of steps, and optionally `average` to draw with the checkpoint's running average of the weights instead
of the weights themselves. Every row draws each text from the same noise:

- `class guided`: the pretrained model, told the class closest to the text, under CLIP guidance. `cgd` today.
- `text`: the run's model with classifier-free guidance from the text's embeddings, and no CLIP.
- `text guided`: the same under CLIP guidance.
- `text+prior`: the run's model told the text's embeddings and the image embedding `finetune.prior` predicts
  from them, and no CLIP: what it draws from a prompt by itself.
- `text+prior guided`: the same, also told the image embedding `finetune.prior` predicts from the text.

It prints the judge's mean score of each row against the texts and the real images, each row's difference from
`class guided` over the pairs with its standard error, and the seconds each row took per image.
"""

import sys
import time
from functools import partial
from pathlib import Path

import torch as th
from PIL import Image

from cgd import clip_models
from cgd.adm import load_model
from cgd.finetuned import predict_image
from finetune import cond, distill, prior
from finetune.embed import encode_images, encode_texts, load_encoder
from cgd.conditioned import AdapterConfig, ConditionedUNet
from finetune.train import CHECKPOINT_NAME, IMAGE_SIZE


def main() -> None:
    out = Path(sys.argv[1])
    which = sys.argv[5] if len(sys.argv) > 5 else "weights"
    args = cond.parse_args(
        ["--out", str(out), "--seed", sys.argv[2], "--samples", sys.argv[3], "--sample-steps", sys.argv[4]]
    )
    th.cuda.set_per_process_memory_fraction(0.9)
    unet, log_alpha_bar = load_model(IMAGE_SIZE, args.device, False, False, False)
    checkpoint = th.load(out / CHECKPOINT_NAME, map_location="cpu", weights_only=True)
    # Without the checkpoint's weights this is the pretrained model. Frozen, with no optimizer: guidance needs the memory.
    model = ConditionedUNet(unet, AdapterConfig(**checkpoint["config"])).to(args.device)
    model = model.to(memory_format=th.channels_last).requires_grad_(False)
    judge = load_encoder(args.judge, "cpu")
    class_features = clip_models.imagenet_text_features(args.models[0], "cpu")
    items, texts = distill.held_out(args)
    embeds = {}
    for name in args.models:
        embeds[cond.slot(name, "image")] = th.stack([item["embeds"][name]["image"] for item in items])
        embeds[cond.slot(name, "text")] = th.stack([item["embeds"][name][cond.SAMPLE_KIND] for item in items])
    labels = cond.anchor_classes(embeds[cond.slot(args.models[0], "text")], class_features)
    real = th.stack([item["image"] for item in items])
    with th.no_grad():
        judged_text, judged_real = encode_texts(judge, texts), encode_images(judge, real)
    samples = cond.Samples(texts, embeds, labels, judged_text, {"real": real})
    priors = th.load(prior.PRIOR_PATH, weights_only=True)
    predicted = dict(embeds)
    for name in args.models:
        predicted[cond.slot(name, "image")] = predict_image(priors[name], embeds[cond.slot(name, "text")])
    with_prior = samples._replace(embeds=predicted)
    guidance = partial(cond.clip_guidance, args, cond.load_guide_towers(args), embeds)
    rows, seconds = {}, {}

    def timed(name: str, denoiser_for, guidance_for=None) -> None:
        th.cuda.synchronize()
        began = time.perf_counter()
        rows[name] = distill.draw(args, log_alpha_bar, denoiser_for, guidance_for)
        th.cuda.synchronize()
        seconds[name] = (time.perf_counter() - began) / args.samples
        print(f"{name}: {seconds[name]:.1f} s an image", flush=True)

    def alone(i: int):
        return cond.class_denoiser(model, True, cond.sample_condition(samples, i, args))

    def told(source: cond.Samples, image: bool):
        return lambda i: cond.cfg_denoiser(
            model, True, cond.sample_condition(source, i, args, image, True), args.sample_scale
        )

    timed("class guided", alone, guidance)
    model.load_state_dict(checkpoint[which], strict=False)
    model.conditioner.load_state_dict(checkpoint["statistics"], strict=False)
    timed("text", told(samples, False))
    timed("text guided", told(samples, False), guidance)
    timed("text+prior", told(with_prior, True))
    timed("text+prior guided", told(with_prior, True), guidance)
    with th.no_grad():
        features = {name: encode_images(judge, images) for name, images in rows.items()}
    for kind, reference in (("text", judged_text), ("image", judged_real)):
        base = (features["class guided"] * reference).sum(1)
        print(f"{kind}: class guided {base.mean():.4f}", flush=True)
        for name in list(rows)[1:]:
            ours = (features[name] * reference).sum(1)
            gap = ours - base
            error = gap.std(unbiased=True) / len(gap) ** 0.5
            print(
                f"{kind}: {name} {ours.mean():.4f}  difference {gap.mean():+.4f} +- {error:.4f}  "
                f"wins {int((gap > 0).sum())} of {len(gap)}",
                flush=True,
            )
    grid = th.cat([th.cat(list(images), dim=2) for images in (real, *rows.values())], dim=1)
    Image.fromarray(grid.permute(1, 2, 0).numpy()).save(out / f"cond_guided_{args.sample_steps}_{which}.jpg", quality=90)
    classes = [clip_models.IMAGENET_CLASSES[label] for label in labels.tolist()]
    lines = [f"{text} [{name}]" for text, name in zip(texts, classes, strict=True)]
    (out / "cond_guided_texts.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

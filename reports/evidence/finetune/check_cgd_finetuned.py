"""`cgd -ft` against `cgd` at 256, by `cgd`'s own generate(), on held-out texts the fine-tune never saw.

    uv run python reports/evidence/finetune/check_cgd_finetuned.py D:/cgd-finetune/cond256-vast/cgd.pt 16

diag_cond_guided.py measured the fine-tune with the training's stored embeddings. This draws through the
command's path instead: the prompt encoded by `cgd`, the class it picks, the prior from the export. Both rows
guide with ViT-B/32 and PE-Core-B-16 at 16 cutouts, 30 steps, each text from its own seed, and SigLIP2, which
neither was trained or guided with, scores each image against its text and its real image. The grid is
`cgd`'s row above `cgd -ft`'s.
"""

import sys
import time
from pathlib import Path

import torch as th
import torchvision.transforms.functional as tvf
from PIL import Image

from cgd import adm
from finetune import cond, distill
from finetune.embed import encode_images, encode_texts, load_encoder

GUIDE = "ViT-B/32+PE-Core-B-16"


def main() -> None:
    export, count = sys.argv[1], int(sys.argv[2])
    out = Path(export).parent / "check_cgd"
    items, texts = distill.held_out(cond.parse_args(["--seed", "1", "--samples", str(count)]))
    rows: dict[str, list[th.Tensor]] = {"cgd": [], "cgd -ft": []}
    seconds = {name: 0.0 for name in rows}
    for i, text in enumerate(texts):
        for name, extra in (("cgd", {"image_size": 256}), ("cgd -ft", {"finetuned": export})):
            began = time.perf_counter()
            saved = list(
                adm.generate([text], clip_model=GUIDE, seed=i, output_dir=out / name, progress=False, **extra)
            )
            seconds[name] += time.perf_counter() - began
            image = tvf.pil_to_tensor(Image.open(saved[-1][1]).convert("RGB"))
            rows[name].append(image)
        print(f"{i}: {text}", flush=True)
    judge = load_encoder("ViT-L-16-SigLIP2-256", "cpu")
    real = th.stack([item["image"] for item in items])
    with th.no_grad():
        references = {"text": encode_texts(judge, texts), "image": encode_images(judge, real)}
        features = {name: encode_images(judge, th.stack(images)) for name, images in rows.items()}
    for name in rows:
        print(f"{name}: {seconds[name] / count:.1f} s an image (the first loads the models)")
    for kind, reference in references.items():
        base = (features["cgd"] * reference).sum(1)
        ours = (features["cgd -ft"] * reference).sum(1)
        gap = ours - base
        error = gap.std(unbiased=True) / len(gap) ** 0.5
        print(
            f"{kind}: cgd {base.mean():.4f}  cgd -ft {ours.mean():.4f}  difference {gap.mean():+.4f} +- {error:.4f}  "
            f"wins {int((gap > 0).sum())} of {len(gap)}"
        )
    # The datasets' own images are not ours to publish, so the grid holds only what the two rows drew.
    grid = th.cat([th.cat(images, dim=2) for images in rows.values()], dim=1)
    Image.fromarray(grid.permute(1, 2, 0).numpy()).save(out / "check_cgd_finetuned.jpg", quality=90)
    (out / "texts.txt").write_text("\n".join(texts), encoding="utf-8")


if __name__ == "__main__":
    main()

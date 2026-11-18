"""`-ft` end to end on the GPU, with a fine-tune at its first step.

That is the pretrained class-conditional 256 model with what the fine-tune adds at zero, exported as
finetune.export exports a run. It needs only the checkpoints cgd downloads, and goes through every part of a
run: the prompts' embeddings, the class, classifier-free guidance under CLIP's, and the export's loading.
tests/unit checks what the parts compute.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch as th

from cgd import adm, finetuned
from cgd.conditioned import AdapterConfig, ConditionedUNet, adapter_checkpoint
from cgd.guide import load_guides
from cgd.prompts import Scene
from tests.integration import test_adm

pytestmark = pytest.mark.cuda

PROMPT, OTHER_PROMPT = test_adm.PROMPT, test_adm.OTHER_PROMPT
EMBED_DIMS = {"ViT-B/32": 512, "PE-Core-B-16": 1024}


@pytest.fixture(name="export", scope="module")
def fixture_export(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """The fine-tune `finetune.cond` starts from, exported, with priors that predict the text's own embedding."""
    unet, _ = adm.build_model(finetuned.SIZE, "cpu", fp16=False)
    dims = {name: dim for model, dim in EMBED_DIMS.items() for name in finetuned.slots([model])}
    model = ConditionedUNet(unet, AdapterConfig(dims, tokens=4, standardised=True, full=True))
    priors = {name: {"weight": th.eye(dim), "bias": th.zeros(dim)} for name, dim in EMBED_DIMS.items()}
    path = tmp_path_factory.mktemp("finetuned") / "cgd.pt"
    finetuned.export(adapter_checkpoint(model), priors, path)
    yield str(path)
    adm.load_finetuned.cache_clear()  # The tests after these need the VRAM more.


def generate(output_dir: Path, export: str, **settings: Any) -> list[tuple[int, Path]]:
    """Every (batch index, path) a run of the fine-tune saves: three steps at four cutouts, and few frames."""
    return test_adm.generate(
        output_dir, **{"finetuned": export, "image_size": None, "steps": 3, "num_cutouts": 4} | settings
    )


def test_at_its_first_step_it_is_the_pretrained_model_told_the_class_cgd_picks(export: str) -> None:
    tuned, _ = adm.load_finetuned(export, "cuda")
    images = [Scene((PROMPT,), ())]
    condition = adm.finetuned_conditions(tuned, images, batch_size=1, device="cuda", fp16=False, progress=False)[0]
    guide = load_guides("ViT-B/32", images, device="cuda")[0]
    labels = adm.closest_class(guide, "ViT-B/32", batch_size=1, device="cuda", fp16=False, progress=False)
    assert th.equal(condition.labels, labels)
    assert condition.keep.all()  # A text prompt gives each model's text, and the prior its image.
    pretrained, _ = adm.load_model(finetuned.SIZE, "cuda", fp16=True)
    generator = th.Generator("cuda").manual_seed(0)
    x, t = th.randn(1, 3, 256, 256, device="cuda", generator=generator), th.tensor([600.0], device="cuda")
    with th.no_grad():
        told = finetuned.cfg_denoiser(tuned.model, True, condition, finetuned.CFG_SCALE)(x, t)
        expected = adm.denoise(pretrained, labels, x, t).float()
    assert (told - expected).abs().max() < 0.02  # Half precision, by two routes.
    adm.load_model.cache_clear()


def test_a_seed_reproduces_a_batch_told_text_image_and_negative_prompts(workdir: Path, export: str, photon: Path):
    """Two images at 256 take the path that recomputes the U-Net's activations in the backward pass."""
    settings = {"prompts": [PROMPT, "blurry:-1"], "image_prompts": [str(photon)], "batch_size": 2}
    first = [path.read_bytes() for _, path in generate(workdir / "first", export, **settings)]
    again = [path.read_bytes() for _, path in generate(workdir / "again", export, **settings)]
    assert first == again
    assert first[0] != first[1]


def test_a_sequence_slides_to_each_image_as_its_prompts_make_it_alone(workdir: Path, export: str) -> None:
    frames = generate(workdir / "sequence", export, prompts=[PROMPT, f">{OTHER_PROMPT}"], transition_frames=2)
    alone = generate(workdir / "alone", export, prompts=[OTHER_PROMPT])
    assert len(frames) == 4  # The first image's first and last frames, and the two that slide to the second.
    assert frames[-1][1].read_bytes() == alone[-1][1].read_bytes()


def test_it_fits_in_8_gb_with_its_defaults(workdir: Path, export: str) -> None:
    th.cuda.reset_peak_memory_stats()
    generate(workdir, export, steps=2, num_cutouts=None)
    assert th.cuda.max_memory_allocated() < 8 * 2**30

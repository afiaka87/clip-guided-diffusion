"""cgd.finetuned: what `-ft` tells the fine-tune for each image of a run, and the export it loads."""

import zlib
from pathlib import Path

import pytest
import torch as th
from torch.nn import functional as tf

from cgd import finetuned as ft
from cgd.conditioned import FusedAttention, adapter_checkpoint
from cgd.prompts import Scene
from tests.unit.test_finetune import EMBED_DIMS, SLOT_DIMS, anchored_inputs, anchored_model, class_unet, randomise

MODELS = list(EMBED_DIMS)


def stand_in(kind: str, calls: list[tuple[str, str]] | None = None) -> ft.Encode:
    """An encoder that gives each prompt its own fixed embedding by each model, noting what it was asked."""

    def encode(name: str, prompt: str) -> th.Tensor:
        if calls is not None:
            calls.append((name, prompt))
        generator = th.Generator().manual_seed(zlib.crc32(f"{kind} {name} {prompt}".encode()))
        return th.randn(1, EMBED_DIMS[name], generator=generator) * 3  # Not unit length, as CLIP's are not.

    return encode


def unit(embedding: th.Tensor) -> th.Tensor:
    """The embedding at unit length."""
    return tf.normalize(embedding, dim=-1)


def priors() -> dict[str, dict[str, th.Tensor]]:
    """A map from text embeddings to image embeddings for each model."""
    generator = th.Generator().manual_seed(4)
    return {
        name: {"weight": th.randn(dim, dim, generator=generator), "bias": th.randn(dim, generator=generator)}
        for name, dim in EMBED_DIMS.items()
    }


def tuned() -> ft.Finetuned:
    """A small fine-tune at its first step, with its priors."""
    return ft.Finetuned(anchored_model(), priors())


TEXT, IMAGE = stand_in("text"), stand_in("image")


def test_the_conditions_are_each_models_image_and_text_in_the_models_order():
    assert ft.model_names(anchored_model().config) == MODELS
    assert ft.slots(MODELS) == list(SLOT_DIMS)


def test_a_text_prompt_is_told_with_the_image_embedding_the_prior_predicts_from_it():
    calls: list[tuple[str, str]] = []
    finetuned = tuned()
    scene = Scene(("a fox:2", "snow", "blurry:-1"), ())
    told = ft.prompt_embeddings(finetuned, [scene], stand_in("text", calls), IMAGE)[0]
    for name in MODELS:
        text = unit(2 * unit(TEXT(name, "a fox")) + unit(TEXT(name, "snow")))
        assert th.allclose(told[ft.slot(name, "text")], text, atol=1e-6)
        assert th.allclose(told[ft.slot(name, "image")], ft.predict_image(finetuned.priors[name], text), atol=1e-6)
    assert "blurry" not in {prompt for _, prompt in calls}  # The condition cannot say "not", so it is not told.


def test_image_prompts_are_the_image_embedding_and_without_a_text_there_is_none():
    finetuned = tuned()
    both = Scene(("a fox",), ("fox.png",))
    alone = Scene((), ("fox.png", "snow.png:0.5"))
    told_both, told_alone = ft.prompt_embeddings(finetuned, [both, alone], TEXT, IMAGE)
    for name in MODELS:
        assert th.allclose(told_both[ft.slot(name, "image")], unit(IMAGE(name, "fox.png")), atol=1e-6)
        expected = unit(unit(IMAGE(name, "fox.png")) + 0.5 * unit(IMAGE(name, "snow.png")))
        assert th.allclose(told_alone[ft.slot(name, "image")], expected, atol=1e-6)
        assert told_alone[ft.slot(name, "text")] is None


def test_each_clip_model_encodes_every_image_of_a_sequence_before_the_next_model():
    calls: list[tuple[str, str]] = []
    ft.prompt_embeddings(tuned(), [Scene(("a castle",), ()), Scene(("a fox",), ())], stand_in("text", calls), IMAGE)
    assert [name for name, _ in calls] == [MODELS[0], MODELS[0], MODELS[1], MODELS[1]]


def test_the_class_is_chosen_by_the_first_models_text_or_without_one_its_image():
    finetuned = tuned()
    text, image = ft.prompt_embeddings(
        finetuned, [Scene(("a fox",), ("fox.png",)), Scene((), ("fox.png",))], TEXT, IMAGE
    )
    assert ft.anchor(finetuned, text) == (MODELS[0], text[ft.slot(MODELS[0], "text")])
    assert ft.anchor(finetuned, image) == (MODELS[0], image[ft.slot(MODELS[0], "image")])
    nothing = ft.prompt_embeddings(finetuned, [Scene(("blurry:-1",), ())], TEXT, IMAGE)[0]
    with pytest.raises(ValueError, match="no prompt of positive weight"):
        ft.anchor(finetuned, nothing)


def test_a_batch_is_told_every_kept_embedding_and_zeros_where_one_is_dropped():
    finetuned = tuned()
    embeds = ft.prompt_embeddings(finetuned, [Scene((), ("fox.png",))], TEXT, IMAGE)[0]
    condition = ft.batch_condition(finetuned, embeds, 7, batch_size=2, device="cpu")
    assert condition.keep.tolist() == [[True, False, True, False]] * 2  # Each model's image, and no text.
    assert condition.labels.tolist() == [7, 7]
    for name, dim in SLOT_DIMS.items():
        assert condition.embeds[name].shape == (2, dim)
        if name.endswith("text"):
            assert not condition.embeds[name].any()
        else:
            assert th.equal(condition.embeds[name][1:], embeds[name])


def test_an_export_draws_what_the_run_drew_and_keeps_the_unets_own_attention(tmp_path: Path):
    model = anchored_model()
    randomise(model)
    x, t, condition = anchored_inputs()
    told = ft.noise_given(model, False, x, t, condition)
    path = tmp_path / "cgd.pt"
    ft.export(adapter_checkpoint(model) | {"steps": 7}, priors(), path)
    # The frozen half comes from the pretrained checkpoint, which the same seed stands in for.
    loaded = ft.load(path, class_unet(), "cpu")
    assert th.allclose(ft.noise_given(loaded.model, False, x, t, condition), told, atol=1e-6)
    assert not [module for module in loaded.model.modules() if isinstance(module, FusedAttention)]
    assert not [parameter for parameter in loaded.model.parameters() if parameter.requires_grad]
    for name, prior in priors().items():
        assert all(th.equal(loaded.priors[name][key], value) for key, value in prior.items())
    assert th.load(path, weights_only=True)["steps"] == 7


def test_an_export_of_the_running_average_draws_with_it(tmp_path: Path):
    model = anchored_model()
    average = {name: value.clone() for name, value in adapter_checkpoint(model)["weights"].items()}  # Before training.
    randomise(model)
    x, t, condition = anchored_inputs()
    path = tmp_path / "cgd.pt"
    ft.export(adapter_checkpoint(model) | {"average": average}, priors(), path, which="average")
    loaded = ft.load(path, class_unet(), "cpu")
    pretrained = class_unet()(x, t, y=condition.labels)[:, :3]  # What an untrained model draws, told anything.
    assert th.allclose(ft.noise_given(loaded.model, False, x, t, condition), pretrained, atol=1e-5)


def test_an_export_needs_a_prior_for_every_model_it_is_told_the_embeddings_of(tmp_path: Path):
    with pytest.raises(ValueError, match="No prior for \\['PE-Core-B-16'\\]"):
        ft.export(adapter_checkpoint(anchored_model()), {MODELS[0]: priors()[MODELS[0]]}, tmp_path / "cgd.pt")


def test_loading_an_export_draws_none_of_the_runs_random_numbers(tmp_path: Path):
    path = tmp_path / "cgd.pt"
    ft.export(adapter_checkpoint(anchored_model()), priors(), path)
    unet = class_unet()
    th.manual_seed(0)
    ft.load(path, unet, "cpu")
    drawn = th.rand(4)
    th.manual_seed(0)
    assert th.equal(th.rand(4), drawn)

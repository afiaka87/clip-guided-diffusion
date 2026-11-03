"""The real CLIP models: loading them, encoding prompts, and what guidance gets from them."""

from pathlib import Path

import pytest
import torch as th
from PIL import Image
from torch.nn import functional as tf

from cgd import clip_models
from cgd.checks import ConfigError
from cgd.clip_models import CLIP_CACHE_PATH, OPEN_CLIP_MODELS, encode_image, encode_texts, load_clip, preprocess
from cgd.guide import load_guide, query
from cgd.imagenet_classes import IMAGENET_CLASSES

pytestmark = pytest.mark.cuda

PERCEPTION_ENCODER = "PE-Core-B-16"
SIGLIP = "ViT-L-16-SigLIP2-256"
PROMPT = "a photograph of a red fox in the snow"


def top_classes(features: th.Tensor, model_name: str, count: int = 10) -> str:
    """The names of the ImageNet classes closest to [1, features], joined with spaces."""
    similarity = tf.normalize(features, dim=-1) @ clip_models.imagenet_text_features(model_name, "cuda").T
    return " ".join(IMAGENET_CLASSES[i] for i in similarity.topk(count).indices[0].tolist())


def test_every_listed_open_clip_model_is_published() -> None:
    import open_clip

    published = set(open_clip.list_pretrained())
    assert all(model in published for model in OPEN_CLIP_MODELS.items())


def test_loading_a_model_leaves_the_random_stream_and_init_functions_alone() -> None:
    init_functions = {name: getattr(th.nn.init, name) for name in clip_models.INIT_FUNCTIONS}
    th.manual_seed(0)
    expected = th.rand(4)
    th.manual_seed(0)
    load_clip.__wrapped__("RN50", "cpu")
    assert th.equal(th.rand(4), expected)
    assert {name: getattr(th.nn.init, name) for name in clip_models.INIT_FUNCTIONS} == init_functions


def test_the_closest_imagenet_class_is_the_subject_of_the_prompt() -> None:
    guide = load_guide("ViT-B/32", [PROMPT], device="cuda")
    assert IMAGENET_CLASSES[clip_models.closest_imagenet_class(query(guide), "ViT-B/32", "cuda")] == "red fox"


class TestOpenClip:
    def test_only_the_vision_tower_is_on_the_gpu(self) -> None:
        model, _ = load_clip(PERCEPTION_ENCODER, "cuda")
        for name, parameter in model.named_parameters():
            assert parameter.device.type == ("cuda" if name.startswith("visual") else "cpu"), name

    def test_fp16_only_changes_the_vision_tower(self) -> None:
        model, _ = load_clip(PERCEPTION_ENCODER, "cuda", True)
        for name, parameter in model.named_parameters():
            assert parameter.dtype == (th.float16 if name.startswith("visual") else th.float32), name

    def test_skipping_random_init_still_loads_every_weight(self) -> None:
        import open_clip

        expected = open_clip.create_model(PERCEPTION_ENCODER, pretrained="meta", cache_dir=str(CLIP_CACHE_PATH))
        loaded = load_clip.__wrapped__(PERCEPTION_ENCODER, "cpu")[0].state_dict()
        assert expected.state_dict().keys() == loaded.keys()
        for name, tensor in expected.state_dict().items():
            assert th.equal(tensor, loaded[name]), name

    def test_preprocess_uses_the_normalization_each_model_was_trained_with(self) -> None:
        white = th.ones(1, 3, 8, 8)
        assert th.equal(preprocess(load_clip.__wrapped__(PERCEPTION_ENCODER, "cpu")[0], white), white)

    @pytest.mark.parametrize("model_name", [PERCEPTION_ENCODER, SIGLIP])
    def test_text_and_images_land_in_the_same_space(self, model_name: str, goose: Path) -> None:
        text = encode_texts(["a photograph of a goose running across a field"], model_name, "cuda")
        image = encode_image(str(goose), model_name, "cuda", image_size=128, num_cutouts=4)
        assert "goose" in top_classes(text, model_name)
        assert "goose" in top_classes(image.mean(0, keepdim=True), model_name)

    def test_gradient_checkpointing_keeps_the_gradient(self) -> None:
        model, size = load_clip(PERCEPTION_ENCODER, "cuda")

        def gradient() -> th.Tensor:
            image = th.linspace(0, 1, 3 * size**2, device="cuda").view(1, 3, size, size).requires_grad_()
            return th.autograd.grad(model.encode_image(preprocess(model, image)).square().sum(), image)[0]

        checkpointed = gradient()
        model.visual.set_grad_checkpointing(False)
        try:
            plain = gradient()
        finally:
            model.visual.set_grad_checkpointing(True)
        assert checkpointed.abs().max().item() > 0
        assert (checkpointed - plain).abs().max().item() < 1e-4 * checkpointed.abs().max().item()


class TestImagePrompts:
    @pytest.mark.parametrize(
        ("size", "features"),
        [
            ((300, 150), 4),
            ((1024, 768), 4),
            ((40, 40), 1),  # Smaller than CLIP's input, so every cutout would be the whole image.
        ],
    )
    def test_any_shape_or_size_of_image_gives_one_feature_per_distinct_cutout(
        self, photon: Path, tmp_path: Path, size: tuple[int, int], features: int
    ) -> None:
        path = tmp_path / "source.png"
        Image.open(photon).convert("RGB").resize(size).save(path)
        assert encode_image(str(path), "ViT-B/32", "cuda", image_size=128, num_cutouts=4).shape == (features, 512)

    def test_an_image_prompt_embeds_like_its_content(self, goose: Path) -> None:
        image = encode_image(str(goose), "ViT-B/32", "cuda", image_size=128, num_cutouts=4)
        assert "goose" in top_classes(image.mean(0, keepdim=True), "ViT-B/32")

    def test_each_model_of_an_ensemble_weights_its_own_targets(self, photon: Path) -> None:
        guide = load_guide("ViT-B/32+PE-Core-B-16", ["a fox:3"], [f"{photon}:1"], device="cuda")
        for tower in guide.towers:
            assert tower.weights.shape == (len(tower.targets),)
            assert tower.weights.sum().item() == pytest.approx(1.0)
            assert tower.weights[0].item() == pytest.approx(0.75)  # The text prompt's share. Each cutout has 1/16.

    def test_weights_that_cancel_out_are_explained(self) -> None:
        with pytest.raises(ConfigError, match="must not sum to 0"):
            load_guide("ViT-B/32", ["cat:1", "dog:-1"], device="cuda")

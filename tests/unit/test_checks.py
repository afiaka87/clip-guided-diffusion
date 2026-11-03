"""Rejecting configurations that cannot work, before anything slow runs."""

from pathlib import Path
from typing import Any

import pytest

from cgd import adm, backbone
from cgd.checks import ConfigError
from cgd.clip_models import MODEL_NAMES, check_model, ensemble_members
from cgd.device import default_device

ADM_SETTINGS: dict[str, Any] = {
    "image_size": 128,
    "height": 128,
    "width": 128,
    "batch_size": 1,
    "steps": 30,
    "guidance_rate": 0.08,
    "guidance_limit": None,
    "aesthetic_scale": 0.0,
    "init_image": None,
    "init_scale": 0.0,
    "skip_timesteps": 0,
    "save_frequency": 10,
}
BACKBONE_SETTINGS: dict[str, Any] = {
    "height": 512,
    "width": 512,
    "batch_size": 1,
    "iterations": 50,
    "aesthetic_scale": 0.05,
    "save_frequency": 10,
}


class TestAdmParameters:
    def test_a_good_configuration_passes(self, photon: Path) -> None:
        settings = ADM_SETTINGS | {"image_size": 256, "width": 288, "init_image": str(photon), "skip_timesteps": 500}
        adm.check_parameters(["a fox"], [f"{photon}:2", "https://example.com/cat.png"], **settings)

    @pytest.mark.parametrize(
        ("prompts", "overrides", "message"),
        [
            ([], {}, "at least one prompt"),
            (["a"], {"batch_size": 0}, "--batch-size"),
            (["a"], {"save_frequency": 0}, "--save-frequency"),
            (["a"], {"aesthetic_scale": -1}, "--aesthetic-scale"),
            (["a"], {"image_size": 100, "height": 100, "width": 100}, "No checkpoint for size 100"),
            (["a"], {"height": 136}, "multiples of 16"),
            (["a"], {"width": 0}, "positive multiples"),
            (["a"], {"steps": 0}, "--steps"),
            (["a"], {"guidance_rate": -0.1}, "--guidance-rate"),
            (["a"], {"guidance_limit": 0.0}, "--guidance-limit"),
            (["a"], {"guidance_limit": 1.5}, "--guidance-limit"),
            (["a"], {"init_image": "missing.png"}, "No such image: missing.png"),
            (["a"], {"skip_timesteps": 500}, "need an --init-image"),
            (["a"], {"init_scale": 1000}, "need an --init-image"),
        ],
    )
    def test_a_bad_one_is_explained(self, prompts: list[str], overrides: dict[str, Any], message: str) -> None:
        with pytest.raises(ConfigError, match=message):
            adm.check_parameters(prompts, [], **ADM_SETTINGS | overrides)

    def test_a_missing_image_prompt_is_explained(self) -> None:
        with pytest.raises(ConfigError, match=r"No such image: missing\.png"):
            adm.check_parameters([], ["missing.png:0.5"], **ADM_SETTINGS)

    @pytest.mark.parametrize("skip", [-1, 1000])
    def test_skipping_off_the_schedule_is_explained(self, photon: Path, skip: int) -> None:
        with pytest.raises(ConfigError, match="between 0 and 999"):
            adm.check_parameters(["a"], [], **ADM_SETTINGS | {"init_image": str(photon), "skip_timesteps": skip})


class TestBackboneParameters:
    def test_any_multiple_of_eight_passes(self) -> None:
        backbone.check_parameters(["a fox"], [], **BACKBONE_SETTINGS | {"height": 136, "width": 200})

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"height": 100}, "multiples of 8"),
            ({"width": -8}, "positive multiples"),
            ({"iterations": -1}, "--iterations"),
            ({"save_frequency": 0}, "--save-frequency"),
        ],
    )
    def test_a_bad_one_is_explained(self, overrides: dict[str, Any], message: str) -> None:
        with pytest.raises(ConfigError, match=message):
            backbone.check_parameters(["a fox"], [], **BACKBONE_SETTINGS | overrides)

    def test_an_unknown_backbone_is_explained_before_anything_loads(self) -> None:
        with pytest.raises(ConfigError, match="Unknown backbone 'dall-e'"):
            next(backbone.generate(["a fox"], backbone="dall-e"))


class TestClipModels:
    @pytest.mark.parametrize("model_name", [*MODEL_NAMES, "ViT-B/32+PE-Core-B-16", __file__])
    def test_listed_models_ensembles_and_checkpoint_files_pass(self, model_name: str) -> None:
        check_model(model_name)

    @pytest.mark.parametrize("model_name", ["ViT-Z/99", "ViT-B/32+ViT-Z/99"])
    def test_an_unknown_model_is_explained(self, model_name: str) -> None:
        with pytest.raises(ConfigError, match="Unknown CLIP model 'ViT-Z/99'"):
            check_model(model_name)

    def test_a_checkpoint_file_is_one_model_even_with_a_plus_in_its_name(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "ViT-B+tuned.pt"
        checkpoint.write_bytes(b"")
        assert ensemble_members(str(checkpoint)) == [str(checkpoint)]
        assert ensemble_members("ViT-B/32+PE-Core-B-16") == ["ViT-B/32", "PE-Core-B-16"]


class TestDevice:
    @pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:1"])
    def test_a_device_given_is_kept(self, device: str) -> None:
        assert default_device(device) == device

    @pytest.mark.parametrize("device", ["gpu", "cuda:", "mps"])
    def test_an_unknown_device_is_explained(self, device: str) -> None:
        with pytest.raises(ConfigError, match="Unknown device"):
            default_device(device)

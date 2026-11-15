"""Perturbed-attention guidance, on a small U-Net of the kind it perturbs, with random weights."""

from functools import partial

import pytest
import torch as th
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from guided_diffusion.unet import AttentionBlock, UNetModel
from torch import nn

from cgd import adm, pag

LAYERS = ("input_blocks.1.1", "middle_block.1")  # Two of the small U-Net's attention blocks.


@pytest.fixture(name="model", scope="module")
def fixture_model() -> UNetModel:
    """A U-Net with attention at both of its resolutions, and random weights, since it starts out predicting zero."""
    with th.random.fork_rng():
        th.manual_seed(0)
        model = UNetModel(16, 3, 32, 6, 1, attention_resolutions=(1, 2), channel_mult=(1, 2), num_head_channels=16)
        for parameter in model.parameters():
            nn.init.normal_(parameter, std=0.2)
    return model.eval().requires_grad_(False)


def unet(model: UNetModel, x: th.Tensor, t: th.Tensor) -> th.Tensor:
    """The noise the U-Net predicts, as adm.denoise() gives it."""
    return adm.denoise(model, None, x, t)


@pytest.fixture(name="noisy")
def fixture_noisy() -> tuple[th.Tensor, th.Tensor]:
    """A batch of two noisy images and their timesteps."""
    return th.randn(2, 3, 16, 16), th.full((2,), 500.0)


def test_a_perturbed_block_returns_each_positions_own_values(model: UNetModel) -> None:
    attention = model.get_submodule("middle_block.1").attention
    qkv = th.randn(2, 3 * 64, 8 * 8)  # Four heads of 16 channels.
    with pag.perturbed_attention(model, LAYERS):
        perturbed = attention(qkv, (8, 8))
    values = qkv.view(2 * 4, 3 * 16, 64)[:, 32:].reshape(2, 64, 64)
    assert th.equal(perturbed, values)
    assert not th.allclose(attention(qkv, (8, 8)), values)


def test_perturbing_changes_the_prediction_only_within_it(model: UNetModel, noisy: tuple[th.Tensor, th.Tensor]) -> None:
    before = unet(model, *noisy)
    with pag.perturbed_attention(model, LAYERS):
        within = unet(model, *noisy)
    assert not th.allclose(within, before)
    assert th.equal(unet(model, *noisy), before)


def test_the_blocks_attend_again_after_an_error_within_it(model: UNetModel, noisy: tuple[th.Tensor, th.Tensor]) -> None:
    before = unet(model, *noisy)
    with pytest.raises(ZeroDivisionError), pag.perturbed_attention(model, LAYERS):
        _ = 1 / 0
    assert th.equal(unet(model, *noisy), before)


def test_a_layer_without_attention_is_refused_and_none_is_left_perturbed(
    model: UNetModel, noisy: tuple[th.Tensor, th.Tensor]
) -> None:
    before = unet(model, *noisy)
    with (
        pytest.raises(TypeError, match=r"input_blocks\.0 is not an attention block"),
        pag.perturbed_attention(model, ("middle_block.1", "input_blocks.0")),
    ):
        pass
    assert th.equal(unet(model, *noisy), before)


def test_the_prediction_moves_away_from_the_perturbed_one(model: UNetModel, noisy: tuple[th.Tensor, th.Tensor]) -> None:
    eps = unet(model, *noisy)
    with pag.perturbed_attention(model, LAYERS):
        perturbed = unet(model, *noisy)
    guided = pag.denoise(partial(unet, model), model, LAYERS, *noisy)
    assert th.allclose(guided, eps + pag.SCALE * (eps - perturbed))
    assert not th.allclose(guided, eps)


def test_only_the_unperturbed_prediction_carries_a_gradient(
    model: UNetModel, noisy: tuple[th.Tensor, th.Tensor]
) -> None:
    x, t = noisy
    x = x.requires_grad_()
    guided = th.autograd.grad(pag.denoise(partial(unet, model), model, LAYERS, x, t).sum(), x)[0]
    plain = th.autograd.grad(unet(model, x, t).sum(), x)[0]
    assert th.allclose(guided, plain, atol=1e-6)


def test_its_layers_are_the_deepest_attention_blocks_of_the_unconditional_256_checkpoint() -> None:
    flags = model_and_diffusion_defaults() | adm.CHECKPOINT_FLAGS[adm.PAG_SIZE]
    with th.device("meta"):  # Only its structure is needed, so nothing is allocated.
        model, _ = create_model_and_diffusion(**flags | {"image_size": adm.PAG_SIZE, "class_cond": False})
    blocks = [model.get_submodule(name) for name in pag.LAYERS]
    assert all(isinstance(block, AttentionBlock) for block in blocks)
    assert [block.attention.spatial for block in blocks] == [(16, 16), (8, 8), (8, 8), (8, 8)]
    with pag.perturbed_attention(model, pag.LAYERS):
        pass

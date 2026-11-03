"""Backward passes that must not depend on the order a GPU sums in, or a seed makes a different image every run."""

import pytest
import torch as th

from cgd.clip_models import load_clip, preprocess
from cgd.cutouts import adaptive_avg_pool

pytestmark = pytest.mark.cuda


def test_pooling_has_a_reproducible_gradient() -> None:
    image = th.rand(4, 3, 128, 128, device="cuda").requires_grad_()
    direction = th.randn(4, 3, 224, 224, device="cuda")
    gradients = {
        th.autograd.grad((adaptive_avg_pool(image, 224) * direction).sum(), image)[0].cpu().numpy().tobytes()
        for _ in range(10)
    }
    assert len(gradients) == 1


def test_an_open_clip_tower_has_a_reproducible_gradient() -> None:
    """The fused attention kernels do not. This is the model that shows it fastest."""
    model, size = load_clip("PE-Core-L-14-336", "cuda")
    images = th.rand(4, 3, size, size, device="cuda", generator=th.Generator("cuda").manual_seed(0))

    def gradient() -> th.Tensor:
        inputs = images.clone().requires_grad_()
        return th.autograd.grad(model.encode_image(preprocess(model, inputs)).square().sum() * 100, inputs)[0]

    first = gradient()
    assert all(th.equal(first, gradient()) for _ in range(3))

"""The guidance losses, and the helpers that take their gradients."""

import math

import pytest
import torch as th

from cgd.losses import finite_gradient, range_loss, spherical_dist_loss, tv_loss, with_gradient


class TestSphericalDistLoss:
    def test_it_only_sees_directions(self) -> None:
        x = th.randn(4, 8)
        assert th.allclose(spherical_dist_loss(x, 3 * x), th.zeros(4), atol=1e-6)

    @pytest.mark.parametrize(("y", "expected"), [([0.0, 1.0], math.pi**2 / 8), ([-1.0, 0.0], math.pi**2 / 2)])
    def test_it_is_twice_the_squared_half_angle(self, y: list[float], expected: float) -> None:
        assert spherical_dist_loss(th.tensor([1.0, 0.0]), th.tensor(y)).item() == pytest.approx(expected, rel=1e-6)

    def test_it_broadcasts_cutouts_against_targets(self) -> None:
        assert spherical_dist_loss(th.randn(4, 2, 1, 8), th.randn(3, 8)).shape == (4, 2, 3)


class TestRegularisers:
    def test_range_loss_is_zero_in_range_and_the_squared_overshoot_outside_it(self) -> None:
        assert th.equal(range_loss(th.rand(2, 3, 8, 8) * 2 - 1), th.zeros(2))
        assert th.equal(range_loss(th.full((1, 3, 8, 8), 3.0)), th.tensor([4.0]))

    def test_tv_loss_is_zero_for_a_flat_image_and_grows_with_detail(self) -> None:
        flat, stripes = th.full((1, 3, 8, 8), 0.5), th.zeros(1, 3, 8, 8)
        stripes[..., ::2] = 1
        assert tv_loss(flat).item() == 0
        assert tv_loss(stripes).item() > tv_loss(stripes * 0.5).item() > 0

    def test_tv_loss_is_per_image(self) -> None:
        assert tv_loss(th.rand(3, 3, 8, 8)).shape == (3,)


class TestGradients:
    def test_finite_gradient_zeroes_what_overflowed_and_keeps_the_rest(self) -> None:
        x = th.tensor([0.0, 4.0], requires_grad=True)
        assert th.equal(finite_gradient(x.sqrt().sum(), x), th.tensor([0.0, 0.25]))

    def test_finite_gradient_zeroes_nan(self) -> None:
        x = th.tensor([1.0, -1.0], requires_grad=True)
        assert th.equal(finite_gradient((x * math.nan).sum(), x), th.zeros(2))

    def test_with_gradient_has_the_value_and_gradient_it_was_given(self) -> None:
        x = th.randn(2, 3, requires_grad=True)
        images = x * 2
        gradient = th.randn(2, 3)
        loss = with_gradient(images, 7.5, gradient)
        assert loss.item() == 7.5
        assert th.allclose(th.autograd.grad(loss, x)[0], 2 * gradient)

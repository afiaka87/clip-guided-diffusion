"""Guidance losses, from Katherine Crowson's CLIP guided diffusion notebooks."""

import torch as th
from torch.nn import functional as tf


def spherical_dist_loss(x: th.Tensor, y: th.Tensor) -> th.Tensor:
    """Squared great circle distance between the directions of x and y, along the last dimension."""
    x = tf.normalize(x, dim=-1)
    y = tf.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def finite_gradient(loss: th.Tensor, x: th.Tensor) -> th.Tensor:
    """The gradient of `loss` with respect to x, with zero wherever it overflowed.

    Both pipelines backpropagate through a model at fp16, and a spiky CLIP gradient can
    overflow that. One inf would make the image NaN for good, which saves as black, so
    whatever overflowed gets no guidance instead. Scaling the loss down would prevent it,
    but any scale changes the rounding and so every image a seed makes.
    """
    return th.nan_to_num(th.autograd.grad(loss, x)[0], nan=0.0, posinf=0.0, neginf=0.0)


def with_gradient(images: th.Tensor, value: float, gradient: th.Tensor) -> th.Tensor:
    """A loss of `value` whose gradient with respect to `images` is `gradient`, both already computed.

    For a loss whose backward pass has to run on its own: one model at a time, or under
    different cuDNN flags. Backpropagating through this only reaches what made `images`.
    """
    surrogate = (images * gradient).sum()
    return surrogate - surrogate.detach() + value


def range_loss(images: th.Tensor) -> th.Tensor:
    """How far each image's pixels are outside [-1, 1], as a mean squared distance."""
    return (images - images.clamp(-1, 1)).pow(2).mean([1, 2, 3])


def tv_loss(images: th.Tensor) -> th.Tensor:
    """Each image's L2 total variation, as in Mahendran et al.: low for smooth images."""
    images = tf.pad(images, (0, 1, 0, 1), "replicate")
    x_diff = images[..., :-1, 1:] - images[..., :-1, :-1]
    y_diff = images[..., 1:, :-1] - images[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])

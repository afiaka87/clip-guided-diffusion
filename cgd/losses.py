import torch as th
from torch.nn import functional as tf


def range_loss(input):
    """(Katherine Crowson) - Spherical distance loss"""
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])


def spherical_dist_loss(x: th.Tensor, y: th.Tensor):
    """(Katherine Crowson) - Spherical distance loss"""
    x = tf.normalize(x, dim=-1)
    y = tf.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input: th.Tensor):
    """(Katherine Crowson) - L2 total variation loss, as in Mahendran et al."""
    input = tf.pad(input, (0, 1, 0, 1), "replicate")
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff ** 2 + y_diff ** 2).mean([1, 2, 3])

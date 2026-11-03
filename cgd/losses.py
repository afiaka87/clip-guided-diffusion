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


def contrastive_loss(x: th.Tensor, y: th.Tensor, negatives: th.Tensor, temperature: float = 100.0, exclude: int = 10):
    """Cross entropy of each prompt in `y` against the text of everything else, `negatives`.

    The spherical loss pulls towards the prompt from anywhere. This one only cares about
    what separates the prompt from other things, which is CLIP's training objective.
    The negatives closest to each prompt are left out: they are likely the same thing
    under another name. `x` is [cutouts, batch, 1, features], and `y` is [prompts, features].
    """
    x = tf.normalize(x, dim=-1).squeeze(2)
    y = tf.normalize(y, dim=-1)
    negatives = tf.normalize(negatives, dim=-1)
    out = []
    for prompt in y:
        keep = (negatives @ prompt).argsort()[:-exclude]
        candidates = th.cat([prompt[None], negatives[keep]])
        out.append(-tf.log_softmax(temperature * x @ candidates.T, dim=-1)[..., 0])
    return th.stack(out, dim=2)


def tv_loss(input: th.Tensor):
    """(Katherine Crowson) - L2 total variation loss, as in Mahendran et al."""
    input = tf.pad(input, (0, 1, 0, 1), "replicate")
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff ** 2 + y_diff ** 2).mean([1, 2, 3])

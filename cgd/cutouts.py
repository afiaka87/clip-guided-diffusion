"""Random square crops of an image, at CLIP's input size, for CLIP to judge the image by."""

from typing import NamedTuple

import torch as th


class Box(NamedTuple):
    """A square crop."""

    left: int
    top: int
    size: int


def adaptive_avg_pool(images: th.Tensor, size: int) -> th.Tensor:
    """Square adaptive average pooling with a reproducible gradient.

    Same result as tf.adaptive_avg_pool2d, but written as a weight matrix. That
    function's CUDA backward pass sums into shared pixels with atomic adds in no
    fixed order, so the same inputs give gradients that differ by rounding error,
    and the guided sampler amplifies that into a visibly different image.
    """
    side = images.shape[-1]
    positions = th.arange(side, device=images.device)
    bins = th.arange(size, device=images.device)
    starts = bins * side // size
    ends = -(-(bins + 1) * side // size)  # Ceiling division.
    in_bin = (positions >= starts[:, None]) & (positions < ends[:, None])
    weights = in_bin.to(images.dtype) / (ends - starts)[:, None]
    return weights @ images @ weights.T


def size_range(height: int, width: int, cut_size: int, min_fraction: float) -> tuple[int, int]:
    """The smallest and largest side of a cutout.

    The largest is the whole shorter side. The smallest is `min_fraction` of it, or CLIP's
    input size if that is smaller, since a cutout that small still loses no detail.
    """
    largest = min(height, width)
    return min(int(largest * min_fraction), cut_size), largest


def random_boxes(
    height: int, width: int, count: int, sizes: tuple[int, int], generator: th.Generator | None = None
) -> list[Box]:
    """`count` random crops of an image, each a side between `sizes` and anywhere inside it."""
    return [_random_box(height, width, sizes, generator) for _ in range(count)]


def _random_box(height: int, width: int, sizes: tuple[int, int], generator: th.Generator | None) -> Box:
    smallest, largest = sizes
    size = int(th.rand([], generator=generator) * (largest - smallest) + smallest)
    left = int(th.randint(0, width - size + 1, (), generator=generator))
    top = int(th.randint(0, height - size + 1, (), generator=generator))
    return Box(left, top, size)


def make_cutouts(
    images: th.Tensor,
    cut_size: int,
    count: int,
    min_fraction: float = 1.0,
    generator: th.Generator | None = None,
) -> th.Tensor:
    """`count` random crops of each image, pooled to `cut_size`, as [count * batch, channels, cut_size, cut_size].

    The crops are in order of crop, then image. A square image whose only possible crop is
    the whole frame gets one, as the rest would be copies of it.
    """
    height, width = images.shape[2:4]
    sizes = size_range(height, width, cut_size, min_fraction)
    if height == width and sizes[0] == sizes[1]:
        count = 1
    boxes = random_boxes(height, width, count, sizes, generator)
    return th.cat(
        [
            adaptive_avg_pool(images[:, :, box.top : box.top + box.size, box.left : box.left + box.size], cut_size)
            for box in boxes
        ]
    )

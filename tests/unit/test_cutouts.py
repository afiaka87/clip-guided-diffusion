"""Random cutouts, and the pooling that shrinks them to CLIP's input size."""

import pytest
import torch as th
from torch.nn import functional as tf

from cgd.cutouts import Box, adaptive_avg_pool, make_cutouts, random_boxes, size_range


class TestSizeRange:
    def test_a_fraction_of_the_shorter_side_when_that_is_below_clips_size(self) -> None:
        assert size_range(128, 192, 224, 0.7) == (89, 128)

    def test_never_smaller_than_clip_needs(self) -> None:
        assert size_range(512, 512, 224, 0.7) == (224, 512)

    def test_the_whole_frame_without_a_fraction(self) -> None:
        assert size_range(128, 128, 224, 1.0) == (128, 128)


class TestRandomBoxes:
    @pytest.mark.parametrize(("height", "width"), [(128, 128), (100, 300), (300, 100)])
    def test_every_box_is_inside_the_image_and_in_range(self, height: int, width: int) -> None:
        sizes = size_range(height, width, 224, 0.7)
        for box in random_boxes(height, width, 64, sizes, th.Generator().manual_seed(0)):
            assert sizes[0] <= box.size <= sizes[1]
            assert 0 <= box.left <= width - box.size
            assert 0 <= box.top <= height - box.size

    def test_sizes_vary(self) -> None:
        boxes = random_boxes(128, 128, 64, (89, 128), th.Generator().manual_seed(0))
        assert len({box.size for box in boxes}) > 1

    def test_a_generator_makes_them_reproducible(self) -> None:
        def boxes() -> list[Box]:
            return random_boxes(128, 192, 8, (89, 128), th.Generator().manual_seed(3))

        assert boxes() == boxes()


class TestMakeCutouts:
    def test_a_frame_with_one_possible_crop_gets_one_cutout(self) -> None:
        assert make_cutouts(th.rand(1, 3, 128, 128), 224, 16).shape == (1, 3, 224, 224)

    @pytest.mark.parametrize(("height", "width", "min_fraction"), [(256, 256, 1.0), (128, 192, 1.0), (128, 128, 0.7)])
    def test_every_cutout_is_kept_when_they_can_differ(self, height: int, width: int, min_fraction: float) -> None:
        assert make_cutouts(th.rand(2, 3, height, width), 224, 16, min_fraction).shape == (32, 3, 224, 224)

    def test_cutouts_are_ordered_by_crop_then_image(self) -> None:
        images = th.cat([th.full((1, 3, 96, 96), 0.2), th.full((1, 3, 96, 96), 0.8)])
        means = make_cutouts(images, 64, 3, 0.7).mean(dim=(1, 2, 3))
        assert th.allclose(means, th.tensor([0.2, 0.8, 0.2, 0.8, 0.2, 0.8]), atol=1e-6)

    def test_cutouts_of_a_wide_image_reach_across_it(self) -> None:
        left_black_right_white = th.zeros(1, 3, 64, 128)
        left_black_right_white[..., 64:] = 1
        cutouts = make_cutouts(left_black_right_white, 64, 32, generator=th.Generator().manual_seed(0))
        assert cutouts.shape == (32, 3, 64, 64)
        means = cutouts.mean(dim=(1, 2, 3))
        assert means.min().item() < 0.5 < means.max().item()


class TestAdaptiveAvgPool:
    @pytest.mark.parametrize("side", [44, 64, 89, 128, 224, 256, 301, 512])
    def test_it_matches_pytorchs(self, side: int) -> None:
        image = th.rand(2, 3, side, side)
        assert (adaptive_avg_pool(image, 224) - tf.adaptive_avg_pool2d(image, 224)).abs().max().item() < 1e-5

    def test_it_keeps_the_mean(self) -> None:
        image = th.rand(1, 3, 300, 300)
        assert adaptive_avg_pool(image, 100).mean().item() == pytest.approx(image.mean().item(), rel=1e-5)

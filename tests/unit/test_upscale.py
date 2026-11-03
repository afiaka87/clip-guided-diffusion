"""Tiling a 4x upscale, with a small network standing in for Real-ESRGAN."""

import pytest
import torch as th
from torch import nn

from cgd.upscale import SCALE, TILE_MARGIN, TILE_SIZE, tiles, upscale


@pytest.mark.parametrize(("height", "width"), [(256, 256), (300, 520), (40, 100), (513, 257)])
def test_tiles_cover_the_output_exactly_once(height: int, width: int) -> None:
    covered = th.zeros(height * SCALE, width * SCALE, dtype=th.int32)
    for tile in tiles(height, width):
        covered[tile.target] += 1
    assert th.equal(covered, th.ones_like(covered))


@pytest.mark.parametrize(("height", "width"), [(300, 520), (40, 100)])
def test_each_tile_sees_its_margin_and_keeps_only_its_own_part(height: int, width: int) -> None:
    for tile in tiles(height, width):
        rows, columns = tile.source
        assert rows.stop - rows.start <= TILE_SIZE + 2 * TILE_MARGIN
        assert columns.stop - columns.start <= TILE_SIZE + 2 * TILE_MARGIN
        for keep, target in zip(tile.keep, tile.target, strict=True):
            assert keep.stop - keep.start == target.stop - target.start


def test_a_tiled_upscale_matches_upscaling_the_whole_image() -> None:
    """Exactly, for a network that sees less context than the margin gives it."""
    th.manual_seed(0)
    network = nn.Sequential(nn.Conv2d(3, 3, 3, padding=1), nn.Upsample(scale_factor=SCALE)).requires_grad_(False)
    images = th.rand(1, 3, 300, 520) * 2 - 1
    whole = network(images.add(1).div(2)).mul(2).sub(1).clamp(-1, 1)
    tiled = upscale(network, images)
    assert tiled.shape == (1, 3, 300 * SCALE, 520 * SCALE)
    assert th.allclose(tiled, whole, atol=1e-5)

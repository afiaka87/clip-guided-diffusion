"""Sliding between two images: what a position means, and where the frames of a transition go."""

import math
from collections.abc import Callable
from itertools import pairwise

import pytest
import torch as th

from cgd import slide

Curve = Callable[[float], float]


def drawn(curve: Curve, asked: list[float] | None = None) -> slide.Draw:
    """A transition whose image at a position is the grey level `curve` gives it. `asked` records the positions."""

    def draw(position: float) -> th.Tensor:
        if asked is not None:
            asked.append(position)
        return th.full((1, 3, 4, 4), curve(position))

    return draw


def placed(curve: Curve, count: int, asked: list[float] | None = None) -> list[float]:
    """The grey level of the start of a transition along `curve`, then of each of its `count` frames."""
    draw = drawn(curve)
    frames = slide.place(drawn(curve, asked), (draw(0.0), draw(1.0)), count)
    return [curve(0.0), *(frame.mean().item() for frame in frames)]


def largest_step(levels: list[float]) -> float:
    """The largest change of grey level from one frame to the next."""
    return max(abs(after - before) for before, after in pairwise(levels))


def steep(position: float) -> float:
    """Nearly all of the change within a tenth of the way, around 0.7."""
    return 1 / (1 + math.exp(-60 * (position - 0.7)))


class TestPosition:
    def test_the_fade_runs_from_the_first_image_to_the_second(self) -> None:
        assert (slide.fade(0.0), slide.fade(0.5), slide.fade(1.0)) == (0.0, 0.5, 1.0)

    @pytest.mark.parametrize("position", [0.05, 0.2, 0.4])
    def test_the_fade_is_the_same_from_either_end(self, position: float) -> None:
        assert slide.fade(position) == pytest.approx(1 - slide.fade(1 - position))

    @pytest.mark.parametrize("position", [0.05, 0.2, 0.4])
    def test_less_of_the_other_image_shows_near_an_end_than_in_a_plain_fade(self, position: float) -> None:
        assert slide.fade(position) < position
        assert slide.fade(1 - position) > 1 - position

    def test_there_is_no_noise_at_either_image_and_the_most_halfway(self) -> None:
        assert slide.noise_timestep(0.0) == 0
        assert slide.noise_timestep(1.0) == pytest.approx(0, abs=1e-9)
        assert slide.noise_timestep(0.5) == slide.PEAK_TIMESTEP
        assert slide.noise_timestep(0.2) == pytest.approx(slide.noise_timestep(0.8))
        assert 0 < slide.noise_timestep(0.2) < slide.PEAK_TIMESTEP


class TestSlerp:
    def test_it_runs_from_the_first_point_to_the_second_and_keeps_their_norm(self) -> None:
        first, second = th.randn(2, 1, 4, 8, 8, generator=th.Generator().manual_seed(0))
        second = second * first.norm() / second.norm()
        assert th.allclose(slide.slerp(first, second, 0.0), first, atol=1e-6)
        assert th.allclose(slide.slerp(first, second, 1.0), second, atol=1e-6)
        for share in (0.25, 0.5, 0.75):
            assert slide.slerp(first, second, share).norm().item() == pytest.approx(first.norm().item(), rel=1e-5)

    def test_a_straight_line_between_the_same_points_is_shorter(self) -> None:
        first, second = th.randn(2, 1, 4, 8, 8, generator=th.Generator().manual_seed(0))
        assert th.lerp(first, second, 0.5).norm() < 0.8 * slide.slerp(first, second, 0.5).norm()

    def test_each_image_of_a_batch_moves_along_its_own_arc(self) -> None:
        first, second = th.randn(2, 2, 4, 8, 8, generator=th.Generator().manual_seed(0))
        together = slide.slerp(first, second, 0.3)
        for index in range(2):
            alone = slide.slerp(first[index : index + 1], second[index : index + 1], 0.3)
            assert th.allclose(together[index : index + 1], alone, atol=1e-6)

    def test_between_a_point_and_itself_is_that_point(self) -> None:
        point = th.randn(1, 4, 8, 8)
        assert th.allclose(slide.slerp(point, point.clone(), 0.5), point, atol=1e-6)


class TestPlace:
    def test_the_frames_come_in_order_and_end_on_the_second_image(self) -> None:
        end = th.ones(1, 3, 4, 4)
        frames = slide.place(drawn(float), (th.zeros(1, 3, 4, 4), end), 12)
        levels = [frame.mean().item() for frame in frames]
        assert len(frames) == 12
        assert levels == sorted(levels)
        assert levels[0] > 0
        assert frames[-1] is end

    def test_one_frame_is_the_second_image_and_nothing_is_drawn(self) -> None:
        asked: list[float] = []
        assert placed(float, 1, asked) == [0.0, 1.0]
        assert not asked

    def test_the_frames_go_where_the_image_changes(self) -> None:
        even = [steep(frame / 16) for frame in range(17)]
        assert largest_step(even) > 0.5  # Evenly spaced, two neighbours hold over half of the whole change.
        assert largest_step(placed(steep, 16)) < 0.2

    def test_a_transition_starts_and_ends_more_slowly_than_it_moves_halfway(self) -> None:
        levels = placed(float, 16)
        steps = [after - before for before, after in pairwise(levels)]
        assert steps[0] < max(steps)
        assert steps[-1] < max(steps)
        assert max(steps) == pytest.approx(steps[len(steps) // 2], rel=1e-6)

    def test_two_images_alike_get_evenly_spaced_frames(self) -> None:
        asked: list[float] = []
        placed(lambda _position: 0.3, 4, asked)
        assert sorted(asked) == [0.25, 0.5, 0.75]

    def test_a_jump_does_not_swallow_the_frames(self) -> None:
        """Halving the gap around a jump never makes it smaller, so past a point the frames go elsewhere."""
        asked: list[float] = []
        placed(lambda position: float(position >= 0.3), 12, asked)
        positions = sorted([0.0, *asked, 1.0])
        closest = min(after - before for before, after in pairwise(positions))
        assert closest > 1 / (2 * slide.FINEST * 12) - 1e-9
        assert sum(abs(position - 0.3) > 0.1 for position in asked) >= 4

    def test_the_image_of_a_batch_that_changes_most_decides(self) -> None:
        alone: list[float] = []
        placed(steep, 10, alone)

        def both(position: float) -> th.Tensor:
            together.append(position)
            return th.cat([th.full((1, 3, 4, 4), 0.3), th.full((1, 3, 4, 4), steep(position))])

        together: list[float] = []
        ends = both(0.0), both(1.0)
        together.clear()
        slide.place(both, ends, 10)
        assert together == alone

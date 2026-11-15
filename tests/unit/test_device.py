"""The random streams a run draws from: keeping a model's construction out of them, and drawing numbers again."""

import torch as th
from torch import nn

from cgd.device import forked_rng, random_state, replayed_rng


def test_a_model_built_in_the_forked_stream_leaves_the_runs_random_numbers_alone() -> None:
    th.manual_seed(0)
    expected = th.randn(4)
    th.manual_seed(0)
    with forked_rng("cpu"):
        nn.Linear(8, 8)
    assert th.equal(th.randn(4), expected)


def test_a_replayed_stream_draws_the_numbers_that_followed_its_state() -> None:
    th.manual_seed(0)
    state = random_state("cpu")
    first = th.randn(4)
    with replayed_rng("cpu", state):
        assert th.equal(th.randn(4), first)
    with replayed_rng("cpu", state):
        assert th.equal(th.randn(4), first)


def test_after_a_replay_the_stream_goes_on_from_where_it_was() -> None:
    th.manual_seed(0)
    th.randn(4)
    expected = th.randn(4)
    th.manual_seed(0)
    state = random_state("cpu")
    th.randn(4)
    with replayed_rng("cpu", state):
        th.randn(100)
    assert th.equal(th.randn(4), expected)

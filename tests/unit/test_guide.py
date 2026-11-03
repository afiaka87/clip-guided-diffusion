"""How a guide turns CLIP embeddings into a loss, on small linear stand-ins for CLIP.

The stand-ins are real modules with CLIP's interface, a random projection of the pixels,
so the weighting, the ensemble's separate backward passes and the scores can be checked
exactly. tests/integration checks the same against the real models.
"""

import pytest
import torch as th
from torch import nn

from cgd.clip_models import INIT_FUNCTIONS, OPENAI_MEAN, OPENAI_STD, preprocess, skip_weight_init
from cgd.guide import Guide, Tower, guide_loss, query, score, tower_distance

SIZE = 16
FEATURES = 8


class LinearClip(nn.Module):  # pylint: disable=abstract-method  # Guidance never calls CLIP's forward().
    """CLIP's image interface over a random projection: encode_image and a visual tower with a resolution."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        with th.random.fork_rng():
            th.manual_seed(seed)
            self.visual = nn.Linear(3 * SIZE * SIZE, FEATURES)
        self.visual.input_resolution = SIZE

    def encode_image(self, images: th.Tensor) -> th.Tensor:
        """One feature vector per image, as CLIP gives."""
        return self.visual(images.flatten(1))


def tower(seed: int, targets: th.Tensor, weights: list[float]) -> Tower:
    """A tower on a stand-in seeded with `seed`, aiming at `targets` with `weights`."""
    return Tower(LinearClip(seed), SIZE, targets, th.tensor(weights))


def two_tower_guide() -> Guide:
    """An ensemble of two stand-ins, aiming at the same two targets."""
    targets = th.randn(2, FEATURES)
    return Guide((tower(1, targets, [0.75, 0.25]), tower(2, targets, [0.75, 0.25])), 4, 0.7)


def test_query_is_the_weighted_sum_of_the_first_models_target_directions() -> None:
    guide = Guide((tower(0, th.tensor([[2.0, 0.0], [0.0, 3.0]]), [0.75, 0.25]),), 4, 0.7)
    assert th.allclose(query(guide), th.tensor([[0.75, 0.25]]))


def test_a_single_model_loss_is_its_cutout_distance_summed_over_the_batch() -> None:
    guide = Guide((tower(1, th.randn(2, FEATURES), [0.5, 0.5]),), 4, 0.7)
    images = th.rand(3, 3, 24, 24) * 2 - 1
    th.manual_seed(0)
    loss = guide_loss(guide, images, 10.0)
    th.manual_seed(0)
    assert loss.item() == pytest.approx(tower_distance(guide, guide.towers[0], images.add(1).div(2)).sum().item() * 10)


def test_an_ensemble_backpropagates_each_model_alone_to_the_same_loss_and_gradient() -> None:
    guide = two_tower_guide()
    images = (th.rand(2, 3, 24, 24) * 2 - 1).requires_grad_()
    th.manual_seed(0)
    separate = guide_loss(guide, images, 10.0)
    separate_gradient = th.autograd.grad(separate, images)[0]
    th.manual_seed(0)
    pixels = images.add(1).div(2)
    joint = sum(tower_distance(guide, member, pixels).sum() * 10 / 2 for member in guide.towers)
    assert separate.item() == pytest.approx(joint.item(), rel=1e-6)
    assert th.allclose(separate_gradient, th.autograd.grad(joint, images)[0], atol=1e-6)


def test_score_prefers_the_image_the_prompt_was_taken_from() -> None:
    model = LinearClip(1)
    images = th.rand(2, 3, SIZE, SIZE) * 2 - 1
    with th.no_grad():
        target = model.encode_image(preprocess(model, images[:1].add(1).div(2)))
    guide = Guide((Tower(model, SIZE, target, th.ones(1)),), 4, 0.7)
    scores = score(guide, images)
    assert scores[0].item() == pytest.approx(1.0, abs=1e-5)
    assert scores[0] > scores[1]


def test_preprocess_uses_each_models_own_normalisation() -> None:
    white = th.ones(1, 3, 2, 2)
    openai = LinearClip(0)
    expected = (white - th.tensor(OPENAI_MEAN).view(3, 1, 1)) / th.tensor(OPENAI_STD).view(3, 1, 1)
    assert th.allclose(preprocess(openai, white), expected)
    own = LinearClip(0)
    own.visual.image_mean, own.visual.image_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    assert th.equal(preprocess(own, white), white)


def test_skip_weight_init_leaves_tensors_as_they_are() -> None:
    tensor = th.zeros(4)
    with skip_weight_init():
        assert nn.init.normal_(tensor) is tensor
    assert th.equal(tensor, th.zeros(4))


def test_skip_weight_init_restores_the_init_functions_even_after_an_error() -> None:
    originals = {name: getattr(nn.init, name) for name in INIT_FUNCTIONS}

    def fail_while_skipping() -> None:
        with skip_weight_init():
            raise RuntimeError

    with pytest.raises(RuntimeError):
        fail_while_skipping()
    assert {name: getattr(nn.init, name) for name in INIT_FUNCTIONS} == originals

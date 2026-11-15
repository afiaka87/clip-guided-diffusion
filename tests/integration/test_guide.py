"""Guidance's loss on the real models: the aesthetic term, and the guides of a sequence."""

from pathlib import Path

import pytest
import torch as th
import torchvision.transforms.functional as tvf
from PIL import Image
from torch.nn import functional as tf

from cgd.clip_models import preprocess
from cgd.guide import Guide, aesthetic_rating, cutout_rating, guide_loss, load_guide, load_guides
from cgd.prompts import Scene

pytestmark = pytest.mark.cuda

PROMPT = "a photograph of a red fox in the snow"
SCALE = 0.5


@pytest.fixture(name="rated", scope="module")
def fixture_rated() -> Guide:
    """A guide with the aesthetic term, loaded once. Its only cutout of a 224px image is the whole image."""
    return load_guide(
        "ViT-B/32",
        [PROMPT],
        image_size=224,
        num_cutouts=1,
        min_cut_fraction=1.0,
        device="cuda",
        aesthetic_scale=SCALE,
    )


class TestAesthetic:
    def test_the_rating_is_the_published_linear_head_on_clip(self, rated: Guide) -> None:
        aesthetic = rated.aesthetic
        assert aesthetic is not None
        images = th.rand(2, 3, 224, 224, device="cuda")
        embeds = tf.normalize(aesthetic.model.encode_image(preprocess(aesthetic.model, images)).float(), dim=-1)
        expected = embeds @ aesthetic.weight.T * 512**0.5 + aesthetic.bias
        rating = aesthetic_rating(aesthetic, images)
        assert rating.shape == (2,)
        assert (rating - expected[:, 0]).abs().max().item() < 1e-2

    def test_it_subtracts_the_scaled_rating_from_the_loss(self, rated: Guide) -> None:
        assert rated.aesthetic is not None
        images = th.rand(1, 3, 224, 224, device="cuda").mul(2).sub(1).requires_grad_()
        plain = guide_loss(rated._replace(aesthetic=None), images)
        combined = guide_loss(rated, images)
        rating = aesthetic_rating(rated.aesthetic, images.add(1).div(2)).sum()
        assert combined.item() == pytest.approx(plain.item() - SCALE * rating.item(), abs=1e-2)
        assert th.autograd.grad(combined, images)[0].abs().sum().item() > 0

    def test_a_photograph_rates_above_a_gray_square_whatever_else_is_in_the_batch(
        self, rated: Guide, goose: Path
    ) -> None:
        aesthetic = rated.aesthetic
        assert aesthetic is not None
        photo = tvf.to_tensor(Image.open(goose).convert("RGB").resize((224, 224)))
        images = th.stack([photo, th.full_like(photo, 0.5)]).cuda()
        whole = aesthetic_rating(aesthetic, images)  # At 224 and this fraction, every cutout is the whole image.
        assert whole[0].item() > whole[1].item() + 1  # 4.5 against 2.7 when measured.
        assert (cutout_rating(aesthetic, images, 3, 1.0) - whole).abs().max().item() < 1e-3


def test_the_guides_of_a_sequence_share_their_models_and_hold_what_each_image_would_alone(photon: Path) -> None:
    images = [Scene((PROMPT,), ()), Scene(("a bowl of ramen", "steam:0.5"), (str(photon),))]
    settings = {"image_size": 224, "num_cutouts": 1, "device": "cuda"}
    first, second = load_guides("RN50+ViT-B/32", images, **settings)
    assert len(first.towers) == len(second.towers) == 2
    assert all(a.model is b.model for a, b in zip(first.towers, second.towers, strict=True))
    alone = load_guide("RN50+ViT-B/32", [PROMPT], **settings)
    for tower, expected in zip(first.towers, alone.towers, strict=True):
        assert th.equal(tower.targets, expected.targets)
        assert th.equal(tower.weights, expected.weights)
    assert [len(tower.weights) for tower in second.towers] == [3, 3]


def test_the_image_prompts_of_a_later_image_do_not_move_the_noise_the_seed_starts_from(photon: Path) -> None:
    """Encoding an image prompt draws random cutouts, which only the first image's may take from the run's stream."""
    settings = {"image_size": 224, "num_cutouts": 4, "device": "cuda"}

    def noise_after(images: list[Scene]) -> th.Tensor:
        th.manual_seed(3)
        load_guides("ViT-B/32", images, **settings)
        return th.randn(4, device="cuda")

    alone = [Scene((PROMPT,), ())]
    assert th.equal(noise_after([*alone, Scene((), (str(photon),))]), noise_after(alone))

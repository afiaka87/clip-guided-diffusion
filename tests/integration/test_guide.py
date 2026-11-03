"""Guidance's loss on the real models: ensembles, the aesthetic term and scoring."""

from pathlib import Path

import pytest
import torch as th
import torchvision.transforms.functional as tvf
from PIL import Image
from torch.nn import functional as tf

from cgd.clip_models import preprocess
from cgd.cutouts import make_cutouts
from cgd.guide import Guide, aesthetic_rating, cutout_rating, guide_loss, load_guide

pytestmark = pytest.mark.cuda

PROMPT = "a photograph of a red fox in the snow"


def guide(model_name: str = "ViT-B/32", aesthetic_scale: float = 0.0) -> Guide:
    """A guide whose only cutout of a 224px image is the whole image, so nothing about it is random."""
    return load_guide(
        model_name,
        [PROMPT],
        image_size=224,
        num_cutouts=1,
        min_cut_fraction=1.0,
        device="cuda",
        aesthetic_scale=aesthetic_scale,
    )


def test_an_ensembles_gradient_is_the_mean_of_its_members() -> None:
    members = ("ViT-B/32", "PE-Core-B-16")
    images = th.randn(1, 3, 224, 224, device="cuda").mul(0.5).clamp(-1, 1).requires_grad_()

    def gradient(model_name: str) -> th.Tensor:
        return th.autograd.grad(guide_loss(guide(model_name), images), images)[0]

    expected = (gradient(members[0]) + gradient(members[1])) / 2
    ensemble = gradient("+".join(members))
    # Not exact: halving the loss changes what underflows on the way back through a 16 bit tower.
    assert tf.cosine_similarity(ensemble.flatten(), expected.flatten(), dim=0).item() > 0.999


class TestAesthetic:
    def test_the_rating_is_the_published_linear_head_on_clip(self) -> None:
        aesthetic = guide(aesthetic_scale=1.0).aesthetic
        assert aesthetic is not None
        images = th.rand(2, 3, 224, 224, device="cuda")
        embeds = tf.normalize(aesthetic.model.encode_image(preprocess(aesthetic.model, images)).float(), dim=-1)
        expected = embeds @ aesthetic.weight.T * 512**0.5 + aesthetic.bias
        rating = aesthetic_rating(aesthetic, images)
        assert rating.shape == (2,)
        assert (rating - expected[:, 0]).abs().max().item() < 1e-2

    def test_following_its_gradient_raises_the_rating(self) -> None:
        aesthetic = guide(aesthetic_scale=1.0).aesthetic
        assert aesthetic is not None
        images = th.rand(1, 3, 224, 224, device="cuda").mul(2).sub(1)
        before = aesthetic_rating(aesthetic, images.add(1).div(2)).item()
        for _ in range(10):
            images = images.requires_grad_()
            gradient = th.autograd.grad(-aesthetic_rating(aesthetic, images.add(1).div(2)).sum(), images)[0]
            images = (images - 0.02 * gradient.sign()).clamp(-1, 1).detach()
        assert aesthetic_rating(aesthetic, images.add(1).div(2)).item() > before + 0.5

    @pytest.mark.parametrize("model_name", ["ViT-B/32", "ViT-B/32+PE-Core-B-16"])
    def test_it_subtracts_the_scaled_rating_from_the_loss(self, model_name: str) -> None:
        images = th.rand(1, 3, 224, 224, device="cuda").mul(2).sub(1).requires_grad_()
        plain = guide_loss(guide(model_name), images)
        rated = guide(model_name, aesthetic_scale=0.5)
        combined = guide_loss(rated, images)
        assert rated.aesthetic is not None
        rating = aesthetic_rating(rated.aesthetic, images.add(1).div(2)).sum()
        assert combined.item() == pytest.approx(plain.item() - 0.5 * rating.item(), abs=1e-2)
        assert th.autograd.grad(combined, images)[0].abs().sum().item() > 0

    def test_the_cutout_rating_keeps_the_images_of_a_batch_apart(self, goose: Path) -> None:
        aesthetic = guide(aesthetic_scale=1.0).aesthetic
        assert aesthetic is not None
        photo = tvf.to_tensor(Image.open(goose).convert("RGB").resize((224, 224)))
        images = th.stack([photo, th.full_like(photo, 0.5)]).cuda()
        whole = aesthetic_rating(aesthetic, images)  # At 224 and this fraction, every cutout is the whole image.
        assert whole[0].item() > whole[1].item() + 1  # 4.5 against 2.7 when measured.
        assert (cutout_rating(aesthetic, images, 3, 1.0) - whole).abs().max().item() < 1e-3

    def test_the_cutout_rating_averages_crops_of_a_larger_image(self) -> None:
        aesthetic = guide(aesthetic_scale=1.0).aesthetic
        assert aesthetic is not None
        images = th.rand(2, 3, 512, 512, device="cuda")
        th.manual_seed(0)
        crops = make_cutouts(images, 224, 4, 0.7)
        th.manual_seed(0)
        rating = cutout_rating(aesthetic, images, 4, 0.7)
        assert rating.shape == (2,)
        assert (rating - aesthetic_rating(aesthetic, crops).view(4, 2).mean(0)).abs().max().item() < 1e-3

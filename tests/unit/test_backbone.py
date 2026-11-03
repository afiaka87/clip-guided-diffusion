"""The parts of --backbone that need no model: the noise optimiser, the prior and the conditioning."""

import torch as th
from torch import nn

from cgd import backbone
from cgd.backbone import BACKBONES, Backbone, optimise_noise


def test_the_noise_penalty_is_lowest_on_the_prior() -> None:
    z = th.randn(1, 4, 64, 64)
    z = z / z.norm() * (z.numel() - 1) ** 0.5  # The mode of the chi distribution of its norm.
    assert backbone.noise_penalty(z) < backbone.noise_penalty(z * 1.1)
    assert backbone.noise_penalty(z) < backbone.noise_penalty(z * 0.9)


def test_the_model_is_only_told_what_to_draw() -> None:
    assert backbone.conditioning_text(["a fox:2", "a hat", "blurry:-1"]) == "a fox, a hat"


def test_latents_are_an_eighth_of_the_image_along_each_side() -> None:
    assert backbone.latent_shape(256, 320) == (1, 4, 32, 40)


def test_an_epsilon_prediction_recovers_the_clean_latents() -> None:
    x0, eps, alpha_bar = th.randn(1, 4, 8, 8), th.randn(1, 4, 8, 8), th.tensor(0.3)
    z = alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * eps
    assert th.allclose(backbone.predicted_x0(z, eps, "epsilon", alpha_bar), x0, atol=1e-5)
    assert backbone.predicted_x0(z, x0, "sample", alpha_bar) is x0


def loaded(name: str, pooled: th.Tensor | None) -> Backbone:
    """A backbone record with a prompt encoded but no networks, for what only reads the prompt."""
    unused = nn.Identity()
    return Backbone(
        BACKBONES[name], unused, unused, unused, th.ones(1, 77, 8), pooled, th.ones(1000), th.float32, "cpu"
    )


def test_stable_diffusion_is_conditioned_on_the_prompt_alone() -> None:
    model = loaded("sd-turbo", None)
    assert backbone.unet_conditioning(model, 256, 320) == {"encoder_hidden_states": model.text}


def test_sdxl_is_also_told_the_size_and_that_the_image_is_uncropped() -> None:
    model = loaded("vega", th.ones(1, 4))
    conditioning = backbone.unet_conditioning(model, 256, 320)
    assert conditioning["added_cond_kwargs"]["text_embeds"] is model.pooled
    assert conditioning["added_cond_kwargs"]["time_ids"].tolist() == [[256, 320, 0, 0, 256, 320]]


class TestOptimiseNoise:
    TARGET = th.randn(1, 4, 8, 8, generator=th.Generator().manual_seed(1))

    @classmethod
    def loss(cls, images: th.Tensor) -> th.Tensor:
        """Lowest where the images are TARGET drawn through tanh."""
        return (images - cls.TARGET.tanh()).square().mean()

    def test_it_lowers_the_loss_and_keeps_the_noise_on_the_prior(self) -> None:
        z = th.randn(1, 4, 8, 8, generator=th.Generator().manual_seed(0))
        steps = list(optimise_noise(th.tanh, self.loss, z, iterations=20, progress=False))
        assert len(steps) == 20
        assert min(step.loss for step in steps[1:]) < steps[0].loss
        assert abs(steps[-1].best_z.norm().item() - z.norm().item()) < 0.1 * z.norm().item()

    def test_it_keeps_the_noise_that_scored_best(self) -> None:
        z = th.randn(1, 4, 8, 8, generator=th.Generator().manual_seed(0))
        steps = list(optimise_noise(lambda z: z * 1.0, self.loss, z, iterations=10, progress=False))
        best = min(steps, key=lambda step: step.loss)
        assert th.equal(steps[-1].best_z, best.images)

    def test_the_first_step_scores_the_starting_noise_and_zero_steps_score_nothing(self) -> None:
        z = th.randn(1, 4, 8, 8)
        first = next(optimise_noise(lambda z: z * 1.0, self.loss, z, iterations=3, progress=False))
        assert th.equal(first.images, z)
        assert th.equal(first.best_z, z)
        assert not list(optimise_noise(th.tanh, self.loss, z, iterations=0, progress=False))

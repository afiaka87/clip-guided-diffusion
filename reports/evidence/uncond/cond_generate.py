"""Sample the CLIP-conditioned 256 U-Net, with classifier-free guidance and CLIP guidance, like adm.generate().

The prompt's CLIP text embedding is the condition. Guidance backpropagates through the
conditional branch only, so a guided call holds one U-Net graph, as the unconditional
pipeline does: the classifier-free part, (scale - 1) * (conditional - unconditional), moves
x0 but is held constant for the gradient.
"""

from collections.abc import Iterator, Sequence
from functools import lru_cache, partial
from pathlib import Path

import torch as th
import torch.nn.functional as tf

from cgd import adm, sampler
from cgd.clip_models import encode_texts
from cgd.guide import NUM_CUTOUTS, load_guide
from cgd.outputs import run_name, save_frames
from conditioning import ClipConditionedUNet


@lru_cache(maxsize=1)
def load_conditioned(checkpoint: str, device: str, use_ema: bool = True) -> tuple[ClipConditionedUNet, object, str]:
    """The fp16 unconditional U-Net with a checkpoint's trained weights, its schedule, and its CLIP model."""
    state = th.load(checkpoint, map_location="cpu", weights_only=False)
    # Uncached: the trained weights replace some of the U-Net's own, which must not leak into adm.load_model()'s copy.
    unet, log_alpha_bar = adm.load_model.__wrapped__(256, device, fp16=True, use_checkpoint=False, unconditional=True)
    weights = state["ema" if use_ema else "trainable"]
    clip_model = state["args"]["clip_model"]
    dim = weights["project.0.weight"].shape[1]
    model = ClipConditionedUNet(unet, dim).to(device)
    with th.no_grad():
        parameters = dict(model.named_parameters())
        for name, value in weights.items():
            parameters[name].copy_(value.to(parameters[name].dtype))
    return model.eval().requires_grad_(False), log_alpha_bar, clip_model


def cfg_denoise(model: ClipConditionedUNet, condition: th.Tensor, scale: float, x: th.Tensor, t: th.Tensor):
    """The classifier-free guided noise, with a gradient through the conditional branch only."""
    eps_c = model(x, t, condition.expand(x.shape[0], -1))[:, :3]
    if scale == 1:
        return eps_c
    with th.no_grad():
        eps_u = model(x.detach(), t)[:, :3]
    return eps_c + (scale - 1) * (eps_c.detach() - eps_u)


def generate(  # pylint: disable=too-many-locals
    prompts: Sequence[str] = (),
    image_prompts: Sequence[str] = (),
    *,
    checkpoint: str,
    cfg_scale: float = 3.0,
    steps: int = sampler.UNCONDITIONAL_STEPS,
    window: tuple[float, float] = sampler.UNCONDITIONAL_GUIDANCE_WINDOW,
    guidance_rate: float = sampler.DEFAULT_GUIDANCE_RATE,
    clip_model: str = adm.DEFAULTS.clip_model,
    num_cutouts: int = NUM_CUTOUTS,
    use_ema: bool = True,
    seed: int = 0,
    output_dir: str | Path = "outputs",
    save_frequency: int = 10,
    progress: bool = False,
    device: str = "cuda",
) -> Iterator[tuple[int, Path]]:
    th.manual_seed(seed)
    guide = load_guide(clip_model, prompts, image_prompts, image_size=256, num_cutouts=num_cutouts, device=device)
    model, log_alpha_bar, condition_model = load_conditioned(checkpoint, device, use_ema)
    condition = tf.normalize(encode_texts(list(prompts), condition_model, device, True, truncate=True), dim=-1)
    condition = tf.normalize(condition.mean(0, keepdim=True), dim=-1)
    ts = sampler.timesteps(log_alpha_bar, steps)
    x = th.randn((1, 3, 256, 256), device=device)
    guidance = sampler.Guidance(partial(adm.guidance_loss, guide), guidance_rate, tuple(window)) if guidance_rate else None
    denoise = partial(cfg_denoise, model, condition, cfg_scale)
    samples = sampler.sample(denoise, log_alpha_bar, ts, x, guidance, progress=progress)
    yield from save_frames(samples, output_dir, run_name(prompts, image_prompts), save_frequency, len(ts) - 2)

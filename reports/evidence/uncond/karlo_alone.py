"""What Karlo makes without the unconditional U-Net, as the reference for the arms that use its decoder as the thumbnail model.

    uv run python run_arm.py karlo_alone karlo_alone:generate '{}'
    uv run python run_arm.py karlo_thumb karlo_alone:generate '{"upsample": false}'

With `upsample`, the whole of Karlo as its authors run it: prior, 64x64 decoder and its own
64 to 256 super-resolution model, at their pipeline's steps and guidance scales. Without, the
bare thumbnail that thumbnail.generate() hands off, drawn the same way and only enlarged
bicubically.
"""

from collections.abc import Iterator, Sequence
from functools import lru_cache, partial
from pathlib import Path

import numpy as np
import torch as th
import torch.nn.functional as tf
import torchvision.transforms.functional as tvf

from cgd.outputs import frame_path, run_name, save_frame
from thumbnail import KARLO, PRIOR_SEED_OFFSET, drawn_thumbnails, karlo_eps, karlo_text, load_karlo


@lru_cache(maxsize=1)
def load_pipeline(device: str):
    """Karlo's own pipeline, at half precision."""
    from diffusers import PriorTransformer, UNet2DConditionModel, UNet2DModel, UnCLIPPipeline, UnCLIPScheduler
    from diffusers.pipelines.deprecated.unclip.text_proj import UnCLIPTextProjModel
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer

    def part(kind, folder: str):
        return kind.from_pretrained(KARLO, subfolder=folder, use_safetensors=True).to(th.float16)

    # Put together by hand: the pipeline's index names the module UnCLIPTextProjModel was in before diffusers moved it.
    pipeline = UnCLIPPipeline(
        prior=part(PriorTransformer, "prior"),
        decoder=part(UNet2DConditionModel, "decoder"),
        text_encoder=part(CLIPTextModelWithProjection, "text_encoder"),
        tokenizer=CLIPTokenizer.from_pretrained(KARLO, subfolder="tokenizer"),
        text_proj=part(UnCLIPTextProjModel, "text_proj"),
        super_res_first=part(UNet2DModel, "super_res_first"),
        super_res_last=part(UNet2DModel, "super_res_last"),
        prior_scheduler=UnCLIPScheduler.from_pretrained(KARLO, subfolder="prior_scheduler"),
        decoder_scheduler=UnCLIPScheduler.from_pretrained(KARLO, subfolder="decoder_scheduler"),
        super_res_scheduler=UnCLIPScheduler.from_pretrained(KARLO, subfolder="super_res_scheduler"),
    ).to(device)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


@th.no_grad()
def generate(
    prompts: Sequence[str] = (),
    image_prompts: Sequence[str] = (),
    *,
    upsample: bool = True,
    scale: float = 3.0,
    seed: int = 0,
    output_dir: str | Path = "outputs",
    save_frequency: int = 10,  # pylint: disable=unused-argument
    progress: bool = False,  # pylint: disable=unused-argument
    device: str = "cuda",
) -> Iterator[tuple[int, Path]]:
    """One 256x256 image of the first prompt from Karlo alone. `scale` is the decoder's guidance for the bare thumbnail."""
    path = frame_path(output_dir, run_name(prompts, image_prompts), 0, 0)
    if upsample:
        generator = th.Generator(device).manual_seed(seed + PRIOR_SEED_OFFSET)
        image = load_pipeline(device)(prompts[0], generator=generator, output_type="np").images[0]  # [256, 256, 3] in [0, 1].
        yield 0, save_frame(th.from_numpy(image).permute(2, 0, 1).float().mul(2).sub(1), path)
        return
    karlo = load_karlo(device)
    text = karlo_text(karlo, prompts[0], device, th.Generator(device).manual_seed(seed + 2 * PRIOR_SEED_OFFSET))
    eps = partial(karlo_eps, karlo.decoder, text)
    ts = list(np.linspace(len(karlo.log_alpha_bar) - 1, 0, 51))
    thumbnail = drawn_thumbnails(eps, karlo.log_alpha_bar, ts, scale, th.Generator(device).manual_seed(seed + PRIOR_SEED_OFFSET))
    image = tf.interpolate(thumbnail.float(), (256, 256), mode="bicubic", align_corners=False).clamp(-1, 1)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tvf.to_pil_image(thumbnail[0].add(1).div(2).clamp(0, 1).cpu()).save(Path(output_dir) / "thumbnail.png")
    yield 0, save_frame(image[0].float(), path)

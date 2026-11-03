"""cgd: make images from text and image prompts with CLIP guided diffusion."""

import argparse
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import torchvision.transforms.functional as tvf
from PIL import Image

from cgd import adm, backbone, sampler, upscale
from cgd.checks import ConfigError
from cgd.clip_models import MODEL_NAMES, OPEN_CLIP_MODELS, ensemble_members
from cgd.device import default_device
from cgd.guide import load_guide, score
from cgd.outputs import save_gif, save_video
from cgd.prompts import split_prompts

DEFAULT_SAVE_FREQUENCY = 10
JUDGE_MODELS = (
    "ViT-B/16+ViT-L-16-SigLIP2-256"  # Models guidance does not use by default, so it cannot have fooled them.
)
# Options only one pipeline has, by the name each pipeline takes them as.
ADM_OPTIONS = {
    "steps": "--steps",
    "guidance_rate": "--guidance-rate",
    "guidance_limit": "--guidance-limit",
    "init_image": "--init-image",
    "skip_timesteps": "--skip-timesteps",
    "init_scale": "--init-scale",
}
BACKBONE_OPTIONS = {"iterations": "--iterations"}
# Options both take. Those left unset get the pipeline's own default.
SHARED_OPTIONS = (
    "image_size",
    "width_offset",
    "height_offset",
    "batch_size",
    "clip_model",
    "clip_fp16",
    "aesthetic_scale",
    "device",
)

Frames = dict[int, list[Path]]
"""Each image of the batch's saved frames, in order. The last is the result."""

Run = Callable[[int, Path], Iterator[tuple[int, Path]]]
"""(seed, output folder) -> (batch index, path) of each image the run saves."""


def build_parser() -> argparse.ArgumentParser:
    """The command's options."""
    parser = argparse.ArgumentParser(prog="cgd", description=__doc__)
    parser.add_argument(
        "prompts",
        nargs="?",
        default="",
        help="Text prompts, joined with |. Each can take a weight, negative to avoid it: 'a fox:2|blurry:-1'.",
    )
    parser.add_argument(
        "-imgs",
        "--image-prompts",
        default="",
        help="Image prompts, paths or URLs, joined with | and weighted the same way.",
    )
    parser.add_argument(
        "-size",
        "--image-size",
        type=int,
        help=f"Image size. 64, 128, 256 or 512, default {adm.DEFAULT_IMAGE_SIZE}. With --backbone, any multiple "
        f"of {backbone.LATENT_FACTOR}, defaulting to what the model was trained for.",
    )
    parser.add_argument(
        "-wd", "--width-offset", type=int, default=0, help="Make the image this much wider. Default %(default)s."
    )
    parser.add_argument(
        "-ht", "--height-offset", type=int, default=0, help="Make the image this much taller. Default %(default)s."
    )
    parser.add_argument("-bs", "--batch-size", type=int, default=1, help="Images to make at once. Default %(default)s.")
    parser.add_argument("-seed", "--seed", type=int, default=0, help="Random seed. Default %(default)s.")
    parser.add_argument(
        "-clip",
        "--clip-model",
        help=f"CLIP model to guide with, one of {', '.join(MODEL_NAMES)}, or a checkpoint file. Join several with + "
        f"to guide with all of them. Default {adm.DEFAULT_CLIP_MODEL}, or {backbone.DEFAULT_CLIP_MODEL} with "
        "--backbone. The README has what each costs.",
    )
    parser.add_argument(
        "-fp16",
        "--clip-fp16",
        action="store_true",
        help="Run a PE-Core or SigLIP2 model at half precision: up to 3x faster for half the CLIP VRAM. "
        "Less exact, but it scored about the same. OpenAI's models always do.",
    )
    parser.add_argument(
        "-aes",
        "--aesthetic-scale",
        type=float,
        help="Also reward what people rate highly, per point of a 1 to 10 rating. Default 0, or "
        f"{backbone.DEFAULT_AESTHETIC_SCALE} with --backbone. The README has what it trades away.",
    )
    parser.add_argument(
        "-best",
        "--candidates",
        type=int,
        default=1,
        help="Make this many images from seeds counting up from --seed, and keep the one that CLIP models which "
        "did not guide them score best. Takes this many times as long. Default %(default)s.",
    )
    parser.add_argument("-dev", "--device", help="cpu, cuda or cuda:<index>. Default cuda when there is a GPU.")

    imagenet = parser.add_argument_group(
        "ImageNet model", "The default pipeline: CLIP guides OpenAI's ImageNet diffusion model."
    )
    imagenet.add_argument("-n", "--steps", type=int, help=f"Model calls. Default {sampler.DEFAULT_STEPS}.")
    imagenet.add_argument(
        "-rate",
        "--guidance-rate",
        type=float,
        help="Guidance strength: the most CLIP may move each step, relative to the noise. Higher follows the "
        f"prompt harder but saturates. Default {sampler.DEFAULT_GUIDANCE_RATE}.",
    )
    imagenet.add_argument(
        "-gl",
        "--guidance-limit",
        type=float,
        help="Hold back whatever part of CLIP's update would push more than this quantile of pixels out of "
        "range, e.g. 0.99. Natural colour and almost no clipped pixels, though the judges prefer the "
        "saturated default.",
    )
    imagenet.add_argument("-init", "--init-image", help="Start from this image, a path or URL, instead of pure noise.")
    imagenet.add_argument(
        "-skip",
        "--skip-timesteps",
        type=int,
        help=f"How much of the {adm.NUM_TIMESTEPS} step schedule an init image skips. Higher stays closer "
        f"to it. Default {adm.NUM_TIMESTEPS // 2}.",
    )
    imagenet.add_argument(
        "-is",
        "--init-scale",
        type=float,
        help="Also keep the image perceptually close to the init image, by LPIPS, e.g. 1000. Default 0.",
    )

    modern = parser.add_argument_group(
        "Modern backbones",
        "Draw with a one-step model and correct it with CLIP by optimising its noise, instead of guiding "
        "the ImageNet model. The best images this makes.",
    )
    modern.add_argument("-bb", "--backbone", choices=tuple(backbone.BACKBONES), help="The model to draw with.")
    modern.add_argument(
        "-iters",
        "--iterations",
        type=int,
        help=f"Steps of noise optimisation. 0 is the plain model. Default {backbone.DEFAULT_ITERATIONS}.",
    )

    output = parser.add_argument_group("Output")
    output.add_argument(
        "-dir",
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Where images go, in a folder named after the prompts. Default %(default)s.",
    )
    output.add_argument(
        "-freq",
        "--save-frequency",
        type=int,
        help=f"Save a frame every this many steps. The result is always saved. Default {DEFAULT_SAVE_FREQUENCY}, "
        "or 1 with --gif or --mp4.",
    )
    output.add_argument("-gif", "--gif", action="store_true", help="Also save the frames as a GIF, then delete them.")
    output.add_argument(
        "-mp4", "--mp4", action="store_true", help="Also save the frames as an H.264 video, then delete them."
    )
    output.add_argument(
        "-up",
        "--upscale",
        action="store_true",
        help=f"Also save a copy {upscale.SCALE}x larger, made with Real-ESRGAN x4plus. Adds little time or VRAM.",
    )
    output.add_argument("-q", "--quiet", action="store_true", help="No progress bars or status messages.")
    return parser


def pipeline(args: argparse.Namespace) -> tuple[Callable[..., Iterator[tuple[int, Path]]], dict[str, Any]]:
    """The generate() the options choose, and the settings to call it with besides the prompts, seed and folder."""
    own, other = (BACKBONE_OPTIONS, ADM_OPTIONS) if args.backbone else (ADM_OPTIONS, BACKBONE_OPTIONS)
    stray = [flag for option, flag in other.items() if getattr(args, option) is not None]
    if stray:
        raise ConfigError(
            f"{', '.join(stray)} only {'apply' if len(stray) > 1 else 'applies'} "
            f"{'without' if args.backbone else 'with'} --backbone."
        )
    settings = {
        option: getattr(args, option) for option in (*SHARED_OPTIONS, *own) if getattr(args, option) is not None
    }
    animate = args.gif or args.mp4
    settings["save_frequency"] = args.save_frequency or (1 if animate else DEFAULT_SAVE_FREQUENCY)
    settings["progress"] = not args.quiet
    if args.backbone:
        return backbone.generate, settings | {"backbone": args.backbone}
    return adm.generate, settings


def collect(saved: Iterable[tuple[int, Path]]) -> Frames:
    """Group the (batch index, path) of each image a run saves by batch index."""
    frames: Frames = {}
    for index, path in saved:
        frames.setdefault(index, []).append(path)
    return frames


def judged_candidates(
    run: Run,
    candidates: int,
    *,
    seed: int,
    output_dir: Path,
    prompts: Sequence[str],
    image_prompts: Sequence[str],
    judges: str = JUDGE_MODELS,
) -> Frames:
    """The frames of the best of `candidates` runs for each image of the batch.

    The judges are CLIP models that did not guide the runs, so guidance cannot have fooled
    them, and they run on the CPU, so they can not push out what guidance needs. Each run
    keeps its frames in a folder of its own seed.
    """
    runs = [
        collect(run(candidate, Path(output_dir) / f"seed_{candidate}")) for candidate in range(seed, seed + candidates)
    ]
    judge = load_guide(judges, prompts, image_prompts, device="cpu")

    def rating(frames: list[Path]) -> float:
        image = tvf.to_tensor(Image.open(frames[-1]).convert("RGB")).unsqueeze(0).mul(2).sub(1)
        return score(judge, image).item()

    return {index: max((frames[index] for frames in runs), key=rating) for index in runs[0]}


def make_images(args: argparse.Namespace) -> Frames:
    """Run the pipeline the options choose, as many times as --candidates asks."""
    generate, settings = pipeline(args)
    prompts, image_prompts = split_prompts(args.prompts), split_prompts(args.image_prompts)

    def run(seed: int, output_dir: Path) -> Iterator[tuple[int, Path]]:
        return generate(prompts, image_prompts, seed=seed, output_dir=output_dir, **settings)

    if args.candidates < 1:
        raise ConfigError("--candidates must be at least 1.")
    if args.candidates == 1:
        return collect(run(args.seed, args.output_dir))
    return judged_candidates(
        run, args.candidates, seed=args.seed, output_dir=args.output_dir, prompts=prompts, image_prompts=image_prompts
    )


def save_outputs(frames: Frames, args: argparse.Namespace) -> None:
    """Upscale each result and make any GIF or video. Frames are only deleted once everything made from them exists."""
    device = default_device(args.device)
    upscaler = upscale.load_upscaler(device) if args.upscale and frames else None
    animations = [
        (save, suffix)
        for wanted, save, suffix in ((args.gif, save_gif, ".gif"), (args.mp4, save_video, ".mp4"))
        if wanted
    ]
    for paths in frames.values():
        if upscaler is not None:
            print(f"Saved {upscale.upscale_file(upscaler, paths[-1], device)}")
        folder = paths[0].parent
        saved = [save(paths, folder.with_suffix(suffix)) for save, suffix in animations]
        if saved and all(saved):
            for path in paths:
                path.unlink()
            if not any(folder.iterdir()):
                folder.rmdir()
        else:
            print(f"Saved {paths[-1]}")


def is_out_of_memory(error: RuntimeError) -> bool:
    """PyTorch's allocator raises OutOfMemoryError, but CUDA failing mid-backward is a plain AcceleratorError."""
    return "out of memory" in str(error).lower()


def out_of_memory_advice(args: argparse.Namespace) -> str:
    """What to change after running out of GPU memory."""
    clip_model = args.clip_model or (backbone.DEFAULT_CLIP_MODEL if args.backbone else adm.DEFAULT_CLIP_MODEL)
    advice = (
        "Ran out of GPU memory. Try a smaller --image-size/-size or --batch-size/-bs.\n"
        f"--clip-model/-clip (currently {clip_model}) has a large impact on VRAM: "
        "RN50 uses the least, ViT-B/32 the second least.\n"
    )
    if not args.clip_fp16 and any(member in OPEN_CLIP_MODELS for member in ensemble_members(clip_model)):
        advice += "--clip-fp16/-fp16 halves what this model needs.\n"
    if args.backbone:
        advice += "--iterations/-iters 0 is the plain model, which needs far less than correcting it with CLIP.\n"
    return advice


def main(argv: Sequence[str] | None = None) -> None:
    """The cgd command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        frames = make_images(args)
    except ConfigError as error:
        parser.error(str(error))
    except RuntimeError as error:
        if not is_out_of_memory(error):
            raise
        # Exit once this block is done with the error: its traceback holds every tensor of
        # the failed run, and an exit raised in here would keep them alive as its context.
        frames = None
    if frames is None:
        parser.exit(1, out_of_memory_advice(args))
    save_outputs(frames, args)

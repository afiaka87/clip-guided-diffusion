"""CLIP Guided Diffusion: generate images from text and image prompts."""
import argparse
from pathlib import Path

import torch as th
from PIL import Image
import torchvision.transforms as tvt

from tqdm.auto import tqdm
from cgd import losses
from cgd import clip_util
from cgd import modern
from cgd import sampling
from cgd import script_util
from cgd import secondary
from cgd import upscale

FAST_NUM_CUTOUTS = 4
FAST_MIN_CUT_FRACTION = 0.7  # Cutouts span 70-100% of the frame, or down to CLIP's input size if that is smaller.
LEGACY_NUM_CUTOUTS = 16
DEFAULT_SAVE_FREQUENCY = 10
# Guidance backpropagates through the U-Net. Above this many pixels per batch, recompute
# its activations instead of storing them: 1.5x the time per guided step for under half
# the memory, which is what lets 512px run in 8 GB.
GRADIENT_CHECKPOINT_PIXELS = 256 ** 2
CLIP_VIEWS = ("blend", "detached", "x0")
DEFAULT_IMAGE_SIZE = 128
DEFAULT_CLIP_MODEL = "ViT-B/32"
JUDGE_MODELS = "ViT-B/16+ViT-L-16-SigLIP2-256"  # Models guidance does not use by default, so it cannot have fooled them.


def clip_guided_diffusion(
    image_size: int = 128,
    num_cutouts: int = None,
    prompts: "list[str]" = (),
    image_prompts: "list[str]" = (),
    clip_guidance_scale: int = 1000,
    tv_scale: float = 150,
    range_scale: float = 50,
    sat_scale: float = 0,
    init_scale: float = 0,
    batch_size: int = 1,
    init_image: Path = None,
    class_cond: bool = True,
    cutout_power: float = 1.0,
    timestep_respacing: str = None,  # Set this to use the original guided-diffusion sampler.
    seed: int = 0,
    skip_timesteps: int = 0,
    checkpoints_dir: str = script_util.CACHE_PATH,
    clip_model_name: str = "ViT-B/32",
    clip_fp16: bool = False,  # Faster and half the CLIP VRAM with open_clip models. Less exact, but scored about the same.
    randomize_class: bool = True,
    prefix_path: Path = Path('./outputs'),
    save_frequency: int = DEFAULT_SAVE_FREQUENCY,
    device: str = '',
    wandb_project: str = None,
    wandb_entity: str = None,
    use_augs: bool = False, # enables augmentation, mostly better for timesteps <= 100
    use_magnitude: bool = False, # clamps the gradient. Original sampler only: the fast one caps guidance itself.
    height_offset: int = 0,
    width_offset: int = 0,
    progress: bool = True,
    sample_steps: int = None,
    guidance_rate: float = sampling.DEFAULT_GUIDANCE_RATE,
    guidance_schedule: str = "flat",  # Shape of the rate over the guidance window: flat, sine or rising.
    x0_threshold: float = None,  # Dynamic thresholding quantile, replacing the x0 clamp. Fast sampler only.
    guidance_limit: float = None,  # Hold back guidance that would push this quantile of pixels out of range. Fast sampler only.
    clip_loss: str = "spherical",  # spherical, or contrastive against the text of the ImageNet classes.
    aesthetic_scale: float = 0,  # Also reward what people rate highly, per point of a 1 to 10 rating.
    clip_view: str = "blend",  # What CLIP sees: blend, blend with no gradient through x, or x0.
    pag_scale: float = 0,  # Perturbed attention guidance. Fast sampler only.
    contrast_scale: float = 0,  # Push away from the second closest ImageNet class. Fast sampler only.
    class_rank: int = 0,  # Condition on this closest ImageNet class, 0 being the closest.
    secondary_guidance: bool = False,  # Guide through a small U-Net: much less VRAM from 256 up. Fast sampler only.
    solver: str = sampling.SOLVERS[0],  # Fast sampler only.
):
    if sample_steps is None:
        sample_steps = sampling.DEFAULT_STEPS if class_cond else sampling.UNCOND_STEPS
    script_util.check_parameters(
        prompts, image_prompts, image_size, class_cond=class_cond, timestep_respacing=timestep_respacing,
        skip_timesteps=skip_timesteps, save_frequency=save_frequency, sample_steps=sample_steps,
        height_offset=height_offset, width_offset=width_offset, init_image=init_image)
    clip_util.check_clip_model(clip_model_name)
    if aesthetic_scale < 0:
        raise script_util.ConfigError("--aesthetic_scale must not be negative.")
    if clip_view not in CLIP_VIEWS:
        raise script_util.ConfigError(f"Unknown CLIP view {clip_view!r}. Choose one of {list(CLIP_VIEWS)}.")
    if solver not in sampling.SOLVERS:
        raise script_util.ConfigError(f"Unknown solver {solver!r}. Choose one of {list(sampling.SOLVERS)}.")
    fast = timestep_respacing is None
    if num_cutouts is None:
        num_cutouts = FAST_NUM_CUTOUTS if fast else LEGACY_NUM_CUTOUTS

    if len(device) == 0:
        device = 'cuda' if th.cuda.is_available() else 'cpu'
    fp32_diffusion = (device == 'cpu')

    wandb_run = None
    if wandb_project is not None:
        # just use local vars for config
        wandb_config = dict(locals())
        import wandb
        wandb_run = wandb.init(project=wandb_project, entity=wandb_entity, config=wandb_config)

    def status(message):
        if progress:
            tqdm.write(message)

    status(f"Using device {device}")
    th.manual_seed(seed)

    if not fast and use_magnitude == False and image_size == 64:
        use_magnitude = True
        status("Enabling magnitude for 64x64 checkpoints.")

    use_saturation = sat_scale != 0
    Path(prefix_path).mkdir(parents=True, exist_ok=True)
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)

    diffusion_path = script_util.download_guided_diffusion(image_size=image_size, checkpoints_dir=checkpoints_dir, class_cond=class_cond)

    # Load CLIP models/Encode text/Create `MakeCutouts`
    guide = clip_util.ClipGuide(clip_model_name, prompts, image_prompts, image_size, num_cutouts, device, clip_fp16,
                                cutout_power, use_augs, FAST_MIN_CUT_FRACTION if fast else 1.0, clip_loss, aesthetic_scale)

    # Load initial image (if provided)
    init_tensor = None
    if init_image:
        pil_image = Image.open(script_util.fetch(init_image)).convert('RGB')
        pil_image = pil_image.resize((image_size + width_offset, image_size + height_offset))
        init_tensor = tvt.ToTensor()(pil_image)
        init_tensor = init_tensor.to(device).unsqueeze(0).mul(2).sub(1)

   # Class randomization requires a starting class index `y`
    model_kwargs = {}
    if class_cond:
        model_kwargs["y"] = th.zeros(
            [batch_size], device=device, dtype=th.long)
        if fast:
            # Let the U-Net do the heavy lifting by conditioning on the closest real class.
            class_indices = clip_util.imagenet_top_n(guide.query(), device=device, n=class_rank + 2,
                                                     clip_model_name=clip_util.ensemble_members(clip_model_name)[0], fp16=clip_fp16)
            class_index = class_indices[class_rank:class_rank + 1]
            runner_up = class_indices[class_rank + 1]
            model_kwargs["y"][:] = class_index
            status(f"Conditioning on ImageNet class: {clip_util.IMAGENET_CLASSES[class_index.item()]}")

    shape = (batch_size, 3, image_size + height_offset, image_size + width_offset)

    # Load guided diffusion
    gd_model, diffusion = script_util.load_guided_diffusion(
        checkpoint_path=diffusion_path,
        image_size=image_size, class_cond=class_cond,
        timestep_respacing="" if fast else timestep_respacing,
        use_fp16=(not fp32_diffusion),
        device=device,
        use_checkpoint=batch_size * shape[2] * shape[3] > GRADIENT_CHECKPOINT_PIXELS,
    )

    # An init image only has an effect if diffusion starts partway down the schedule.
    if init_tensor is not None and skip_timesteps == 0:
        skip_timesteps = diffusion.num_timesteps // 2
        status(f"Skipping {skip_timesteps} timesteps to blend in the init image. Set --skip_timesteps to change this.")

    # This is initialized lazily as it can use a bit of VRAM
    if init_tensor is not None and init_scale != 0:
        import lpips
        script_util.download_vgg16()
        lpips_vgg = lpips.LPIPS(net='vgg').to(device)
    current_timestep = None

    def guidance_loss(x, pred_xstart, fac):
        """Scaled guidance losses for what CLIP sees: a blend of pred_xstart and x."""
        log = {}
        sigmas = 1 - fac
        if clip_view == "x0":
            x_in = pred_xstart
        else:
            x_in = pred_xstart * fac + (x.detach() if clip_view == "detached" else x) * sigmas
        if wandb_project is not None:
            def preview(images, caption):  # The first image of the batch, as wandb wants it.
                return wandb.Image(images[0].detach().add(1).div(2).clamp(0, 1), caption=caption)

            log[f'Generations - {timestep_respacing or sample_steps}'] = [
                preview(x, "Noisy Sample"),
                preview(pred_xstart, "Denoised Prediction"),
                preview(x_in, "Blended (what CLIP sees)"),
            ]

        clip_losses = guide.loss(x_in, clip_guidance_scale)
        range_losses = losses.range_loss(pred_xstart)
        tv_losses = losses.tv_loss(x_in)

        range_losses = range_losses.sum() * range_scale
        tv_losses = tv_losses.sum() * tv_scale

        log['CLIP Loss'] = clip_losses.item()
        log['Range Loss'] = range_losses.item()
        log['TV Loss'] = tv_losses.item()

        loss = clip_losses + tv_losses + range_losses

        if use_saturation:
            sat_losses = th.abs(x_in - x_in.clamp(min=-1, max=1)).mean()
            sat_losses = sat_losses.sum() * sat_scale
            log['Saturation Loss'] = sat_losses.item()
            loss = loss + sat_losses

        if init_tensor is not None and init_scale != 0:
            init_losses = lpips_vgg(x_in, init_tensor)
            init_losses = init_losses.sum() * init_scale
            log['Init VGG Loss'] = init_losses.item()
            loss = loss + init_losses

        log['Total Loss'] = loss.item()
        return loss, log

    def report(log):
        if wandb_project is not None:
            wandb_run.log(log)

    def fast_loss(x, pred_xstart, fac):
        loss, log = guidance_loss(x, pred_xstart, fac)
        report(log)
        return loss

    def cond_fn(x, t, out, y=None):
        fac = diffusion.sqrt_one_minus_alphas_cumprod[current_timestep]
        loss, log = guidance_loss(x, out["pred_xstart"], fac)

        final_loss = -sampling.loss_gradient(loss, x)  # negative gradient
        if use_magnitude:
            magnitude = final_loss.square().mean().sqrt()  # TODO experimental clamping?
            log["Magnitude"] = magnitude.item()
            final_loss = final_loss * magnitude.clamp(max=0.05) / magnitude.clamp(min=1e-20)
        log['Grad'] = final_loss.mean().item()
        status("\t".join(f"{k}: {v:.3f}" for k, v in log.items() if "loss" in k.lower()))
        report(log)
        return final_loss

    def eps_adjustment(x, t, eps):
        adjustment = 0
        if pag_scale != 0:
            with sampling.perturbed_attention(gd_model):
                adjustment = adjustment + pag_scale * (eps - gd_model(x, t, **model_kwargs)[:, :shape[1]])
        if contrast_scale != 0 and class_cond:
            other = {**model_kwargs, "y": runner_up.expand(x.shape[0])}
            adjustment = adjustment + contrast_scale * (eps - gd_model(x, t, **other)[:, :shape[1]])
        return adjustment

    def legacy_samples():
        nonlocal current_timestep
        # Choose between normal or DDIM
        if timestep_respacing.startswith("ddim"):
            diffusion_sample_loop = diffusion.ddim_sample_loop_progressive
        else:
            diffusion_sample_loop = diffusion.p_sample_loop_progressive

        current_timestep = diffusion.num_timesteps - 1 - skip_timesteps
        for sample in diffusion_sample_loop(
            gd_model,
            shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            progress=progress,
            skip_timesteps=skip_timesteps,
            init_image=init_tensor,
            randomize_class=randomize_class,
            cond_fn_with_grad=True,
        ):
            current_timestep -= 1
            yield {**sample, "final": current_timestep == -1}

    def fast_samples():
        # Only an init image can start partway down the schedule, as it supplies the signal.
        start_timestep = None
        if init_tensor is not None:
            start_timestep = diffusion.num_timesteps - 1 - skip_timesteps
        return sampling.sample_loop_progressive(
            gd_model,
            diffusion,
            shape,
            loss_fn=fast_loss,
            steps=sample_steps,
            guidance_rate=guidance_rate,
            guidance_window=sampling.GUIDANCE_WINDOW if class_cond else sampling.UNCOND_GUIDANCE_WINDOW,
            guidance_schedule=guidance_schedule,
            x0_threshold=x0_threshold,
            guidance_limit=guidance_limit,
            eps_adjustment=eps_adjustment if pag_scale != 0 or (contrast_scale != 0 and class_cond) else None,
            guide_model=secondary.load_secondary(device, checkpoints_dir) if secondary_guidance else None,
            solver=solver,
            init_image=init_tensor,
            start_timestep=start_timestep,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        )

    run_name = script_util.run_name(prompts, image_prompts)
    # The secondary model's convolutions are nondeterministic going backward unless cuDNN is
    # told otherwise. The default path is reproducible without this, and it would change its images.
    with th.backends.cudnn.flags(enabled=True, deterministic=secondary_guidance):
        for step, sample in enumerate(fast_samples() if fast else legacy_samples()):
            if step % save_frequency == 0 or sample["final"]:
                for batch_idx, image_tensor in enumerate(sample["pred_xstart"]):
                    yield batch_idx, script_util.log_image(image_tensor, prefix_path, run_name, step, batch_idx)


def is_out_of_memory(error: RuntimeError) -> bool:
    """PyTorch's allocator raises OutOfMemoryError, but CUDA failing mid-backward is a plain AcceleratorError."""
    return "out of memory" in str(error).lower()


def judged_candidates(generate, candidates: int, seed: int, prefix_path: Path, prompts, image_prompts, judges: str = JUDGE_MODELS):
    """Return {batch_idx: frames} of the best of `candidates` runs, scored by models that did not guide them.

    `generate(seed, prefix_path)` is one run, yielding (batch_idx, image_path). Each run
    keeps its frames under its own seed. The judges run on the CPU, so they can not push
    out what guidance needs.
    """
    runs = []
    for candidate_seed in range(seed, seed + candidates):
        frames = {}
        for batch_idx, path in generate(candidate_seed, Path(prefix_path) / f"seed_{candidate_seed}"):
            frames.setdefault(batch_idx, []).append(path)
        runs.append(frames)
    judge = clip_util.ClipGuide(judges, prompts, image_prompts, device="cpu")

    def score(path):
        image = tvt.ToTensor()(Image.open(path).convert('RGB')).unsqueeze(0).mul(2).sub(1)
        return judge.score(image).item()

    return {batch_idx: max((run[batch_idx] for run in runs), key=lambda frames: score(frames[-1])) for batch_idx in runs[0]}


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("prompt", nargs="?", default='',
                   help="the prompt/s to reward, same format as --prompts")
    p.add_argument("--prompts", "-txts", type=str, default='',
                   help="the prompt/s to reward paired with weights. e.g. 'My text:0.5|Other text:-0.5' ")
    p.add_argument("--image_prompts", "-imgs", type=str, default='',
                   help="the image prompt/s to reward paired with weights. e.g. 'img1.png:0.5|img2.png:-0.5'")
    p.add_argument("--image_size", "-size", type=int, default=None,
                   help=f"Image size. Without --backbone one of 64, 128, 256 or 512, defaulting to {DEFAULT_IMAGE_SIZE}. With it, any multiple of {modern.LATENT_FACTOR}, defaulting to what the model was trained for.")
    p.add_argument("--init_image", "-init", type=str, default='',
                   help="Start from this image instead of pure noise.")
    p.add_argument("--init_scale", "-is", type=float, default=0,
                   help="(optional) Perceptual loss scale for init image. ")
    p.add_argument("--skip_timesteps", "-skip", type=int, default=0,
                   help="How much of the schedule to skip with an init image. Higher stays closer to it. 0 skips half.")
    p.add_argument("--prefix", "-dir", default="outputs",
                   type=Path, help="output directory")
    p.add_argument("--checkpoints_dir", "-ckpts", default=script_util.CACHE_PATH,
                   type=Path, help="Path subdirectory containing checkpoints.")
    p.add_argument("--batch_size", "-bs", type=int,
                   default=1, help="the batch size")
    p.add_argument("--clip_guidance_scale", "-cgs", type=float, default=1000,
                   help="Scale for CLIP spherical distance loss. Values will need tinkering for different settings.",)
    p.add_argument("--tv_scale", "-tvs", type=float,
                   default=150., help="Controls the smoothness of the final output.",)
    p.add_argument("--range_scale", "-rs", type=float,
                   default=50., help="Controls how far out of RGB range values may get.",)
    p.add_argument("--sat_scale", "-sats", type=float, default=0.,
                   help="Controls how much saturation is allowed. Used for ddim. From @nshepperd.",)
    p.add_argument("--seed", "-seed", type=int,
                   default=0, help="Random number seed")
    p.add_argument("--save_frequency", "-freq", type=int, default=None,
                   help=f"Save a frame every n steps. The final image is always saved. Defaults to {DEFAULT_SAVE_FREQUENCY}, or 1 with --save-as-gif/--save-as-video.")
    p.add_argument("--timestep_respacing", "-respace", type=str, default=None,
                   help="Use the original guided-diffusion sampler with this respacing, e.g. 1000 or ddim50. Much slower.")
    p.add_argument("--sample_steps", "-n", type=int, default=None,
                   help=f"Model calls for the fast sampler. Defaults to {sampling.DEFAULT_STEPS}, or {sampling.UNCOND_STEPS} with --uncond. Ignored when --timestep_respacing is set.")
    p.add_argument("--guidance_rate", "-rate", type=float, default=sampling.DEFAULT_GUIDANCE_RATE,
                   help="Fast sampler guidance strength, as the most CLIP may move each step relative to the noise. Higher follows the prompt harder but saturates.")
    p.add_argument("--num_cutouts", "-cutn", type=int, default=None,
                   help=f"Number of randomly cut patches to distort from diffusion. Defaults to {FAST_NUM_CUTOUTS}, or {LEGACY_NUM_CUTOUTS} with --timestep_respacing.")
    p.add_argument("--cutout_power", "-cutpow", type=float,
                   default=1.0, help="Cutout size power")
    p.add_argument("--clip_model", "-clip", type=str, default=None,
                   help=f"clip model name. Should be one of: {clip_util.CLIP_MODEL_NAMES} or a checkpoint filename ending in `.pt`. Join several with + to guide with all of them. "
                        f"Defaults to {DEFAULT_CLIP_MODEL}, or {modern.DEFAULT_CLIP_MODEL} with --backbone.")
    p.add_argument("--clip_fp16", "-fp16", action="store_true",
                   help="Run a PE-Core or SigLIP2 model at half precision: up to 3x faster for half the CLIP VRAM. Less exact, but it scored about the same. OpenAI's models always do.")
    p.add_argument("--uncond", "-uncond", action="store_true",
                   help='Use finetuned unconditional checkpoints from OpenAI (256px) and Katherine Crowson (512px)')
    p.add_argument("--device", "-dev", default='', type=str,
                   help="Device to use. Either cpu or cuda.")
    p.add_argument('--wandb_project', '-proj', default=None,
                   help='Name W&B will use when saving results.\ne.g. `--wandb_project "my_project"`')
    p.add_argument('--wandb_entity', '-ent', default=None,
                   help='(optional) Name of W&B team/entity to log to.')
    p.add_argument('--height_offset', '-ht', default=0, type=int, help='Height offset for image')
    p.add_argument('--width_offset', '-wd', default=0, type=int, help='Width offset for image')
    p.add_argument('--quiet', '-q', action='store_true',
                   help='Suppress output.')
    p.add_argument('--save-as-gif', '-gif', action='store_true',
                   help='Save output as high-quality GIF using ffmpeg. Deletes individual frames.')
    p.add_argument('--save-as-video', '-mp4', action='store_true',
                   help='Save output as high-quality MP4 video using ffmpeg. Deletes individual frames.')
    p.add_argument("--clip_loss", "-loss", choices=clip_util.CLIP_LOSSES, default="spherical",
                   help="spherical is distance to the prompt. contrastive also pushes away from the text of the ImageNet classes.")
    p.add_argument("--aesthetic_scale", "-aes", type=float, default=None,
                   help=f"Also reward what people rate highly, per point of a 1 to 10 rating. Default 0, or {modern.DEFAULT_AESTHETIC_SCALE} with --backbone. The README has what it trades away.")
    p.add_argument("--upscale", "-up", action="store_true",
                   help=f"Also save a copy {upscale.SCALE}x larger, made with Real-ESRGAN x4plus. Adds little time or VRAM.")
    p.add_argument("--candidates", "-best", type=int, default=1,
                   help="Run this many seeds, starting at --seed, and report the one that models which did not guide it score best. Takes this many times as long.")

    fast_sampler = p.add_argument_group("fast sampler", "Options of the fast sampler. Ignored with --timestep_respacing or --backbone.")
    fast_sampler.add_argument("--x0_threshold", "-dt", type=float, default=None,
                              help="Dynamic thresholding quantile, e.g. 0.999, instead of clamping each denoised image. Keeps contrast in what guidance saturates.")
    fast_sampler.add_argument("--guidance_limit", "-gl", type=float, default=None,
                              help="Hold back whatever part of CLIP's update would push more than this quantile of pixels out of range, e.g. 0.99. Natural colour and almost no clipped pixels, though the judges prefer the saturated default.")
    fast_sampler.add_argument("--guidance_schedule", "-gs", choices=sampling.GUIDANCE_SCHEDULES, default="flat",
                              help="Shape of --guidance_rate over the guidance window, keeping its total.")
    fast_sampler.add_argument("--clip_view", "-view", choices=CLIP_VIEWS, default="blend",
                              help="What CLIP sees: the blend of the denoised and the noisy image, that blend with no gradient through the noisy image, or the denoised image.")
    fast_sampler.add_argument("--pag_scale", "-pag", type=float, default=0,
                              help="Perturbed attention guidance. One more model call per guided step.")
    fast_sampler.add_argument("--contrast_scale", "-contrast", type=float, default=0,
                              help="Push away from the second closest ImageNet class. One more model call per guided step.")
    fast_sampler.add_argument("--class_rank", "-class", type=int, default=0,
                              help="Condition on this closest ImageNet class, 0 being the closest.")
    fast_sampler.add_argument("--secondary", "-sec", action="store_true",
                              help="Take guidance through a small U-Net instead of the diffusion model: from 5.3 to 1.9 GiB at 256 and 4.1 to 2.6 at 512, at no measurable loss. It scored lower at 128.")
    fast_sampler.add_argument("--solver", "-solver", choices=sampling.SOLVERS, default=sampling.SOLVERS[0],
                              help="The sampler's solver.")

    backbone = p.add_argument_group("modern backbones", "Draw with a modern one-step model and correct it with CLIP by optimising its noise, instead of guiding an ImageNet model. "
                                                        "Of the options above, this uses the prompts, the size, offsets, batch size, seed, CLIP options, --aesthetic_scale, --upscale and --candidates.")
    backbone.add_argument("--backbone", "-bb", choices=tuple(modern.BACKBONES), default=None,
                          help="The model to draw with.")
    backbone.add_argument("--iterations", "-iters", type=int, default=modern.DEFAULT_ITERATIONS,
                          help="Steps of noise optimisation. 0 is the plain model.")
    backbone.add_argument("--lr", "-lr", type=float, default=modern.DEFAULT_LR,
                          help="Learning rate of the noise optimisation.")
    backbone.add_argument("--noise_penalty", "-np", type=float, default=modern.DEFAULT_NOISE_PENALTY,
                          help="How hard to keep the noise on the prior.")
    backbone.add_argument("--render_steps", "-steps", type=int, default=1,
                          help="Model calls for the final image.")
    backbone.add_argument("--full_decoder", "-fulldec", action="store_true",
                          help="Backpropagate through the real decoder instead of a tiny one. No better by score, for about twice the time and 1 GiB more at 512.")
    args = p.parse_args()

    _class_cond = not args.uncond
    prefix_path = args.prefix
    make_animation = args.save_as_gif or args.save_as_video
    save_frequency = args.save_frequency
    if save_frequency is None:
        save_frequency = 1 if make_animation else DEFAULT_SAVE_FREQUENCY

    prompts = [prompt for prompt in f"{args.prompt}|{args.prompts}".split('|') if prompt]
    image_prompts = [image_prompt for image_prompt in args.image_prompts.split('|') if image_prompt]
    clip_model = args.clip_model or (modern.DEFAULT_CLIP_MODEL if args.backbone else DEFAULT_CLIP_MODEL)

    def generate(seed, prefix):
        if args.backbone:
            return modern.generate(
                prompts=prompts,
                image_prompts=image_prompts,
                backbone=args.backbone,
                image_size=args.image_size,
                height_offset=args.height_offset,
                width_offset=args.width_offset,
                clip_model_name=clip_model,
                clip_fp16=args.clip_fp16,
                clip_loss=args.clip_loss,
                aesthetic_scale=modern.DEFAULT_AESTHETIC_SCALE if args.aesthetic_scale is None else args.aesthetic_scale,
                num_cutouts=args.num_cutouts or modern.DEFAULT_NUM_CUTOUTS,
                cutout_power=args.cutout_power,
                iterations=args.iterations,
                lr=args.lr,
                noise_penalty_scale=args.noise_penalty,
                render_steps=args.render_steps,
                tiny_decoder=not args.full_decoder,
                batch_size=args.batch_size,
                seed=seed,
                prefix_path=prefix,
                save_frequency=save_frequency,
                device=args.device,
                progress=not args.quiet,
            )
        return clip_guided_diffusion(
            prompts=prompts,
            image_prompts=image_prompts,
            batch_size=args.batch_size,
            tv_scale=args.tv_scale,
            init_scale=args.init_scale,
            range_scale=args.range_scale,
            sat_scale=args.sat_scale,
            image_size=args.image_size or DEFAULT_IMAGE_SIZE,
            class_cond=_class_cond,
            randomize_class=(_class_cond),
            save_frequency=save_frequency,
            clip_guidance_scale=args.clip_guidance_scale,
            cutout_power=args.cutout_power,
            num_cutouts=args.num_cutouts,
            timestep_respacing=args.timestep_respacing,
            seed=seed,
            skip_timesteps=args.skip_timesteps,
            init_image=args.init_image,
            checkpoints_dir=args.checkpoints_dir,
            clip_model_name=clip_model,
            clip_fp16=args.clip_fp16,
            device=args.device,
            prefix_path=prefix,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            height_offset=args.height_offset,
            width_offset=args.width_offset,
            progress=not args.quiet,
            sample_steps=args.sample_steps,
            guidance_rate=args.guidance_rate,
            guidance_schedule=args.guidance_schedule,
            x0_threshold=args.x0_threshold,
            guidance_limit=args.guidance_limit,
            clip_loss=args.clip_loss,
            aesthetic_scale=args.aesthetic_scale or 0,
            clip_view=args.clip_view,
            pag_scale=args.pag_scale,
            contrast_scale=args.contrast_scale,
            class_rank=args.class_rank,
            secondary_guidance=args.secondary,
            solver=args.solver,
        )

    if args.candidates < 1:
        p.error("--candidates must be at least 1.")
    frames = {}  # Only this run's frames: the folder may hold older ones for the same prompt.
    try:
        if args.candidates > 1:
            frames = judged_candidates(generate, args.candidates, args.seed, prefix_path, prompts, image_prompts)
        else:
            for batch_idx, frame_path in generate(args.seed, prefix_path):
                frames.setdefault(batch_idx, []).append(frame_path)
    except script_util.ConfigError as error:
        p.error(str(error))
    except RuntimeError as error:
        if not is_out_of_memory(error):
            raise
        advice = (f"Ran out of GPU memory. Try lowering --image_size/-size, --batch_size/-bs or --num_cutouts/-cutn.\n"
                  f"--clip_model/-clip (currently {clip_model}) has a large impact on VRAM: RN50 uses the least, ViT-B/32 the second least.\n")
        if any(member in clip_util.OPEN_CLIP_MODELS for member in clip_util.ensemble_members(clip_model)) and not args.clip_fp16:
            advice += "--clip_fp16/-fp16 halves what this model needs.\n"
        if args.backbone:
            advice += "--iterations/-iters 0 is the plain model, which needs far less than correcting it with CLIP.\n"
        p.exit(1, advice)

    upscaler = None
    if args.upscale and frames:
        device = args.device or ('cuda' if th.cuda.is_available() else 'cpu')
        upscaler = upscale.load_upscaler(device, args.checkpoints_dir)

    for batch_idx, paths in frames.items():
        if upscaler is not None:
            print(f"Saved {upscale.upscale_file(upscaler, paths[-1], device)}")
        run_dir = Path(paths[0]).parent
        outputs = []
        if args.save_as_gif:
            outputs.append(script_util.create_gif_ffmpeg(paths, run_dir.with_suffix('.gif')))
        if args.save_as_video:
            outputs.append(script_util.create_video_ffmpeg(paths, run_dir.with_suffix('.mp4')))
        # The frames are the only copy of the result until every requested output exists.
        if make_animation and all(outputs):
            for frame_path in paths:
                Path(frame_path).unlink()
            if not any(run_dir.iterdir()):
                run_dir.rmdir()
        else:
            print(f"Saved {paths[-1]}")


if __name__ == "__main__":
    main()

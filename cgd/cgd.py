"""CLIP Guided Diffusion: generate images from text and image prompts."""
import argparse
from pathlib import Path

import torch as th
import torch.nn.functional as tf
from PIL import Image
import torchvision.transforms as tvt

from tqdm.auto import tqdm
from cgd import losses
from cgd import clip_util
from cgd import sampling
from cgd import script_util

FAST_NUM_CUTOUTS = 4
FAST_MIN_CUT_FRACTION = 0.7  # Cutouts span 70-100% of the frame.
LEGACY_NUM_CUTOUTS = 16
DEFAULT_SAVE_FREQUENCY = 10
# Guidance backpropagates through the U-Net. Above this many pixels per batch, recompute
# its activations instead of storing them: 1.5x the time per guided step for under half
# the memory, which is what lets 512px run in 8 GB.
GRADIENT_CHECKPOINT_PIXELS = 256 ** 2


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
):
    if sample_steps is None:
        sample_steps = sampling.DEFAULT_STEPS if class_cond else sampling.UNCOND_STEPS
    script_util.check_parameters(
        prompts, image_prompts, image_size, class_cond=class_cond, timestep_respacing=timestep_respacing,
        skip_timesteps=skip_timesteps, save_frequency=save_frequency, sample_steps=sample_steps,
        height_offset=height_offset, width_offset=width_offset, init_image=init_image)
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

    # Load CLIP model/Encode text/Create `MakeCutouts`
    embeds_list = []
    weights_list = []
    clip_model, clip_size = clip_util.load_clip(clip_model_name, device)

    for prompt in prompts:
        text, weight = script_util.parse_prompt(prompt)
        text, weight = clip_util.encode_text_prompt(text, weight, clip_model_name, device)
        embeds_list.append(text)
        weights_list.append(weight)

    for image_prompt in image_prompts:
        img, weight = script_util.parse_prompt(image_prompt)
        image_prompt, batched_weight = clip_util.encode_image_prompt(
            img, weight, image_size, num_cutouts=num_cutouts, clip_model_name=clip_model_name, device=device)
        embeds_list.append(image_prompt)
        weights_list.extend(batched_weight)

    target_embeds = th.cat(embeds_list)

    weights = th.tensor(weights_list, device=device)
    if weights.sum().abs() < 1e-3:  # smart :)
        raise script_util.ConfigError('The prompt weights must not sum to 0.')
    weights /= weights.sum().abs()

    make_cutouts = clip_util.MakeCutouts(cut_size=clip_size, num_cutouts=num_cutouts,
                                         cutout_size_power=cutout_power, use_augs=use_augs,
                                         min_cut_fraction=FAST_MIN_CUT_FRACTION if fast else 1.0)

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
            query = (weights[:, None] * tf.normalize(target_embeds, dim=-1)).sum(0, keepdim=True)
            class_index = clip_util.imagenet_top_n(query, device=device, n=1, clip_model_name=clip_model_name)
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
        x_in = pred_xstart * fac + x * sigmas
        if wandb_project is not None:
            def preview(images, caption):  # The first image of the batch, as wandb wants it.
                return wandb.Image(images[0].detach().add(1).div(2).clamp(0, 1), caption=caption)

            log[f'Generations - {timestep_respacing or sample_steps}'] = [
                preview(x, "Noisy Sample"),
                preview(pred_xstart, "Denoised Prediction"),
                preview(x_in, "Blended (what CLIP sees)"),
            ]

        clip_in = clip_util.CLIP_NORMALIZE(make_cutouts(x_in.add(1).div(2)))
        cutout_embeds = clip_model.encode_image(clip_in).float()
        cutout_embeds = cutout_embeds.view(-1, x.shape[0], cutout_embeds.shape[-1])  # [cutouts, batch, features]
        dists = losses.spherical_dist_loss(cutout_embeds.unsqueeze(2), target_embeds)

        clip_losses = dists.mul(weights).sum(2).mean(0)
        range_losses = losses.range_loss(pred_xstart)
        tv_losses = losses.tv_loss(x_in)

        clip_losses = clip_losses.sum() * clip_guidance_scale
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

        final_loss = -th.autograd.grad(loss, x)[0]  # negative gradient
        if use_magnitude:
            magnitude = final_loss.square().mean().sqrt()  # TODO experimental clamping?
            log["Magnitude"] = magnitude.item()
            final_loss = final_loss * magnitude.clamp(max=0.05) / magnitude
        log['Grad'] = final_loss.mean().item()
        status("\t".join(f"{k}: {v:.3f}" for k, v in log.items() if "loss" in k.lower()))
        report(log)
        return final_loss

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
            init_image=init_tensor,
            start_timestep=start_timestep,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        )

    run_name = script_util.run_name(prompts, image_prompts)
    for step, sample in enumerate(fast_samples() if fast else legacy_samples()):
        if step % save_frequency == 0 or sample["final"]:
            for batch_idx, image_tensor in enumerate(sample["pred_xstart"]):
                yield batch_idx, script_util.log_image(image_tensor, prefix_path, run_name, step, batch_idx)


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
    p.add_argument("--image_size", "-size", type=int, default=128, choices=(64, 128, 256, 512),
                   help="Diffusion image size.")
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
    p.add_argument("--clip_model", "-clip", type=str, default="ViT-B/32",
                   help=f"clip model name. Should be one of: {clip_util.CLIP_MODEL_NAMES} or a checkpoint filename ending in `.pt`")
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
    args = p.parse_args()

    _class_cond = not args.uncond
    prefix_path = args.prefix
    make_animation = args.save_as_gif or args.save_as_video
    save_frequency = args.save_frequency
    if save_frequency is None:
        save_frequency = 1 if make_animation else DEFAULT_SAVE_FREQUENCY

    prompts = [prompt for prompt in f"{args.prompt}|{args.prompts}".split('|') if prompt]
    image_prompts = [image_prompt for image_prompt in args.image_prompts.split('|') if image_prompt]
    cgd_generator = clip_guided_diffusion(
        prompts=prompts,
        image_prompts=image_prompts,
        batch_size=args.batch_size,
        tv_scale=args.tv_scale,
        init_scale=args.init_scale,
        range_scale=args.range_scale,
        sat_scale=args.sat_scale,
        image_size=args.image_size,
        class_cond=_class_cond,
        randomize_class=(_class_cond),
        save_frequency=save_frequency,
        clip_guidance_scale=args.clip_guidance_scale,
        cutout_power=args.cutout_power,
        num_cutouts=args.num_cutouts,
        timestep_respacing=args.timestep_respacing,
        seed=args.seed,
        skip_timesteps=args.skip_timesteps,
        init_image=args.init_image,
        checkpoints_dir=args.checkpoints_dir,
        clip_model_name=args.clip_model,
        device=args.device,
        prefix_path=prefix_path,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        height_offset=args.height_offset,
        width_offset=args.width_offset,
        progress=not args.quiet,
        sample_steps=args.sample_steps,
        guidance_rate=args.guidance_rate,
    )
    frames = {}  # Only this run's frames: the folder may hold older ones for the same prompt.
    try:
        for batch_idx, frame_path in cgd_generator:
            frames.setdefault(batch_idx, []).append(frame_path)
    except script_util.ConfigError as error:
        p.error(str(error))
    except th.cuda.OutOfMemoryError:
        p.exit(1, f"Ran out of GPU memory. Try lowering --image_size/-size, --batch_size/-bs or --num_cutouts/-cutn.\n"
                  f"--clip_model/-clip (currently {args.clip_model}) has a large impact on VRAM: RN50 uses the least, ViT-B/32 the second least.\n")

    for batch_idx, paths in frames.items():
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

import argparse
from pathlib import Path

import lpips
import torch as th
import wandb
from PIL import Image
import torchvision.transforms as tvt

from tqdm.auto import tqdm
from cgd import losses
from cgd import clip_util
from cgd import script_util


# Define necessary functions

def clip_guided_diffusion(
    image_size: int = 128,
    num_cutouts: int = 16,
    prompts: "list[str]" = [],
    image_prompts: "list[str]" = [],
    clip_guidance_scale: int = 1000,
    tv_scale: float = 150,
    range_scale: float = 50,
    sat_scale: float = 0,
    init_scale: float = 0,
    batch_size: int = 1,
    init_image: Path = None,
    class_cond: bool = True,
    cutout_power: float = 1.0,
    timestep_respacing: str = "1000",
    seed: int = 0,
    diffusion_steps: int = 1000,
    skip_timesteps: int = 0,
    checkpoints_dir: str = script_util.CACHE_PATH,
    clip_model_name: str = "ViT-B/32",
    randomize_class: bool = True,
    prefix_path: Path = Path('./outputs'),
    save_frequency: int = 25,
    noise_schedule: str = "linear",
    dropout: float = 0.0,
    device: str = '',
    wandb_project: str = None,
    wandb_entity: str = None,
    use_augs: bool = False, # enables augmentation, mostly better for timesteps <= 100
    use_magnitude: bool = False, # enables magnitude of the gradient
    height_offset: int = 0,
    width_offset: int = 0,
    progress: bool = True,
):
    if len(device) == 0:
        device = 'cuda' if th.cuda.is_available() else 'cpu'
        print(f"Using device {device}. You can specify a device manually with `--device/-dev`")
    else:
        print(f"Using device {device}")
    fp32_diffusion = (device == 'cpu')

    wandb_run = None
    if wandb_project is not None:
        # just use local vars for config
        wandb_run = wandb.init(project=wandb_project, entity=wandb_entity, config=locals())
    else:
        print(f"--wandb_project not specified. Skipping W&B integration.")

    th.manual_seed(seed)

    if use_magnitude == False and image_size == 64:
        use_magnitude = True
        tqdm.write("Enabling magnitude for 64x64 checkpoints.")
    
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
        raise RuntimeError('The weights must not sum to 0.')
    weights /= weights.sum().abs()

    if use_augs: tqdm.write( f"Augmentations enabled." )
    make_cutouts = clip_util.MakeCutouts(cut_size=clip_size, num_cutouts=num_cutouts,
                                         cutout_size_power=cutout_power, use_augs=use_augs)

    # Load initial image (if provided)
    init_tensor = None
    if init_image:
        pil_image = Image.open(script_util.fetch(init_image)).convert('RGB').resize((image_size, image_size))
        init_tensor = tvt.ToTensor()(pil_image)
        init_tensor = init_tensor.to(device).unsqueeze(0).mul(2).sub(1)

   # Class randomization requires a starting class index `y`
    model_kwargs = {}
    if class_cond:
        model_kwargs["y"] = th.zeros(
            [batch_size], device=device, dtype=th.long)

    # Load guided diffusion
    gd_model, diffusion = script_util.load_guided_diffusion(
        checkpoint_path=diffusion_path,
        image_size=image_size, class_cond=class_cond,
        diffusion_steps=diffusion_steps,
        timestep_respacing=timestep_respacing,
        use_fp16=(not fp32_diffusion),
        device=device,
        noise_schedule=noise_schedule,
        dropout=dropout,
    )
    # This is initialized lazily as it can use a bit of VRAM
    if init_tensor is not None and init_scale != 0:
        lpips_vgg = lpips.LPIPS(net='vgg').to(device)
    current_timestep = None

    def cond_fn(x, t, out, y=None):
        log = {}
        n = x.shape[0]
        fac = diffusion.sqrt_one_minus_alphas_cumprod[current_timestep]
        sigmas = 1 - fac
        x_in = out["pred_xstart"] * fac + x * sigmas
        if wandb_project is not None:
            log[f'Generations - {timestep_respacing}'] = [
                wandb.Image(x, caption=f"Noisy Sample"),
                wandb.Image(out['pred_xstart'],
                            caption=f"Denoised Prediction"),
                wandb.Image(x_in, caption=f"Blended (what CLIP sees)"),
            ]

        clip_in = clip_util.CLIP_NORMALIZE(make_cutouts(x_in.add(1).div(2)))
        cutout_embeds = clip_model.encode_image(
            clip_in).float().view([num_cutouts, n, -1])
        dists = losses.spherical_dist_loss(
            cutout_embeds.unsqueeze(0), target_embeds.unsqueeze(0))
        dists = dists.view([num_cutouts, n, -1])

        clip_losses = dists.mul(weights).sum(2).mean(0)
        range_losses = losses.range_loss(out["pred_xstart"])
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

        final_loss = -th.autograd.grad(loss, x)[0]  # negative gradient
        if use_magnitude:
            magnitude = final_loss.square().mean().sqrt()  # TODO experimental clamping?
            log["Magnitude"] = magnitude.item()
            final_loss = final_loss * magnitude.clamp(max=0.05) / magnitude
        log['Grad'] = final_loss.mean().item()
        if progress:
            tqdm.write(
                "\t".join([f"{k}: {v:.3f}" for k, v in log.items() if "loss" in k.lower()]))
        if wandb_project is not None:
            wandb_run.log(log)
        return final_loss

    # Choose between normal or DDIM
    if timestep_respacing.startswith("ddim"):
        diffusion_sample_loop = diffusion.ddim_sample_loop_progressive
    else:
        diffusion_sample_loop = diffusion.p_sample_loop_progressive

    # def denoised_fn(image): return image

    try:
        cgd_samples = diffusion_sample_loop(
            gd_model,
            (batch_size, 3, image_size + height_offset, image_size + width_offset),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            progress=progress,
            skip_timesteps=skip_timesteps,
            init_image=init_tensor,
            randomize_class=randomize_class,
            cond_fn_with_grad=True,
            # denoised_fn=denoised_fn,
        )

        # Gather generator for diffusion
        current_timestep = diffusion.num_timesteps - 1
        for step, sample in enumerate(cgd_samples):
            current_timestep -= 1
            if step % save_frequency == 0 or current_timestep == -1:
                for batch_idx, image_tensor in enumerate(sample["pred_xstart"]):
                    yield batch_idx, script_util.log_image(image_tensor, prefix_path, prompts, step, batch_idx)
                    # if wandb_project is not None: wandb.log({"image": wandb.Image(image_tensor, caption="|".join(prompts))})

        for batch_idx in range(batch_size):
            script_util.create_gif(prefix_path, prompts, batch_idx)

    except (RuntimeError, KeyboardInterrupt) as runtime_ex:
        if "CUDA out of memory" in str(runtime_ex):
            print(f"CUDA OOM error occurred.")
            print(
                f"Try lowering --image_size/-size, --batch_size/-bs, --num_cutouts/-cutn")
            print(
                f"--clip_model/-clip (currently {clip_model_name}) can have a large impact on VRAM usage.")
            print(f"'RN50' will use the least VRAM. 'ViT-B/32' the second least and is good for its memory/runtime constraints.")
        else:
            raise runtime_ex


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--prompts", "-txts", type=str, default='',
                   help="the prompt/s to reward paired with weights. e.g. 'My text:0.5|Other text:-0.5' ")
    p.add_argument("--image_prompts", "-imgs", type=str, default='',
                   help="the image prompt/s to reward paired with weights. e.g. 'img1.png:0.5,img2.png:-0.5'")
    p.add_argument("--image_size", "-size", type=int, default=128,
                   help="Diffusion image size. Must be one of [64, 128, 256, 512].")
    p.add_argument("--init_image", "-init", type=str, default='',
                   help="Blend an image with diffusion for n steps")
    p.add_argument("--init_scale", "-is", type=int, default=0,
                   help="(optional) Perceptual loss scale for init image. ")
    p.add_argument("--skip_timesteps", "-skip", type=int, default=0,
                   help="Number of timesteps to blend image for. CLIP guidance occurs after this.")
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
    p.add_argument("--save_frequency", "-freq", type=int,
                   default=1, help="Save frequency")
    p.add_argument("--diffusion_steps", "-steps", type=int,
                   default=1000, help="Diffusion steps")
    p.add_argument("--timestep_respacing", "-respace", type=str,
                   default="1000", help="Timestep respacing")
    p.add_argument("--num_cutouts", "-cutn", type=int, default=16,
                   help="Number of randomly cut patches to distort from diffusion.")
    p.add_argument("--cutout_power", "-cutpow", type=float,
                   default=1.0, help="Cutout size power")
    p.add_argument("--clip_model", "-clip", type=str, default="ViT-B/32",
                   help=f"clip model name. Should be one of: {clip_util.CLIP_MODEL_NAMES} or a checkpoint filename ending in `.pt`")
    p.add_argument("--uncond", "-uncond", action="store_true",
                   help='Use finetuned unconditional checkpoints from OpenAI (256px) and Katherine Crowson (512px)')
    p.add_argument("--noise_schedule", "-sched", default='linear', type=str,
                   help="Specify noise schedule. Either 'linear' or 'cosine'.")
    p.add_argument("--dropout", "-drop", default=0.0, type=float,
                   help="Amount of dropout to apply. ")
    p.add_argument("--device", "-dev", default='', type=str,
                   help="Device to use. Either cpu or cuda.")
    p.add_argument('--wandb_project', '-proj', default=None,
                   help='Name W&B will use when saving results.\ne.g. `--wandb_project "my_project"`')
    p.add_argument('--wandb_entity', '-ent', default=None,
                   help='(optional) Name of W&B team/entity to log to.')
    p.add_argument('--height_offset', '-ht', default=0, type=int, help='Height offset for image')
    p.add_argument('--width_offset', '-wd', default=0, type=int, help='Width offset for image')
    p.add_argument('--use_augs', '-augs', action='store_true', help="Uses augmentations from the `quick` clip guided diffusion notebook")
    p.add_argument('--use_magnitude', '-mag', action='store_true', help="Uses magnitude of the gradient")
    p.add_argument('--quiet', '-q', action='store_true',
                   help='Suppress output.')
    args = p.parse_args()

    _class_cond = not args.uncond
    prefix_path = args.prefix

    Path(prefix_path).mkdir(exist_ok=True)

    if len(args.prompts) > 0:
        prompts = args.prompts.split('|')
    else:
        prompts = []

    if len(args.image_prompts) > 0:
        image_prompts = args.image_prompts.split('|')
    else:
        image_prompts = []
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
        save_frequency=args.save_frequency,
        clip_guidance_scale=args.clip_guidance_scale,
        cutout_power=args.cutout_power,
        num_cutouts=args.num_cutouts,
        timestep_respacing=args.timestep_respacing,
        seed=args.seed,
        diffusion_steps=args.diffusion_steps,
        skip_timesteps=args.skip_timesteps,
        init_image=args.init_image,
        checkpoints_dir=args.checkpoints_dir,
        clip_model_name=args.clip_model,
        noise_schedule=args.noise_schedule,
        dropout=args.dropout,
        device=args.device,
        prefix_path=prefix_path,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        use_augs=False,
        use_magnitude=False,
        height_offset=args.height_offset,
        width_offset=args.width_offset,
        progress=not args.quiet,
    )
    list(enumerate(cgd_generator))  # iterate over generator


if __name__ == "__main__":
    main()

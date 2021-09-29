import argparse
import os
import sys
import time
from pathlib import Path

import clip
import lpips
import torch as th
import wandb
from PIL import Image
from torch.nn.functional import normalize
from torchvision import transforms as T
from torchvision.transforms import functional as tvf
from torchvision.transforms.transforms import ToTensor
from tqdm.auto import tqdm

from cgd.clip_util import (CLIP_MODEL_NAMES, CLIP_NORMALIZE, MakeCutouts,
                           load_clip)
from cgd.loss_util import spherical_dist_loss, tv_loss
from cgd.util import (CACHE_PATH, create_gif, download_guided_diffusion, fetch,
                      load_guided_diffusion, log_image)

sys.path.append(os.path.join(os.getcwd(), "guided-diffusion"))

TIMESTEP_RESPACINGS = ("25", "50", "100", "250", "500", "1000",
                       "ddim25", "ddim50", "ddim100", "ddim250", "ddim500", "ddim1000")
DIFFUSION_SCHEDULES = (25, 50, 100, 250, 500, 1000)
IMAGE_SIZES = (64, 128, 256, 512)


def check_parameters(
    prompts: list,
    image_prompts: list,
    image_size: int,
    timestep_respacing: str,
    diffusion_steps: int,
    clip_model_name: str,
    save_frequency: int,
    noise_schedule: str,
):
    if not (len(prompts) > 0 or len(image_prompts) > 0):
        raise ValueError("Must provide at least one prompt, text or image.")
    if not (noise_schedule in ['linear', 'cosine']):
        raise ValueError('Noise schedule should be one of: linear, cosine')
    if not (image_size in IMAGE_SIZES):
        raise ValueError(f"--image size should be one of {IMAGE_SIZES}")
    if not (0 < save_frequency <= int(timestep_respacing.replace('ddim', ''))):
        raise ValueError(
            "--save_frequency must be greater than 0 and less than `timestep_respacing`")
    if not (diffusion_steps in DIFFUSION_SCHEDULES):
        print('(warning) Diffusion steps should be one of:', DIFFUSION_SCHEDULES)
    if not (timestep_respacing in TIMESTEP_RESPACINGS):
        print(
            f"Pausing run. `timestep_respacing` should be one of {TIMESTEP_RESPACINGS}. CTRL-C if this was a mistake.")
        time.sleep(5)
        print("Resuming run.")
    if clip_model_name.endswith('.pt') or clip_model_name.endswith('.pth'):
        assert os.path.isfile(
            clip_model_name), f"{clip_model_name} does not exist"
        print(f"Loading custom model from {clip_model_name}")
    elif not (clip_model_name in CLIP_MODEL_NAMES):
        print(
            f"--clip model name should be one of: {CLIP_MODEL_NAMES} unless you are trying to use your own checkpoint.")
        print(f"Loading OpenAI CLIP - {clip_model_name}")


# Define necessary functions

def parse_prompt(prompt):  # parse a single prompt in the form "<text||img_url>:<weight>"
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)  # theres two colons, so we grab the 2nd
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)  # grab weight after colon
    vals = vals + ['', '1'][len(vals):]  # if no weight, use 1
    return vals[0], float(vals[1])  # return text, weight


def encode_text_prompt(txt, weight, clip_model_name="ViT-B/32", device="cpu"):
    clip_model, _ = load_clip(clip_model_name, device)
    txt_tokens = clip.tokenize(txt).to(device)
    txt_encoded = clip_model.encode_text(txt_tokens).float()
    return txt_encoded, weight


def encode_image_prompt(image: str, weight: float, diffusion_size: int, num_cutouts, clip_model_name: str = "ViT-B/32", device: str = "cpu"):
    clip_model, clip_size = load_clip(clip_model_name, device)
    make_cutouts = MakeCutouts(cut_size=clip_size, num_cutouts=num_cutouts)
    pil_img = Image.open(fetch(image)).convert('RGB')
    smallest_side = min(diffusion_size, *pil_img.size)
    # You can ignore the type warning caused by pytorch resize having
    # an incorrect type hint for their resize signature. which does indeed support PIL.Image
    pil_img = tvf.resize(pil_img, [smallest_side],
                         tvf.InterpolationMode.LANCZOS)
    batch = make_cutouts(tvf.to_tensor(pil_img).unsqueeze(0).to(device))
    batch_embed = clip_model.encode_image(normalize(batch)).float()
    batch_weight = [weight / make_cutouts.cutn] * make_cutouts.cutn
    return batch_embed, batch_weight


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])


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
    init_image: str = "",
    class_cond: bool = True,
    cutout_power: float = 1.0,
    timestep_respacing: str = "1000",
    seed: int = 0,
    diffusion_steps: int = 1000,
    skip_timesteps: int = 0,
    checkpoints_dir: str = CACHE_PATH,
    clip_model_name: str = "ViT-B/32",
    randomize_class: bool = True,
    prefix_path: str = 'outputs',
    save_frequency: int = 25,
    noise_schedule: str = "linear",
    dropout: float = 0.0,
    device: str = '',
    wandb_project: str = None,
    wandb_entity: str = None,
    progress: bool = True,
):

    if len(device) == 0:
        device = 'cuda' if th.cuda.is_available() else 'cpu'
        print(
            f"Using device {device}. You can specify a device manually with `--device/-dev`")
    else:
        print(f"Using device {device}")
    fp32_diffusion = (device == 'cpu')

    if wandb_project is not None:
        # just use local vars for config
        wandb_run = wandb.init(project=wandb_project, entity=wandb_entity, config=locals())

    if seed:
        th.manual_seed(seed)

    # only use magnitude for low timestep_respacing
    use_magnitude = (int(timestep_respacing.replace("ddim", "")) <= 25 or image_size == 64)
    # only use saturation loss on ddim
    use_saturation = ("ddim" in timestep_respacing or image_size == 64)

    Path(prefix_path).mkdir(parents=True, exist_ok=True)
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)

    diffusion_path = download_guided_diffusion(
        image_size=image_size, checkpoints_dir=checkpoints_dir, class_cond=class_cond)

    # Load CLIP model/Encode text/Create `MakeCutouts`
    embeds_list = []
    weights_list = []
    clip_model, clip_size = load_clip(clip_model_name, device)

    for prompt in prompts:
        text, weight = parse_prompt(prompt)
        text, weight = encode_text_prompt(
            text, weight, clip_model_name, device)
        embeds_list.append(text)
        weights_list.append(weight)

    for image_prompt in image_prompts:
        img, weight = parse_prompt(image_prompt)
        image_prompt, batched_weight = encode_image_prompt(
            img, weight, image_size, num_cutouts=num_cutouts, clip_model_name=clip_model_name, device=device)
        embeds_list.append(image_prompt)
        weights_list.extend(batched_weight)

    target_embeds = th.cat(embeds_list)

    weights = th.tensor(weights_list, device=device)
    if weights.sum().abs() < 1e-3:  # smart :)
        raise RuntimeError('The weights must not sum to 0.')
    weights /= weights.sum().abs()

    # Add noise, translation, affine, etc. if under 100 diffusion steps
    use_augs = (int(timestep_respacing.replace("ddim", "")) <= 100)
    if use_augs:
        tqdm.write(
            f"Using augmentations to improve performance for lower timestep_respacing of {timestep_respacing}")
    make_cutouts = MakeCutouts(cut_size=clip_size, num_cutouts=num_cutouts,
                               cutout_size_power=cutout_power, use_augs=use_augs)

    # Load initial image (if provided)
    init_tensor = None
    if len(init_image) > 0:
        pil_image = Image.open(fetch(init_image)).convert(
            "RGB").resize((image_size, image_size), Image.LANCZOS)
        init_tensor = ToTensor()(pil_image).to(device).unsqueeze(0).mul(2).sub(1)

   # Class randomization requires a starting class index `y`
    model_kwargs = {}
    if class_cond:
        model_kwargs["y"] = th.zeros(
            [batch_size], device=device, dtype=th.long)

    # Load guided diffusion
    gd_model, diffusion = load_guided_diffusion(
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
        wandb_run = None
        if wandb_project is not None:
            log['Generations'] = [
                wandb.Image(x, caption=f"Noisy Sample"),
                wandb.Image(out['pred_xstart'],
                            caption=f"Denoised Prediction"),
                wandb.Image(x_in, caption=f"Blended (what CLIP sees)"),
            ]

        clip_in = CLIP_NORMALIZE(make_cutouts(x_in.add(1).div(2)))
        cutout_embeds = clip_model.encode_image(
            clip_in).float().view([num_cutouts, n, -1])
        dists = spherical_dist_loss(
            cutout_embeds.unsqueeze(0), target_embeds.unsqueeze(0))
        dists = dists.view([num_cutouts, n, -1])

        clip_losses = dists.mul(weights).sum(2).mean(0)
        range_losses = range_loss(out["pred_xstart"])
        tv_losses = tv_loss(x_in)

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
        if wandb_run is not None:
            wandb_run.log(log)
        return final_loss

    # Choose between normal or DDIM
    if timestep_respacing.startswith("ddim"):
        diffusion_sample_loop = diffusion.ddim_sample_loop_progressive
    else:
        diffusion_sample_loop = diffusion.p_sample_loop_progressive

    try:
        cgd_samples = diffusion_sample_loop(
            gd_model,
            (batch_size, 3, image_size, image_size),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            progress=progress,
            skip_timesteps=skip_timesteps,
            init_image=init_tensor,
            randomize_class=randomize_class,
            cond_fn_with_grad=True,
        )

        # Gather generator for diffusion
        current_timestep = diffusion.num_timesteps - 1
        for step, sample in enumerate(cgd_samples):
            current_timestep -= 1
            if step % save_frequency == 0 or current_timestep == -1:
                for batch_idx, image_tensor in enumerate(sample["pred_xstart"]):
                    yield batch_idx, log_image(image_tensor, prefix_path, prompts, step, batch_idx)
                    # if wandb_project is not None: wandb.log({"image": wandb.Image(image_tensor, caption="|".join(prompts))})

        for batch_idx in range(batch_size):
            create_gif(prefix_path, prompts, batch_idx)

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
    p.add_argument("--checkpoints_dir", "-ckpts", default=CACHE_PATH,
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
                   help=f"clip model name. Should be one of: {CLIP_MODEL_NAMES} or a checkpoint filename ending in `.pt`")
    p.add_argument("--uncond", "-uncond", action="store_true",
                   help='Use finetuned unconditional checkpoints from OpenAI (256px) and Katherine Crowson (512px)')
    p.add_argument("--noise_schedule", "-sched", default='linear', type=str,
                   help="Specify noise schedule. Either 'linear' or 'cosine'.")
    p.add_argument("--dropout", "-drop", default=0.0, type=float,
                   help="Amount of dropout to apply. ")
    p.add_argument("--device", "-dev", default='', type=str,
                   help="Device to use. Either cpu or cuda.")
    p.add_argument('--wandb_project', '-proj', default=None,
                   help='Name W&B will use when saving results.\ne.g. `--wandb_name "my_project"`')
    p.add_argument('--wandb_entity', '-ent', default=None,
                   help='(optional) Name of W&B team/entity to log to.')
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
        progress=not args.quiet,
    )
    list(enumerate(cgd_generator))  # iterate over generator


if __name__ == "__main__":
    main()
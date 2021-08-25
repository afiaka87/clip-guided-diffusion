import argparse
import time
import glob
import re
from pathlib import Path

from data.imagenet1000_clsidx_to_labels import IMAGENET_CLASSES

import torch as th
from PIL import Image
from torchvision.transforms.transforms import ToTensor
from tqdm.auto import tqdm, trange

from cgd.clip_util import (CLIP_NORMALIZE, MakeCutouts, clip_encode_text, imagenet_top_n,
                           load_clip)
from cgd.util import (ALPHANUMERIC_REGEX, CACHE_PATH, alphanumeric_filter, create_gif,
                      download_guided_diffusion, fetch, load_guided_diffusion,
                      log_image, spherical_dist_loss, tv_loss)

import sys
import os
sys.path.append(os.path.join(os.getcwd(), "guided-diffusion"))

TIMESTEP_RESPACINGS = ("25", "50", "100", "250", "500", "1000",
                       "ddim25", "ddim50", "ddim100", "ddim250", "ddim500", "ddim1000")
DIFFUSION_SCHEDULES = (25, 50, 100, 250, 500, 1000)
IMAGE_SIZES = (64, 128, 256, 512)
CLIP_MODEL_NAMES = ("ViT-B/16", "ViT-B/32", "RN50",
                    "RN101", "RN50x4", "RN50x16")


def check_parameters(
    prompt: str,
    top_n: int,
    image_size: int,
    class_score: bool,
    timestep_respacing: str,
    diffusion_steps: int,
    skip_timesteps: int,
    init_image: str,
    clip_model_name: str,
    save_frequency: int,
    noise_schedule: str,
):
    if class_score: # TODO re-enable after class_score is fixed
        raise ValueError('Class scoring is currently a work-in-progress feature. It is unclear if the current implementation is working. Sorry for the confusion; please disable for now.')
    if not (diffusion_steps in DIFFUSION_SCHEDULES):
        print('(warning) Diffusion steps should be one of:', DIFFUSION_SCHEDULES)
    if not (noise_schedule in ['linear', 'cosine']):
        raise ValueError('Noise schedule should be one of: linear, cosine')
    if not (clip_model_name in CLIP_MODEL_NAMES):
        raise ValueError(f"clip model name should be one of: {CLIP_MODEL_NAMES}")
    if not (image_size in IMAGE_SIZES):
        raise ValueError(f"image size should be one of {IMAGE_SIZES}")
    if not (len(prompt) > 0):
        raise ValueError("prompt/-txt cant be empty")
    if not (0 < top_n <= len(IMAGENET_CLASSES)):
        raise ValueError(f"top_n must be less than or equal to the number of classes: {top_n} > {len(IMAGENET_CLASSES)}")
    if not (0 < save_frequency <= int(timestep_respacing.replace('ddim', ''))):
        raise ValueError("`save_frequency` must be greater than 0 and less than `timestep_respacing`")
    if len(init_image) > 0 and skip_timesteps != 0:
        raise ValueError("skip_timesteps/-skip must be greater than 0 when using init_image")
    if not (timestep_respacing in TIMESTEP_RESPACINGS):
        print(f"Pausing run. `timestep_respacing` should be one of {TIMESTEP_RESPACINGS}. CTRL-C if this was a mistake.")
        time.sleep(5)


def clip_guided_diffusion(
    prompt: str = '',
    prompt_min: str = '',
    min_weight: float = 0.1,
    batch_size: int = 1,
    tv_scale: float = 100,
    image_size: int = 128,
    class_cond: bool = True,
    clip_guidance_scale: int = 1000,
    cutout_power: float = 1.0,
    num_cutouts: int = 16,
    timestep_respacing: str = "1000",
    seed: int = 0,
    diffusion_steps: int = 1000,
    skip_timesteps: int = 0,
    init_image: str = "",
    checkpoints_dir: str = CACHE_PATH,
    clip_model_name: str = "ViT-B/32",
    augs: list = [],
    randomize_class: bool = True,
    prefix_path: str = 'outputs',
    save_frequency: int = 1,
    noise_schedule: str = "linear",
    dropout: float = 0.0,
    device: str = 'cuda',
):
    if seed:
        th.manual_seed(seed)

    Path(prefix_path).mkdir(parents=True, exist_ok=True)
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)
    if (fp32_diffusion := device == 'cpu'):
        print("Enabling FP32 mode for CPU compatibility.")

    diffusion_path = download_guided_diffusion(image_size=image_size, checkpoints_dir=checkpoints_dir, class_cond=class_cond)

    # Load CLIP model/Encode text/Create `MakeCutouts`
    clip_model, clip_size = load_clip(clip_model_name, device)
    make_cutouts = MakeCutouts(cut_size=clip_size, num_cutouts=num_cutouts, cutout_size_power=cutout_power, augment_list=augs)
    text_embed = clip_encode_text(clip_model_name, prompt, device)
    penalty_text_embed = clip_encode_text(clip_model_name, prompt_min, device)

    # Load initial image (if provided)
    init_tensor = None
    if len(init_image) > 0:
        pil_image = Image.open(fetch(init_image)).convert(
            "RGB").resize((image_size, image_size), Image.LANCZOS)
        init_tensor = ToTensor()(pil_image).to(device).unsqueeze(0).mul(2).sub(1)

   # Class randomization requires a starting class index `y`
    model_kwargs = {}
    if class_cond:
        model_kwargs["y"] = th.zeros([batch_size], device=device, dtype=th.long)

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

    current_timestep = None
    def cond_fn(x, t, y=None):
        with th.enable_grad():
            x = x.detach().requires_grad_()
            n = x.shape[0]
            my_t = th.ones([n], device=device, dtype=th.long) * \
                current_timestep
            out = diffusion.p_mean_variance(
                gd_model, x, my_t, clip_denoised=False, model_kwargs={"y": y})
            fac = diffusion.sqrt_one_minus_alphas_cumprod[current_timestep]
            # Blend denoised prediction with noisey sample
            x_in = out["pred_xstart"] * fac + x * (1 - fac)
            clip_in = CLIP_NORMALIZE(make_cutouts(x_in.add(1).div(2)))
            cutout_embeds = clip_model.encode_image(
                clip_in).float().view([num_cutouts, n, -1])
            max_dists = spherical_dist_loss(
                cutout_embeds, text_embed.unsqueeze(0))
            if len(prompt_min) > 0:  # Implicit comparison to None is not supported by pytorch tensors
                min_dists = spherical_dist_loss(
                    cutout_embeds, penalty_text_embed.unsqueeze(0))
                dists = max_dists - (min_weight * min_dists)
            else:
                dists = max_dists
            losses = dists.mean(0)
            tv_losses = tv_loss(x_in)
            loss = losses.sum() * clip_guidance_scale + tv_losses.sum() * tv_scale
            final_loss = -th.autograd.grad(loss, x)[0]
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
            progress=False,
            skip_timesteps=skip_timesteps,
            init_image=init_tensor,
            clip_scores=None,
            randomize_class=randomize_class,
        )
        # Gather generator for diffusion
        current_timestep = diffusion.num_timesteps - 1
        for step, sample in enumerate(cgd_samples):
            current_timestep -= 1
            if step % save_frequency == 0 or current_timestep == -1:
                for batch_idx, image_tensor in enumerate(sample["pred_xstart"]):
                    yield batch_idx, log_image(image_tensor, prefix_path, prompt, prompt_min, step, batch_idx)
        for batch_idx in range(batch_size):
            create_gif(prefix_path, prompt, prompt_min, batch_idx)

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
    p.add_argument("--prompt", "-txt", type=str,
                   default='', help="the prompt to reward")
    p.add_argument("--prompt_min", "-min", type=str,
                   default="", help="the prompt to penalize")
    p.add_argument("--min_weight", "-min_wt", type=float,
                   default=0.1, help="the prompt to penalize")
    p.add_argument("--image_size", "-size", type=int, default=128,
                   help="Diffusion image size. Must be one of [64, 128, 256, 512].")
    p.add_argument("--init_image", "-init", type=str, default='',
                   help="Blend an image with diffusion for n steps")
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
                   default=100, help="Scale for denoising loss",)
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
                   default=0.5, help="Cutout size power")
    p.add_argument("--clip_model", "-clip", type=str, default="ViT-B/32",
                   help=f"clip model name. Should be one of: {CLIP_MODEL_NAMES}")
    p.add_argument("--uncond", "-uncond", action="store_true",
                   help='Use finetuned unconditional checkpoints from OpenAI (256px) and Katherine Crowson (512px)')
    p.add_argument("--noise_schedule", "-sched", default='linear', type=str,
                   help="Specify noise schedule. Either 'linear' or 'cosine'.")
    p.add_argument("--dropout", "-drop", default=0.0, type=float,
                   help="Specify noise schedule. Either 'linear' or 'cosine'.")

    args = p.parse_args()

    _class_cond = not args.uncond
    prefix_path = args.prefix

    Path(prefix_path).mkdir(exist_ok=True)

    cgd_generator = clip_guided_diffusion(
        prompt=args.prompt,
        prompt_min=args.prompt_min,
        min_weight=args.min_weight,
        batch_size=args.batch_size,
        tv_scale=args.tv_scale,
        image_size=args.image_size,
        class_cond=_class_cond,
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
        randomize_class=(_class_cond),
        noise_schedule=args.noise_schedule,
        dropout=args.dropout,
        augs=[]
    )
    prefix_path.mkdir(exist_ok=True)
    list(enumerate(tqdm(cgd_generator))) # iterate over generator
    for batch_idx in range(args.batch_size):
        create_gif(base=prefix_path,prompt=args.prompt,prompt_min=args.prompt_min,batch_idx=batch_idx)


if __name__ == "__main__":
    main()
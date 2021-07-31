import argparse
import os
import sys
from pathlib import Path

from IPython import display
from PIL import Image

sys.path.append("./guided-diffusion")
import clip
import torch
from guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from kornia import augmentation as K
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm

# Model settings

def load_guided_diffusion(
    diffusion_steps=None,
    timestep_respacing=None,
    device=None,
    class_cond=False,
    rescale_timesteps=True,
):
    assert device is not None, "device must be set"
    model_config = model_and_diffusion_defaults()
    model_config.update({
        "attention_resolutions": "32, 16, 8",
        "class_cond": class_cond,
        "diffusion_steps": diffusion_steps,
        "rescale_timesteps": rescale_timesteps,
        "timestep_respacing": timestep_respacing,
        "image_size": 256,
        "learn_sigma": True,
        "noise_schedule": "linear",
        "num_channels": 256,
        "num_head_channels": 64,
        "num_res_blocks": 2,
        "resblock_updown": True,
        "use_fp16": True,
        "use_scale_shift_norm": True,
    })
    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(torch.load("checkpoints/256x256_diffusion_uncond.pt", map_location="cpu"))
    model.requires_grad_(False).eval().to(device)
    for name, param in model.named_parameters():
        if "qkv" in name or "norm" in name or "proj" in name:
            param.requires_grad_()
    if model_config["use_fp16"]:
        model.convert_to_fp16()
    return model, diffusion


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, num_cutouts, cutout_size_power=1.0, augment_list=[]):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = num_cutouts
        self.cut_pow = cutout_size_power
        self.augs = nn.Sequential(*augment_list)

    def forward(self, input):
        side_x, side_y = input.shape[2:4]
        max_size = min(side_y, side_x)
        min_size = min(side_y, side_x, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(
                torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size
            )
            offsetx = torch.randint(0, side_y - size + 1, ())
            offsety = torch.randint(0, side_x - size + 1, ())
            cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
            cutout = F.interpolate(
                cutout,
                (self.cut_size, self.cut_size),
                mode="bilinear",
                align_corners=False,
            )
            cutouts.append(cutout)
        return self.augs(torch.cat(cutouts))


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), "replicate")
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff ** 2 + y_diff ** 2).mean([1, 2, 3])


"""
[Generate an image from a specified text prompt.]
"""
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("prompt", type=str, help="the prompt")
    p.add_argument("--num_cutouts", "--cutn", type=int, default=8, help="Number of randomly cut patches to distort from diffusion.")
    p.add_argument("--prefix", "--output_dir", default="outputs", type=str, help="output directory")
    p.add_argument("--batch_size", "-bs", type=int, default=1, help="the batch size")
    p.add_argument("--clip_guidance_scale", "-cgs", type=int, default=500, help="clip guidance scale.",)
    p.add_argument("--tv_scale", "-tvs", type=int, default=100, help="tv scale")
    p.add_argument("--seed", type=int, default=0, help="random number seed")
    p.add_argument("--save_frequency", "-sf", type=int, default=100, help="save frequency")
    p.add_argument("--device", type=str, help="device")
    p.add_argument("--diffusion_steps", type=int, default=1000, help="diffusion steps")
    p.add_argument("--timestep_respacing", type=str, default='250', help="timestep respacing")
    p.add_argument('--cutout_power', '--cutpow', type=float, default=1.0, help='cutout size power')
    p.add_argument('--clip_model', type=str, default='ViT-B/16', help='clip model name. Should be one of: [ViT-B/16, ViT-B/32, RN50, RN101, RN50x4, RN50x16]')
    # p.add_argument("--image_size", type=int, default=256, help="image size") TODO - image size only works @ 256; need to fix
    args = p.parse_args()

    # Initialize
    prompt = args.prompt 
    batch_size = args.batch_size
    clip_guidance_scale = args.clip_guidance_scale
    seed = args.seed
    save_frequency = args.save_frequency
    cutout_power = args.cutout_power
    num_cutouts = args.num_cutouts
    image_size = 256 # TODO - support other image sizes

    prefix = args.prefix
    prefix_path = Path(prefix)
    os.makedirs(prefix_path, exist_ok=True)

    diffusion_steps = args.diffusion_steps
    timestep_respacing = args.timestep_respacing
    assert timestep_respacing in ['25', '50', '100', '250', '500', '1000', 'ddim25', 'ddim50', 'ddim100', 'ddim250', 'ddim500', 'ddim1000'], 'timestep_respacing should be one of [25, 50, 100, 250, 500, 1000, ddim25, ddim50, ddim100, ddim250, ddim500, ddim1000]'

    tv_scale = args.tv_scale
    clip_model_name = args.clip_model
    assert clip_model_name in ['ViT-B/16', 'ViT-B/32', 'RN50', 'RN101', 'RN50x4', 'RN50x16'], 'clip model name should be one of: [ViT-B/16, ViT-B/32, RN50, RN101, RN50x4, RN50x16]'

    if args.device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
    else:
        print(f"Using user-specified device {args.device}.")
        device = torch.device(args.device)

    # Setup
    if seed is not None:
        torch.manual_seed(seed)

    # Load guided-diffusion model
    gd_model, diffusion = load_guided_diffusion(
        diffusion_steps=diffusion_steps,
        timestep_respacing=timestep_respacing,
        device=device,
        class_cond=False,
        rescale_timesteps=True,
    )
    # Load CLIP model
    clip_model = (
        clip.load(clip_model_name, jit=False)[0].eval().requires_grad_(False).to(device)
    )
    clip_size = clip_model.visual.input_resolution
    # Normalize applied to images before going into the CLIP model
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    # Embed text with CLIP model
    text_embed = clip_model.encode_text(clip.tokenize(prompt).to(device)).float()

    # Load MakeCutouts model for generating random cutouts from diffusion,
    # embedding with CLIP model, and comparing with text embedding
    # optionally apply some augments to the cutouts.
    make_cutouts = MakeCutouts(
        clip_size,
        num_cutouts,
        cutout_size_power=cutout_power,
        augment_list=[
            # K.RandomHorizontalFlip(p=0.2),
            # K.RandomVerticalFlip(p=0.5),
            # K.RandomElasticTransform(p=1.0),
            # K.RandomGaussianNoise(mean=0.4, std=0.2, p=0.5),
            # K.RandomPerspective(distortion_scale=0.1, p=0.5),
            # K.RandomMotionBlur(3, 15, 0.5, p=0.25),
            # K.RandomThinPlateSpline(p=0.25),
            # K.RandomSharpness(p=0.25),
            # K.RandomChannelShuffle(p=0.25),
            # K.RandomGrayscale(p=0.25),
            # K.RandomAffine(degrees=15, p=0.5, padding_mode="border"),
            K.RandomErasing((0.1, 0.4), (0.3, 1 / 0.3), same_on_batch=True, p=0.5),
            # K.RandomSolarize(0.01, 0.01, p=0.25),
        ]
    )

    # Customize guided-diffusion model with function that uses CLIP guidance.
    def cond_fn(x, t, y=None):
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            n = x.shape[0]
            my_t = torch.ones([n], device=device, dtype=torch.long) * current_timestep
            out = diffusion.p_mean_variance(
                gd_model, x, my_t, clip_denoised=False, model_kwargs={"y": y}
            )
            fac = diffusion.sqrt_one_minus_alphas_cumprod[current_timestep]
            x_in = out["pred_xstart"] * fac + x * (1 - fac)
            clip_in = normalize(make_cutouts(x_in.add(1).div(2)))
            image_embeds = (
                clip_model.encode_image(clip_in).float().view([num_cutouts, n, -1])
            )
            dists = spherical_dist_loss(image_embeds, text_embed.unsqueeze(0))
            losses = dists.mean(0)
            tv_losses = tv_loss(x_in)
            loss = losses.sum() * clip_guidance_scale + tv_losses.sum() * tv_scale
            return -torch.autograd.grad(loss, x)[0]

    if timestep_respacing.startswith("ddim"):
        diffusion_sample_loop = diffusion.ddim_sample_loop_progressive
    else:
        diffusion_sample_loop = diffusion.p_sample_loop_progressive

    samples = diffusion_sample_loop(
        gd_model,
        (batch_size, 3, image_size, image_size),
        clip_denoised=False,
        cond_fn=cond_fn,
        progress=True,
        # Pass in {'y': imagenet_class_labels_idx} to sample from a specific class
        model_kwargs={},
    )

    print(f"Attempting to generate the caption:")
    print(prompt)
    try:
        current_timestep = diffusion.num_timesteps - 1
        for step, sample in enumerate(samples):
            current_timestep -= 1
            if step % save_frequency == 0 or current_timestep == -1:
                for j, image in enumerate(sample["pred_xstart"]):
                    filename = os.path.join(
                        prefix_path, f"batch_idx_{j:05}_iteration_{step}.png"
                    )
                    TF.to_pil_image(image.add(1).div(2).clamp(0, 1)).save(filename)
                    tqdm.write(f"Step {step}, output {j}:")
                    display.display(display.Image(filename))
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"CUDA OOM error occurred. Lower the batch_size or num_cutouts and try again.")
        else:
            raise e


if __name__ == "__main__":
    main()

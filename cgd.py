import argparse
import os
import re
import sys
from pathlib import Path

from guided_diffusion.nn import checkpoint
from PIL import Image

from torch_util import (IMAGENET_CLASSES, spherical_dist_loss, tv_loss)
from util import fetch, load_guided_diffusion

sys.path.append("./guided-diffusion")
import clip
import torch
from torch import clip_, nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm


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

# (edit by afiaka87)
# 
# Compare `prompt` with 1000 imagenet label transcriptions for its classes.
# `imagenet_classes` list taken from https://github.com/openai/CLIP/notebooks/

def top_imagenet_class(target_text, clip_model=None, device=None, top:int=16):
    target_text = clip.tokenize(target_text).to(device)
    text_tokenized = clip.tokenize(IMAGENET_CLASSES).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokenized).float()
        target_text_features = clip_model.encode_text(target_text).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)
    target_text_features /= target_text_features.norm(dim=-1, keepdim=True)
    text_probs = (100.0 * target_text_features @ text_features.T).softmax(dim=-1)
    sorted_probs, sorted_classes = text_probs.cpu().topk(top, dim=-1, sorted=True)
    return sorted_classes[0][0]

clip_guidance_scale_dict = {64: 5, 128: 750, 256: 1000, 512: 1500}
tv_scale_dict = {64: 0, 128: 100, 256: 100, 512: 150}
"""
[Generate an image from a specified text prompt.]
"""
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("prompt", type=str, help="the prompt")
    p.add_argument("--image_size", type=int, default=256, help="Diffusion image size. Must be one of [64, 128, 256, 512].")
    p.add_argument("--init_image", type=str, help="Blend an image with diffusion for n steps")
    p.add_argument('--skip_timesteps', type=int, default=0, help='Number of timesteps to blend image for. CLIP guidance occurs after this.')
    p.add_argument("--num_cutouts", "-cutn", type=int, default=8, help="Number of randomly cut patches to distort from diffusion.")
    p.add_argument("--prefix", "--output_dir", default="outputs", type=str, help="output directory")
    p.add_argument("--batch_size", "-bs", type=int, default=1, help="the batch size")
    p.add_argument("--clip_guidance_scale", "-cgs", type=int, default=1000, help="Scale for CLIP spherical distance loss. Default value varies depending on image size.")
    p.add_argument("--tv_scale", "-tvs", type=int, default=100, help="Scale for denoising loss. Disabled by default for 64 and 128")
    p.add_argument("--seed", type=int, default=0, help="Random number seed")
    p.add_argument("--save_frequency", "-sf", type=int, default=25, help="Save frequency")
    p.add_argument("--device", type=str, help="device to run on .e.g. cuda:0 or cpu")
    p.add_argument("--diffusion_steps", type=int, default=1000, help="Diffusion steps")
    p.add_argument("--timestep_respacing", type=str, default='1000', help="Timestep respacing")
    p.add_argument('--cutout_power', '-cutpow', type=float, default=0.5, help='Cutout size power')
    p.add_argument('--clip_model', type=str, default='ViT-B/32', help='clip model name. Should be one of: [ViT-B/16, ViT-B/32, RN50, RN101, RN50x4, RN50x16]')
    p.add_argument('--class_cond', type=bool, default=True, help='Use class conditional. Required for image sizes other than 256')
    p.add_argument('--clip_class_search', action='store_true', help='Lookup imagenet class with CLIP rather than changing them throughout run. Use `--clip_class_search` on its own to enable. ')
    args = p.parse_args()

    # Initialize
    prompt = args.prompt 
    clip_guidance_scale = args.clip_guidance_scale
    init_image = args.init_image
    skip_timesteps = args.skip_timesteps
    image_size = args.image_size
    batch_size = args.batch_size
    seed = args.seed
    save_frequency = args.save_frequency
    cutout_power = args.cutout_power
    num_cutouts = args.num_cutouts
    class_cond = args.class_cond
    diffusion_steps = args.diffusion_steps
    timestep_respacing = args.timestep_respacing
    tv_scale = args.tv_scale
    clip_model_name = args.clip_model
    clip_class_search = args.clip_class_search
    # Disable class randomize from RiversHaveWings if we are passing a class in.
    randomize_class = not clip_class_search

    prefix = args.prefix

    if args.device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
    else:
        print(f"Using user-specified device {args.device}.")
        device = torch.device(args.device)

    assert clip_model_name in [
        'ViT-B/16', 'ViT-B/32', 'RN50', 'RN101', 'RN50x4', 'RN50x16'
    ], 'clip model name should be one of: [ViT-B/16, ViT-B/32, RN50, RN101, RN50x4, RN50x16]'
    assert timestep_respacing in [
        '25', '50', '100', '250', '500', '1000', 'ddim25', 'ddim50', 'ddim100', 'ddim250', 'ddim500', 'ddim1000'
    ], 'timestep_respacing should be one of [25, 50, 100, 250, 500, 1000, ddim25, ddim50, ddim100, ddim250, ddim500, ddim1000]'
    assert image_size in [ 64, 128, 256, 512 ], 'image size should be one of [64, 128, 256, 512]'

    # Setup
    prompt_as_subdir = re.sub(r'[^\w\s]', '', prompt).replace(' ', '_')[:100] # Remove non-alphabet characters
    prefix_path = Path(f'{prefix}/{prompt_as_subdir}')
    os.makedirs(prefix_path, exist_ok=True)

    if image_size == 64 and clip_guidance_scale > 500:
        clip_guidance_scale = clip_guidance_scale_dict[image_size]
        tv_scale = tv_scale_dict[image_size]
        print("CLIP guidance scale and TV scale too high for 64x64 image. Overriding.")
        print("Setting TV scale to {}".format(tv_scale))
        print("Setting CLIP guidance scale to {}".format(clip_guidance_scale))

    if args.class_cond:
        diffusion_path = f'./checkpoints/{image_size}x{image_size}_diffusion.pt'
    else:
        diffusion_path = f'checkpoints/256x256_diffusion_uncond.pt'

    if seed is not None:
        torch.manual_seed(seed)
    # Load guided-diffusion model
    gd_model, diffusion = load_guided_diffusion(checkpoint_path=diffusion_path, image_size=image_size, diffusion_steps=diffusion_steps, timestep_respacing=timestep_respacing, device=device, class_cond=class_cond)
    # Load CLIP model
    clip_model = clip.load(clip_model_name, jit=False)[0].eval().requires_grad_(False).to(device)
    clip_size = clip_model.visual.input_resolution
    # Normalize applied to images before going into the CLIP model
    normalize = transforms.Normalize( mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

    # Random or CLIP-selected imagenet class.
    model_kwargs = {}
    if not clip_class_search:
        model_kwargs["y"] = torch.zeros([batch_size], device=device, dtype=torch.long)
        print("Randomizing class as generation occurs.")
    else:
        imagenet_class = top_imagenet_class(prompt, clip_model=clip_model, device=device)
        model_kwargs["y"] = torch.Tensor([imagenet_class]).to(torch.long).to(device)
        print(f"The prompt: '{prompt}'")
        print(f"Scored well with the ImageNet class idx: '{imagenet_class.item()}'")
        print(f"This index roughly maps to the label:")
        print(IMAGENET_CLASSES[imagenet_class])
    make_cutouts = MakeCutouts(clip_size, num_cutouts, cutout_size_power=cutout_power, augment_list=[])
    # Embed text with CLIP model
    text_embed = clip_model.encode_text(clip.tokenize(prompt).to(device)).float()

    init = None
    if init_image is not None:
        init = Image.open(fetch(init_image)).convert('RGB')
        init = init.resize((image_size, image_size), Image.LANCZOS)
        init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)

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
            cutout_embeds = clip_model.encode_image(clip_in).float().view([num_cutouts, n, -1])
            dists = spherical_dist_loss(cutout_embeds, text_embed.unsqueeze(0))
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
        model_kwargs=model_kwargs,
        cond_fn=cond_fn,
        progress=True,
        skip_timesteps=skip_timesteps,
        init_image=init,
        randomize_class=randomize_class,
    )

    print(f"Attempting to generate the caption: '{prompt}'")
    print(f"Using initial image: {init_image}")
    print(f"Using {num_cutouts} cutouts.")
    print(f"Using {timestep_respacing} iterations.")
    print(f"Using {clip_guidance_scale} for text clip guidance scale.")
    print(f"Using {tv_scale} for denoising loss.")
    try:
        current_timestep = diffusion.num_timesteps - 1
        for step, sample in enumerate(samples):
            current_timestep -= 1
            if step % save_frequency == 0 or current_timestep == -1:
                for j, image in enumerate(sample["pred_xstart"]):
                    filename = os.path.join(
                        prefix_path, f"{j:04}_iteration_{step:04}.png"
                    )
                    TF.to_pil_image(image.add(1).div(2).clamp(0, 1)).save(filename)
                    tqdm.write(f"Step {step}, output {j}:")
                    # display.display(display.Image(filename))
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"CUDA OOM error occurred. Lower the batch_size or num_cutouts and try again.")
        else:
            raise e


if __name__ == "__main__":
    main()

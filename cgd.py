import argparse
from data.imagenet1000_clsidx_to_labels import IMAGENET_CLASSES
import kornia
import kornia.augmentation as kaug
import os
import re
import sys
import time
from functools import lru_cache
from pathlib import Path
from kornia.augmentation.augmentation import RandomBoxBlur

from kornia.augmentation.container import augment

import clip
import torch as th
from guided_diffusion.nn import checkpoint
from PIL import Image
from torch import clip_, nn
from torch.distributions import multinomial
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms.transforms import RandomAffine, RandomHorizontalFlip, RandomVerticalFlip
from tqdm import tqdm

from torch_util import spherical_dist_loss, tv_loss
from util import fetch, load_guided_diffusion


sys.path.append("./guided-diffusion")

TIMESTEP_RESPACINGS = ["25", "50", "100", "250", "500", "1000", "ddim25", "ddim50", "ddim100", "ddim250", "ddim500", "ddim1000",]
IMAGE_SIZES = [ 64, 128, 256, 512, ]
CLIP_MODEL_NAMES = ["ViT-B/16", "ViT-B/32", "RN50", "RN101", "RN50x4", "RN50x16",]

def log_image(image, prefix_path, current_step, batch_idx):
    filename = os.path.join(prefix_path, f"{batch_idx:04}_iteration_{current_step:04}.png")
    pil_image = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
    pil_image.save(filename)
    pil_image.save("current.png")


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
            size = int(th.rand([]) ** self.cut_pow *
                       (max_size - min_size) + min_size)
            offsetx = th.randint(0, side_y - size + 1, ())
            offsety = th.randint(0, side_x - size + 1, ())
            cutout = input[:, :, offsety: offsety +
                size, offsetx: offsetx + size]
            cutout = F.interpolate(
                cutout,
                (self.cut_size, self.cut_size),
                mode="bilinear",
                align_corners=False,
            )
            cutouts.append(cutout)
        return self.augs(th.cat(cutouts))


# - Update (afiaka87): CLIP score against imagenet transcribed classes. 
@lru_cache(maxsize=None)
def imagenet_top_n(prompt, prompt_min='', clip_model=None, device=None, n: int = len(IMAGENET_CLASSES)):
    with th.no_grad():
        imagenet_lbl_tokens = clip.tokenize(IMAGENET_CLASSES).to(device)
        prompt_tokens = clip.tokenize(prompt).to(device)

        imagenet_features = clip_model.encode_text(imagenet_lbl_tokens).float()
        prompt_features = clip_model.encode_text(prompt_tokens).float()

        imagenet_features /= imagenet_features.norm(dim=-1, keepdim=True)
        prompt_features /= prompt_features.norm(dim=-1, keepdim=True)

        if len(prompt_min) > 0:
            prompt_min_tokens = clip.tokenize(prompt_min).to(device)
            prompt_min_features = clip_model.encode_text(prompt_min_tokens).float()
            prompt_min_features /= prompt_min_features.norm(dim=-1, keepdim=True)
            prompt_features = prompt_features - prompt_min_features

        text_probs = (100.0 * prompt_features @ imagenet_features.T).softmax(dim=-1)
        sorted_probs, sorted_classes = text_probs.cpu().topk(n, dim=-1, sorted=True)
        categorical_clip_scores = th.distributions.Categorical(sorted_probs)
        return (sorted_classes[0], categorical_clip_scores)


"""
[Generate an image from a specified text prompt.]
"""
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--prompt", type=str, help="the prompt to reward")
    p.add_argument("--prompt_min", type=str, default="", help="the prompt to penalize")
    p.add_argument("--image_size", "-size", type=int, default=128, help="Diffusion image size. Must be one of [64, 128, 256, 512].")
    p.add_argument("--init_image", type=str, help="Blend an image with diffusion for n steps")
    p.add_argument("--skip_timesteps", "-skipt", type=int, default=0, help="Number of timesteps to blend image for. CLIP guidance occurs after this.")
    p.add_argument("--prefix", "-dir", default="outputs", type=str, help="output directory")
    p.add_argument("--batch_size", "-bs", type=int, default=1, help="the batch size")
    p.add_argument("--clip_guidance_scale", "-cgs", type=int, default=900,
        help="Scale for CLIP spherical distance loss. Default value varies depending on image size.",
    )
    p.add_argument("--tv_scale", "-tvs", type=float, default=0., help="Scale for denoising loss",)
    p.add_argument("--class_score", "-score", default=True, help="Enables CLIP guided class randomization. Use `-score False` to disable CLIP guided class generation.",)
    p.add_argument("--top_n", "-tn", type=int, default=len(IMAGENET_CLASSES), help="Top n imagenet classes compared to phrase by CLIP",)
    p.add_argument("--seed", type=int, default=0, help="Random number seed")
    p.add_argument("--save_frequency", "-sf", type=int, default=5, help="Save frequency")
    p.add_argument("--device", type=str, help="device to run on .e.g. cuda:0 or cpu")
    p.add_argument("--diffusion_steps", "-steps", type=int, default=1000, help="Diffusion steps")
    p.add_argument("--timestep_respacing", "-respace", type=str, default="1000", help="Timestep respacing")
    p.add_argument("--num_cutouts", "-cutn", type=int, default=16, help="Number of randomly cut patches to distort from diffusion.")
    p.add_argument("--cutout_power", "-cutpow", type=float, default=1.0, help="Cutout size power")
    p.add_argument("--clip_model", "-clip", type=str, default="ViT-B/16", help=f"clip model name. Should be one of: {CLIP_MODEL_NAMES}")
    p.add_argument("--class_cond", "-cond", type=bool, default=True, help="Use class conditional. Required for image sizes other than 256")
    args = p.parse_args()

    # Initialize
    prompt = args.prompt
    prompt_min = args.prompt_min
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
    clip_class_score = args.class_score
    prefix = args.prefix
    top_n = args.top_n

    if args.device is None:
        device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    else:
        device = th.device(args.device)

    assert clip_model_name in CLIP_MODEL_NAMES, f"clip model name should be one of: {CLIP_MODEL_NAMES}"
    assert timestep_respacing in TIMESTEP_RESPACINGS, f"timestep_respacing should be one of {TIMESTEP_RESPACINGS}"
    assert image_size in IMAGE_SIZES, f"image size should be one of {IMAGE_SIZES}"

    # Setup
    prompt_as_subdir = prompt
    if len(prompt_min) > 0:
        prompt_as_subdir = f"{prompt_as_subdir}_MIN_{prompt_min}"
    prompt_as_subdir = re.sub(r"[^\w\s]", "", f"{prompt_as_subdir}").replace(" ", "_")[:256]  # Remove non-alphabet characters
    prefix_path = Path(f"{prefix}/{prompt_as_subdir}")

    os.makedirs(prefix_path, exist_ok=True)

    if image_size == 64 and clip_guidance_scale > 500:
        print("CLIP guidance scale and TV scale may be too high for 64x64 image. Press CTRL-C to exit. ")
        time.sleep(5)

    if args.class_cond:
        diffusion_path = f"./checkpoints/{image_size}x{image_size}_diffusion.pt"
    else:
        assert image_size == 256, "Class unconditional requires image size to be 256"
        diffusion_path = f"checkpoints/256x256_diffusion_uncond.pt"

    if seed is not None:
        th.manual_seed(seed)
    clip_model = clip.load(clip_model_name, jit=False)[0].eval().requires_grad_(False).to(device)
    clip_size = clip_model.visual.input_resolution
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

    # - Update (Clay M): Use clip scores as weights for random class selection.
    model_kwargs = {}
    model_kwargs["y"] = th.zeros([batch_size], device=device, dtype=th.long)
    clip_scores = None
    if clip_class_score:
        clip_scores = imagenet_top_n(prompt, prompt_min, clip_model, device, top_n)
        tqdm.write(f"Using ImageNet CLIP scores for class randomization for top {top_n} classes.")
    else:
        clip_scores = None
        print("Randomizing class as generation occurs.")

    make_cutouts = MakeCutouts(clip_size, num_cutouts, cutout_size_power=cutout_power, augment_list=[
        # kaug.RandomChannelShuffle(p=0.1),
        # kaug.RandomBoxBlur(p=0.1),
        # kaug.RandomSharpness(sharpness=2, p=0.1),
    ])

    # Embed text with CLIP model
    text_embed = clip_model.encode_text(
        clip.tokenize(prompt).to(device)).float()

    # Embed penalty text with CLIP model
    text_min_embed = None
    if len(prompt_min) > 0:
        text_min_embed = clip_model.encode_text(
            clip.tokenize(prompt_min).to(device)
        ).float()
    # (Optional) Load image
    init = None
    if init_image is not None:
        init = Image.open(fetch(init_image)).convert("RGB")
        init = init.resize((image_size, image_size), Image.LANCZOS)
        init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)
    
    # Load diffusion and CLIP models
    gd_model, diffusion = load_guided_diffusion(
        checkpoint_path=diffusion_path,
        image_size=image_size,
        diffusion_steps=diffusion_steps,
        timestep_respacing=timestep_respacing,
        device=device,
        class_cond=class_cond,
    )
    # Customize guided-diffusion model with function that uses CLIP guidance.
    current_timestep = diffusion.num_timesteps - 1
    def cond_fn(x, t, y=None):
        with th.enable_grad():
            x = x.detach().requires_grad_()
            n = x.shape[0]
            my_t = th.ones([n], device=device, dtype=th.long) * current_timestep
            out = diffusion.p_mean_variance(
                gd_model, x, my_t, clip_denoised=False, model_kwargs={"y": y}
            )
            fac = diffusion.sqrt_one_minus_alphas_cumprod[current_timestep]
            x_in = out["pred_xstart"] * fac + x * (1 - fac)
            clip_in = normalize(make_cutouts(x_in.add(1).div(2)))
            cutout_embeds = clip_model.encode_image(clip_in).float().view([num_cutouts, n, -1])
            max_dists = spherical_dist_loss(cutout_embeds, text_embed.unsqueeze(0))
            min_dists = 0
            if len(prompt_min) > 0:
                min_dists = spherical_dist_loss(cutout_embeds, text_min_embed.unsqueeze(0))
                # dists = (0.5 * max_dists) - (0.5 * min_dists) # TODO make these kwargs
                dists = max_dists - min_dists
            else:
                dists = max_dists
            losses = dists.mean(0)
            tv_losses = tv_loss(x_in)
            loss = losses.sum() * clip_guidance_scale + tv_losses.sum() * tv_scale
            return -th.autograd.grad(loss, x)[0]

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
        randomize_class=True,
        clip_scores=clip_scores,
    )

    try:
        current_timestep = diffusion.num_timesteps - 1
        for step, sample in enumerate(samples):
            current_timestep -= 1
            if step % save_frequency == 0 or current_timestep == -1:
                for j, image in enumerate(sample["pred_xstart"]):
                    log_image(image, prefix_path, step, j)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(
                f"CUDA OOM error occurred. Lower the batch_size or num_cutouts and try again."
            )
        else:
            raise e


if __name__ == "__main__":
    main()

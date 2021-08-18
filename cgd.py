import argparse
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

from guided_diffusion.nn import timestep_embedding

import clip
import torch as th
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as tvt
from torchvision.transforms import functional as tf
from tqdm import tqdm

from data.imagenet1000_clsidx_to_labels import IMAGENET_CLASSES
from util import fetch, load_guided_diffusion

sys.path.append("./guided-diffusion")

TIMESTEP_RESPACINGS = ("25", "50", "100", "250", "500", "1000", "ddim25", "ddim50", "ddim100", "ddim250", "ddim500", "ddim1000")
DIFFUSION_SCHEDULES = (25, 50, 100, 250, 500, 1000)
IMAGE_SIZES = (64, 128, 256, 512)
CLIP_MODEL_NAMES = ("ViT-B/16", "ViT-B/32", "RN50", "RN101", "RN50x4", "RN50x16")
CLIP_NORMALIZE = tvt.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

def log_image(image, prefix_path, current_step, batch_idx):
    filename = os.path.join(prefix_path, f"{batch_idx:04}_iteration_{current_step:04}.png")
    pil_image = tf.to_pil_image(image.add(1).div(2).clamp(0, 1))
    pil_image.save(filename)
    pil_image.save("current.png")

def txt_to_dir(base_path, txt, txt_min=None):
    dir_name = f"{txt}_MIN_{txt_min}" if txt_min else txt
    dir_name = Path(re.sub(r"[^\w\s]", "", f"{dir_name}").replace(" ", "_")[:256])
    return Path(os.path.join(base_path, dir_name))

def resize_image(image, out_size):
    """Resize image"""
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)

def spherical_dist_loss(x, y):
    """Spherical distance loss"""
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), "replicate")
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff ** 2 + y_diff ** 2).mean([1, 2, 3])

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
def imagenet_top_n(prompt, prompt_min='', min_weight=0.1, clip_model=None, device=None, n: int = len(IMAGENET_CLASSES)):
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
            prompt_features = prompt_features - (min_weight * prompt_min_features)
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
    p.add_argument("--prompt", "-txt", type=str, default='', help="the prompt to reward")
    p.add_argument("--prompt_min", "-min", type=str, default=None, help="the prompt to penalize")
    p.add_argument("--min_weight", "-min_wt", type=str, default=0.1, help="the prompt to penalize")
    p.add_argument("--image_size", "-size", type=int, default=128, help="Diffusion image size. Must be one of [64, 128, 256, 512].")
    p.add_argument("--init_image", "-init", type=str, help="Blend an image with diffusion for n steps")
    p.add_argument("--skip_timesteps", "-skip", type=int, default=0, help="Number of timesteps to blend image for. CLIP guidance occurs after this.")
    p.add_argument("--prefix", "-dir", default="outputs", type=str, help="output directory")
    p.add_argument("--checkpoints_dir", "-ckpts", default='checkpoints', type=str, help="Path subdirectory containing checkpoints.")
    p.add_argument("--batch_size", "-bs", type=int, default=1, help="the batch size")
    p.add_argument("--clip_guidance_scale", "-cgs", type=float, default=1000,
        help="Scale for CLIP spherical distance loss. Values will need tinkering for different settings.",)
    p.add_argument("--tv_scale", "-tvs", type=float, default=100, help="Scale for denoising loss",)
    p.add_argument("--class_score", "-score", action="store_true", help="Enables CLIP guided class randomization.",)
    p.add_argument("--top_n", "-top", type=int, default=len(IMAGENET_CLASSES), help="Top n imagenet classes compared to phrase by CLIP",)
    p.add_argument("--seed", "-seed", type=int, default=0, help="Random number seed")
    p.add_argument("--save_frequency", "-freq", type=int, default=5, help="Save frequency")
    p.add_argument("--device", type=str, help="device to run on .e.g. cuda:0 or cpu")
    p.add_argument("--diffusion_steps", "-steps", type=int, default=1000, help="Diffusion steps")
    p.add_argument("--timestep_respacing", "-respace", type=str, default="1000", help="Timestep respacing")
    p.add_argument("--num_cutouts", "-cutn", type=int, default=32, help="Number of randomly cut patches to distort from diffusion.")
    p.add_argument("--cutout_power", "-cutpow", type=float, default=0.5, help="Cutout size power")
    p.add_argument("--clip_model", "-clip", type=str, default="ViT-B/32", help=f"clip model name. Should be one of: {CLIP_MODEL_NAMES}")
    p.add_argument("--class_cond", "-cond", type=bool, default=True, help="Use class conditional. Required for image sizes other than 256")
    args = p.parse_args()

    # Assertions
    assert len(args.prompt) > 0, "`--prompt` / `-txt` cant be empty"
    assert args.timestep_respacing in TIMESTEP_RESPACINGS, f"timestep_respacing should be one of {TIMESTEP_RESPACINGS}"
    assert args.diffusion_steps in DIFFUSION_SCHEDULES, f"Diffusion steps should be one of: {DIFFUSION_SCHEDULES}"
    assert args.clip_model in CLIP_MODEL_NAMES, f"clip model name should be one of: {CLIP_MODEL_NAMES}"
    assert args.image_size in IMAGE_SIZES, f"image size should be one of {IMAGE_SIZES}"
    assert args.num_cutouts > 0, "`--num_cutouts` / `-cutn` must greater than zero."
    assert 0 < args.top_n <= len(IMAGENET_CLASSES), \
        f"top_n must be less than or equal to the number of classes: {args.top_n} > {len(IMAGENET_CLASSES)}"
    assert 0 < args.save_frequency <= int(args.timestep_respacing.replace("ddim", "")), \
        "`--save_frequency` / `--freq` must be greater than 0and less than `--timestep_respacing`"
    assert 0.0 <= args.min_weight <= 1.0, f"min_weight must be between 0 and 1: {args.min_weight} not in [0, 1]"
    assert (not args.class_cond and args.image_size == 256) or args.class_cond, \
        f"Image size must be 256 when `--class_cond` / `-cond` is False."
    if args.init_image:
        # Check skip timesteps logic
        assert args.skip_timesteps > 0 and args.skip_timesteps < int(args.timestep_respacing.replace("ddim", "")), \
        f"`--skip_timesteps` / `-skip` (currently {args.skip_timesteps}) must be greater than 0 and less than `--timestep_respacing` / `-respace` (currently {args.timestep_respacing}) when `--init_image` / `-init` is not None."
        assert Path(args.init_image).exists(), f"{args.init_image} does not exist. Check spelling or provide another path."
    assert Path(args.checkpoints_dir).is_dir(), f"`--checkpoints_dir` / `-ckpts` {args.checkpoints_dir} is a file, not a directory. Please provide a directory."
    assert Path(args.checkpoints_dir).exists(), f"`--checkpoints_dir` / `-ckpts` {args.checkpoints_dir} does not exist. Create it or provide another directory."
    assert Path(args.prefix).is_dir(), f"`--prefix` / `-dir` {args.prefix} is a file, not a directory. Please provide a directory."

    # Initialize
    batch_size = args.batch_size
    clip_guidance_scale = args.clip_guidance_scale
    cutout_power = args.cutout_power
    save_frequency = args.save_frequency
    num_cutouts = args.num_cutouts
    tv_scale = args.tv_scale
    top_n = args.top_n
    min_weight = args.min_weight
    prompt = args.prompt
    prompt_min = args.prompt_min
    image_size = args.image_size
    class_cond = args.class_cond
    timestep_respacing = args.timestep_respacing
    seed = args.seed
    diffusion_steps = args.diffusion_steps
    skip_timesteps = args.skip_timesteps
    device = args.device
    init_image = args.init_image 
    checkpoints_dir = args.checkpoints_dir
    clip_model_name = args.clip_model
    prefix = args.prefix
    class_score = args.class_score
    
    ## Setup
    if seed: th.manual_seed(seed)

    device = th.device("cuda:0" if th.cuda.is_available() else "cpu") if device is None else th.device(device)
    init_image = Path(init_image) if args.init_image else None
    checkpoints_dir = Path(checkpoints_dir)

    ### Check that the checkpoint directory and file exist. 
    # TODO - implement download of checkpoint files
    diffusion_filename = f"{image_size}x{image_size}_diffusion.pt" if class_cond else f"256x256_diffusion_uncond.pt"
    diffusion_path = checkpoints_dir.joinpath(Path(diffusion_filename))
    assert diffusion_path.exists(), f"{diffusion_path} does not exist. See README for download links."

    ### Remove non-alphanumeric and white space characters from prompt and prompt_min for directory name
    outputs_path = txt_to_dir(base_path=Path(prefix), txt=prompt, txt_min=prompt_min)
    if not outputs_path.exists():
        outputs_path.mkdir(exist_ok=True)

    ### Load CLIP model
    clip_model = clip.load(clip_model_name, jit=False)[0].eval().requires_grad_(False).to(device)
    clip_size = clip_model.visual.input_resolution

    ### Use CLIP scores as weights for random class selection.
    model_kwargs = {}
    model_kwargs["y"] = th.zeros([batch_size], device=device, dtype=th.long)
    ### Rank the classes by their CLIP score
    clip_scores = imagenet_top_n(prompt, prompt_min, min_weight, clip_model, device, top_n) if class_score else None
    if clip_scores is not None:
        tqdm.write(f"Ranking top {top_n} ImageNet classes by their CLIP score.")
    else:
        tqdm.write("Ranking all ImageNet classes uniformly. Use `--class_score` / `-score` to enable CLIP guided class selection instead.")
    
    ### Setup CLIP cutouts/embeds
    make_cutouts = MakeCutouts(clip_size, num_cutouts, cutout_size_power=cutout_power, augment_list=[])
    text_embed = clip_model.encode_text(clip.tokenize(prompt).to(device)).float()
    text_min_embed = clip_model.encode_text(clip.tokenize(prompt_min).to(device)).float() if prompt_min else None
    
    ### Load initial image (if provided)
    init_tensor = None
    if init_image:
        pil_image = Image.open(fetch(init_image)).convert("RGB").resize((image_size, image_size), Image.LANCZOS)
        init_tensor = tf.to_tensor(pil_image).to(device).unsqueeze(0).mul(2).sub(1)
    
    ### Load guided diffusion
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
            out = diffusion.p_mean_variance( gd_model, x, my_t, clip_denoised=False, model_kwargs={"y": y})
            fac = diffusion.sqrt_one_minus_alphas_cumprod[current_timestep]
            x_in = out["pred_xstart"] * fac + x * (1 - fac)
            clip_in = CLIP_NORMALIZE(make_cutouts(x_in.add(1).div(2)))
            cutout_embeds = clip_model.encode_image(clip_in).float().view([num_cutouts, n, -1])
            max_dists = spherical_dist_loss(cutout_embeds, text_embed.unsqueeze(0))
            if text_min_embed is not None: # Implicit comparison to None is not supported by pytorch tensors
                min_dists = spherical_dist_loss(cutout_embeds, text_min_embed.unsqueeze(0))
                dists = max_dists - (min_weight * min_dists)
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
        gd_model, (batch_size, 3, image_size, image_size), 
        clip_denoised=False, model_kwargs=model_kwargs, cond_fn=cond_fn, 
        progress=True, skip_timesteps=skip_timesteps, init_image=init_tensor, 
        randomize_class=class_cond, clip_scores=clip_scores,
    )

    try:
        current_timestep = diffusion.num_timesteps - 1
        for step, sample in enumerate(samples):
            current_timestep -= 1
            if step % save_frequency == 0 or current_timestep == -1:
                for j, image in enumerate(sample["pred_xstart"]):
                    log_image(image, outputs_path, step, j)
    except RuntimeError as runtime_ex:
        if "CUDA out of memory" in str(runtime_ex):
            tqdm.write(f"CUDA OOM error occurred.")
            tqdm.write(f"Try lowering `--image_size`/`-size`, `--batch_size`/`-bs`, `--num_cutouts`/`-cutn`")
            tqdm.write(f"`--clip_model` / `-clip` (currently {clip_model_name}) can have a large impact on VRAM usage.")
            tqdm.write(f"`RN50` will use the least VRAM. `ViT-B/32` is the best bang for your buck.")
        else:
            raise runtime_ex


if __name__ == "__main__":
    main()

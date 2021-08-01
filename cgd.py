import argparse
import os
import sys
from pathlib import Path

from IPython import display
from PIL import Image
from guided_diffusion.nn import checkpoint

sys.path.append("./guided-diffusion")
import clip
import torch
from guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from kornia import augmentation as K
from torch import clip_, nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm

def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)

# Model settings
# SAMPLE_FLAGS="--batch_size 4 --num_samples 100 --timestep_respacing 250"
# MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
# python classifier_sample.py $MODEL_FLAGS --classifier_scale 1.0 --classifier_path models/64x64_classifier.pt --model_path models/64x64_diffusion.pt $SAMPLE_FLAGS

def load_guided_diffusion(
    checkpoint_path,
    image_size,
    diffusion_steps=None,
    timestep_respacing=None,
    device=None,
    class_cond=False,
    rescale_timesteps=True,
):
    assert device is not None, "device must be set"

    num_channels = 192 if class_cond else 256
    noise_schedule = "cosine" if class_cond else "linear"
    num_res_blocks = 3 if class_cond else 2
    use_new_attention_order = True if class_cond else False

    model_config = model_and_diffusion_defaults()
    model_config.update({
        "attention_resolutions": "32, 16, 8",
        "class_cond": class_cond,
        "diffusion_steps": diffusion_steps,
        "rescale_timesteps": rescale_timesteps,
        "timestep_respacing": timestep_respacing,
        "image_size": image_size,
        "learn_sigma": True,
        "noise_schedule": 'linear', # noise_schedule,
        "num_channels": num_channels,
        "num_head_channels": 64,
        "num_res_blocks": num_res_blocks,
        "resblock_updown": True,
        "use_new_attention_order": use_new_attention_order,
        "use_fp16": False,
        "use_scale_shift_norm": True,
    })
    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.requires_grad_(False).eval().to(device)
    for name, param in model.named_parameters():
        if "qkv" in name or "norm" in name or "proj" in name:
            param.requires_grad_()
    if model_config["use_fp16"]:
        model.convert_to_fp16()
    return model, diffusion

def parse_prompt(prompt):                                               # NR: Weights after colons
    vals = prompt.rsplit(':', 2)
    vals = vals + ['', '1', '-inf'][len(vals):]
    return vals[0], float(vals[1]), float(vals[2])

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

custom_augment_list = [
    # K.RandomAffine(degrees=25, translate=0.1, p=0.7, padding_mode='reflection'),
    # K.RandomElasticTransform(p=0.1, alpha=(10.0, 10.0)),
    # K.RandomGaussianNoise(0.1, 0.08, p=0.5),
    # K.RandomPerspective(distortion_scale=0.3, p=0.7),
    # K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
    # K.RandomChannelShuffle(p=0.25),
    # K.RandomGrayscale(p=0.5),
    # K.RandomMotionBlur(3, 15, 0.5, p=0.25),
    # K.RandomSharpness(p=0.5),
    # K.RandomHorizontalFlip(p=0.1),
    # K.RandomThinPlateSpline(p=0.25),
    # K.RandomAffine(degrees=7, p=0.4, padding_mode="border"),
    # K.RandomSolarize(0.01, 0.01, p=0.25),
]

"""
[Generate an image from a specified text prompt.]
"""
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("prompt", type=str, help="the prompt")
    p.add_argument("--img_prompts", type=str, help="Generation will be similar to specified comma-separated image paths", default=[], dest='img_prompts')
    p.add_argument("--num_cutouts", "--cutn", type=int, default=8, help="Number of randomly cut patches to distort from diffusion.")
    p.add_argument("--prefix", "--output_dir", default="outputs", type=str, help="output directory")
    p.add_argument("--batch_size", "-bs", type=int, default=1, help="the batch size")
    p.add_argument("--text_prompt_weight", "-tpw", type=int, default=500, help="Scale for CLIP's text prompt based loss.")
    p.add_argument("--img_prompt_weight", "-ipw", type=int, default=50, help="Scale for CLIP's image prompt based loss.")
    p.add_argument("--tv_weight", "-tvw", type=int, default=100, help="Scale for denoising loss")
    p.add_argument("--seed", type=int, default=0, help="Random number seed")
    p.add_argument("--save_frequency", "-sf", type=int, default=100, help="Save frequency")
    p.add_argument("--device", type=str, help="device")
    p.add_argument("--diffusion_steps", type=int, default=1000, help="Diffusion steps")
    p.add_argument("--timestep_respacing", type=str, default='250', help="Timestep respacing")
    p.add_argument('--cutout_power', '--cutpow', type=float, default=1.0, help='Cutout size power')
    p.add_argument('--clip_model', type=str, default='ViT-B/16', help='clip model name. Should be one of: [ViT-B/16, ViT-B/32, RN50, RN101, RN50x4, RN50x16]')
    p.add_argument('--class_cond', type=bool, default=True, help='Class condition')
    p.add_argument("--image_size", type=int, default=64, help="image size") # TODO - image size only works @ 256; need to fix
    args = p.parse_args()

    # Initialize

    prompt = args.prompt 
    text_prompt_weight = args.text_prompt_weight

    img_prompts = []
    if len(args.img_prompts) > 0:
        img_prompts = [i.strip() for i in args.img_prompts.split(',')]

    img_prompt_weight = args.img_prompt_weight
    
    image_size = args.image_size # TODO - support other image sizes
    batch_size = args.batch_size
    seed = args.seed
    save_frequency = args.save_frequency
    cutout_power = args.cutout_power
    num_cutouts = args.num_cutouts
    class_cond = args.class_cond
    prefix = args.prefix

    prefix_path = Path(prefix)
    os.makedirs(prefix_path, exist_ok=True)

    diffusion_steps = args.diffusion_steps
    timestep_respacing = args.timestep_respacing
    assert timestep_respacing in ['25', '50', '100', '250', '500', '1000', 'ddim25', 'ddim50', 'ddim100', 'ddim250', 'ddim500', 'ddim1000'], 'timestep_respacing should be one of [25, 50, 100, 250, 500, 1000, ddim25, ddim50, ddim100, ddim250, ddim500, ddim1000]'

    tv_weight = args.tv_weight
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
        checkpoint_path=f'checkpoints/64x64_diffusion.pt',
        image_size=image_size,
        diffusion_steps=diffusion_steps,
        timestep_respacing=timestep_respacing,
        device=device,
        class_cond=True,
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

    # Convert images to tensors
    img_tensors = []
    for img_prompt in img_prompts:
        img_path = Path(img_prompt)
        assert img_path.exists(), f"Image {img_prompt} does not exist."
        img = resize_image(Image.open(img_path).convert("RGB"), (image_size, image_size))
        img_tensors.append(TF.to_tensor(img))

    print(f'Found {len(img_tensors)} images to use as guides for generation.')

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

            augmented_make_cutouts = MakeCutouts(clip_size, num_cutouts, cutout_size_power=cutout_power, augment_list=custom_augment_list)
            normal_make_cutouts = MakeCutouts(clip_size, num_cutouts, cutout_size_power=cutout_power)
            clip_in = normalize(augmented_make_cutouts(x_in.add(1).div(2)))
            cutout_embeds = clip_model.encode_image(clip_in).float().view([num_cutouts, n, -1])

            # spherical distance between diffusion cutouts and text
            text_prompt_spherical_loss = spherical_dist_loss(cutout_embeds, text_embed.unsqueeze(0)).mean(0).sum()
            text_prompt_spherical_loss = text_prompt_spherical_loss.mean() * text_prompt_weight

            img_prompt_spherical_loss = torch.zeros([num_cutouts, n], device=device)
            # spherical distance between diffusion cutouts and image (if any)
            if len(img_tensors) > 0:
                for img in img_tensors:
                    img_prompt_cutouts = normalize( normal_make_cutouts(img.unsqueeze(0).to(device)))
                    img_clip_embed = clip_model.encode_image(img_prompt_cutouts).float().view([num_cutouts, n, -1])
                    img_prompt_spherical_loss += spherical_dist_loss(cutout_embeds, img_clip_embed.unsqueeze(0)).mean(0).sum()
                img_prompt_spherical_loss /= len(img_tensors) 
                img_prompt_spherical_loss = img_prompt_spherical_loss.mean() * img_prompt_weight

            if len(img_tensors) > 0:
                clip_based_loss = (text_prompt_spherical_loss + img_prompt_spherical_loss) / 2
            else:
                clip_based_loss = text_prompt_spherical_loss
            # denoising loss
            tv_denoise_loss = tv_loss(x_in).sum() * tv_weight
            loss = clip_based_loss + tv_denoise_loss

            print(f"Text loss: {text_prompt_spherical_loss}")
            print(f"Image loss: {img_prompt_spherical_loss.sum()}")
            print(f"CLIP loss: {clip_based_loss}")
            print(f"Denoise loss: {tv_denoise_loss}")
            print("---")
            print(f"Total loss: {loss}")
            return -torch.autograd.grad(loss, x)[0]

    if timestep_respacing.startswith("ddim"):
        diffusion_sample_loop = diffusion.ddim_sample_loop_progressive
    else:
        diffusion_sample_loop = diffusion.p_sample_loop_progressive
 
    model_kwargs = {}
    if class_cond:
        classes = torch.randint(low=0, high=1000, size=(batch_size,), device=device)
        model_kwargs["y"] = classes

    samples = diffusion_sample_loop(
        gd_model,
        (batch_size, 3, image_size, image_size),
        clip_denoised=False,
        cond_fn=cond_fn,
        progress=True,
        model_kwargs=model_kwargs,
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
                        prefix_path, f"batch_idx_{j:04}_iteration_{step:04}.png"
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
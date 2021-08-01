import argparse
import os
import sys
from pathlib import Path
from torch.utils.data.dataloader import DataLoader

from torch.utils.data.dataset import TensorDataset
from torch_util import load_tokenized, resize_image, spherical_dist_loss, tv_loss
from util import load_guided_diffusion

from IPython import display
from PIL import Image
from guided_diffusion.nn import checkpoint

sys.path.append("./guided-diffusion")
import clip
import torch
from kornia import augmentation as K
from torch import clip_, nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm


# def find_imagenet_class_with_clip(batch_size, num_workers, shuffle=False, path="data/imagenet.pkl", clip_model_name='ViT-B/16', device='cuda:0'):
#     tokens = load_tokenized(path)
#     dataset = TensorDataset(tokens)
#     # Load CLIP model
#     clip_perceptor = clip.load(clip_model_name, jit=False)[0].eval().requires_grad_(False).to(device)
#     # Embed text with CLIP model
#     imagenet_clip_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
#     for batch_idx, (clip_token,) in enumerate(imagenet_clip_loader):
#         clip_token = clip_token.to(device)
#         embed = F.normalize(clip_token.float().unsqueeze(0).unsqueeze(0).neg())



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
    p.add_argument("--text_prompt_weight", "-tpw", type=int, default=1000, help="Scale for CLIP's text prompt based loss.")
    p.add_argument("--img_prompt_weight", "-ipw", type=int, default=50, help="Scale for CLIP's image prompt based loss.")
    p.add_argument("--tv_weight", "-tvw", type=int, default=100, help="Scale for denoising loss")
    p.add_argument("--seed", type=int, default=0, help="Random number seed")
    p.add_argument("--save_frequency", "-sf", type=int, default=100, help="Save frequency")
    p.add_argument("--device", type=str, help="device")
    p.add_argument("--diffusion_steps", type=int, default=1000, help="Diffusion steps")
    p.add_argument("--timestep_respacing", type=str, default='1000', help="Timestep respacing")
    p.add_argument('--cutout_power', '--cutpow', type=float, default=1.0, help='Cutout size power')
    p.add_argument('--clip_model', type=str, default='ViT-B/32', help='clip model name. Should be one of: [ViT-B/16, ViT-B/32, RN50, RN101, RN50x4, RN50x16]')
    p.add_argument('--class_cond', type=bool, default=True, help='Class condition')
    p.add_argument("--image_size", type=int, default=256, help="image size") # TODO - image size only works @ 256; need to fix
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
            checkpoint_path='./checkpoints/256x256_diffusion_uncond.pt',
        image_size=image_size,
        diffusion_steps=diffusion_steps,
        timestep_respacing=timestep_respacing,
        device=device,
        class_cond=False,
        rescale_timesteps=False,
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

            normal_make_cutouts = MakeCutouts(clip_size, num_cutouts, cutout_size_power=cutout_power)
            clip_in = normalize(normal_make_cutouts(x_in.add(1).div(2)))
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
#    if class_cond:
#        classes = torch.randint(low=0, high=1000, size=(batch_size,), device=device)
#        model_kwargs["y"] = classes

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

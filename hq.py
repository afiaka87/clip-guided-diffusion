import argparse
import math
import sys
from pathlib import Path

from IPython import display
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm

sys.path.append('./CLIP')
sys.path.append('./guided-diffusion')

import clip
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

# Model settings

model_config = model_and_diffusion_defaults()
model_config.update({
    'attention_resolutions': '32, 16, 8',
    'class_cond': False,
    'diffusion_steps': 1000,
    'rescale_timesteps': True,
    'timestep_respacing': '250',
    'image_size': 256,
    'learn_sigma': True,
    'noise_schedule': 'linear',
    'num_channels': 256,
    'num_head_channels': 64,
    'num_res_blocks': 2,
    'resblock_updown': True,
    'use_fp16': True,
    'use_scale_shift_norm': True,
})

# Load models

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

model, diffusion = create_model_and_diffusion(**model_config)
model.load_state_dict(torch.load('256x256_diffusion_uncond.pt', map_location='cpu'))
model.requires_grad_(False).eval().to(device)
for name, param in model.named_parameters():
    if 'qkv' in name or 'norm' in name or 'proj' in name:
        param.requires_grad_()
if model_config['use_fp16']:
    model.convert_to_fp16()

clip_model = clip.load('ViT-B/16', jit=False)[0].eval().requires_grad_(False).to(device)
clip_size = clip_model.visual.input_resolution
normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

# Define necessary functions

class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=4.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutout = F.interpolate(cutout, (self.cut_size, self.cut_size),
                                   mode='bilinear', align_corners=False)
            cutouts.append(cutout)
        return torch.cat(cutouts)


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])



def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('prompt', type=str, help='the prompt')
    p.add_argument('--prefix', default='outputs', type=str, help='output directory')
    p.add_argument('--batch_size', '-bs', type=int, default=1, help='the batch size')
    p.add_argument('--clip_guidance_scale', '-cgs', type=int, default=2750, help='clip guidance scale.')
    p.add_argument('--tv_scale', '-tvs', type=int, default=100, help='tv scale')
    p.add_argument('--cutn', type=int, default=16, help='tv scale')
    p.add_argument('--seed', type=int, default=0, help='random number seed')
    args = p.parse_args()
    prompt = args.prompt # 'fearful symmetry by Odilon Redon'
    batch_size = args.batch_size  
    clip_guidance_scale = args.clip_guidance_scale # 2750
    seed = args.seed # 0
    prefix = args.prefix
    prefix_path = Path(prefix)
    import os
    os.makedirs(prefix_path, exist_ok=True)


    if seed is not None:
        torch.manual_seed(seed)

    """### Actually do the run..."""

    if seed is not None:
        torch.manual_seed(seed)

    def cond_fn(x, t, y=None):
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            n = x.shape[0]
            my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
            out = diffusion.p_mean_variance(model, x, my_t, clip_denoised=False, model_kwargs={'y': y})
            fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
            x_in = out['pred_xstart'] * fac + x * (1 - fac)
            clip_in = normalize(make_cutouts(x_in.add(1).div(2)))
            image_embeds = clip_model.encode_image(clip_in).float().view([args.cutn, n, -1])
            dists = spherical_dist_loss(image_embeds, text_embed.unsqueeze(0))
            losses = dists.mean(0)
            tv_losses = tv_loss(x_in)
            loss = losses.sum() * clip_guidance_scale + tv_losses.sum() * args.tv_scale
            return -torch.autograd.grad(loss, x)[0]

    text_embed = clip_model.encode_text(clip.tokenize(prompt).to(device)).float() 
    make_cutouts = MakeCutouts(clip_size, args.cutn)

    cur_t = diffusion.num_timesteps - 1 
    if model_config['timestep_respacing'].startswith('ddim'):
        sample_fn = diffusion.ddim_sample_loop_progressive
    else:
        sample_fn = diffusion.p_sample_loop_progressive

    samples = sample_fn(
        model,
        (batch_size, 3, model_config['image_size'], model_config['image_size']),
        clip_denoised=False,
        model_kwargs={},
        cond_fn=cond_fn,
        progress=True,
    )

    for i, sample in enumerate(samples):
        cur_t -= 1
        if i % 100 == 0 or cur_t == -1:
            print()
            for j, image in enumerate(sample['pred_xstart']):
                filename = os.path.join(prefix_path, 'progress_{j:05}.png') 
                TF.to_pil_image(image.add(1).div(2).clamp(0, 1)).save(filename)
                tqdm.write(f'Step {i}, output {j}:')
                display.display(display.Image(filename))


if __name__ == '__main__':
    main()

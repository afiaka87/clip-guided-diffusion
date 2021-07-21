# Imports

import math
import sys

from IPython import display
from kornia import augmentation, filters
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
    'rescale_timesteps': False,
    'timestep_respacing': '500',
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

# Load models and define necessary functions

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

model, diffusion = create_model_and_diffusion(**model_config)
model.load_state_dict(torch.load('256x256_diffusion_uncond.pt', map_location='cpu'))
model.eval().requires_grad_(False).to(device)
model.convert_to_fp16()

clip_model = clip.load('ViT-B/16', jit=False)[0].eval().requires_grad_(False).to(device)
clip_size = clip_model.visual.input_resolution
normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])

def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

prompt = 'fearful symmetry by Odilon Redon'
batch_size = 1
clip_guidance_scale = 2750
seed = 0

if seed is not None:
    torch.manual_seed(seed)

text_embed = clip_model.encode_text(clip.tokenize(prompt).to(device)).float()

sigmas = 0.25 * diffusion.sqrt_one_minus_alphas_cumprod / diffusion.sqrt_alphas_cumprod
translate_by = 8 / clip_size
if translate_by:
    aug = augmentation.RandomAffine(0, (translate_by, translate_by),
                                    padding_mode='border', p=1)
else:
    aug = nn.Identity()

cur_t = diffusion.num_timesteps - 1

def cond_fn(x, t, y=None):
    with torch.enable_grad():
        x_in = x.detach().requires_grad_()
        n = x_in.shape[0]
        sigma = min(24, sigmas[cur_t])
        kernel_size = max(math.ceil((sigma * 6 + 1) / 2) * 2 - 1, 3)
        x_blur = filters.gaussian_blur2d(x_in, (kernel_size, kernel_size), (sigma, sigma))
        clip_in = F.interpolate(aug(x_blur.add(1).div(2)), (clip_size, clip_size),
                                mode='bilinear', align_corners=False)
        image_embed = clip_model.encode_image(normalize(clip_in)).float()
        losses = spherical_dist_loss(image_embed, text_embed)
        grad = -torch.autograd.grad(losses.sum(), x_in)[0]
        return grad * clip_guidance_scale

samples = diffusion.p_sample_loop_progressive(
    model,
    (batch_size, 3, model_config['image_size'], model_config['image_size']),
    clip_denoised=True,
    model_kwargs={},
    cond_fn=cond_fn,
    progress=True,
)

for i, sample in enumerate(samples):
    cur_t -= 1
    if i % 100 == 0 or cur_t == -1:
        print()
        for j, image in enumerate(sample['sample']):
            filename = f'progress_{j:05}.png'
            TF.to_pil_image(image.add(1).div(2).clamp(0, 1)).save(filename)
            tqdm.write(f'Step {i}, output {j}:')
            display.display(display.Image(filename))


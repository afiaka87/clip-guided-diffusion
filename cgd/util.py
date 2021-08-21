from urllib import request
import subprocess
import io
import requests
import os
import re
from pathlib import Path

import requests
import torch as th
from guided_diffusion.script_util import (create_model_and_diffusion, model_and_diffusion_defaults)
from PIL import Image
from torch import nn
from torch.nn import functional as tf
from torchvision.transforms import functional as tvf
from tqdm.autonotebook import tqdm

CACHE_PATH = os.path.expanduser("~/.cache/clip-guided-diffusion")
UNCOND_512_URL = 'https://the-eye.eu/public/AI/models/512x512_diffusion_unconditional_ImageNet/512x512_diffusion_uncond_finetune_008100.pt'
UNCOND_256_URL = f'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt'

DIFFUSION_64_MODEL_FLAGS = {
    "attention_resolutions": '32,16,8',
    "class_cond": True,
    "diffusion_steps": 1000,
    "dropout": 0.1,
    "image_size": 64,
    "learn_sigma": True,
    "noise_schedule": 'cosine',
    "num_channels": 192,
    "num_head_channels": 64,
    "num_res_blocks": 3,
    "resblock_updown": True,
    "use_new_attention_order": True,
    "use_fp16": True,
    "use_scale_shift_norm": True
}
DIFFUSION_128_MODEL_FLAGS = {
    "attention_resolutions": '32,16,8',
    "class_cond": True,
    "diffusion_steps": 1000,
    "image_size": 128,
    "learn_sigma": True,
    "noise_schedule": 'linear',
    "num_channels": 256,
    "num_heads": 4,
    "num_res_blocks": 2,
    "resblock_updown": True,
    "use_fp16": True,
    "use_scale_shift_norm": True
}
DIFFUSION_256_MODEL_FLAGS = {
    "attention_resolutions": "32,16,8",
    "class_cond": True,
    "diffusion_steps": 1000,
    "image_size": 256,
    "learn_sigma": True,
    "noise_schedule": "linear",
    "num_channels": 256,
    "num_head_channels": 64,
    "num_res_blocks": 2,
    "resblock_updown": True,
    "use_fp16": True,
    "use_scale_shift_norm": True
}
DIFFUSION_512_MODEL_FLAGS = {
    'attention_resolutions': '32, 16, 8',
    'class_cond': True,
    'diffusion_steps': 1000,
    'rescale_timesteps': True,
    'timestep_respacing': '1000',
    'image_size': 512,
    'learn_sigma': True,
    'noise_schedule': 'linear',
    'num_channels': 256,
    'num_head_channels': 64,
    'num_res_blocks': 2,
    'resblock_updown': True,
    'use_fp16': True,
    'use_scale_shift_norm': True,
}

# modified from https://github.com/lucidrains/DALLE-pytorch/blob/d355100061911b13e1f1c22de8c2b5deb44e65f8/dalle_pytorch/vae.py
def download(url: str, filename:str, root: str=CACHE_PATH) -> str:
    os.makedirs(root, exist_ok=True)
    download_target = Path(os.path.join(root, filename))
    download_target_tmp = download_target.with_suffix('.tmp')
    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")
    if os.path.isfile(download_target):
        return str(download_target)
    with request.urlopen(url) as source, open(download_target_tmp, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80) as loop:
            while True:
                buffer = source.read(4096)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))
    os.rename(download_target_tmp, download_target)
    return str(download_target)


def download_guided_diffusion(image_size: int, class_cond: bool=False, checkpoints_dir:str=CACHE_PATH, overwrite: bool=False):
    gd_path = None
    if class_cond:
        Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)
        gd_path = Path(checkpoints_dir).joinpath(Path(f'{image_size}x{image_size}_diffusion.pt'))
        gd_url = f'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/{image_size}x{image_size}_diffusion.pt'
    else:
        print("Using finetuned uncond checkpoint.")
        assert image_size in [256,512], "class_cond=False only works for 256x256 images"
        if image_size == 256:
            gd_path = Path(checkpoints_dir).joinpath(Path(f"{image_size}x{image_size}_diffusion_uncond.pt"))
        elif image_size == 512:
            gd_path = Path(checkpoints_dir).joinpath(Path(f"512x512_diffusion_uncond_finetune_008100.pt"))
        gd_url = UNCOND_512_URL if image_size == 512 else UNCOND_256_URL

    if gd_path and gd_path.exists() and not overwrite:
        print(f"Found guided diffusion model for {image_size}x{image_size} at {gd_path}. Skipping download.")
        return gd_path

    print(f'Downloading {gd_url} to {gd_path}')
    download(gd_url, str(gd_path))
    return gd_path


def load_guided_diffusion(
    checkpoint_path: str,
    image_size: int,
    class_cond: bool,
    diffusion_steps: int=None,
    timestep_respacing: str=None,
    device:str=None,
):
    '''
    checkpoint_path: path to the checkpoint to load.
    image_size: size of the images to be used.
    class_cond: whether to condition on the class label
    diffusion_steps: number of diffusion steps
    timestep_respacing: whether to use timestep-respacing or not
    '''
    assert device is not None, "device must be set"
    model_config = model_and_diffusion_defaults()
    if image_size == 64:
        model_config.update(DIFFUSION_64_MODEL_FLAGS)
    elif image_size == 128:
        model_config.update(DIFFUSION_128_MODEL_FLAGS)
    elif image_size == 256:
        model_config.update(DIFFUSION_256_MODEL_FLAGS)
    elif image_size == 512:
        model_config.update(DIFFUSION_512_MODEL_FLAGS)
    else:
        raise ValueError("Invalid image size")

    model_config.update({
        'class_cond': class_cond,
        'diffusion_steps': diffusion_steps,
        'timestep_respacing': timestep_respacing,
    })

    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(th.load(checkpoint_path, map_location="cpu"))
    model.requires_grad_(False).eval().to(device)
    for name, param in model.named_parameters():
        if "qkv" in name or "norm" in name or "proj" in name:
            param.requires_grad_()
    if model_config["use_fp16"]:
        model.convert_to_fp16()
    return model, diffusion

def create_video_from_image_dir(image_dir:str, target:str="current.mp4", fps:int=30) -> str:
    assert Path(target).suffix == ".mp4", "target must be a .mp4 file"
    result = subprocess.call([
        "ffmpeg", 
        "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", "", "-pattern_type", "glob",
        "-i", f"{image_dir}/*.png", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", str(target)
    ])
    return target


def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')

def log_image(image: th.Tensor, base_path: str, txt: str, txt_min: str, current_step: int, batch_idx: int) -> str:
    cleaned_txt = f"{txt}_MIN_{txt_min}" if txt_min else txt
    cleaned_txt = re.sub(r"[^\w\s]", "", str(cleaned_txt)).replace(" ", "_")[:256]
    dirname = Path(base_path).joinpath(cleaned_txt)
    dirname.mkdir(parents=True, exist_ok=True)
    filename = dirname.joinpath(f"j_{batch_idx:04}_i_{current_step:04}.png")
    pil_image = tvf.to_pil_image(image.add(1).div(2).clamp(0, 1))
    pil_image.save('current.png')
    pil_image.save(filename)
    return str(filename)

def resize_image(image:th.Tensor, out_size: int):
    """Resize image"""
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)


def spherical_dist_loss(x:th.Tensor, y: th.Tensor):
    """Spherical distance loss"""
    x = tf.normalize(x, dim=-1)
    y = tf.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input: th.Tensor):
    """L2 total variation loss, as in Mahendran et al."""
    input = tf.pad(input, (0, 1, 0, 1), "replicate")
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff ** 2 + y_diff ** 2).mean([1, 2, 3])


class MakeCutouts(nn.Module):
    def __init__(self, cut_size: int, num_cutouts: int, cutout_size_power: float=1.0, augment_list: list=[]):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = num_cutouts
        self.cut_pow = cutout_size_power
        self.augs = nn.Sequential(*augment_list)

    def forward(self, input: th.Tensor):
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
            cutout = tf.interpolate(
                cutout,
                (self.cut_size, self.cut_size),
                mode="bilinear",
                align_corners=False,
            )
            cutouts.append(cutout)
        return self.augs(th.cat(cutouts))
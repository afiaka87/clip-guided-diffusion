import glob
import io
import os
import re
import subprocess
from pathlib import Path
from typing import Tuple, Union
from urllib import request
from PIL import Image

import requests
import torch as th
from guided_diffusion.script_util import (create_model_and_diffusion,
                                          model_and_diffusion_defaults)
from torch.nn import functional as tf
from torchvision.transforms import functional as tvf
from tqdm.autonotebook import tqdm

from data.diffusion_model_flags import DIFFUSION_LOOKUP

CACHE_PATH = os.path.expanduser("~/.cache/clip-guided-diffusion")
ALPHANUMERIC_REGEX = r"[^\w\s]"

# modified from https://github.com/lucidrains/DALLE-pytorch/blob/d355100061911b13e1f1c22de8c2b5deb44e65f8/dalle_pytorch/vae.py


def download(url: str, filename: str, root: str = CACHE_PATH) -> str:
    os.makedirs(root, exist_ok=True)
    download_target = Path(os.path.join(root, filename))
    download_target_tmp = download_target.with_suffix('.tmp')
    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(
            f"{download_target} exists and is not a regular file")
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


def download_guided_diffusion(image_size: int, class_cond: bool, checkpoints_dir: str = CACHE_PATH, overwrite: bool = False) -> str:
    cond_key = 'cond' if class_cond else 'uncond'
    diffusion_model_info = DIFFUSION_LOOKUP[cond_key][image_size]
    if not overwrite:
        target_path = Path(checkpoints_dir).joinpath(diffusion_model_info["filename"])
        if target_path.exists():
            return str(target_path)
    return download(diffusion_model_info["url"], diffusion_model_info["filename"], checkpoints_dir)


def load_guided_diffusion(
    checkpoint_path: str,
    image_size: int,
    class_cond: bool,
    diffusion_steps: int = None,
    timestep_respacing: str = None,
    use_fp16: bool = True,
    device: str = '',
    noise_schedule: str = 'linear',
    dropout: float = 0.0,
):
    '''
    checkpoint_path: path to the checkpoint to load.
    image_size: size of the images to be used.
    class_cond: whether to condition on the class label
    diffusion_steps: number of diffusion steps
    timestep_respacing: whether to use timestep-respacing or not
    '''
    if not (len(device) > 0): raise ValueError("device must be set")
    if not (noise_schedule in [ "linear", "cosine"]): raise ValueError("linear_or_cosine must be set")

    cond_key = 'cond' if class_cond else 'uncond'
    diffusion_model_info = DIFFUSION_LOOKUP[cond_key][image_size]
    model_config: dict = model_and_diffusion_defaults()
    model_config.update(diffusion_model_info['model_flags'])
    model_config.update(**{  # Custom params from user
        'diffusion_steps': diffusion_steps,
        'timestep_respacing': timestep_respacing,
        "use_fp16": use_fp16,
        "noise_schedule": noise_schedule,
        "dropout": dropout,
    })
    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(th.load(checkpoint_path, map_location='cpu'))
    model.requires_grad_(False).eval().to(device)
    for name, param in model.named_parameters():
        if "qkv" in name or "norm" in name or "proj" in name:
            param.requires_grad_()
    if model_config["use_fp16"]:
        model.convert_to_fp16()
    return model.to(device), diffusion


def create_video_from_image_dir(image_dir: str, target: str = "current.mp4", fps: int = 30) -> str:
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


def alphanumeric_filter(s: str) -> str:
    return re.sub(ALPHANUMERIC_REGEX, "", s).replace(" ", "_")

def combine_txt(txt, txt_min):
    txt_clean = alphanumeric_filter(txt)
    txt_min_clean = alphanumeric_filter(txt_min)
    if len(txt_min) > 0:
        return f"{txt_clean}_MIN_{txt_min_clean}"
    return txt_clean

def get_dir_for_prompt(base_path, txt, txt_min, batch_idx) -> str:
    return os.path.join(base_path, combine_txt(txt, txt_min), f"{batch_idx:02}")

def log_image(image: th.Tensor, base_path: str, txt: str, txt_min: str, current_step: int, batch_idx: int) -> str:
    dirname = get_dir_for_prompt(base_path, txt, txt_min, batch_idx)
    os.makedirs(dirname, exist_ok=True)
    stem = f"{current_step:04}"
    filename = os.path.join(dirname, f'{stem}.png')
    pil_image = tvf.to_pil_image(image.add(1).div(2).clamp(0, 1))
    pil_image.save(os.path.join(os.getcwd(), f'current.png'))
    pil_image.save(filename)
    return str(filename)

def create_gif(base, prompt, prompt_min, batch_idx):
    dirname = get_dir_for_prompt(base, prompt, prompt_min, batch_idx)
    images_glob = os.path.join(dirname, "*.png")
    imgs = [Image.open(f) for f in sorted(glob.glob(images_glob))]
    gif_filename = f"{base}/{combine_txt(prompt, prompt_min)}_{batch_idx:02}.gif"
    imgs[0].save(fp=gif_filename, format='GIF', append_images=imgs, save_all=True, duration=200, loop=0)
    return gif_filename

def resize_image(image: th.Tensor, out_size: Union[int, Tuple[int, int]]) -> th.Tensor:
    """(Katherine Crowson) - Resize image"""
    outsize_x = out_size if isinstance(out_size, int) else out_size[0]
    outsize_y = out_size if isinstance(out_size, int) else out_size[1]
    ratio = image.size(1) / image.size(1)
    area = min(image.size(0) * image.size(1), outsize_x * outsize_y)
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.reshape(size)


def spherical_dist_loss(x: th.Tensor, y: th.Tensor):
    """(Katherine Crowson) - Spherical distance loss"""
    x = tf.normalize(x, dim=-1)
    y = tf.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input: th.Tensor):
    """(Katherine Crowson) - L2 total variation loss, as in Mahendran et al."""
    input = tf.pad(input, (0, 1, 0, 1), "replicate")
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff ** 2 + y_diff ** 2).mean([1, 2, 3])

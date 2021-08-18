
from urllib import request
import io
import requests
import os
import re
from pathlib import Path

import requests
import torch as th
from guided_diffusion.script_util import (
    create_model_and_diffusion, model_and_diffusion_defaults)
from PIL import Image
from torch import nn
from torch.nn import functional as tf
from torchvision.transforms import functional as tvf
from tqdm import tqdm

CACHE_PATH = os.path.expanduser("~/.cache/clip-guided-diffusion")

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
DIFFUSION_256_UNCOND_MODEL_FLAGS = {
    "attention_resolutions": '32,16,8',
    "class_cond": False,
    "diffusion_steps": 1000,
    "image_size": 256,
    "learn_sigma": True,
    "noise_schedule": 'linear',
    "num_channels": 256,
    "num_head_channels": 64,
    "num_res_blocks": 2,
    "resblock_updown": True,
    "use_fp16": True,
    "use_scale_shift_norm": True
}
DIFFUSION_512_MODEL_FLAGS = {
    'attention_resolutions': '32,16,8',
    'class_cond': True,
    'diffusion_steps': 1000,
    'image_size': 512,
    'learn_sigma': True,
    'noise_schedule': 'linear',
    'num_channels': 256,
    'num_head_channels': 64,
    'num_res_blocks': 2,
    'resblock_updown': True,
    'use_fp16': False,
    'use_scale_shift_norm': True
}

# modified from https://github.com/lucidrains/DALLE-pytorch/blob/d355100061911b13e1f1c22de8c2b5deb44e65f8/dalle_pytorch/vae.py
def download(url, filename=None, root=CACHE_PATH):
    os.makedirs(root, exist_ok=True)
    filename = filename if os.path.basename(
        url) is not None else os.path.basename(url)
    download_target = os.path.join(root, filename)
    download_target_tmp = Path(os.path.join(root, f'tmp_{filename}'))
    download_target_tmp.mkdir(parents=True, exist_ok=True)
    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(
            f"{download_target} exists and is not a regular file")
    if os.path.isfile(download_target):
        return download_target
    os.rename(download_target_tmp, download_target)
    with request.urlopen(url) as source, open(download_target_tmp, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))
    return download_target


def download_guided_diffusion(image_size, checkpoints_dir=CACHE_PATH, class_cond=True, overwrite=False):
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    if class_cond:
        gd_path = checkpoints_dir.joinpath(
            Path(f'{image_size}x{image_size}_diffusion.pt'))
        gd_url = f'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/{image_size}x{image_size}_diffusion.pt'
    else:
        assert image_size == 256, "class_cond=False only works for 256x256 images"
        gd_path = checkpoints_dir.joinpath(Path("256x256_diffusion_uncond.pt"))
        gd_url = f'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt'

    if gd_path.exists() and not overwrite:
        print("Already downloaded")
        return gd_path

    print(f'Downloading {gd_url} to {gd_path}')
    download(gd_url, gd_path)
    return gd_path


def load_guided_diffusion(
    checkpoint_path,
    image_size,
    diffusion_steps=None,
    timestep_respacing=None,
    device=None,
    class_cond=True,
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
        if class_cond:
            model_config.update(DIFFUSION_256_MODEL_FLAGS)
        else:
            model_config.update(DIFFUSION_256_UNCOND_MODEL_FLAGS)
    elif image_size == 512:
        model_config.update(DIFFUSION_512_MODEL_FLAGS)
    else:
        raise ValueError("Invalid image size")

    model_config['diffusion_steps'] = diffusion_steps
    model_config['timestep_respacing'] = timestep_respacing

    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(th.load(checkpoint_path, map_location="cpu"))
    model.requires_grad_(False).eval().to(device)
    for name, param in model.named_parameters():
        if "qkv" in name or "norm" in name or "proj" in name:
            param.requires_grad_()
    if model_config["use_fp16"]:
        model.convert_to_fp16()
    return model, diffusion


def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')


def log_image(image, prefix_path, current_step, batch_idx):
    filename = os.path.join(
        prefix_path, f"{batch_idx:04}_iteration_{current_step:04}.png")
    pil_image = tvf.to_pil_image(image.add(1).div(2).clamp(0, 1))
    pil_image.save("current.png")
    return filename


def txt_to_dir(base_path, txt, txt_min=None):
    dir_name = f"{txt}_MIN_{txt_min}" if txt_min else txt
    dir_name = Path(
        re.sub(r"[^\w\s]", "", f"{dir_name}").replace(" ", "_")[:256])
    return Path(os.path.join(base_path, dir_name))


def resize_image(image, out_size):
    """Resize image"""
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)


def spherical_dist_loss(x, y):
    """Spherical distance loss"""
    x = tf.normalize(x, dim=-1)
    y = tf.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = tf.pad(input, (0, 1, 0, 1), "replicate")
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
            cutout = tf.interpolate(
                cutout,
                (self.cut_size, self.cut_size),
                mode="bilinear",
                align_corners=False,
            )
            cutouts.append(cutout)
        return self.augs(th.cat(cutouts))

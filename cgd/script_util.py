import glob
import io
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from urllib import request

import requests
import torch as th
import torchvision.transforms.functional as tvf
from tqdm.auto import tqdm
from data.diffusion_model_flags import DIFFUSION_LOOKUP
from PIL import Image

from cgd import clip_util
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

CACHE_PATH = os.path.expanduser("~/.cache/clip-guided-diffusion")
TIMESTEP_RESPACINGS = ("25", "50", "100", "250", "500", "1000",
                       "ddim25", "ddim50", "ddim100", "ddim250", "ddim500", "ddim1000")
DIFFUSION_SCHEDULES = (25, 50, 100, 250, 500, 1000)
IMAGE_SIZES = (64, 128, 256, 512)

def check_parameters(
    prompts: list,
    image_prompts: list,
    image_size: int,
    timestep_respacing: str,
    diffusion_steps: int,
    clip_model_name: str,
    save_frequency: int,
    noise_schedule: str,
):
    if not (len(prompts) > 0 or len(image_prompts) > 0):
        raise ValueError("Must provide at least one prompt, text or image.")
    if not (noise_schedule in ['linear', 'cosine']):
        raise ValueError('Noise schedule should be one of: linear, cosine')
    if not (image_size in IMAGE_SIZES):
        raise ValueError(f"--image size should be one of {IMAGE_SIZES}")
    if not (0 < save_frequency <= int(timestep_respacing.replace('ddim', ''))):
        raise ValueError(
            "--save_frequency must be greater than 0 and less than `timestep_respacing`")
    if not (diffusion_steps in DIFFUSION_SCHEDULES):
        print('(warning) Diffusion steps should be one of:', DIFFUSION_SCHEDULES)
    if not (timestep_respacing in TIMESTEP_RESPACINGS):
        print(
            f"Pausing run. `timestep_respacing` should be one of {TIMESTEP_RESPACINGS}. CTRL-C if this was a mistake.")
        time.sleep(5)
        print("Resuming run.")
    if clip_model_name.endswith('.pt') or clip_model_name.endswith('.pth'):
        assert os.path.isfile(
            clip_model_name), f"{clip_model_name} does not exist"
        print(f"Loading custom model from {clip_model_name}")
    elif not (clip_model_name in clip_util.CLIP_MODEL_NAMES):
        print(
            f"--clip model name should be one of: {clip_util.CLIP_MODEL_NAMES} unless you are trying to use your own checkpoint.")
        print(f"Loading OpenAI CLIP - {clip_model_name}")


def parse_prompt(prompt):  # parse a single prompt in the form "<text||img_url>:<weight>"
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)  # theres two colons, so we grab the 2nd
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)  # grab weight after colon
    vals = vals + ['', '1'][len(vals):]  # if no weight, use 1
    return vals[0], float(vals[1])  # return text, weight


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
    # regex to remove non-alphanumeric characters
    ALPHANUMERIC_REGEX = r"[^\w\s]"
    return re.sub(ALPHANUMERIC_REGEX, "", s).replace(" ", "_")


def clean_and_combine_prompts(base_path, txts, batch_idx, max_length=255) -> str:
    clean_txt = "_".join([alphanumeric_filter(txt)
                         for txt in txts])[:max_length]
    return os.path.join(base_path, clean_txt, f"{batch_idx:02}")


def log_image(image: th.Tensor, base_path: str, txts: list, current_step: int, batch_idx: int) -> str:
    dirname = clean_and_combine_prompts(base_path, txts, batch_idx)
    os.makedirs(dirname, exist_ok=True)
    stem = f"{current_step:04}"
    filename = os.path.join(dirname, f'{stem}.png')
    pil_image = tvf.to_pil_image(image.add(1).div(2).clamp(0, 1))
    pil_image.save(os.path.join(os.getcwd(), f'current.png'))
    pil_image.save(filename)
    return str(filename)


def create_gif(base, prompts, batch_idx):
    io_safe_prompts = clean_and_combine_prompts(base, prompts, batch_idx)
    images_glob = os.path.join(io_safe_prompts, "*.png")
    imgs = [Image.open(f) for f in sorted(glob.glob(images_glob))]
    gif_filename = f"{io_safe_prompts}_{batch_idx:02}.gif"
    imgs[0].save(fp=gif_filename, format='GIF', append_images=imgs,
                 save_all=True, duration=200, loop=0)
    return gif_filename


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
        target_path = Path(checkpoints_dir).joinpath(
            diffusion_model_info["filename"])
        if target_path.exists():
            return str(target_path)
    return download(diffusion_model_info["url"], diffusion_model_info["filename"], checkpoints_dir)


@lru_cache(maxsize=1)
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
    if not (len(device) > 0):
        raise ValueError("device must be set")
    if not (noise_schedule in ["linear", "cosine"]):
        raise ValueError("linear_or_cosine must be set")

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

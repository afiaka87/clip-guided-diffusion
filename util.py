import io
from functools import lru_cache

import requests
import torch
from guided_diffusion.script_util import (create_model_and_diffusion,
                                          model_and_diffusion_defaults)

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
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.requires_grad_(False).eval().to(device)
    for name, param in model.named_parameters():
        if "qkv" in name or "norm" in name or "proj" in name:
            param.requires_grad_()
    if model_config["use_fp16"]:
        model.convert_to_fp16()
    return model, diffusion


def parse_prompt(prompt):  # NR: Weights after colons
    vals = prompt.rsplit(":", 2)
    vals = vals + ["", "1", "-inf"][len(vals):]
    return vals[0], float(vals[1]), float(vals[2])


def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith(
            'https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')

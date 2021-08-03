import argparse
import io
import requests
import json
import os
import sys
import random
from functools import lru_cache
import torch

from guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)

# load the list of classes from yaml file
imagenet_class_labels_filename = "imagenet1000_clsidx_to_labels.json"


@lru_cache(maxsize=1)
def get_idx_to_class_map():
    with open(imagenet_class_labels_filename, "r") as infile:
        mapping = json.load(infile)
        return mapping


@lru_cache(maxsize=None)
def get_classes_for_idx(idx):
    idx_to_class_mapping = get_idx_to_class_map()
    return idx_to_class_mapping[str(idx)]


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
    # TODO Docs says to use cosine but loss goes to NaN
    model_config = model_and_diffusion_defaults()
    model_config.update(
        {
            "attention_resolutions": "32, 16, 8",
            "class_cond": class_cond,
            "diffusion_steps": diffusion_steps,
            "rescale_timesteps": rescale_timesteps,
            "timestep_respacing": timestep_respacing,
            "image_size": image_size,
            "learn_sigma": True,
            "noise_schedule": "linear", 
            "num_channels": 192 if class_cond else 256,
            "num_head_channels": 64,
            "num_res_blocks": 3 if class_cond else 2,
            "resblock_updown": True,
            "use_new_attention_order": True if class_cond else False,
            "use_fp16": True,
            "use_scale_shift_norm": True,
        }
    )
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
    vals = vals + ["", "1", "-inf"][len(vals) :]
    return vals[0], float(vals[1]), float(vals[2])

def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')
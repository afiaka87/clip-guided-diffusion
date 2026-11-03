"""Scratch: how the 64 to 256 upsampler responds to its step count and to softening its input."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import torch.nn.functional as tf
from PIL import Image
import torchvision.transforms.functional as tvf
from guided_diffusion.script_util import sr_model_and_diffusion_defaults, sr_create_model_and_diffusion
from cgd import clip_util, sampling, script_util

DEVICE = "cuda"
config = sr_model_and_diffusion_defaults()
config.update(large_size=256, small_size=64, class_cond=True, learn_sigma=True, diffusion_steps=1000, noise_schedule="linear", num_channels=192, num_res_blocks=2,
              resblock_updown=True, use_fp16=True, use_scale_shift_norm=True, timestep_respacing="", use_checkpoint=False, attention_resolutions="32,16,8", num_heads=4)
with th.device("meta"):
    model, diffusion = sr_create_model_and_diffusion(**config)
model.load_state_dict(th.load(os.path.join(SP, "weights_sr", "64_256_upsampler.pt"), map_location="cpu", mmap=True, weights_only=True), assign=True)
model.requires_grad_(False).eval()
model.convert_to_fp16()
model = model.to(DEVICE)
base = paths("d64")
classes = {}
for prompt in PROMPTS:
    target, _ = clip_util.encode_text_prompt(prompt, 1.0, "ViT-B/32", DEVICE, False)
    classes[prompt] = clip_util.imagenet_top_n(tf.normalize(target, dim=-1), device=DEVICE, n=1, clip_model_name="ViT-B/32", fp16=False).reshape(1)


def variant(name, steps=30, blur=0.0, eta=0.5):
    for prompt in PROMPTS:
        low = tvf.to_tensor(Image.open(base[prompt]).convert("RGB")).unsqueeze(0).to(DEVICE)
        if blur:
            low = tvf.gaussian_blur(low, [5, 5], [blur, blur])
        kwargs = {"y": classes[prompt], "low_res": low.mul(2).sub(1)}

        def run():
            th.manual_seed(0)
            for sample in sampling.sample_loop_progressive(model, diffusion, (1, 3, 256, 256), steps=steps, eta=eta, model_kwargs=kwargs, device=DEVICE, progress=False):
                final = sample["pred_xstart"]
            return final
        image, stats = measured(run)
        folder = os.path.join(HERE, "out", name)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, script_util.alphanumeric_filter(prompt) + ".png")
        tvf.to_pil_image(image[0].add(1).div(2).clamp(0, 1).float().cpu()).save(path)
        record(name, prompt, path, stats)


variant("sr256_n10", steps=10)
variant("sr256_n60", steps=60)
variant("sr256_blur06", blur=0.6)
variant("sr256_blur10", blur=1.0)
print("run3 done", flush=True)

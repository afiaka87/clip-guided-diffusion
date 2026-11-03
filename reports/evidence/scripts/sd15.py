import json, os, sys, time
import torch as th
from diffusers import StableDiffusionPipeline
S = sys.argv[1]
exec(open(f"{S}/scripts/prompts.py").read())
pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=th.float16, safety_checker=None, requires_safety_checker=False).to("cuda")
pipe.set_progress_bar_config(disable=True)
d = f"{S}/turbo/sd15_cfg7.5_20"; os.makedirs(d, exist_ok=True); out = []
t0 = time.perf_counter()
for i, p in enumerate(PROMPTS):
    img = pipe(p, num_inference_steps=20, guidance_scale=7.5, generator=th.Generator("cuda").manual_seed(0)).images[0]
    img.save(f"{d}/{i}.png"); out.append(f"{d}/{i}.png")
print((time.perf_counter() - t0) / 12, "s/img", th.cuda.max_memory_allocated() / 2**30, "GiB")
paths = json.load(open(f"{S}/results/turbo_paths.json")); paths["SD 1.5 CFG 7.5 20 steps"] = out
json.dump(paths, open(f"{S}/results/turbo_paths.json", "w"), indent=0)

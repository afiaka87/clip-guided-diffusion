"""CLIP forward and backward cost per cutout: seconds and VRAM, with a linear fit."""
import sys
import time

import numpy as np
import torch as th

sys.path.insert(0, "/home/sam/GitHub/clip-guided-diffusion")
from cgd import clip_util  # noqa: E402

device = "cuda"
th.manual_seed(0)
models = [("ViT-B/32", False), ("ViT-B/16", False), ("PE-Core-B-16", False), ("PE-Core-B-16", True),
          ("ViT-L-16-SigLIP2-256", False), ("ViT-L-16-SigLIP2-256", True), ("PE-Core-L-14-336", False), ("PE-Core-L-14-336", True),
          ("ViT-SO400M-16-SigLIP2-384", True)]
print("gpu", th.cuda.get_device_name(0), flush=True)
for name, fp16 in models:
    clip_util.load_clip.cache_clear()
    th.cuda.empty_cache()
    model, size = clip_util.load_clip(name, device, fp16)
    base = th.cuda.memory_allocated()
    rows = []
    for n in (1, 2, 4, 8, 16, 32):
        try:
            th.cuda.reset_peak_memory_stats()
            times = []
            for repeat in range(6):
                x = th.rand(n, 3, size, size, device=device, requires_grad=True)
                th.cuda.synchronize()
                start = time.perf_counter()
                embeds = model.encode_image(clip_util.preprocess(model, x)).float()
                grad = th.autograd.grad(embeds.square().sum(), x)[0]
                th.cuda.synchronize()
                if repeat:
                    times.append(time.perf_counter() - start)
            rows.append((n, float(np.median(times)), (th.cuda.max_memory_allocated() - base) / 2**20))
            del x, embeds, grad
        except th.cuda.OutOfMemoryError:
            th.cuda.empty_cache()
            break
    ns, seconds, memory = map(np.array, zip(*rows))
    time_fit = np.polyfit(ns, seconds, 1)
    memory_fit = np.polyfit(ns, memory, 1)
    print(f"{name:28s} fp16={fp16!s:5s} in={size} weights={base / 2**20:6.0f} MiB  "
          f"{time_fit[0] * 1e3:7.1f} ms/cut + {time_fit[1] * 1e3:5.1f} ms  {memory_fit[0]:6.1f} MiB/cut + {memory_fit[1]:6.1f} MiB  "
          f"rows={[(int(n), round(s * 1e3, 1), round(m)) for n, s, m in rows]}", flush=True)

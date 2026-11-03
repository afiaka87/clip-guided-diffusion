"""Which stage of Lightning at 1024 runs out of memory. argv: <REPO>."""
import sys, time
sys.path.insert(0, sys.argv[1])
import torch as th
from cgd import modern, clip_util
G = 2 ** 30
def mem(tag): print(f"{tag}: now {th.cuda.memory_allocated() / G:.2f} peak {th.cuda.max_memory_allocated() / G:.2f} reserved {th.cuda.memory_reserved() / G:.2f}", flush=True)
B = modern.load_backbone("sdxl-lightning", "a red fox in the snow", "cuda"); mem("loaded")
shape = B.latent_shape(1024, 1024); th.manual_seed(1); z = th.randn(shape, device="cuda")
th.cuda.reset_peak_memory_stats()
with th.no_grad():
    t = time.perf_counter(); lat = B.sample(z, 1); mem(f"unet {time.perf_counter() - t:.1f}s")
    try:
        t = time.perf_counter(); img = B.decode(lat); mem(f"decode with unet on card {time.perf_counter() - t:.1f}s")
    except th.cuda.OutOfMemoryError as e:
        print("decode OOM:", str(e)[:120], flush=True); del e
th.cuda.empty_cache(); th.cuda.reset_peak_memory_stats()
guide = clip_util.ClipGuide("ViT-B/32+PE-Core-B-16", ["a red fox in the snow"], (), 1024, 4, "cuda", min_cut_fraction=.7, aesthetic_scale=.1); mem("guide loaded")
try:
    t = time.perf_counter()
    for step in modern.optimise_noise(B, guide, z, iterations=3, progress=False): pass
    mem(f"3 guided steps {time.perf_counter() - t:.1f}s")
except th.cuda.OutOfMemoryError as e:
    print("guidance OOM:", str(e)[:160], flush=True); mem("at failure")

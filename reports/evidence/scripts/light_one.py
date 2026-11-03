"""One Lightning 1024 image through generate(). argv: <REPO> <OUT> <clip> <aes>."""
import sys, time
sys.path.insert(0, sys.argv[1])
import torch as th
from cgd import modern
t = time.perf_counter()
try:
    for _ in modern.generate(["a red fox in the snow"], backbone="sdxl-lightning", image_size=1024, seed=1, prefix_path=sys.argv[2], progress=False, save_frequency=1000,
                             clip_model_name=sys.argv[3], aesthetic_scale=float(sys.argv[4])):
        pass
    print("RESULT", sys.argv[3], sys.argv[4], f"ok {time.perf_counter() - t:.1f}s peak {th.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
except th.cuda.OutOfMemoryError:
    print("RESULT", sys.argv[3], sys.argv[4], f"OOM after {time.perf_counter() - t:.1f}s peak {th.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)

"""Plain model through the real generate(), time and peak per backbone and size. argv: <REPO> <OUT>."""
import sys, time
sys.path.insert(0, sys.argv[1])
import torch as th
from cgd import modern
exec(open(sys.argv[3]).read())
for name, size in (("sd-turbo", 512), ("vega", 768), ("vega", 1024), ("sdxl-lightning", 768)):
    th.cuda.reset_peak_memory_stats(); t0 = time.perf_counter(); stamps = []
    for i, p in enumerate(PROMPTS[:3]):
        t = time.perf_counter()
        for _ in modern.generate([p], backbone=name, image_size=size, seed=1, iterations=0, prefix_path=f"{sys.argv[2]}/{name}{size}/{i}", progress=False, save_frequency=1000):
            pass
        stamps.append(time.perf_counter() - t)
    print("RESULT", name, size, "seconds per run incl load", [round(v, 1) for v in stamps], f"peak {th.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)

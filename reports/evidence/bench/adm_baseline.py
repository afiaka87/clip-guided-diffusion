"""Check the method against the README: the real 256px ADM checkpoint, one guided step, same loss shape."""
import os, time
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch as th, open_clip
from cgd import script_util
GIB=1024**3
for size in (128, 256, 512):
    path = script_util.download_guided_diffusion(size, True)
    model, _ = script_util.load_guided_diffusion(path, size, True, use_fp16=True, device="cuda", use_checkpoint=size>256)
    clip = open_clip.create_model("ViT-B-32", device="cuda").float().requires_grad_(False).eval()
    th.cuda.empty_cache(); th.cuda.reset_peak_memory_stats()
    x = th.randn(1,3,size,size,device="cuda",requires_grad=True)
    t = th.full((1,),500.0,device="cuda"); y = th.zeros((1,),device="cuda",dtype=th.long)
    start=time.perf_counter()
    eps = model(x,t,y=y)[:,:3]; x0 = x-eps
    batch = th.nn.functional.interpolate(x0.float().add(1).div(2),size=224,mode="bilinear").repeat(4,1,1,1)
    th.autograd.grad(clip.encode_image(batch).square().mean(), x)
    th.cuda.synchronize()
    print(size, "params_M", round(sum(p.numel() for p in model.parameters())/1e6,1), "peak_alloc", round(th.cuda.max_memory_allocated()/GIB,2), "peak_res", round(th.cuda.max_memory_reserved()/GIB,2), "s", round(time.perf_counter()-start,2), flush=True)
    del model, clip, x, eps, x0, batch; script_util.load_guided_diffusion.cache_clear(); th.cuda.empty_cache()

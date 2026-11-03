import sys, hashlib
sys.path.insert(0, sys.argv[1])
import torch as th
from cgd import clip_util, modern
P = "a photograph of a red fox in the snow"; dev = "cuda"
b = modern.load_backbone("sd-turbo", P, dev, False)
guide = clip_util.ClipGuide("ViT-B/32", [P], (), 256, 4, dev, min_cut_fraction=.7)
def h(t): return hashlib.sha1(t.detach().cpu().numpy().tobytes()).hexdigest()[:8]
th.manual_seed(0); z0 = th.randn(1, 4, 32, 32, device=dev)
def rep(name, f, n=6):
    print(name, sorted({f() for _ in range(n)}), flush=True)
def forward():
    with th.no_grad(): return h(b.denoise(z0))
rep("unet forward", forward)
lat = None
with th.no_grad(): lat = b.denoise(z0)
def vae_forward():
    with th.no_grad(): return h(b.decode(lat))
rep("vae forward", vae_forward)
def unet_back():
    z = z0.clone().requires_grad_(); return h(th.autograd.grad(b.denoise(z).square().sum(), z)[0])
rep("unet backward", unet_back)
def vae_back():
    l = lat.clone().requires_grad_(); return h(th.autograd.grad(b.decode(l).square().sum(), l)[0])
rep("vae backward", vae_back)
img = b.decode(lat)
def clip_back():
    th.manual_seed(1); x = img.clone().requires_grad_(); return h(th.autograd.grad(guide.loss(x), x)[0])
rep("clip backward", clip_back)
b2 = None
import torch.nn.functional as F

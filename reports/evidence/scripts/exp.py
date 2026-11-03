"""SD-Turbo on the pilot prompts: plain, CLIP noise optimisation, and refining ADM output. Prototype."""
import json, os, sys, time
S, REPO = sys.argv[1], sys.argv[2]; ONLY = sys.argv[3].split(",") if len(sys.argv) > 3 else None
SEEDS = [int(v) for v in os.environ.get("SEEDS", "0").split(",")]
sys.path.insert(0, REPO)
import torch as th, torch.nn.functional as tf
from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler
from diffusers.models.attention_processor import AttnProcessor
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision.transforms.functional import to_tensor
from PIL import Image
from cgd import clip_util, losses, modules
exec(open(f"{S}/scripts/prompts.py").read())
dev = "cuda"; HF = "stabilityai/sd-turbo"
tok = CLIPTokenizer.from_pretrained(HF, subfolder="tokenizer")
te = CLIPTextModel.from_pretrained(HF, subfolder="text_encoder", variant="fp16", torch_dtype=th.float16).to(dev)
with th.no_grad():
    emb = {p: te(tok(p, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(dev))[0] for p in PROMPTS}
del te; th.cuda.empty_cache()
unet = UNet2DConditionModel.from_pretrained(HF, subfolder="unet", variant="fp16", torch_dtype=th.float16).to(dev).eval().requires_grad_(False)
vae = AutoencoderKL.from_pretrained(HF, subfolder="vae", variant="fp16", torch_dtype=th.float16).to(dev).eval().requires_grad_(False)
unet.set_attn_processor(AttnProcessor()); vae.set_attn_processor(AttnProcessor())
unet.enable_gradient_checkpointing(); vae.enable_gradient_checkpointing()
sch = EulerDiscreteScheduler.from_pretrained(HF, subfolder="scheduler")
SF = vae.config.scaling_factor
ab = sch.alphas_cumprod.to(dev)
def sig(t): return ((1 - ab[t]) / ab[t]).sqrt().float()
def denoise(z_scaled, t, e):
    """z_scaled is the VP sample, z = a x0 + s eps. Returns x0 in latent units."""
    eps = unet(z_scaled.to(th.float16), t, encoder_hidden_states=e).sample.float()
    a, s = ab[t].sqrt().float(), (1 - ab[t]).sqrt().float()
    return (z_scaled - s * eps) / a
def decode(z): return vae.decode((z / SF).to(th.float16)).sample.float().add(1).div(2)
def steps_sample(noise, e, ts):
    x0 = None; z = noise
    for i, t in enumerate(ts):
        x0 = denoise(z, t, e)
        if i + 1 < len(ts):
            n = ts[i + 1]; z = ab[n].sqrt() * x0 + (1 - ab[n]).sqrt() * th.randn_like(z)
    return x0
TOWERS = {}
def tower(name):
    if name not in TOWERS:
        clip_util.load_clip.cache_clear()
        m, s = clip_util.load_clip(name, dev); TOWERS[name] = (m, s, modules.MakeCutouts(s, 4, min_cut_fraction=.7))
    return TOWERS[name]
TGT = {}
def target(name, p):
    if (name, p) not in TGT:
        clip_util.load_clip.cache_clear(); clip_util.load_clip = (lambda real: real)(clip_util.load_clip)
        m, _, _ = tower(name)
        with th.no_grad():
            if name in clip_util.OPEN_CLIP_MODELS: t = m.encode_text(clip_util._open_clip_tokenizer(name)([p])).float().to(dev)
            else:
                import clip as oc; t = m.encode_text(oc.tokenize([p]).to(dev)).float()
        TGT[(name, p)] = tf.normalize(t, dim=-1)
    return TGT[(name, p)]
def clip_loss(img, names, p):
    out = 0
    for n in names:
        m, s, cut = tower(n)
        e = m.encode_image(clip_util.preprocess(m, cut(img))).float()
        out = out + losses.spherical_dist_loss(e.unsqueeze(1), target(n, p)).mean()
    return out / len(names)
def reno(p, seed, names, iters=50, lr=5., clip=.1, reg=.01, scale=1., tv=0., keep_best=True, multi=False):
    th.manual_seed(seed)
    z = th.randn(1, 4, 64, 64, device=dev).requires_grad_(True)
    opt = th.optim.SGD([z], lr=lr, momentum=.9, nesterov=True)
    best, best_loss = None, 1e9
    D = z.numel()
    for i in range(iters):
        opt.zero_grad()
        img = decode(denoise(z, th.tensor(999, device=dev), emb[p]))
        l = clip_loss(img, names, p); v = l.item()
        if tv: l = l + tv * losses.tv_loss(img * 2 - 1).mean()
        if v < best_loss: best_loss, best = v, (img.detach().clone(), z.detach().clone())
        if i == iters - 1: break
        n = z.norm(); total = scale * l + reg * (.5 * n ** 2 - (D - 1) * n.log())
        total.backward(); th.nn.utils.clip_grad_norm_([z], clip); opt.step()
    return (best[0] if keep_best else img.detach()), best_loss
def pil(x): return Image.fromarray(x[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy())
base = json.load(open(f"{S}/results/pilot_paths.json"))["base ViT-B/32 tv150"]
def refine(i, p, seed, strength, up=Image.BICUBIC, steps=1):
    th.manual_seed(seed)
    src = Image.open(base[i]).convert("RGB").resize((512, 512), up)
    x = to_tensor(src).unsqueeze(0).to(dev).mul(2).sub(1).to(th.float16)
    z0 = vae.encode(x).latent_dist.mean.float() * SF
    t0 = int(strength * 999); ts = th.linspace(t0, 0, steps + 1).long()[:-1].to(dev)
    z = ab[t0].sqrt() * z0 + (1 - ab[t0]).sqrt() * th.randn_like(z0)
    return decode(steps_sample(z, emb[p], ts))
B32, PE = "ViT-B/32", "PE-Core-B-16"
V = {
 "turbo 1step": lambda i, p, s: (th.manual_seed(s), decode(denoise(th.randn(1, 4, 64, 64, device=dev), th.tensor(999, device=dev), emb[p])))[1],
 "turbo 4step": lambda i, p, s: (th.manual_seed(s), decode(steps_sample(th.randn(1, 4, 64, 64, device=dev), emb[p], th.tensor([999, 749, 499, 249], device=dev))))[1],
 "reno B32": lambda i, p, s: reno(p, s, [B32])[0],
 "reno PE": lambda i, p, s: reno(p, s, [PE])[0],
 "reno B32+PE": lambda i, p, s: reno(p, s, [B32, PE])[0],
 "reno B32 x100": lambda i, p, s: reno(p, s, [B32], scale=100)[0],
 "refine .3": lambda i, p, s: refine(i, p, s, .3),
 "refine .5": lambda i, p, s: refine(i, p, s, .5),
 "refine .7": lambda i, p, s: refine(i, p, s, .7),
 "refine .7 2step": lambda i, p, s: refine(i, p, s, .7, steps=2),
}
paths = {}
out_json = f"{S}/results/turbo_paths.json"
if os.path.exists(out_json): paths = json.load(open(out_json))
for name, f in V.items():
    if ONLY and name not in ONLY: continue
    for seed in SEEDS:
        key = name if seed == 0 else f"{name} s{seed}"
        d = f"{S}/turbo/{key.replace(' ', '_')}"; os.makedirs(d, exist_ok=True); paths[key] = []
        th.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
        for i, p in enumerate(PROMPTS):
            with (th.enable_grad() if "reno" in name else th.no_grad()):
                img = f(i, p, seed)
            pil(img).save(f"{d}/{i}.png"); paths[key].append(f"{d}/{i}.png")
        print(key, f"{(time.perf_counter() - t0) / len(PROMPTS):.1f}s/img peak {th.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
        json.dump(paths, open(out_json, "w"), indent=0)
print("done")

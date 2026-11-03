"""OpenAI's noise-aware classifier added to CLIP guidance in the repo's fast sampler at 128px. Only the repo's own dependencies."""
import inspect, json, os, sys, time
S, REPO = sys.argv[1], sys.argv[2]
sys.path.insert(0, REPO)
import torch as th, torch.nn.functional as tf
from PIL import Image
from cgd import clip_util, losses, modules, sampling, script_util
from data.diffusion_model_flags import DIFFUSION_LOOKUP
from guided_diffusion.script_util import classifier_defaults, create_classifier

exec(open(os.path.join(S, "scripts", "prompts.py")).read())
SMOKE = os.environ.get("PILOT_SMOKE", "")
DEV = "cpu" if SMOKE else "cuda"
STEPS = 4 if SMOKE else sampling.DEFAULT_STEPS
if SMOKE:
    PROMPTS = PROMPTS[:1]
CKPT = os.path.expanduser("~/.cache/clip-guided-diffusion")
W = os.path.join(S, "weights_noise")

gd_model, diffusion = script_util.load_guided_diffusion(
    checkpoint_path=os.path.join(CKPT, DIFFUSION_LOOKUP["cond"][128]["filename"]), image_size=128, class_cond=True,
    timestep_respacing="", use_fp16=DEV != "cpu", device=DEV, use_checkpoint=False)


class NoBackward(th.nn.Module):
    """The U-Net with no graph, for runs with no guidance through it."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        with th.no_grad():
            return self.model(*args, **kwargs)


# This fork's EncoderUNetModel passes a flag ResBlock does not take. Dropping it changes nothing the checkpoint holds.
import guided_diffusion.unet as gd_unet
real_res_init = gd_unet.ResBlock.__init__
def res_init(self, *args, use_neighborhood_attention=False, **kwargs):
    real_res_init(self, *args, **kwargs)
gd_unet.ResBlock.__init__ = res_init
# It also never gives the encoder's attention layers the `spatial` they need for neighborhood attention, which is unused here.
for attention in (gd_unet.QKVAttention, gd_unet.QKVAttentionLegacy):
    attention.__init__ = (lambda real: lambda self, n_heads, spatial=None, *a: real(self, n_heads, spatial or (0, 0), *a))(attention.__init__)

kw = classifier_defaults()
kw.update(image_size=128, classifier_attention_resolutions="32,16,8", classifier_depth=2, classifier_width=128,
          classifier_pool="attention", classifier_resblock_updown=True, classifier_use_scale_shift_norm=True,
          classifier_use_fp16=DEV != "cpu")
classifier = create_classifier(**kw)
classifier.load_state_dict(th.load(os.path.join(W, "128x128_classifier.pt"), map_location="cpu", weights_only=True))
if DEV != "cpu":
    classifier.convert_to_fp16()
classifier = classifier.to(DEV).eval().requires_grad_(False)

EXTRA = {"fn": None}
source = inspect.getsource(sampling.sample_loop_progressive).replace(
    "        guided_x0 = x0.detach()\n",
    "        guided_x0 = x0.detach()\n"
    "        if EXTRA['fn'] is not None:\n"
    "            guided_x0 = guided_x0 + sigma ** 2 / alpha * EXTRA['fn'](x, batch_t, model_kwargs)\n")
sampling.EXTRA = EXTRA
exec(source, sampling.__dict__)


def classifier_grad(scale):
    def f(x, t, kwargs):
        with th.enable_grad():
            x_ = x.detach().requires_grad_(True)
            log_probs = tf.log_softmax(classifier(x_, t), dim=-1)
            return scale * th.nan_to_num(th.autograd.grad(log_probs[range(len(x_)), kwargs["y"]].sum(), x_)[0])
    return f


# The default guidance of the fast sampler, ViT-B/32 with the repo's defaults, to add the classifier to.
clip_model, clip_size = clip_util.load_clip("ViT-B/32", DEV)
make_cutouts = modules.MakeCutouts(cut_size=clip_size, num_cutouts=4, min_cut_fraction=0.7)


def clip_loss(target):
    def f(x, pred_xstart, fac):
        x_in = pred_xstart * fac + x * (1 - fac)
        embeds = clip_model.encode_image(clip_util.preprocess(clip_model, make_cutouts(x_in.add(1).div(2)))).float()
        clip_l = losses.spherical_dist_loss(embeds.view(4, 1, 1, -1), target).sum(2).mean(0).sum() * 1000
        return clip_l + losses.range_loss(pred_xstart).sum() * 50 + losses.tv_loss(x_in).sum() * 150
    return f


def run(class_index, loss_fn, model, window, rate, extra, out):
    th.manual_seed(0)
    EXTRA["fn"] = extra
    for sample in sampling.sample_loop_progressive(model, diffusion, (1, 3, 128, 128), loss_fn=loss_fn, steps=STEPS, guidance_rate=rate,
                                                   guidance_window=window, model_kwargs={"y": th.tensor([class_index], device=DEV)},
                                                   device=DEV, progress=False):
        pass
    arr = sample["pred_xstart"][0].add(1).div(2).clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(arr).save(out)


FREE = NoBackward(gd_model)
V = {
    "class only, no CLIP": lambda c: (None, FREE, sampling.GUIDANCE_WINDOW, 0.0, None),
    "class + classifier 1, no CLIP": lambda c: (None, FREE, sampling.GUIDANCE_WINDOW, 0.0, classifier_grad(1.0)),
    "base + classifier .5": lambda c: (clip_loss(c), gd_model, sampling.GUIDANCE_WINDOW, 0.08, classifier_grad(0.5)),
    "base + classifier 1": lambda c: (clip_loss(c), gd_model, sampling.GUIDANCE_WINDOW, 0.08, classifier_grad(1.0)),
}

Z = {}
for p in PROMPTS:
    with th.no_grad():
        target = clip_util.encode_texts([p], "ViT-B/32", DEV)
        Z[p] = (target, clip_util.imagenet_top_n(tf.normalize(target, dim=-1), device=DEV, n=1, clip_model_name="ViT-B/32").item())

paths = {}
for name, make in V.items():
    d = os.path.join(S, "pilot2", name.replace(" ", "_").replace(",", ""))
    os.makedirs(d, exist_ok=True)
    paths[name] = []
    if not SMOKE:
        th.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for i, p in enumerate(PROMPTS):
        target, cls = Z[p]
        loss_fn, model, window, rate, extra = make(target)
        out = os.path.join(d, f"{i}.png")
        run(cls, loss_fn, model, window, rate, extra, out)
        paths[name].append(out)
    peak = 0 if SMOKE else th.cuda.max_memory_allocated() / 2**30
    print(name, f"{(time.perf_counter() - t0) / len(PROMPTS):.1f}s/img, peak {peak:.2f} GiB", flush=True)
if not SMOKE:
    json.dump(paths, open(os.path.join(S, "results", "pilot_paths2.json"), "w"), indent=0)
print("done", flush=True)

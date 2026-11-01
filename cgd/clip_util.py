from contextlib import contextmanager
from functools import lru_cache
import os
import re
from pathlib import Path

import clip
import torch as th
import torch.nn.functional as tf
import torchvision.transforms as tvt
import torchvision.transforms.functional as tvf
from data.imagenet1000_clsidx_to_labels import IMAGENET_CLASSES
from PIL import Image

from cgd import script_util
from cgd.modules import MakeCutouts
from cgd.ResizeRight import resize_right
from cgd.ResizeRight.interp_methods import lanczos3

OPENAI_CLIP_MODEL_NAMES = ("ViT-B/16", "ViT-B/32", "RN50", "RN101", "RN50x4", "RN50x16", "ViT-L/14")

# Perception Encoder and SigLIP 2 models from open_clip, lightest first, and the weights
# to load for each. The README lists what each one costs.
OPEN_CLIP_MODELS = {
    "PE-Core-B-16": "meta",
    "ViT-L-16-SigLIP2-256": "webli",
    "PE-Core-L-14-336": "meta",
    "ViT-SO400M-16-SigLIP2-384": "webli",
    "ViT-gopt-16-SigLIP2-384": "webli",
    "PE-Core-bigG-14-448": "meta",
}

CLIP_MODEL_NAMES = (*OPENAI_CLIP_MODEL_NAMES, *OPEN_CLIP_MODELS)
CLIP_CACHE_PATH = os.path.join(script_util.CACHE_PATH, "clip")

# CLIP model URLs from OpenAI
CLIP_MODEL_URLS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


def download_clip_model(model_name: str) -> str:
    """Download CLIP model using our robust download function."""
    if model_name not in CLIP_MODEL_URLS:
        raise ValueError(f"Unknown CLIP model: {model_name}. Available: {list(CLIP_MODEL_URLS.keys())}")

    url = CLIP_MODEL_URLS[model_name]
    # Convert model name to filename (e.g., "ViT-B/32" -> "ViT-B-32.pt")
    filename = model_name.replace("/", "-") + ".pt"

    return script_util.download(url, filename, root=CLIP_CACHE_PATH)


CLIP_NORMALIZE = tvt.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

_INIT_FUNCTIONS = ("normal_", "uniform_", "constant_", "kaiming_uniform_", "xavier_uniform_", "xavier_normal_")


@contextmanager
def skip_weight_init():
    """Skip random init while building a model the checkpoint will overwrite anyway.

    The `clip` library initialises every parameter before loading the real ones, which
    makes up most of its load time.
    """
    originals = {name: getattr(th.nn.init, name) for name in _INIT_FUNCTIONS}
    for name in _INIT_FUNCTIONS:
        setattr(th.nn.init, name, lambda tensor, *args, **kwargs: tensor)
    try:
        yield
    finally:
        for name, function in originals.items():
            setattr(th.nn.init, name, function)


def check_clip_model(model_name: str):
    """Reject a CLIP model that cannot be loaded, before spending time on anything else."""
    if model_name not in OPEN_CLIP_MODELS and model_name not in clip.available_models() and not os.path.isfile(model_name):
        raise script_util.ConfigError(f"Unknown CLIP model {model_name!r}. Choose one of {list(CLIP_MODEL_NAMES)}, "
                                      "or give the path of a checkpoint.")


def _load_open_clip(model_name: str, device: str, fp16: bool):
    import open_clip  # Slow to import, and only these models need it.
    from huggingface_hub.utils import logging as hub_logging

    hub_logging.set_verbosity_error()  # The Hub otherwise asks for a token on every run.
    with skip_weight_init():
        clip_model = open_clip.create_model(model_name, pretrained=OPEN_CLIP_MODELS[model_name], cache_dir=CLIP_CACHE_PATH)
    clip_model = clip_model.eval().requires_grad_(False)
    # Text is only encoded once, up front, so that tower stays on the CPU. It is up to
    # two thirds of the weights and guidance has no use for it.
    # fp32 is the exact computation, so it is the default. fp16's gradient can be far from
    # it, as low as a cosine of 0.11, but on these towers noise of about that size on the
    # input moves the exact gradient as far, and fp16 scored about the same on held out
    # prompts. OpenAI's models have a smoother gradient, and always use fp16.
    clip_model.visual.to(device, th.float16 if fp16 and device != "cpu" else th.float32)
    # Guidance backpropagates through the vision tower. Recomputing its activations
    # during the backward pass takes 30% more time and a quarter of the memory.
    clip_model.visual.set_grad_checkpointing(True)
    # The fused attention kernels have a nondeterministic backward pass, which makes a
    # seed produce a different image every run. timm's plain path computes the same
    # gradient for up to 3% more time, and does not depend on which kernel a GPU gets.
    for module in clip_model.visual.modules():
        if hasattr(module, "fused_attn"):
            module.fused_attn = False
    return clip_model


@lru_cache(maxsize=1)
def load_clip(model_name='ViT-B/32', device="cpu", fp16=False):
    if "cuda" not in device and device != "cpu":
        raise ValueError("Invalid or unspecified device: {}".format(device))

    if model_name in OPEN_CLIP_MODELS:
        with th.random.fork_rng(devices=[]):
            clip_model = _load_open_clip(model_name, device, fp16)
        return clip_model, clip_model.visual.image_size[0]

    # Use our robust download function for known models
    if model_name in CLIP_MODEL_URLS:
        model_path = download_clip_model(model_name)
    else:
        # Fall back to clip library's download for unknown models
        model_path = model_name

    # Building CLIP draws random numbers. Fork the stream so a seed gives the same
    # image whether or not this call was a cache hit.
    with th.random.fork_rng(devices=[]), skip_weight_init():
        clip_model = clip.load(model_path, jit=False, device=device)[0]
    clip_model = clip_model.eval().requires_grad_(False)
    return clip_model, clip_model.visual.input_resolution


def preprocess(clip_model, images: th.Tensor) -> th.Tensor:
    """Prepare images in [0, 1] for `clip_model`: its normalization, in its precision."""
    visual = clip_model.visual
    normalized = tvf.normalize(images, *_normalization(visual))
    return normalized.to(next(visual.parameters()).dtype)


def _normalization(visual):
    """The (mean, std) a vision tower was trained with. Only open_clip's record one."""
    if hasattr(visual, "image_mean"):
        return visual.image_mean, visual.image_std
    return CLIP_NORMALIZE.mean, CLIP_NORMALIZE.std


@lru_cache(maxsize=1)
def _open_clip_tokenizer(model_name: str):
    import open_clip
    return open_clip.get_tokenizer(model_name, cache_dir=CLIP_CACHE_PATH)


def encode_texts(texts, clip_model_name: str, device: str, truncate: bool = False, fp16: bool = False) -> th.Tensor:
    """CLIP text features for `texts`, on `device`."""
    clip_model, _ = load_clip(clip_model_name, device, fp16)
    if clip_model_name in OPEN_CLIP_MODELS:
        tokens = _open_clip_tokenizer(clip_model_name)(texts)  # The text tower is on the CPU.
    else:
        tokens = clip.tokenize(texts, truncate=truncate).to(device)
    with th.no_grad():
        return th.cat([clip_model.encode_text(chunk).float() for chunk in tokens.split(256)]).to(device)


def imagenet_text_features(clip_model_name: str = "ViT-B/32", device: str = "cpu", fp16: bool = False) -> th.Tensor:
    """Normalized CLIP text features for every ImageNet class, cached on disk."""
    safe_name = re.sub(r"[^\w.-]", "-", clip_model_name)
    cache_path = Path(script_util.CACHE_PATH) / f"imagenet_text_features_{safe_name}.pt"
    if cache_path.is_file():
        return th.load(cache_path, map_location=device, weights_only=True)

    if clip_model_name in OPEN_CLIP_MODELS:
        print(f"Encoding the ImageNet classes for {clip_model_name} on the CPU. This only happens once.")
    labels = [f"a photo of a {img_cls}" for img_cls in IMAGENET_CLASSES]
    features = tf.normalize(encode_texts(labels, clip_model_name, device, truncate=True, fp16=fp16), dim=-1)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(f".{os.getpid()}.tmp")
    th.save(features.cpu(), tmp_path)
    os.replace(tmp_path, cache_path)
    return features


def imagenet_top_n(text_encodes, device: str = 'cuda', n: int = len(IMAGENET_CLASSES), clip_model_name: str = "ViT-B/32", fp16: bool = False):
    """
    Returns the top n classes for a given clip model.
    """
    imagenet_features = imagenet_text_features(clip_model_name, device, fp16)
    prompt_features = tf.normalize(text_encodes.detach(), dim=-1)
    similarity = prompt_features @ imagenet_features.T
    return similarity.topk(n, dim=-1, sorted=True).indices[0].to(device)


def encode_image_prompt(image: str, weight: float, diffusion_size: int, num_cutouts, clip_model_name: str = "ViT-B/32", device: str = "cpu", fp16: bool = False):
    clip_model, clip_size = load_clip(clip_model_name, device, fp16)
    make_cutouts = MakeCutouts(cut_size=clip_size, num_cutouts=num_cutouts)
    pil_img = Image.open(script_util.fetch(image)).convert('RGB')
    img = tvf.to_tensor(pil_img).unsqueeze(0).to(device)
    height, width = img.shape[2:]
    scale = diffusion_size / min(height, width)
    if scale < 1:  # Match the detail level of the generated image, keeping the aspect ratio.
        out_shape = (round(height * scale), round(width * scale))
        img = resize_right.resize(img, out_shape=out_shape,
                                  interp_method=lanczos3, support_sz=None,
                                  antialiasing=True, by_convs=False, scale_tolerance=None).clamp(0, 1)
    batch_embed = clip_model.encode_image(preprocess(clip_model, make_cutouts(img))).float()
    batch_weight = [weight / len(batch_embed)] * len(batch_embed)
    return batch_embed, batch_weight


def encode_text_prompt(txt, weight, clip_model_name="ViT-B/32", device="cpu", fp16=False):
    return encode_texts(txt, clip_model_name, device, fp16=fp16), weight

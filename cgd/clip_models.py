"""Loading CLIP models, OpenAI's and open_clip's, and encoding prompts with them."""

import os
import re
import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import torch as th
import torchvision.transforms.functional as tvf
from PIL import Image
from torch import nn
from torch.nn import functional as tf
from torchvision.transforms import InterpolationMode

from cgd.checks import require
from cgd.cutouts import make_cutouts
from cgd.download import CACHE_PATH, download, fetch
from cgd.imagenet_classes import IMAGENET_CLASSES

with warnings.catch_warnings():  # It imports pkg_resources, which is deprecated and nothing to act on.
    warnings.simplefilter("ignore", UserWarning)
    import clip

CLIP_CACHE_PATH = CACHE_PATH / "clip"
TEXT_BATCH_SIZE = 256

OPENAI_MODELS = {
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}
OPENAI_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_STD = (0.26862954, 0.26130258, 0.27577711)

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
MODEL_NAMES = (*OPENAI_MODELS, *OPEN_CLIP_MODELS)

INIT_FUNCTIONS = ("normal_", "uniform_", "constant_", "kaiming_uniform_", "xavier_uniform_", "xavier_normal_")


def ensemble_members(model_name: str) -> list[str]:
    """The models of a `+` separated ensemble such as `ViT-B/32+PE-Core-B-16`."""
    return [model_name] if Path(model_name).is_file() else model_name.split("+")


def check_model(model_name: str) -> None:
    """Reject a CLIP model that cannot be loaded, before spending time on anything else."""
    for name in ensemble_members(model_name):
        require(
            name in MODEL_NAMES or Path(name).is_file(),
            f"Unknown CLIP model {name!r}. Choose one of {list(MODEL_NAMES)}, or give the path of a checkpoint. "
            "Join several with + to guide with all of them.",
        )


@contextmanager
def skip_weight_init() -> Iterator[None]:
    """Skip random init while building a model the checkpoint will overwrite anyway.

    The `clip` library initialises every parameter before loading the real ones, which
    makes up most of its load time.
    """
    originals = {name: getattr(nn.init, name) for name in INIT_FUNCTIONS}
    for name in INIT_FUNCTIONS:
        setattr(nn.init, name, lambda tensor, *_args, **_kwargs: tensor)
    try:
        yield
    finally:
        for name, function in originals.items():
            setattr(nn.init, name, function)


@lru_cache(maxsize=1)
def load_clip(model_name: str = "ViT-B/32", device: str = "cpu", fp16: bool = False) -> tuple[nn.Module, int]:
    """A CLIP model, frozen, and the size of image it takes.

    Building a model draws random numbers. The stream is forked, so a seed gives the same
    image whether or not this call was a cache hit.
    """
    with th.random.fork_rng(devices=[]):
        if model_name in OPEN_CLIP_MODELS:
            model = _load_open_clip(model_name, device, fp16)
            return model, model.visual.image_size[0]
        model = _load_openai(model_name, device)
    return model, model.visual.input_resolution


def _load_openai(model_name: str, device: str) -> nn.Module:
    """One of OpenAI's models, or a checkpoint file of one. They always run at half precision on a GPU."""
    path = (
        download(OPENAI_MODELS[model_name], model_name.replace("/", "-") + ".pt", CLIP_CACHE_PATH)
        if model_name in OPENAI_MODELS
        else Path(model_name)
    )
    with skip_weight_init():
        model = clip.load(str(path), jit=False, device=device)[0]
    return model.eval().requires_grad_(False)


def _load_open_clip(model_name: str, device: str, fp16: bool) -> nn.Module:
    import open_clip  # Slow to import, and only these models need it.
    from huggingface_hub.utils import logging as hub_logging

    hub_logging.set_verbosity_error()  # The Hub otherwise asks for a token on every run.
    with skip_weight_init():
        model = open_clip.create_model(
            model_name, pretrained=OPEN_CLIP_MODELS[model_name], cache_dir=str(CLIP_CACHE_PATH)
        )
    model = model.eval().requires_grad_(False)
    # Text is only encoded once, up front, so that tower stays on the CPU. It is up to
    # two thirds of the weights and guidance has no use for it.
    # fp32 is the exact computation, so it is the default. fp16's gradient can be far from
    # it, as low as a cosine of 0.11, but on these towers noise of about that size on the
    # input moves the exact gradient as far, and fp16 scored about the same on held out
    # prompts. OpenAI's models have a smoother gradient, and always use fp16.
    model.visual.to(device, th.float16 if fp16 and device != "cpu" else th.float32)
    # Guidance backpropagates through the vision tower. Recomputing its activations
    # during the backward pass takes 30% more time and a quarter of the memory.
    model.visual.set_grad_checkpointing(True)
    # The fused attention kernels have a nondeterministic backward pass, which makes a
    # seed produce a different image every run. timm's plain path computes the same
    # gradient for up to 3% more time, and does not depend on which kernel a GPU gets.
    for module in model.visual.modules():
        if hasattr(module, "fused_attn"):
            module.fused_attn = False
    return model


def preprocess(model: nn.Module, images: th.Tensor) -> th.Tensor:
    """Prepare images in [0, 1] for `model`: its normalization, in its precision."""
    visual = model.visual
    mean, std = (visual.image_mean, visual.image_std) if hasattr(visual, "image_mean") else (OPENAI_MEAN, OPENAI_STD)
    return tvf.normalize(images, mean, std).to(next(visual.parameters()).dtype)


@lru_cache(maxsize=1)
def _open_clip_tokenizer(model_name: str) -> Callable[[list[str]], th.Tensor]:
    import open_clip

    return open_clip.get_tokenizer(model_name, cache_dir=str(CLIP_CACHE_PATH))


def encode_texts(
    texts: Sequence[str], model_name: str, device: str, fp16: bool = False, *, truncate: bool = False
) -> th.Tensor:
    """CLIP text features of `texts`, on `device`."""
    model, _ = load_clip(model_name, device, fp16)
    if model_name in OPEN_CLIP_MODELS:
        tokens = _open_clip_tokenizer(model_name)(list(texts))  # The text tower is on the CPU.
    else:
        tokens = clip.tokenize(list(texts), truncate=truncate).to(device)
    with th.no_grad():
        return th.cat([model.encode_text(chunk).float() for chunk in tokens.split(TEXT_BATCH_SIZE)]).to(device)


def encode_image(
    path: str, model_name: str, device: str, fp16: bool = False, *, image_size: int, num_cutouts: int
) -> th.Tensor:
    """CLIP features of an image prompt's cutouts, at the detail level of an `image_size` image."""
    model, clip_size = load_clip(model_name, device, fp16)
    image = tvf.to_tensor(Image.open(fetch(path)).convert("RGB")).unsqueeze(0).to(device)
    height, width = image.shape[2:]
    scale = image_size / min(height, width)
    if scale < 1:  # Match the detail level of the generated image, keeping the aspect ratio.
        size = [round(height * scale), round(width * scale)]
        image = tvf.resize(image, size, interpolation=InterpolationMode.BICUBIC, antialias=True).clamp(0, 1)
    return model.encode_image(preprocess(model, make_cutouts(image, clip_size, num_cutouts))).float()


def imagenet_text_features(model_name: str = "ViT-B/32", device: str = "cpu", fp16: bool = False) -> th.Tensor:
    """Normalized CLIP text features of every ImageNet class, cached on disk."""
    safe_name = re.sub(r"[^\w.-]", "-", model_name)
    cache_path = CACHE_PATH / f"imagenet_text_features_{safe_name}.pt"
    if cache_path.is_file():
        return th.load(cache_path, map_location=device, weights_only=True)
    if model_name in OPEN_CLIP_MODELS:
        print(f"Encoding the ImageNet classes for {model_name} on the CPU. This only happens once.")
    labels = [f"a photo of a {name}" for name in IMAGENET_CLASSES]
    features = tf.normalize(encode_texts(labels, model_name, device, fp16, truncate=True), dim=-1)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    partial = cache_path.with_suffix(f".{os.getpid()}.tmp")
    th.save(features.cpu(), partial)
    partial.replace(cache_path)
    return features


def closest_imagenet_class(
    query: th.Tensor, model_name: str = "ViT-B/32", device: str = "cpu", fp16: bool = False
) -> int:
    """The ImageNet class whose text is closest to a [1, features] query of `model_name`."""
    similarity = tf.normalize(query.detach(), dim=-1) @ imagenet_text_features(model_name, device, fp16).T
    return int(similarity.topk(1, dim=-1).indices[0, 0])

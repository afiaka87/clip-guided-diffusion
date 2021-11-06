from functools import lru_cache

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

CLIP_MODEL_NAMES = ("ViT-B/16", "ViT-B/32", "RN50", "RN101", "RN50x4", "RN50x16")
CLIP_NORMALIZE = tvt.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

@lru_cache(maxsize=1)
def load_clip(model_name='ViT-B/32', device="cpu"):
    print(f"Loading clip model\t{model_name}\ton device\t{device}.")
    if device == "cpu":
        clip_model = clip.load(model_name, jit=False)[
            0].eval().to(device=device).float()
        clip_size = clip_model.visual.input_resolution
        return clip_model, clip_size
    elif "cuda" in device:
        clip_model = clip.load(model_name, jit=False)[
            0].eval().requires_grad_(False).to(device)
        clip_size = clip_model.visual.input_resolution
        return clip_model, clip_size
    else:
        raise ValueError("Invalid or unspecified device: {}".format(device))


def imagenet_top_n(text_encodes, device: str = 'cuda', n: int = len(IMAGENET_CLASSES), clip_model_name: str = "ViT-B/32"):
    """
    Returns the top n classes for a given clip model.
    """
    clip_model, _ = load_clip(model_name=clip_model_name, device=device)
    with th.no_grad():
        engineered_pronmpts = [
            f"an image of a {img_cls}" for img_cls in IMAGENET_CLASSES]
        imagenet_lbl_tokens = clip.tokenize(engineered_pronmpts).to(device)
        imagenet_features = clip_model.encode_text(imagenet_lbl_tokens).float()
        imagenet_features /= imagenet_features.norm(dim=-1, keepdim=True)
        prompt_features = text_encodes / \
            text_encodes.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * prompt_features @
                      imagenet_features.T).softmax(dim=-1)
        return text_probs.topk(n, dim=-1, sorted=True).indices[0].to(device)


def encode_image_prompt(image: str, weight: float, diffusion_size: int, num_cutouts, clip_model_name: str = "ViT-B/32", device: str = "cpu"):
    clip_model, clip_size = load_clip(clip_model_name, device)
    make_cutouts = MakeCutouts(cut_size=clip_size, num_cutouts=num_cutouts)
    pil_img = Image.open(script_util.fetch(image)).convert('RGB')
    smallest_side = min(diffusion_size, *pil_img.size)
    pil_img = resize_right.resize(pil_img, out_shape=[smallest_side],
                                  interp_method=lanczos3, support_sz=None,
                                  antialiasing=True, by_convs=False, scale_tolerance=None)
    batch = make_cutouts(tvf.to_tensor(pil_img).unsqueeze(0).to(device))
    batch_embed = clip_model.encode_image(tf.normalize(batch)).float()
    batch_weight = [weight / make_cutouts.cutn] * make_cutouts.cutn
    return batch_embed, batch_weight


def encode_text_prompt(txt, weight, clip_model_name="ViT-B/32", device="cpu"):
    clip_model, _ = load_clip(clip_model_name, device)
    txt_tokens = clip.tokenize(txt).to(device)
    txt_encoded = clip_model.encode_text(txt_tokens).float()
    return txt_encoded, weight

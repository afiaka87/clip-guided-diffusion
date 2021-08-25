import clip
import torch as th
import torch.nn.functional as tf
import torchvision.transforms as tvt
from data.imagenet1000_clsidx_to_labels import IMAGENET_CLASSES

def imagenet_top_n(prompt: str = '', prompt_min: str = '', min_weight: float = 0.1, device:str='cpu', n: int = len(IMAGENET_CLASSES), clip_model_name: str = "ViT-B/32"):
    """
    Returns the top n classes for a given clip model.
    """
    clip_model, _ = load_clip(model_name=clip_model_name, device=device)
    clip_model = clip_model.to(device)
    clip_model.eval()
    with th.no_grad():
        imagenet_lbl_tokens = clip.tokenize(IMAGENET_CLASSES).to(device)
        prompt_tokens = clip.tokenize(prompt).to(device)
        prompt_min_features = None
        prompt_min_tokens = None
        if prompt_min is not None:
            prompt_min_tokens = clip.tokenize(prompt_min).to(device)

        imagenet_features = clip_model.encode_text( imagenet_lbl_tokens).float()
        prompt_features = clip_model.encode_text(prompt_tokens).float()
        if prompt_min_tokens is not None:
            prompt_min_features = clip_model.encode_text(
                prompt_min_tokens).float()
            prompt_min_features /= prompt_min_features.norm(
                dim=-1, keepdim=True)

        imagenet_features /= imagenet_features.norm(dim=-1, keepdim=True)
        prompt_features /= prompt_features.norm(dim=-1, keepdim=True)
        if prompt_min_features is not None:
            prompt_features = prompt_features - \
                (min_weight * prompt_min_features)
        text_probs = (100.0 * prompt_features @
                      imagenet_features.T).softmax(dim=-1)
        return text_probs.cpu().topk(n, dim=-1, sorted=True)


# CLIP functions

CLIP_NORMALIZE = tvt.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[
                               0.26862954, 0.26130258, 0.27577711])


# clip_model = clip.load(clip_model_name, jit=False)[0].eval().requires_grad_(False).to(device)
def load_clip(model_name='ViT-B/32', device="cpu", jit=False):
    if device == "cpu":
        clip_model = clip.load(model_name, jit=False)[0].eval().to(device=device).float()
        # raise ValueError("CPU not supported at the moment due to a bug.")
    else:
        clip_model = clip.load(model_name, jit=False)[0].eval().requires_grad_(False).to(device=device)
    clip_size = clip_model.visual.input_resolution
    return clip_model, clip_size


def clip_encode_text(clip_model_name, text, device="cuda:0", truncate: bool = True):
    clip_model, _ = load_clip(clip_model_name, device=device, jit=False)
    return clip_model.encode_text(clip.tokenize(text, truncate=truncate).to(device))
    

class MakeCutouts(th.nn.Module):
    def __init__(self, cut_size: int, num_cutouts: int, cutout_size_power: float = 1.0, augment_list: list = []):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = num_cutouts
        self.cut_pow = cutout_size_power
        self.augs = th.nn.Sequential(*augment_list)

    def forward(self, input: th.Tensor):
        side_x, side_y = input.shape[2:4]
        max_size = min(side_y, side_x)
        min_size = min(side_y, side_x, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(th.rand([]) ** self.cut_pow *
                       (max_size - min_size) + min_size)
            offsetx = th.randint(0, side_y - size + 1, ())
            offsety = th.randint(0, side_x - size + 1, ())
            cutout = input[:, :, offsety: offsety +
                           size, offsetx: offsetx + size]
            cutout = tf.interpolate(
                cutout,
                (self.cut_size, self.cut_size),
                mode="bilinear",
                align_corners=False,
            )
            cutouts.append(cutout)
        return self.augs(th.cat(cutouts))

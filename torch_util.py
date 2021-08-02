import glob
import clip
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import functional as F
from PIL import Image

def resize_image(image, out_size):
    """Resize image"""
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)

def spherical_dist_loss(x, y):
    """Spherical distance loss"""
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), "replicate")
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff ** 2 + y_diff ** 2).mean([1, 2, 3])

# From https://github.com/mehdidc/feed_forward_vqgan_clip/blob/master/main.py
def tokenize(paths, out="tokenized.pkl", max_length:int=None):
    """
    tokenize and save to a pkl file
    path: str
        can be either a text file where each line is a text prompt
        or a glob pattern where each file is a text prompt
    out: str
        output pkl file
    max_length: int
        this can be used to filter text prompts and retain only
        ones that are up to `max_length`
    """
    if "*" in paths:
        texts = [open(f).read().strip() for f in glob(paths)]
    else:
        texts = [l.strip() for l in open(paths).readlines()]
        if max_length:
            texts = [text for text in texts if len(text) <= max_length]
    T = clip.tokenize(texts, truncate=True)
    torch.save(T, out)

def load_tokenized(path):
    # path can be the following:
    # - a path to a text file where each line is a text prompt
    # - a glob pattern (*) of text files where each file is a text prompt
    # - a pkl file created using `tokenize`, where each text prompt is already tokenized
    # returns:
    # - a TensorDataset of clip tokenized text prompts
    if path.endswith("pkl"):
        tokens = torch.load(path)
    elif "*" in path:
        texts = [open(f).read().strip() for f in glob(path)]
        tokens = clip.tokenize(texts, truncate=True)
    else:
        texts = [t.strip() for t in open(path).readlines()]
        tokens = clip.tokenize(texts, truncate=True)
    return tokens


def load_tensor_dataset(batch_size, num_workers, shuffle=False, path="data/imagenet.pkl"):
    tokens = load_tokenized(path)
    dataset = TensorDataset(tokens)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

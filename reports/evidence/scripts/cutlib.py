"""Cutout schemes for sweeps. Set CONFIG before building a guide; clip_util.MakeCutouts is replaced by Scheme."""
import torch as th, torch.nn.functional as tf
from cgd import modules
CONFIG = {}

class Scheme(modules.MakeCutouts):
    def __init__(self, cut_size, num_cutouts, cutout_size_power=1., use_augs=False, min_cut_fraction=1.):
        super().__init__(cut_size, num_cutouts, cutout_size_power, use_augs, min_cut_fraction)
        self.c = dict(CONFIG); self.calls = 0

    def spans(self, n):
        c = self.c
        low, high = c.get("low", .7), c.get("high", 1.)
        anneal = c.get("anneal")
        if anneal:
            f = min(self.calls / c.get("total", 30), 1.)
            low = anneal[0] + (anneal[2] - anneal[0]) * f
            high = anneal[1] + (anneal[3] - anneal[1]) * f
        if c.get("strat"):
            u = (th.randperm(n) + th.rand(n)) / n
        else:
            u = th.rand(n)
        return low + u * (high - low)

    def crop(self, x, s, ax, ay):
        return modules.adaptive_avg_pool(x[:, :, ay:ay + s, ax:ax + s], self.cut_size)

    def forward(self, x):
        c = self.c; h, w = x.shape[2:]; side = min(h, w); out = []
        for _ in range(c.get("whole", 0)):
            out.append(modules.adaptive_avg_pool(x, self.cut_size))
        n = c.get("crops", 0)
        if n:
            fracs = self.spans(n)
            if c.get("strat"):  # Latin hypercube: each crop takes its own row and column of the image.
                px, py = (th.randperm(n) + th.rand(n)) / n, (th.randperm(n) + th.rand(n)) / n
            else:
                px, py = th.rand(n), th.rand(n)
            for i in range(n):
                s = max(int(fracs[i].item() * side), 1)
                if c.get("pad"):  # Centre uniform over the image, with reflection past the edge, so every pixel is covered equally.
                    m = s // 2
                    p = tf.pad(x, (m, m, m, m), mode="reflect")
                    ax, ay = int(px[i].item() * (w + 2 * m - s)), int(py[i].item() * (h + 2 * m - s))
                    view = self.crop(p, s, ax, ay)
                else:
                    ax, ay = int(px[i].item() * (w - s + 1)), int(py[i].item() * (h - s + 1))
                    view = self.crop(x, s, ax, ay)
                out.append(view)
                if c.get("flip"): out.append(view.flip(-1))
        self.calls += 1
        return th.cat(out)

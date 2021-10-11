import torch as th
import torch.nn.functional as tf
import torchvision.transforms as tvt

class MakeCutouts(th.nn.Module):
    def __init__(self, cut_size: int, num_cutouts: int, cutout_size_power: float = 1.0, use_augs: bool = False):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = num_cutouts
        self.cut_pow = cutout_size_power
        if use_augs:
            self.augs = tvt.Compose([
                tvt.RandomHorizontalFlip(p=0.5),
                tvt.Lambda(lambda x: x + th.randn_like(x) * 0.01),
                tvt.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                tvt.Lambda(lambda x: x + th.randn_like(x) * 0.01),
                tvt.RandomPerspective(distortion_scale=0.4, p=0.7),
                tvt.Lambda(lambda x: x + th.randn_like(x) * 0.01),
                tvt.RandomGrayscale(p=0.15),
                tvt.Lambda(lambda x: x + th.randn_like(x) * 0.01),
            ])
        else:
            self.augs = tvt.Compose([])

    def forward(self, input: th.Tensor):
        side_x, side_y = input.shape[2:4]
        max_size = min(side_y, side_x)
        min_size = min(side_y, side_x, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(th.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = th.randint(0, side_x - size + 1, ())
            offsety = th.randint(0, side_y - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutout = self.augs(cutout)
            cutout = tf.adaptive_avg_pool2d(cutout, self.cut_size)
            cutouts.append(cutout)
    
        return th.cat(cutouts)

import torch as th
import torch.nn.functional as tf
import torchvision.transforms as tvt

class MakeCutouts(th.nn.Module):
    def __init__(self, cut_size: int, num_cutouts: int, cutout_size_power: float = 1.0, use_augs: bool = False):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = num_cutouts
        self.cut_pow = cutout_size_power
        self.cached_coords = None
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

    def cache_coordinates(self, side_x: int, side_y: int):
        """Pre-compute and cache cutout coordinates for reuse across steps."""
        max_size = min(side_y, side_x)
        min_size = min(side_y, side_x, self.cut_size)
        coords = []
        for _ in range(self.cutn):
            size = int(th.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = th.randint(0, side_x - size + 1, ()).item()
            offsety = th.randint(0, side_y - size + 1, ()).item()
            coords.append((offsetx, offsety, size))
        self.cached_coords = coords

    def _generate_coords(self, side_x: int, side_y: int, cutn: int):
        """Generate fresh random coordinates."""
        max_size = min(side_y, side_x)
        min_size = min(side_y, side_x, self.cut_size)
        coords = []
        for _ in range(cutn):
            size = int(th.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = th.randint(0, side_x - size + 1, ()).item()
            offsety = th.randint(0, side_y - size + 1, ()).item()
            coords.append((offsetx, offsety, size))
        return coords

    def forward(self, input: th.Tensor, use_cache: bool = False, num_cutouts_override: int = None):
        cutn = num_cutouts_override if num_cutouts_override is not None else self.cutn
        side_x, side_y = input.shape[2:4]

        if use_cache and self.cached_coords is not None:
            coords = self.cached_coords[:cutn]
        else:
            coords = self._generate_coords(side_x, side_y, cutn)

        cutouts = []
        for offsetx, offsety, size in coords:
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutout = self.augs(cutout)
            cutout = tf.adaptive_avg_pool2d(cutout, self.cut_size)
            cutouts.append(cutout)

        return th.cat(cutouts)

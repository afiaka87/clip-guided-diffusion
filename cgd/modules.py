import torch as th
import torch.nn.functional as tf
import torchvision.transforms as tvt

class MakeCutouts(th.nn.Module):
    def __init__(self, cut_size: int, num_cutouts: int, cutout_size_power: float = 1.0, use_augs: bool = False,
                 min_cut_fraction: float = 1.0):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = num_cutouts
        self.cut_pow = cutout_size_power
        self.min_cut_fraction = min_cut_fraction  # Smallest cutout, as a fraction of the shorter side.
        self.use_augs = use_augs
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

    def _size_range(self, side_x: int, side_y: int):
        max_size = min(side_y, side_x)
        min_size = min(int(max_size * self.min_cut_fraction), self.cut_size)
        return min_size, max_size

    def distinct_cutouts(self, side_x: int, side_y: int) -> int:
        """Number of cutouts worth computing for an image of this size.

        When the size range is a single value on a square image, the only possible
        cutout is the whole frame. Without augmentations, every cutout after the
        first is then a pixel-identical copy and just burns CLIP time.
        """
        min_size, max_size = self._size_range(side_x, side_y)
        if side_x == side_y and min_size == max_size and not self.use_augs:
            return 1
        return self.cutn

    def _generate_coords(self, side_x: int, side_y: int, cutn: int):
        min_size, max_size = self._size_range(side_x, side_y)
        coords = []
        for _ in range(cutn):
            size = int(th.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = th.randint(0, side_x - size + 1, ()).item()
            offsety = th.randint(0, side_y - size + 1, ()).item()
            coords.append((offsetx, offsety, size))
        return coords

    def forward(self, input: th.Tensor):
        side_y, side_x = input.shape[2:4]
        cutouts = []
        for offsetx, offsety, size in self._generate_coords(side_x, side_y, self.distinct_cutouts(side_x, side_y)):
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutout = self.augs(cutout)
            cutout = tf.adaptive_avg_pool2d(cutout, self.cut_size)
            cutouts.append(cutout)

        return th.cat(cutouts)

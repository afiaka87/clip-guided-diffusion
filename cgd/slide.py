"""Sliding from one finished image of a sequence to the next, for the animation.

Every image of a sequence is made as its prompts would make it alone, from the same noise.
A transition is then a path between two finished images, and each pipeline has a
function that draws the image at a position along it, from 0 at the first to 1 at the
second. This module is what they share: where a position puts the fade and the noise,
and place(), which chooses the positions.

A frame only depends on its position, never on the frame before it, so nothing builds
up along the way and frames can be drawn in any order. place() uses that: it draws a
frame in the middle of whichever gap between two neighbours shows the largest change,
again and again. A transition spends most of its change in a short stretch, where the
subject turns over, and that is where the frames go.

What a position means is the pipeline's. A backbone's image is a smooth function of
its noise and its text embedding, so its frame is the image of the noise and the text
partway between the two images' (backbone.between()). The ImageNet model has no such
handle. Its frame is the two finished images faded into each other and redrawn by the
U-Net: noised, more the further the frame is from either image, and denoised without
CLIP (adm.between()). So both ends are the images themselves, and halfway the U-Net
makes one picture of the double exposure.

That was the fourth thing tried on the ImageNet model. On a castle and a fox at 128,
by the RMS difference between neighbouring frames, of pixels in [-1, 1]:

- Drawing every frame in full from the same noise, with the prompts and classes blended:
  0.3 to 0.5 between any two frames, and as much when they were 0.004 apart. CLIP guidance
  is chaotic in its inputs, so every frame is another image. For the same reason CLIP
  does not guide the redrawn frames: with it, frames 0.004 apart still differed by 0.15 to 0.25.
- Redrawing each frame from the frame before it: what guidance adds piles up, and within 15
  frames the image is a poster of flat saturated colour.
- Taking both images back to noise with the U-Net's own ODE and sliding between the
  noises: smooth and clean at 128, 0.07 at most. But the unconditional checkpoint's
  images are flat and saturated, the ODE stretches whatever is mixed into their noise,
  and every frame in between came out as grain.
"""

import math
from collections.abc import Callable

import torch as th
from tqdm.auto import tqdm

PEAK_TIMESTEP = 600  # How far up the schedule the frame halfway is noised. By 800 the U-Net draws subjects of its own.
FADE_SHARPNESS = 2  # Above 1, less of the other image shows near either end, where too little noise hides it.
EASE = 0.5  # The pace at either image, next to 1 more than it halfway: a transition starts and ends gently.
FINEST = 8  # No two frames are closer than a FINEST-th of an even spacing, so a jump cannot swallow them all.

Draw = Callable[[float], th.Tensor]
"""position -> the batch of images, in [-1, 1], at that position of a transition."""


def fade(position: float) -> float:
    """How much of the second image the frame at `position` is: an S-curve from 0 to 1, as sharp as FADE_SHARPNESS."""
    rising, falling = position**FADE_SHARPNESS, (1 - position) ** FADE_SHARPNESS
    return rising / (rising + falling)


def noise_timestep(position: float) -> float:
    """The timestep the frame at `position` is noised to: 0 at either image, PEAK_TIMESTEP halfway."""
    return PEAK_TIMESTEP * math.sin(math.pi * position)


def slerp(first: th.Tensor, second: th.Tensor, share: float) -> th.Tensor:
    """The point `share` of the way along the arc from `first` to `second`, for each of a batch.

    Unlike a straight line it keeps the norm, which a model reads as the noise level.
    """
    a, b = first.flatten(1), second.flatten(1)
    cosine = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1))
    angle = cosine.clamp(-1, 1).acos().view(-1, *[1] * (first.dim() - 1))
    arc = ((1 - share) * angle).sin() / angle.sin() * first + (share * angle).sin() / angle.sin() * second
    return th.where(angle < 1e-4, th.lerp(first, second, share), arc)  # The same point, to within rounding.


def change(first: th.Tensor, second: th.Tensor) -> float:
    """How far apart two batches of images are: the RMS difference of their pixels, of the image that changed most."""
    return float((first.float() - second.float()).square().mean(dim=(1, 2, 3)).sqrt().max())


def pace(position: float) -> float:
    """How fast a transition should move at `position`, relative to its other positions."""
    return EASE + math.sin(math.pi * position)


def place(draw: Draw, ends: tuple[th.Tensor, th.Tensor], count: int, progress: bool = False) -> list[th.Tensor]:
    """The `count` frames of a transition, in order, ending on the second of `ends`: the images at 0 and 1.

    Each frame after that goes halfway between the two neighbours with the largest
    change() between them for the pace() there, so the change is spread evenly
    over the frames, a little slower near either end.
    """
    positions, frames = [0.0, 1.0], list(ends)
    finest = 1 / (FINEST * count)

    def gap(i: int) -> tuple[float, float]:
        """How much a frame is wanted between frames i and i + 1, then how far apart they are, which settles ties."""
        width = positions[i + 1] - positions[i]
        if width <= finest:
            return -1.0, width
        return change(frames[i], frames[i + 1]) / pace((positions[i] + positions[i + 1]) / 2), width

    for _ in tqdm(range(count - 1), disable=not progress):
        widest = max(range(len(positions) - 1), key=gap)
        middle = (positions[widest] + positions[widest + 1]) / 2
        positions.insert(widest + 1, middle)
        frames.insert(widest + 1, draw(middle))
    return frames[1:]

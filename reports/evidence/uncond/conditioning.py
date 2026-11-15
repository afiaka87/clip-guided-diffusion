"""The unconditional 256 U-Net with a CLIP embedding added to its timestep embedding.

guided-diffusion conditions its class models with emb = time_embed(t) + label_emb(y). This
adds project(c) in the same place, for a normalised CLIP embedding c, through every ResBlock's
scale and shift. The projection's last layer starts at zero, so before training the model is
exactly the unconditional checkpoint, and without a condition it always runs as itself plus
whatever training changed in the shared layers.
"""

from collections.abc import Iterable

import torch as th
from guided_diffusion.nn import timestep_embedding
from torch import nn
from torch.utils.checkpoint import checkpoint

TRAINABLE = {
    "project": ("project.",),
    "emb": ("project.", "unet.time_embed.", ".emb_layers."),
    # Also the 16 attention blocks, so more of the model can move from ImageNet towards the training images.
    "emb_attn": ("project.", "unet.time_embed.", ".emb_layers.", ".qkv.", ".proj_out."),
}


class ClipConditionedUNet(nn.Module):
    """An ADM U-Net whose timestep embedding also carries a CLIP embedding."""

    def __init__(self, unet: nn.Module, condition_dim: int):
        super().__init__()
        self.unet = unet
        width = unet.time_embed[-1].out_features
        self.project = nn.Sequential(nn.Linear(condition_dim, width), nn.SiLU(), nn.Linear(width, width))
        nn.init.zeros_(self.project[-1].weight)
        nn.init.zeros_(self.project[-1].bias)
        self.gradient_checkpointing = False

    def forward(self, x: th.Tensor, t: th.Tensor, condition: th.Tensor | None = None,
                present: th.Tensor | None = None) -> th.Tensor:
        """The U-Net's output for x at timesteps t, given each image's condition.

        `present` is a [batch] boolean mask of images that have one. The rest, or all of them
        with no condition, run unconditioned.
        """
        unet = self.unet
        emb = unet.time_embed(timestep_embedding(t, unet.model_channels))
        if condition is not None:
            projected = self.project(condition.to(emb.dtype))
            if present is not None:
                projected = projected * present[:, None].to(projected.dtype)
            emb = emb + projected
        h, skips = x.type(unet.dtype), []
        for module in unet.input_blocks:
            h = self._block(module, h, emb)
            skips.append(h)
        h = self._block(unet.middle_block, h, emb)
        for module in unet.output_blocks:
            h = self._block(module, th.cat([h, skips.pop()], dim=1), emb)
        return unet.out(h.type(x.dtype))

    def _block(self, module: nn.Module, h: th.Tensor, emb: th.Tensor) -> th.Tensor:
        if self.gradient_checkpointing and self.training:
            return checkpoint(module, h, emb, use_reentrant=False)
        return module(h, emb)


def set_trainable(model: ClipConditionedUNet, which: str) -> list[nn.Parameter]:
    """Freeze everything but the parameters `which` names in TRAINABLE, and return those."""
    prefixes = TRAINABLE[which]
    chosen = []
    for name, parameter in model.named_parameters():
        train = any(name.startswith(p) if not p.startswith(".") else p in name for p in prefixes)
        parameter.requires_grad_(train)
        if train:
            chosen.append(parameter)
    return chosen


def trainable_state(model: ClipConditionedUNet, names: Iterable[str]) -> dict[str, th.Tensor]:
    """The parameters a fine-tune changed, by name, on the CPU."""
    wanted = set(names)
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if name in wanted}

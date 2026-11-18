"""The ImageNet U-Net with a CLIP embedding as its condition, added so that it starts as the pretrained model.

`finetune/` trains it, and `cgd -ft` draws with what that trained, see finetuned.py.

Nothing of the U-Net is trained. What is new:

- **The condition's path in.** `Conditioner` projects each CLIP model's embedding and adds the result to the
  timestep embedding, where a class label goes in the class-conditional checkpoints. Every residual block
  already scales and shifts its features from that embedding. Its last layer starts at zero.
- **Tokens.** The same projection also makes a few tokens, and a `TokenAttention` after each attention block
  lets the image attend to them, as unCLIP's decoder does. Its output layer starts at zero.
- **LoRA.** A low-rank update on each attention block's projections and on each residual block's embedding
  layer, so the frozen weights can respond to the new signal. Its second factor starts at zero.

With all three at zero the output is the pretrained model's exactly, and with every embedding dropped the
condition's path contributes nothing, which is the unconditional half of classifier-free guidance.

Four more are off unless the configuration asks for them, and each also starts at zero. They answer what the
first runs measured, see the README:

- **Standardised embeddings** (`standardised`). A unit embedding's entries are about 0.03, which left the first
  layer's activations near 0.01 and the condition a twentieth of the timestep embedding's size. This centres
  and scales each entry by statistics kept with the weights, and normalises the hidden layer.
- **A scale and shift per block** (`film`). Each residual block gets its own projection of the condition, beside
  the shared one that goes through its frozen embedding layer.
- **A layout** (`layout`). What an embedding predicts of an image is a thumbnail, and nothing above knows
  position. This projects the condition to a small feature map, added after the middle block and the first
  output blocks.
- **Convolution updates** (`conv_rank`). A low-rank update on each residual block's last convolution.

By default only the middle and the decoder half are adapted. The encoder half then runs as pretrained, without
gradients, which saves about a third of a training step.

The attention blocks use PyTorch's fused kernel. It computes what the U-Net's own einsum does with less memory.
Its backward pass is nondeterministic on a GPU, which matters to sampling under guidance and not to training,
so `fused=False` keeps the original.
"""

import re
import warnings
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass

import torch as th
from torch import nn
from torch.nn import functional as tf

with warnings.catch_warnings():  # It uses a deprecated spelling of torch.amp, which is nothing to act on.
    warnings.simplefilter("ignore", FutureWarning)
    from guided_diffusion.nn import timestep_embedding
    from guided_diffusion.unet import AttentionBlock, QKVAttentionLegacy, ResBlock, TimestepBlock, UNetModel


def slug(model_name: str) -> str:
    """A CLIP model's name as part of a file or module name."""
    return re.sub(r"\W", "_", model_name)


@dataclass(frozen=True)
class AdapterConfig:  # pylint: disable=too-many-instance-attributes  # Each is one option of what is added.
    """What was added to the U-Net. Saved with the weights, so a checkpoint rebuilds itself."""

    embed_dims: dict[str, int]  # Each CLIP model's name and the size of its embedding, in the order they are given.
    rank: int = 16
    tokens: int = 4
    token_dim: int = 512
    hidden: int = 1024
    train_encoder: bool = False
    standardised: bool = False
    film: bool = False
    layout: int = 0  # The side of the layout's feature map. 0 for none.
    layout_channels: int = 64
    layout_blocks: int = 6  # How many of the first output blocks the layout is added after, besides the middle.
    conv_rank: int = 0  # 0 for none.
    full: bool = False  # Train the adapted blocks' own weights, with no low-rank updates beside them.
    align_dim: int = 0  # The size of the patch features the middle block also predicts. 0 for none.


class LoRA(nn.Module):
    """A frozen linear layer or 1x1 convolution plus a trainable low-rank update that starts at zero."""

    def __init__(self, base: nn.Linear | nn.Conv1d, rank: int):
        super().__init__()
        self.base = base
        if isinstance(base, nn.Linear):
            self.down = nn.Linear(base.in_features, rank, bias=False)
            self.up = nn.Linear(rank, base.out_features, bias=False)
        else:
            self.down = nn.Conv1d(base.in_channels, rank, 1, bias=False)
            self.up = nn.Conv1d(rank, base.out_channels, 1, bias=False)
        nn.init.zeros_(self.up.weight)
        self.extra: th.Tensor | None = None  # Added for one forward pass: a block's own scale and shift.

    def forward(self, x: th.Tensor) -> th.Tensor:
        """The layer's output with the update's added."""
        updated = self.base(x) + self.up(self.down(x))
        return updated if self.extra is None else updated + self.extra.to(updated.dtype)


class ConvLoRA(nn.Module):
    """A frozen convolution plus a trainable low-rank one with the same kernel, which starts at zero."""

    def __init__(self, base: nn.Conv2d, rank: int):
        super().__init__()
        self.base = base
        self.down = nn.Conv2d(base.in_channels, rank, base.kernel_size, padding=base.padding, bias=False)
        self.up = nn.Conv2d(rank, base.out_channels, 1, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """The convolution's output with the update's added."""
        return self.base(x) + self.up(self.down(x))


def zeroed(layer: nn.Module) -> nn.Module:
    """The layer with its weight and bias at zero, so that what it adds starts as nothing."""
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class FusedAttention(nn.Module):
    """`QKVAttentionLegacy` through `scaled_dot_product_attention`: the same result from the fused kernel."""

    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv: th.Tensor, _spatial: object = None) -> th.Tensor:
        """Attention over [N, heads * 3 * channels, length] queries, keys and values."""
        batch, width, length = qkv.shape
        channels = width // (3 * self.n_heads)
        # The legacy layout splits heads first, then each head into its queries, keys and values.
        q, k, v = qkv.reshape(batch, self.n_heads, 3 * channels, length).transpose(2, 3).split(channels, dim=3)
        return tf.scaled_dot_product_attention(q, k, v).transpose(2, 3).reshape(batch, -1, length)


class TokenAttention(nn.Module):
    """Cross-attention from an image's features to the condition's tokens, as a residual that starts at zero."""

    def __init__(self, channels: int, token_dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.norm = nn.GroupNorm(32, channels)
        self.query = nn.Linear(channels, channels)
        self.key_value = nn.Linear(token_dim, 2 * channels)
        self.out = nn.Linear(channels, channels)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, features: th.Tensor, tokens: th.Tensor) -> th.Tensor:
        """What to add to [N, channels, H, W] features, from [N, tokens, token_dim] tokens."""
        batch, channels, *_ = features.shape

        def heads(sequence: th.Tensor) -> th.Tensor:
            return sequence.reshape(batch, -1, self.n_heads, channels // self.n_heads).transpose(1, 2)

        image = self.norm(features.float()).to(features.dtype).flatten(2).transpose(1, 2)
        key, value = self.key_value(tokens.to(image.dtype)).chunk(2, dim=-1)
        attended = tf.scaled_dot_product_attention(heads(self.query(image)), heads(key), heads(value))
        update = self.out(attended.transpose(1, 2).reshape(batch, -1, channels))
        return update.transpose(1, 2).reshape(features.shape)


class Conditioner(nn.Module):
    """CLIP embeddings to what the U-Net takes: an addition to the timestep embedding, and tokens."""

    def __init__(self, config: AdapterConfig, emb_channels: int):
        super().__init__()
        self.names = list(config.embed_dims)
        self.tokens, self.token_dim = config.tokens, config.token_dim
        self.inputs = nn.ModuleDict(
            {slug(name): nn.Linear(dim, config.hidden) for name, dim in config.embed_dims.items()}
        )
        for name, dim in config.embed_dims.items():
            self.register_buffer(f"mean_{slug(name)}", th.zeros(dim))
            self.register_buffer(f"scale_{slug(name)}", th.ones(dim))
        self.mix: nn.Module = nn.Identity()
        if config.standardised:
            self.mix = nn.Sequential(nn.LayerNorm(config.hidden), nn.Linear(config.hidden, config.hidden), nn.SiLU())
        self.to_emb = zeroed(nn.Linear(config.hidden, emb_channels))
        self.to_tokens = nn.Linear(config.hidden, config.tokens * config.token_dim) if config.tokens else None

    def set_statistics(self, statistics: Mapping[str, tuple[th.Tensor, th.Tensor]]) -> None:
        """Each CLIP model's embedding mean and standard deviation, which the embeddings are centred and scaled by."""
        for name, (mean, deviation) in statistics.items():
            getattr(self, f"mean_{slug(name)}").copy_(mean)
            getattr(self, f"scale_{slug(name)}").copy_(deviation)

    def standardise(self, name: str, embedding: th.Tensor) -> th.Tensor:
        """An embedding centred and scaled by its model's statistics, which leave it as it is until they are set."""
        mean, scale = getattr(self, f"mean_{slug(name)}"), getattr(self, f"scale_{slug(name)}")
        return (embedding.float() - mean) / scale

    def make_tokens(self, hidden: th.Tensor) -> th.Tensor | None:
        """The tokens the attention layers read, [N, tokens, token_dim], or None without any."""
        if self.to_tokens is None:
            return None
        return self.to_tokens(hidden).reshape(len(hidden), self.tokens, self.token_dim)

    def forward(self, embeds: Mapping[str, th.Tensor], keep: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """The condition as one hidden vector per image, and which images have any condition at all.

        `keep` is [N, models], false where an image's embedding from that model is dropped. Everything made
        from the hidden vector is multiplied by the second result, so an image with every embedding dropped
        gets exactly nothing.
        """
        hidden = sum(
            keep[:, i : i + 1] * tf.silu(self.inputs[slug(name)](self.standardise(name, embeds[name])))
            for i, name in enumerate(self.names)
        )
        return self.mix(hidden), keep.any(dim=1, keepdim=True).to(hidden.dtype)


def adapted_blocks(unet: UNetModel, train_encoder: bool) -> Iterator[tuple[str, nn.Module]]:
    """The blocks of the U-Net that are adapted, each with a name for what is attached to it."""
    if train_encoder:
        yield from ((f"input_{i}", block) for i, block in enumerate(unet.input_blocks))
    yield "middle", unet.middle_block
    yield from ((f"output_{i}", block) for i, block in enumerate(unet.output_blocks))


class ConditionedUNet(nn.Module):
    """A frozen `UNetModel` and its adapters. `forward` without embeddings is the U-Net with its LoRA updates."""

    def __init__(self, unet: UNetModel, config: AdapterConfig, fused: bool = True):
        super().__init__()
        if config.full and (config.film or config.conv_rank):
            raise ValueError("A block whose own weights are trained takes no low-rank update or scale of its own.")
        self.config = config
        self.unet = unet.requires_grad_(False)
        emb_channels = unet.time_embed[-1].out_features
        self.conditioner = Conditioner(config, emb_channels)
        self.token_attention, self.film, self.layout_proj = nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict()
        self._film_targets: dict[str, LoRA] = {}  # The embedding layer each block's own scale and shift joins.
        for name, block in adapted_blocks(unet, config.train_encoder):
            block.requires_grad_(config.full)
            for number, layer in enumerate(block):
                if isinstance(layer, ResBlock) and not config.full:
                    self._adapt_residual(f"{name}_{number}", layer)
                elif isinstance(layer, AttentionBlock):
                    if not config.full:
                        layer.qkv, layer.proj_out = LoRA(layer.qkv, config.rank), LoRA(layer.proj_out, config.rank)
                    if config.tokens:
                        self.token_attention[f"{name}_{number}"] = TokenAttention(
                            layer.channels, config.token_dim, layer.num_heads
                        )
        unet.out.requires_grad_(config.full)
        self._add_heads()
        if fused:
            for module in unet.modules():
                if isinstance(module, AttentionBlock) and isinstance(module.attention, QKVAttentionLegacy):
                    module.attention = FusedAttention(module.num_heads)

    def _add_heads(self) -> None:
        config, unet = self.config, self.unet
        if config.align_dim:
            channels = unet.middle_block[0].out_channels
            self.align = nn.Sequential(
                nn.Conv2d(channels, config.hidden, 1), nn.SiLU(), nn.Conv2d(config.hidden, config.align_dim, 1)
            )
        if config.layout:
            self.to_layout = nn.Linear(config.hidden, config.layout_channels * config.layout**2)
            outputs = [(f"output_{i}", block) for i, block in enumerate(unet.output_blocks)]
            for name, block in [("middle", unet.middle_block), *outputs[: config.layout_blocks]]:
                self.layout_proj[name] = zeroed(nn.Conv2d(config.layout_channels, block[0].out_channels, 1))

    def _adapt_residual(self, key: str, layer: ResBlock) -> None:
        config = self.config
        layer.emb_layers[1] = LoRA(layer.emb_layers[1], config.rank)
        if config.film:
            self.film[key] = zeroed(nn.Linear(config.hidden, layer.emb_layers[1].base.out_features))
            self._film_targets[key] = layer.emb_layers[1]
        if config.conv_rank:
            layer.out_layers[-1] = ConvLoRA(layer.out_layers[-1], config.conv_rank)

    def _with_layout(self, name: str, h: th.Tensor, layout: th.Tensor | None) -> th.Tensor:
        if layout is None or name not in self.layout_proj:
            return h
        resized = tf.interpolate(layout, size=h.shape[-2:], mode="bilinear", align_corners=False)
        return h + self.layout_proj[name](resized).to(h.dtype)

    def _run(
        self,
        name: str,
        block: nn.Module,
        h: th.Tensor,
        emb: th.Tensor,
        *,
        tokens: th.Tensor | None,
        conditioned: th.Tensor,
    ) -> th.Tensor:
        for number, layer in enumerate(block):
            h = layer(h, emb) if isinstance(layer, TimestepBlock) else layer(h)
            key = f"{name}_{number}"
            if tokens is not None and key in self.token_attention:
                h = h + self.token_attention[key](h, tokens) * conditioned[:, :, None, None].to(h.dtype)
        return h

    def forward(  # pylint: disable=too-many-locals  # It is the U-Net's own forward pass, spelled out.
        self,
        x: th.Tensor,
        timesteps: th.Tensor,
        embeds: Mapping[str, th.Tensor] | None = None,
        keep: th.Tensor | None = None,
        *,
        labels: th.Tensor | None = None,
        middle: bool = False,
    ) -> th.Tensor | tuple[th.Tensor, th.Tensor]:
        """The U-Net's output for noisy images `x`, conditioned on each CLIP model's [N, dim] embedding.

        `keep` is [N, models] and drops embeddings where false. Without it every embedding given is used.
        `labels` are the classes a class-conditional U-Net draws, which both of its halves are told as before.
        With `middle`, the patch features the middle block predicts come back too, [N, align_dim, h, w].
        """
        unet = self.unet
        time_emb = unet.time_embed(timestep_embedding(timesteps, unet.model_channels))
        if labels is not None:
            time_emb = time_emb + unet.label_emb(labels)
        emb, tokens, layout, conditioned = time_emb, None, None, x.new_zeros(len(x), 1)
        if embeds is not None:
            if keep is None:
                keep = th.ones(len(x), len(self.conditioner.names), dtype=th.bool, device=x.device)
            hidden, conditioned = self.conditioner(embeds, keep)
            emb = time_emb + self.conditioner.to_emb(hidden) * conditioned
            tokens = self.conditioner.make_tokens(hidden)
            for key, target in self._film_targets.items():
                target.extra = self.film[key](hidden) * conditioned
            if self.config.layout:
                side = self.config.layout
                layout = self.to_layout(hidden).reshape(len(x), -1, side, side) * conditioned[:, :, None, None]
        try:
            output, bottleneck = self._denoise(x, time_emb, emb, tokens=tokens, layout=layout, conditioned=conditioned)
            return (output, self.align(bottleneck)) if middle else output
        finally:
            for target in self._film_targets.values():
                target.extra = None

    def _denoise(
        self,
        x: th.Tensor,
        time_emb: th.Tensor,
        emb: th.Tensor,
        *,
        tokens: th.Tensor | None,
        layout: th.Tensor | None,
        conditioned: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        unet, encoder = self.unet, self.config.train_encoder
        skips = []
        h = x
        # The frozen encoder half needs no gradient, unless the image itself does, as under CLIP guidance.
        with th.set_grad_enabled((encoder or x.requires_grad) and th.is_grad_enabled()):
            for i, block in enumerate(unet.input_blocks):
                h = (
                    self._run(f"input_{i}", block, h, emb, tokens=tokens, conditioned=conditioned)
                    if encoder
                    else block(h, time_emb)
                )
                skips.append(h)
        h = bottleneck = self._run("middle", unet.middle_block, h, emb, tokens=tokens, conditioned=conditioned)
        h = self._with_layout("middle", h, layout)
        for i, block in enumerate(unet.output_blocks):
            h = self._run(
                f"output_{i}", block, th.cat([h, skips.pop()], dim=1), emb, tokens=tokens, conditioned=conditioned
            )
            h = self._with_layout(f"output_{i}", h, layout)
        return unet.out(h.to(x.dtype)), bottleneck


def trainable(model: nn.Module) -> dict[str, nn.Parameter]:
    """The parameters that are trained, by name: everything added to the U-Net."""
    return {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}


def adapter_checkpoint(model: ConditionedUNet, weights: Mapping[str, th.Tensor] | None = None) -> dict:
    """What to save of a model: its configuration and the added weights, or `weights` in their place."""
    weights = trainable(model) if weights is None else weights
    return {
        "config": asdict(model.config),
        "weights": {name: value.detach().cpu() for name, value in weights.items()},
        "statistics": {name: value.cpu() for name, value in model.conditioner.named_buffers()},
    }


def load_adapter(unet: UNetModel, checkpoint: Mapping, fused: bool = True) -> ConditionedUNet:
    """Rebuild a conditioned model on a freshly loaded U-Net from `adapter_checkpoint`'s result."""
    model = ConditionedUNet(unet, AdapterConfig(**checkpoint["config"]), fused)
    expected = set(trainable(model))
    if set(checkpoint["weights"]) != expected:
        raise ValueError("The checkpoint's weights are not this configuration's.")
    model.load_state_dict(checkpoint["weights"], strict=False)
    model.conditioner.load_state_dict(checkpoint.get("statistics", {}), strict=False)
    return model

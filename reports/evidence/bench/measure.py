"""Peak VRAM of one CLIP guided step, on real architectures with random weights.

Weights do not change memory, so this needs only each model's config.json.
Each mode runs one step shaped like cgd/sampling.py: denoiser, x0, decode, four
cutouts through CLIP ViT-B/32 at fp32, then the gradient with respect to the input.

Modes:
  full      backward through denoiser and the full VAE decoder
  ckpt      the same, with gradient checkpointing in both
  tiny      backward through the denoiser and a tiny decoder
  tiny_ckpt the same, with a checkpointed denoiser
  x0_vae    no denoiser backward: the gradient stops at x0, through the full decoder
  x0_tiny   no denoiser backward: the gradient stops at x0, through a tiny decoder
  forward   no gradient at all: plain inference with the full decoder, for comparison
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch as th
import torch.utils.checkpoint
import diffusers
from diffusers import (AuraFlowTransformer2DModel, AutoencoderDC, AutoencoderKL, AutoencoderKLFlux2,
                       AutoencoderTiny, DiTTransformer2DModel, Flux2Transformer2DModel,
                       FluxTransformer2DModel, Lumina2Transformer2DModel, PixArtTransformer2DModel,
                       PRXTransformer2DModel, SanaTransformer2DModel, SD3Transformer2DModel,
                       UNet2DConditionModel, UNet2DModel, VQModel, ZImageTransformer2DModel)
from huggingface_hub import hf_hub_download

import open_clip

DEVICE = "cuda"
DTYPE = {"fp16": th.float16, "bf16": th.bfloat16}[os.environ.get("BENCH_DTYPE", "fp16")]
GIB = 1024 ** 3
FLUX_VAE = dict(in_channels=3, out_channels=3, down_block_types=["DownEncoderBlock2D"] * 4,
                up_block_types=["UpDecoderBlock2D"] * 4, block_out_channels=[128, 256, 512, 512],
                layers_per_block=2, latent_channels=16, use_quant_conv=False, use_post_quant_conv=False)
SD35M = dict(sample_size=128, patch_size=2, in_channels=16, out_channels=16, num_layers=24,
             attention_head_dim=64, num_attention_heads=24, joint_attention_dim=4096,
             caption_projection_dim=1536, pooled_projection_dim=2048, pos_embed_max_size=384,
             dual_attention_layers=tuple(range(13)), qk_norm="rms_norm")
SD3M = dict(sample_size=128, patch_size=2, in_channels=16, out_channels=16, num_layers=24,
            attention_head_dim=64, num_attention_heads=24, joint_attention_dim=4096,
            caption_projection_dim=1536, pooled_projection_dim=2048, pos_embed_max_size=192)
FLUX1 = dict(patch_size=1, in_channels=64, num_layers=19, num_single_layers=38, attention_head_dim=128,
             num_attention_heads=24, joint_attention_dim=4096, pooled_projection_dim=768,
             guidance_embeds=False)

# taef2 ships no diffusers config: the same layout as taef1 with its 32 latent channels.
TAEF2 = dict(in_channels=3, out_channels=3, encoder_block_out_channels=(64, 64, 64, 64),
             decoder_block_out_channels=(64, 64, 64, 64), num_encoder_blocks=(1, 3, 3, 3),
             num_decoder_blocks=(3, 3, 3, 1), latent_channels=32)

# name: family, denoiser class, (repo, subfolder) or inline config, vae class, vae source, tiny repo
SPECS = {
    "sd15": ("unet", UNet2DConditionModel, ("stable-diffusion-v1-5/stable-diffusion-v1-5", "unet"),
             AutoencoderKL, ("stable-diffusion-v1-5/stable-diffusion-v1-5", "vae"), "madebyollin/taesd"),
    "sd21": ("unet", UNet2DConditionModel, ("sd2-community/stable-diffusion-2-1", "unet"),
             AutoencoderKL, ("sd2-community/stable-diffusion-2-1", "vae"), "madebyollin/taesd"),
    "sdturbo": ("unet", UNet2DConditionModel, ("stabilityai/sd-turbo", "unet"),
                AutoencoderKL, ("stabilityai/sd-turbo", "vae"), "madebyollin/taesd"),
    "sdxl": ("unet", UNet2DConditionModel, ("stabilityai/stable-diffusion-xl-base-1.0", "unet"),
             AutoencoderKL, ("stabilityai/stable-diffusion-xl-base-1.0", "vae"), "madebyollin/taesdxl"),
    "ssd1b": ("unet", UNet2DConditionModel, ("segmind/SSD-1B", "unet"),
              AutoencoderKL, ("segmind/SSD-1B", "vae"), "madebyollin/taesdxl"),
    "vega": ("unet", UNet2DConditionModel, ("segmind/Segmind-Vega", "unet"),
             AutoencoderKL, ("segmind/Segmind-Vega", "vae"), "madebyollin/taesdxl"),
    "kolors": ("unet", UNet2DConditionModel, ("Kwai-Kolors/Kolors-diffusers", "unet"),
               AutoencoderKL, ("Kwai-Kolors/Kolors-diffusers", "vae"), "madebyollin/taesdxl"),
    "kandinsky22": ("kandinsky", UNet2DConditionModel, ("kandinsky-community/kandinsky-2-2-decoder", "unet"),
                    VQModel, ("kandinsky-community/kandinsky-2-2-decoder", "movq"), None),
    "pixart_a512": ("pixart", PixArtTransformer2DModel, ("PixArt-alpha/PixArt-XL-2-512x512", "transformer"),
                    AutoencoderKL, ("PixArt-alpha/PixArt-XL-2-512x512", "vae"), "madebyollin/taesd"),
    "pixart_s512": ("pixart", PixArtTransformer2DModel, ("PixArt-alpha/PixArt-Sigma-XL-2-512-MS", "transformer"),
                    AutoencoderKL, ("PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", "vae"), "madebyollin/taesdxl"),
    "pixart_s1024": ("pixart", PixArtTransformer2DModel, ("PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", "transformer"),
                     AutoencoderKL, ("PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", "vae"), "madebyollin/taesdxl"),
    "sana600_512": ("sana", SanaTransformer2DModel, ("Efficient-Large-Model/Sana_600M_512px_diffusers", "transformer"),
                    AutoencoderDC, ("Efficient-Large-Model/Sana_600M_512px_diffusers", "vae"), None),
    "sana1600_512": ("sana", SanaTransformer2DModel, ("Efficient-Large-Model/Sana_1600M_512px_diffusers", "transformer"),
                     AutoencoderDC, ("Efficient-Large-Model/Sana_1600M_512px_diffusers", "vae"), None),
    "sana1600_1024": ("sana", SanaTransformer2DModel, ("Efficient-Large-Model/Sana_1600M_1024px_diffusers", "transformer"),
                      AutoencoderDC, ("Efficient-Large-Model/Sana_1600M_1024px_diffusers", "vae"), None),
    "sprint600": ("sana", SanaTransformer2DModel, ("Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers", "transformer"),
                  AutoencoderDC, ("Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers", "vae"), None),
    "sprint1600": ("sana", SanaTransformer2DModel, ("Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers", "transformer"),
                   AutoencoderDC, ("Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers", "vae"), None),
    "sd35m": ("sd3", SD3Transformer2DModel, SD35M, AutoencoderKL, FLUX_VAE, "madebyollin/taesd3"),
    "sd3m": ("sd3", SD3Transformer2DModel, SD3M, AutoencoderKL, FLUX_VAE, "madebyollin/taesd3"),
    "flux1": ("flux", FluxTransformer2DModel, FLUX1, AutoencoderKL, FLUX_VAE, "madebyollin/taef1"),
    "klein4b": ("flux2", Flux2Transformer2DModel, ("black-forest-labs/FLUX.2-klein-4B", "transformer"),
                AutoencoderKLFlux2, ("black-forest-labs/FLUX.2-klein-4B", "vae"), TAEF2),
    "zimage": ("zimage", ZImageTransformer2DModel, ("Tongyi-MAI/Z-Image-Turbo", "transformer"),
               AutoencoderKL, ("Tongyi-MAI/Z-Image-Turbo", "vae"), "madebyollin/taef1"),
    "auraflow": ("aura", AuraFlowTransformer2DModel, ("fal/AuraFlow-v0.3", "transformer"),
                 AutoencoderKL, ("fal/AuraFlow-v0.3", "vae"), "madebyollin/taesdxl"),
    "lumina2": ("lumina", Lumina2Transformer2DModel, ("Alpha-VLLM/Lumina-Image-2.0", "transformer"),
                AutoencoderKL, ("Alpha-VLLM/Lumina-Image-2.0", "vae"), "madebyollin/taef1"),
    "dit256": ("dit", DiTTransformer2DModel, ("facebook/DiT-XL-2-256", "transformer"),
               AutoencoderKL, ("facebook/DiT-XL-2-256", "vae"), "madebyollin/taesd"),
    "dit512": ("dit", DiTTransformer2DModel, ("facebook/DiT-XL-2-512", "transformer"),
               AutoencoderKL, ("facebook/DiT-XL-2-512", "vae"), "madebyollin/taesd"),
    "prx512": ("prx", PRXTransformer2DModel, ("Photoroom/prx-512-t2i-sft", "transformer"),
               AutoencoderDC, ("Photoroom/prx-512-t2i-sft", "vae"), None),
    # A proxy, not the model: EDM2-S and XS channel layouts in diffusers' plain U-Net.
    "edm2s_proxy": ("uncond_unet", UNet2DModel,
                    dict(sample_size=64, in_channels=4, out_channels=4, layers_per_block=3,
                         block_out_channels=(192, 384, 576, 768),
                         down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
                         up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
                         attention_head_dim=64),
                    AutoencoderKL, ("stabilityai/sd-vae-ft-mse", ""), "madebyollin/taesd"),
    "edm2xs_proxy": ("uncond_unet", UNet2DModel,
                     dict(sample_size=64, in_channels=4, out_channels=4, layers_per_block=3,
                          block_out_channels=(128, 256, 384, 512),
                          down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
                          up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
                          attention_head_dim=64),
                     AutoencoderKL, ("stabilityai/sd-vae-ft-mse", ""), "madebyollin/taesd"),
}


def load_config(source):
    repo, sub = source
    name = f"{sub}/config.json" if sub else "config.json"
    return json.load(open(hf_hub_download(repo, name)))


def build(cls, source):
    config = dict(source) if isinstance(source, dict) else load_config(source)
    config = {k: v for k, v in config.items() if not k.startswith("_")}
    with th.device("meta"):
        model = cls.from_config(config)
    model = model.to_empty(device=DEVICE)
    model = model.to(DTYPE)
    with th.no_grad():
        for tensor in model.parameters():
            if tensor.dim() >= 2:
                th.nn.init.normal_(tensor, std=0.01)
            else:
                th.nn.init.zeros_(tensor)
        for tensor in model.buffers():
            if tensor.is_floating_point() and not tensor.isfinite().all():
                tensor.zero_()
    return model.requires_grad_(False).eval()


def tensor(*shape, dtype=DTYPE):
    return th.zeros(shape, device=DEVICE, dtype=dtype)


class Denoiser:
    """Calls a model with zero conditioning of the right shape. Returns its output."""

    def __init__(self, family, model):
        self.family, self.model, c = family, model, model.config
        self.c = c
        if family == "unet":
            dim = c.encoder_hid_dim or c.cross_attention_dim
            dim = dim[-1] if isinstance(dim, (list, tuple)) else dim
            self.ctx = tensor(1, 256 if c.encoder_hid_dim else 77, dim)
            self.added = None
            if c.addition_embed_type == "text_time":
                text = c.projection_class_embeddings_input_dim - 6 * c.addition_time_embed_dim
                self.added = dict(text_embeds=tensor(1, text), time_ids=tensor(1, 6))
        elif family == "kandinsky":
            self.added = dict(image_embeds=tensor(1, 1280))
        elif family == "pixart":
            self.ctx = tensor(1, 300, c.caption_channels)
        elif family == "sana":
            self.ctx = tensor(1, 300, c.caption_channels)
        elif family == "sd3":
            self.ctx, self.pooled = tensor(1, 333, c.joint_attention_dim), tensor(1, c.pooled_projection_dim)
        elif family == "flux":
            self.ctx, self.pooled = tensor(1, 256, c.joint_attention_dim), tensor(1, c.pooled_projection_dim)
            self.txt_ids = tensor(256, 3)
        elif family == "flux2":
            self.ctx = tensor(1, 512, c.joint_attention_dim)
            self.txt_ids = tensor(512, 4)
        elif family == "zimage":
            self.ctx = [tensor(128, c.cap_feat_dim)]
        elif family == "aura":
            self.ctx = tensor(1, 256, c.joint_attention_dim)
        elif family == "lumina":
            self.ctx = tensor(1, 256, c.cap_feat_dim)
            self.mask = th.ones((1, 256), device=DEVICE, dtype=th.bool)
        elif family == "prx":
            self.ctx = tensor(1, 256, c.context_in_dim)
            self.mask = th.ones((1, 256), device=DEVICE, dtype=th.bool)

    def __call__(self, x):
        m, f, c = self.model, self.family, self.c
        t = th.full((1,), 500.0, device=DEVICE, dtype=th.float32)
        if f == "unet":
            return m(x, t, encoder_hidden_states=self.ctx, added_cond_kwargs=self.added).sample
        if f == "kandinsky":
            return m(x, t, encoder_hidden_states=None, added_cond_kwargs=self.added).sample[:, :4]
        if f == "uncond_unet":
            return m(x, t).sample
        if f == "pixart":
            added = dict(resolution=None, aspect_ratio=None)
            if getattr(m, "use_additional_conditions", False):
                added = dict(resolution=tensor(1, 2), aspect_ratio=tensor(1, 1))
            return m(x, self.ctx, timestep=t, added_cond_kwargs=added).sample[:, :x.shape[1]]
        if f == "sana":
            kwargs = {"guidance": tensor(1)} if getattr(c, "guidance_embeds", False) else {}
            return m(x, self.ctx, timestep=t, **kwargs).sample
        if f == "sd3":
            return m(hidden_states=x, encoder_hidden_states=self.ctx, pooled_projections=self.pooled, timestep=t / 1000 * 1000).sample
        if f == "flux":
            b, ch, h, w = x.shape
            packed = x.view(b, ch, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(b, (h // 2) * (w // 2), ch * 4)
            ids = tensor((h // 2) * (w // 2), 3)
            out = m(hidden_states=packed, encoder_hidden_states=self.ctx, pooled_projections=self.pooled,
                    timestep=t / 1000, img_ids=ids, txt_ids=self.txt_ids).sample
            return out.reshape(b, h // 2, w // 2, ch, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(b, ch, h, w)
        if f == "flux2":
            b, ch, h, w = x.shape
            packed = x.view(b, ch // 4 if False else ch, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(b, (h // 2) * (w // 2), ch * 4)
            ids = tensor((h // 2) * (w // 2), 4)
            kwargs = {"guidance": tensor(1)} if getattr(c, "guidance_embeds", False) else {"guidance": None}
            out = m(hidden_states=packed, encoder_hidden_states=self.ctx, timestep=t / 1000,
                    img_ids=ids, txt_ids=self.txt_ids, **kwargs).sample
            return out.reshape(b, h // 2, w // 2, ch, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(b, ch, h, w)
        if f == "zimage":
            out = m([x[0].unsqueeze(1)], t / 1000, self.ctx).sample
            return th.stack([o.squeeze(1) for o in out])
        if f == "aura":
            return m(x, self.ctx, timestep=t / 1000).sample
        if f == "lumina":
            return m(x, t / 1000, self.ctx, self.mask).sample
        if f == "dit":
            return m(x, t, class_labels=th.zeros((1,), device=DEVICE, dtype=th.long)).sample[:, :4]
        if f == "prx":
            return m(x, t / 1000, self.ctx, attention_mask=self.mask).sample
        raise KeyError(f)


def latent_shape(family, denoiser, size, vae, source):
    """Latent channels and spatial factor from the configs."""
    if isinstance(vae, AutoencoderDC):
        channels, factor = vae.config.latent_channels, 2 ** (len(vae.config.encoder_block_out_channels) - 1)
    elif isinstance(vae, AutoencoderKLFlux2):
        channels, factor = vae.config.latent_channels, 8
    elif isinstance(vae, VQModel):
        channels, factor = 4, 8
    else:
        channels = vae.config.latent_channels
        factor = 2 ** (len(vae.config.block_out_channels) - 1)
    return channels, size // factor


def cutouts_clip(clip, image):
    """Four resized full-frame cutouts, like the fast sampler, through CLIP at fp32."""
    image = image.float().add(1).div(2)
    batch = th.nn.functional.interpolate(image, size=224, mode="bilinear", align_corners=False)
    batch = batch.repeat(4, 1, 1, 1)
    features = clip.encode_image(batch)
    return features.square().mean()


def enable_ckpt(model):
    if isinstance(model, AutoencoderDC):  # diffusers has no checkpointing for it: wrap each up block.
        for block in model.decoder.up_blocks:
            plain = block.forward
            block.forward = lambda x, plain=plain: th.utils.checkpoint.checkpoint(plain, x, use_reentrant=False)
        return
    if hasattr(model, "enable_gradient_checkpointing"):
        try:
            model.enable_gradient_checkpointing()
        except ValueError as error:
            print(json.dumps(dict(no_checkpointing=type(model).__name__, error=str(error)[:100])), flush=True)
            return
        model.train()  # diffusers only checkpoints in training mode for some classes
        for module in model.modules():
            if isinstance(module, th.nn.Dropout):
                module.p = 0.0


def disable_ckpt(model):
    if isinstance(model, AutoencoderDC):
        for block in model.decoder.up_blocks:
            block.__dict__.pop("forward", None)
        return
    if hasattr(model, "disable_gradient_checkpointing"):
        try:
            model.disable_gradient_checkpointing()
        except ValueError:
            return
        model.eval()


def run_mode(mode, family, model, vae, tiny, clip, channels, side):
    th.cuda.empty_cache()
    th.cuda.reset_peak_memory_stats()
    grad_through_denoiser = mode in ("full", "ckpt", "tiny", "tiny_ckpt")
    use_tiny = mode in ("tiny", "tiny_ckpt", "x0_tiny")
    if use_tiny and tiny is None:
        return None
    ckpt = mode in ("ckpt", "tiny_ckpt")
    if ckpt:
        enable_ckpt(model)
        if not use_tiny:
            enable_ckpt(vae)
    denoiser = Denoiser(family, model)
    start = time.perf_counter()
    x = th.randn(1, channels, side, side, device=DEVICE, dtype=DTYPE)
    if mode == "forward":
        with th.no_grad():
            eps = denoiser(x)
            image = vae.decode(eps).sample if not isinstance(vae, VQModel) else vae.decode(eps, force_not_quantize=True).sample
    else:
        x.requires_grad_(grad_through_denoiser)
        with th.set_grad_enabled(grad_through_denoiser):
            eps = denoiser(x)
            x0 = x - eps  # The sigma and alpha values do not change the memory.
        if not grad_through_denoiser:
            x0 = x0.detach().requires_grad_()
        decoder = tiny if use_tiny else vae
        if use_tiny:
            image = decoder.decode(x0).sample
        elif isinstance(decoder, VQModel):
            image = decoder.decode(x0, force_not_quantize=True).sample
        else:
            image = decoder.decode(x0).sample
        loss = cutouts_clip(clip, image)
        th.autograd.grad(loss, x if grad_through_denoiser else x0)
    th.cuda.synchronize()
    seconds = time.perf_counter() - start
    if ckpt:
        disable_ckpt(model)
        disable_ckpt(vae)
    return dict(mode=mode, side=side, peak_alloc=th.cuda.max_memory_allocated() / GIB,
                peak_reserved=th.cuda.max_memory_reserved() / GIB, seconds=seconds, image=image.shape[-1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("--sizes", type=int, nargs="+", default=[512])
    p.add_argument("--modes", nargs="+", default=["forward", "full", "ckpt", "tiny", "tiny_ckpt", "x0_vae", "x0_tiny"])
    args = p.parse_args()
    family, cls, source, vae_cls, vae_source, tiny_repo = SPECS[args.name]
    th.manual_seed(0)
    model = build(cls, source)
    vae = build(vae_cls, vae_source)
    tiny = None
    if tiny_repo:
        try:
            tiny = build(AutoencoderTiny, tiny_repo if isinstance(tiny_repo, dict) else (tiny_repo, ""))
        except Exception as error:  # A tiny decoder without a diffusers config is reported, not fatal.
            print(json.dumps(dict(name=args.name, tiny_error=str(error)[:120])), flush=True)
    clip = open_clip.create_model("ViT-B-32", device=DEVICE).float().requires_grad_(False).eval()
    params = sum(t.numel() for t in model.parameters())
    decoder_params = sum(t.numel() for t in (vae.decoder if hasattr(vae, "decoder") else vae).parameters())
    tiny_params = sum(t.numel() for t in tiny.decoder.parameters()) if tiny else 0
    base = th.cuda.memory_allocated() / GIB
    print(json.dumps(dict(name=args.name, params_M=round(params / 1e6, 1), decoder_M=round(decoder_params / 1e6, 1),
                          tiny_decoder_M=round(tiny_params / 1e6, 2), resident_GiB=round(base, 3))), flush=True)
    for size in args.sizes:
        channels, side = latent_shape(family, model, size, vae, vae_source)
        for mode in args.modes:
            try:
                result = run_mode(mode, family, model, vae, tiny, clip, channels, side)
                if result:
                    print(json.dumps(dict(name=args.name, px=size, **{k: round(v, 3) if isinstance(v, float) else v for k, v in result.items()})), flush=True)
            except th.OutOfMemoryError:
                th.cuda.empty_cache()
                disable_ckpt(model)
                disable_ckpt(vae)
                print(json.dumps(dict(name=args.name, px=size, mode=mode, oom=True)), flush=True)
            except Exception as error:
                print(json.dumps(dict(name=args.name, px=size, mode=mode, error=f"{type(error).__name__}: {str(error)[:200]}")), flush=True)
                disable_ckpt(model)
                disable_ckpt(vae)


if __name__ == "__main__":
    sys.exit(main())

"""The same guided step as measure.py, counted by the CPU allocator instead of on the GPU.

Used when the GPU is busy with other work. Half precision on this CPU is far too slow, so
every tensor runs at twice the bytes: denoiser and decoder at fp32 instead of fp16, CLIP
at fp64 instead of fp32. The peak of the profiler's allocation timeline is then halved.
That counts live bytes the same way torch.cuda.max_memory_allocated does, so the cross
check is sd15 against its measured values on the card.
"""
import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import torch as th

import measure
import open_clip

measure.DEVICE = "cpu"
measure.DTYPE = th.float32
measure.tensor.__kwdefaults__ = {"dtype": th.float32}  # Its default was bound at import.
SCALE = 0.5  # Every tensor here is twice the bytes of the same tensor in the real run.
GIB = 1024 ** 3


def cutouts_clip(clip, image):
    image = image.double().add(1).div(2)
    batch = th.nn.functional.interpolate(image, size=224, mode="bilinear", align_corners=False)
    return clip.encode_image(batch.repeat(4, 1, 1, 1)).square().mean()


def shrink(config, depth):
    """A copy of a transformer's config with a fraction of its layers, for a fit in depth."""
    out = dict(config)
    for key in ("num_layers", "num_single_layers", "n_layers"):
        if key in out:
            out[key] = max(1, round(out[key] * depth))
    if "dual_attention_layers" in out:
        out["dual_attention_layers"] = tuple(i for i in out["dual_attention_layers"] if i < out["num_layers"])
    return out


def build_denoiser(cls, source, depth):
    config = dict(source) if isinstance(source, dict) else measure.load_config(source)
    config = {k: v for k, v in config.items() if not k.startswith("_")}
    with th.device("meta"):
        full = sum(t.numel() for t in cls.from_config(config).parameters())
    if depth < 1:
        config = shrink(config, depth)
    if os.environ.get("BENCH_LAZY"):  # Untouched pages take no RAM, and sizes do not depend on weights.
        with th.device("meta"):
            model = cls.from_config(config)
        return model.to_empty(device="cpu").requires_grad_(False).eval(), full, config
    return measure.build(cls, config), full, config


def peak_bytes(fn):
    with th.profiler.profile(activities=[th.profiler.ProfilerActivity.CPU], profile_memory=True) as prof:
        fn()
    events = [e for e in prof.profiler.kineto_results.events() if e.name() == "[memory]"]
    events.sort(key=lambda e: e.start_ns())
    live = peak = 0
    for event in events:
        live += event.nbytes()
        peak = max(peak, live)
    return peak


def step(mode, family, model, vae, tiny, clip, channels, side):
    grad_through_denoiser = mode in ("full", "ckpt", "tiny", "tiny_ckpt")
    use_tiny = mode in ("tiny", "tiny_ckpt", "x0_tiny")
    denoiser = measure.Denoiser(family, model)
    x = th.randn(1, channels, side, side, dtype=measure.DTYPE)
    if mode == "forward":
        with th.no_grad():
            eps = denoiser(x)
            if isinstance(vae, measure.VQModel):
                vae.decode(eps, force_not_quantize=True)
            else:
                vae.decode(eps)
        return
    x.requires_grad_(grad_through_denoiser)
    with th.set_grad_enabled(grad_through_denoiser):
        x0 = x - denoiser(x)
    if not grad_through_denoiser:
        x0 = x0.detach().requires_grad_()
    decoder = tiny if use_tiny else vae
    if isinstance(decoder, measure.VQModel):
        image = decoder.decode(x0, force_not_quantize=True).sample
    else:
        image = decoder.decode(x0).sample
    loss = cutouts_clip(clip, image)
    th.autograd.grad(loss, x if grad_through_denoiser else x0)


def main():
    name = sys.argv[1]
    sizes = [int(s) for s in sys.argv[2].split(",")]
    modes = sys.argv[3].split(",") if len(sys.argv) > 3 else ["forward", "full", "ckpt", "tiny", "tiny_ckpt", "x0_vae", "x0_tiny"]
    th.set_num_threads(int(os.environ.get("BENCH_THREADS", "10")))
    family, cls, source, vae_cls, vae_source, tiny_repo = measure.SPECS[name]
    th.manual_seed(0)
    depth = float(os.environ.get("BENCH_DEPTH", "1"))
    model, full_params, config = build_denoiser(cls, source, depth)
    layers = {k: config[k] for k in ("num_layers", "num_single_layers", "n_layers") if k in config}
    if os.environ.get("BENCH_CAST"):  # Stands for fp8 storage with bf16 compute, at twice the bytes.
        model.enable_layerwise_casting(storage_dtype=th.bfloat16, compute_dtype=th.float32)
    vae = measure.build(vae_cls, vae_source)
    tiny = measure.build(measure.AutoencoderTiny, tiny_repo if isinstance(tiny_repo, dict) else (tiny_repo, "")) if tiny_repo else None
    clip = open_clip.create_model("ViT-B-32", device="cpu").double().requires_grad_(False).eval()
    resident = SCALE * sum(t.numel() * t.element_size() for m in (model, vae, tiny, clip) if m is not None for t in list(m.parameters()) + list(m.buffers()))
    print(json.dumps(dict(name=name, resident_GiB=round(resident / GIB, 3), full_params_M=round(full_params / 1e6, 1), layers=layers, depth=depth,
                          decoder_M=round(sum(t.numel() for t in (vae.decoder if hasattr(vae, "decoder") else vae).parameters()) / 1e6, 1),
                          tiny_decoder_M=round(sum(t.numel() for t in tiny.decoder.parameters()) / 1e6, 2) if tiny else 0)), flush=True)
    for size in sizes:
        channels, side = measure.latent_shape(family, model, size, vae, vae_source)
        for mode in modes:
            if mode in ("tiny", "tiny_ckpt", "x0_tiny") and tiny is None:
                continue
            ckpt = mode in ("ckpt", "tiny_ckpt")
            start = time.perf_counter()
            try:
                if ckpt:
                    measure.enable_ckpt(model)
                    if mode == "ckpt":
                        measure.enable_ckpt(vae)
                peak = SCALE * peak_bytes(lambda: step(mode, family, model, vae, tiny, clip, channels, side))
                print(json.dumps(dict(name=name, px=size, mode=mode, side=side, depth=depth, layers=layers, resident=round(resident / GIB, 3),
                                      extra=round(peak / GIB, 3), peak_alloc=round((resident + peak) / GIB, 3),
                                      seconds=round(time.perf_counter() - start, 1), source="cpu_alloc")), flush=True)
            except Exception as error:
                print(json.dumps(dict(name=name, px=size, mode=mode, error=f"{type(error).__name__}: {str(error)[:200]}", source="cpu_alloc")), flush=True)
            if ckpt:
                measure.disable_ckpt(model)
                measure.disable_ckpt(vae)


if __name__ == "__main__":
    main()

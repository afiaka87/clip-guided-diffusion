"""Scratch: GAN and transformer upscalers, cost at each size and quality from the same bases."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import torchvision.transforms.functional as tvf
from PIL import Image
from spandrel import ModelLoader
from cgd import script_util

MODELS = {"esrgan": "RealESRGAN_x4plus.pth", "ultrasharp": "4x-UltraSharp.pth", "swinirL": "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
          "hat": "HAT_SRx4_ImageNet-pretrain.pth", "compact": "realesr-general-x4v3.pth"}
for tag, filename in MODELS.items():
    desc = ModelLoader().load_from_file(os.path.join(SP, "gan_weights", filename))
    half = desc.supports_half
    print(tag, desc.architecture.name, "scale", desc.scale, "half", half, "params", sum(p.numel() for p in desc.model.parameters()), flush=True)
    model = desc.model.eval().requires_grad_(False).cuda()
    if half:
        model = model.half()
    for base, out_size, sample in (("d64", 256, 12), ("d128", 512, 12), ("d256", 1024, 3), ("d512", 2048, 3)):
        for prompt, path in list(paths(base).items())[:sample]:
            x = tvf.to_tensor(Image.open(path).convert("RGB")).unsqueeze(0).cuda()
            x = x.half() if half else x

            def run():
                with th.no_grad():
                    return desc(x)
            try:
                y, stats = measured(run, retries=3)
            except RuntimeError as error:
                print(tag, base, "failed", error, flush=True)
                continue
            folder = os.path.join(HERE, "out", f"{tag}{out_size}")
            os.makedirs(folder, exist_ok=True)
            target = os.path.join(folder, script_util.alphanumeric_filter(prompt) + ".png")
            tvf.to_pil_image(y[0].clamp(0, 1).float().cpu()).save(target)
            record(f"{tag}{out_size}", prompt, target, stats)
            del y
    del model, desc
    gc.collect(); th.cuda.empty_cache()
print("gan done", flush=True)

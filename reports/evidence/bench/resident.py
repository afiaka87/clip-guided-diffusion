"""Exact weight bytes per model from configs on the meta device: no memory, no GPU."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch as th
import measure
GIB = 1024 ** 3
CLIP_BYTES = None
def count(cls, source):
    config = dict(source) if isinstance(source, dict) else measure.load_config(source)
    config = {k: v for k, v in config.items() if not k.startswith("_")}
    with th.device("meta"):
        model = cls.from_config(config)
    return model
import open_clip
with th.device("meta"):
    clip = open_clip.create_model("ViT-B-32", device="meta")
clip_gib = sum(t.numel() * 4 for t in list(clip.parameters()) + list(clip.buffers())) / GIB
print("clip_gib", round(clip_gib, 3))
out = {}
for name, (family, cls, source, vae_cls, vae_source, tiny_repo) in measure.SPECS.items():
    try:
        model = count(cls, source)
        vae = count(vae_cls, vae_source)
        tiny = count(measure.AutoencoderTiny, tiny_repo if isinstance(tiny_repo, dict) else (tiny_repo, "")) if tiny_repo else None
    except Exception as error:
        print(name, "ERR", str(error)[:100]); continue
    p = sum(t.numel() for t in model.parameters())
    v = sum(t.numel() for t in vae.parameters())
    d = sum(t.numel() for t in (vae.decoder if hasattr(vae, "decoder") else vae).parameters())
    t = sum(x.numel() for x in tiny.parameters()) if tiny else 0
    td = sum(x.numel() for x in tiny.decoder.parameters()) if tiny else 0
    out[name] = dict(denoiser_M=round(p/1e6,1), vae_M=round(v/1e6,1), decoder_M=round(d/1e6,1), tiny_dec_M=round(td/1e6,2),
                     denoiser_gib=round(p*2/GIB,3), decoder_gib=round(d*2/GIB,3), tiny_dec_gib=round(td*2/GIB,3))
    print(name, out[name], flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "resident.json"), "w"), indent=0)

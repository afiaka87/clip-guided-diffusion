import os, time, json, torch, numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
d = json.load(open(r"D:\laion2b-filter\bench_sample\out\captions.json"))
paths = d["paths"][:192]; ims = [Image.open(p).convert("RGB") for p in paths]
rid = "florence-community/Florence-2-base"
proc = AutoProcessor.from_pretrained(rid)
print("attrs:", {k: getattr(proc, k) for k in dir(proc) if "token" in k.lower() and "num" in k.lower()})
model = AutoModelForImageTextToText.from_pretrained(rid, dtype=torch.float16).cuda().eval()
for size, ntok, bs in ((384, 145, 64), (512, 257, 64)):
    proc.num_image_tokens = ntok
    if hasattr(proc, "image_seq_length"): proc.image_seq_length = ntok
    out = []
    try:
        for i in range(0, len(ims), bs):
            if i == bs: torch.cuda.synchronize(); t = time.time()
            x = proc(text=["<DETAILED_CAPTION>"] * len(ims[i:i+bs]), images=ims[i:i+bs], size={"height": size, "width": size}, return_tensors="pt").to("cuda", torch.float16)
            with torch.no_grad(): g = model.generate(**x, max_new_tokens=96, num_beams=1, do_sample=False)
            out += [s.strip() for s in proc.batch_decode(g, skip_special_tokens=True)]
        torch.cuda.synchronize(); print(size, f"{(len(ims)-bs)/(time.time()-t):.1f} img/s", "peak", torch.cuda.max_memory_allocated()/1e9)
        for i in (0, 5, 11, 23): print("  ", out[i][:300])
        d[f"florence2_base_{size}_detailed"] = out
    except Exception as e:
        print(size, "failed", repr(e)[:300])
json.dump(d, open(r"D:\laion2b-filter\bench_sample\out\captions_384.json", "w"), indent=1)

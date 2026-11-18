"""Captioner speed and output on the same images: Florence-2-base (768 and 384 input), SmolVLM-256M, Qwen3-VL-2B."""
import os, time, json, torch, numpy as np
from PIL import Image
from common import sample, log, OUT
df = sample(); n = int(os.environ.get("N", 512))
rng = np.random.default_rng(0); idx = rng.choice(len(df), n, replace=False)
paths = df.path.iloc[idx].tolist(); keys = df.key.iloc[idx].tolist()
ims = [Image.open(p).convert("RGB") for p in paths]
import sys
WHICH = sys.argv[1]
capf = os.path.join(OUT, 'captions.json')
caps = {k: v for k, v in json.load(open(capf)).items() if k not in ('keys', 'paths')} if os.path.exists(capf) else {}
MAXTOK = int(os.environ.get("MAXTOK", 96))

def timed(name, fn, bs):
    out = []; torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    out += fn(ims[:bs])  # warm-up batch, not timed
    t = time.time()
    for i in range(bs, n, bs): out += fn(ims[i:i+bs])
    torch.cuda.synchronize(); r = (n - bs) / (time.time() - t)
    toks = np.mean([len(c.split()) for c in out])
    log(f"{name}: {r:.1f} img/s (batch {bs}), mean {toks:.0f} words, peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    caps[name] = out
    json.dump({"keys": keys, "paths": paths, **caps}, open(os.path.join(OUT, "captions.json"), "w"), indent=1)

from transformers import AutoProcessor, AutoModelForImageTextToText
if WHICH.startswith("florence"):
  rid = "florence-community/Florence-2-base"
  proc = AutoProcessor.from_pretrained(rid)
  model = AutoModelForImageTextToText.from_pretrained(rid, dtype=torch.float16).cuda().eval()
def flo(batch, size=None, task="<MORE_DETAILED_CAPTION>"):
    kw = {"size": {"height": size, "width": size}} if size else {}
    x = proc(text=[task] * len(batch), images=batch, return_tensors="pt", **kw).to("cuda", torch.float16)
    with torch.no_grad():
        g = model.generate(**x, max_new_tokens=MAXTOK, num_beams=1, do_sample=False)
    return [s.strip() for s in proc.batch_decode(g, skip_special_tokens=True)]
if WHICH == "florence384": timed("florence2_base_384_more_detailed", lambda b: flo(b, 384), 32)
if WHICH == "florence384_detailed": timed("florence2_base_384_detailed", lambda b: flo(b, 384, "<DETAILED_CAPTION>"), 48)
if WHICH == "florence768_detailed": timed("florence2_base_768_detailed", lambda b: flo(b, None, "<DETAILED_CAPTION>"), 32)

PROMPT = "Describe this image in one or two sentences."
def chat_model(rid, bs, **pkw):
    proc = AutoProcessor.from_pretrained(rid, **pkw); proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(rid, dtype=torch.bfloat16).cuda().eval()
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    def fn(batch):
        x = proc(text=[text] * len(batch), images=[[im] for im in batch], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            g = model.generate(**x, max_new_tokens=MAXTOK, do_sample=False)
        return [s.strip() for s in proc.batch_decode(g[:, x["input_ids"].shape[1]:], skip_special_tokens=True)]
    return model, fn
if WHICH == "smolvlm":
    model, fn = chat_model("HuggingFaceTB/SmolVLM-256M-Instruct", 32, size={"longest_edge": 512}); timed("smolvlm_256m", fn, 32)
if WHICH == "qwen":
    model, fn = chat_model("Qwen/Qwen3-VL-2B-Instruct", 32, min_pixels=256*256, max_pixels=256*256); timed("qwen3vl_2b_b32", fn, 32)
log("bench3 done")

"""Scratch: contact sheets to look at the pipelines side by side."""
import json, os, sys
from PIL import Image, ImageDraw
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
rows = [r for r in map(json.loads, open("results.jsonl")) if "pipe" in r]
table = {}
for r in rows:
    table.setdefault(r["pipe"], {})[r["prompt"]] = r["path"]
def sheet(name, pipes, prompts, cell=256):
    W = Image.new("RGB", (cell * len(pipes), (cell + 14) * len(prompts)), "white")
    d = ImageDraw.Draw(W)
    for j, prompt in enumerate(prompts):
        for i, pipe in enumerate(pipes):
            if prompt in table.get(pipe, {}):
                W.paste(Image.open(table[pipe][prompt]).convert("RGB").resize((cell, cell), Image.LANCZOS), (i * cell, j * (cell + 14) + 14))
            d.text((i * cell + 2, j * (cell + 14) + 2), pipe, fill="black")
    W.save(name)
ps = ["a red fox in the snow", "a lighthouse on a rocky coast at sunset", "a pencil sketch of an old man with a beard", "a macro photograph of a dew covered dragonfly"]
sheet("sheet256.png", ["d64", "d256", "sr256_u", "sr256_g", "i2i256_s700", "esrgan256"], ps)
sheet("sheet512.png", ["d128", "d512", "sr512_u", "sr512_g", "i2i512_s700", "esrgan512"], ps)
sheet("sheetvar.png", ["v_base", "v_secondary", "v_dt995", "v_orth", "v_freeu13", "pp_hm50"], ps)

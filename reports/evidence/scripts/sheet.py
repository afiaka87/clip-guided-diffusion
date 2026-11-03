"""Contact sheet: argv: <out.png> <tile> <arm...>; arms read from JSON maps listed in env JSONS (colon separated). Row per arm, column per prompt."""
import json, os, sys
from PIL import Image, ImageDraw
out, tile, arms = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
paths = {}
for f in os.environ["JSONS"].split(":"): paths.update(json.load(open(f)))
cols = [int(c) for c in os.environ.get("COLS", "0,1,2,3,4,5,6,7,8,9,10,11").split(",")]
sheet = Image.new("RGB", (len(cols) * tile, len(arms) * (tile + 14)), "white")
d = ImageDraw.Draw(sheet)
for r, a in enumerate(arms):
    y = r * (tile + 14); d.text((2, y + 1), a, fill="black")
    for c, i in enumerate(cols):
        sheet.paste(Image.open(paths[a][i]).convert("RGB").resize((tile, tile), Image.LANCZOS), (c * tile, y + 14))
sheet.save(out); print(sheet.size)

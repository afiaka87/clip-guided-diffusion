"""Contact sheet of given (shard, key) samples, with a label under each."""
import io, tarfile, collections
from PIL import Image, ImageDraw
def sheet(items, out, cols=8, labels=None, cell=256):
    by = collections.defaultdict(set)
    for s, k in items: by[int(s)].add(k)
    ims = {}
    for s, keys in by.items():
        with tarfile.open(rf"D:\laion2b-en-aesthetic\{s:05d}.tar") as tf:
            for m in tf:
                if m.name.endswith(".jpg") and m.name[:-4] in keys:
                    ims[(s, m.name[:-4])] = Image.open(io.BytesIO(tf.extractfile(m).read())).convert("RGB")
    rows = (len(items) + cols - 1) // cols; lh = 14 if labels else 0
    sh = Image.new("RGB", (cols * (cell + 4), rows * (cell + 4 + lh)), (255, 0, 255))
    d = ImageDraw.Draw(sh)
    for i, (s, k) in enumerate(items):
        x, y = (i % cols) * (cell + 4), (i // cols) * (cell + 4 + lh)
        im = ims.get((int(s), k))
        if im: sh.paste(im.resize((cell, cell)), (x + 2, y + 2))
        if labels: d.text((x + 2, y + cell + 3), str(labels[i])[:42], fill=(0, 0, 0))
    sh.save(out)

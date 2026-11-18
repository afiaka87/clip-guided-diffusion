"""Stricter padding and collage detection on a random subset of shards.

Padding: lines from an edge that are flat to within JPEG noise (std < 2), all one colour (within 3 of the
edge line), ending in a sharp step into content. Collage gutters: interior full-length flat lines.
"""
import io, os, sys, glob, tarfile, random
from multiprocessing import Pool
import numpy as np
from PIL import Image
import pyarrow as pa, pyarrow.parquet as pq
LUMA = np.array([0.299, 0.587, 0.114], np.float32)

def edge(std, mean):
    ref = mean[0]; n = 0
    while n < len(std) and std[n] < 2.0 and abs(mean[n] - ref) < 3.0:
        n += 1
    if n == 0 or n == len(std):
        return n, -1.0, 0.0
    step = abs(mean[min(n + 1, len(mean) - 1)] - ref) + std[min(n + 1, len(std) - 1)]
    return n, float(ref), float(step)

def gutters(std, lo=24, hi=232):
    s = std[lo:hi] < 2.0
    # count runs of flat interior lines (each run = one gutter)
    return int(np.sum(s[1:] & ~s[:-1]) + (1 if s.size and s[0] else 0))

def one(path):
    shard = int(os.path.basename(path)[:5]); rows = []
    with tarfile.open(path) as tf:
        for m in tf:
            if not m.name.endswith(".jpg"):
                continue
            a = np.asarray(Image.open(io.BytesIO(tf.extractfile(m).read())).convert("RGB"), np.float32)
            g = a @ LUMA
            r = {"shard": shard, "key": m.name[:-4]}
            rs, rm, cs, cm = g.std(1), g.mean(1), g.std(0), g.mean(0)
            for name, s, mu in (("top", rs, rm), ("bottom", rs[::-1], rm[::-1]), ("left", cs, cm), ("right", cs[::-1], cm[::-1])):
                r[f"p_{name}"], r[f"p_{name}_mean"], r[f"p_{name}_step"] = edge(s, mu)
            r["gut_h"] = gutters(rs); r["gut_v"] = gutters(cs)
            rows.append(r)
    return pa.Table.from_pylist(rows)

if __name__ == "__main__":
    tars = sorted(glob.glob(r"D:\laion2b-en-aesthetic\*.tar"))[17:-1]
    random.seed(0); tars = random.sample(tars, int(sys.argv[2]))
    with Pool(int(os.environ.get("PROCS", 30))) as p:
        tab = pa.concat_tables(p.map(one, tars))
    pq.write_table(tab, sys.argv[1]); print(tab.num_rows)

"""One pass over every shard: each sample's .json/.txt metadata plus classical image statistics.

Writes one parquet per shard into OUT (skipping shards already written), so it resumes and can be read while running.
"""
import io, os, sys, glob, json, tarfile, time
from multiprocessing import Pool
import numpy as np
from PIL import Image
import pyarrow as pa, pyarrow.parquet as pq

D = r"D:\laion2b-en-aesthetic"
N = 256
yy, xx = np.meshgrid(np.fft.fftfreq(N), np.fft.rfftfreq(N), indexing="ij")
RADIUS = np.sqrt(yy**2 + xx**2)
BANDS = [0.1, 0.2, 0.3, 0.4]
MASKS = [RADIUS > b for b in BANDS]
NONDC = RADIUS > 0.02
HANN = np.outer(np.hanning(N), np.hanning(N)).astype(np.float32)
RING = np.ones((N, N), bool); RING[4:-4, 4:-4] = False
LUMA = np.array([0.299, 0.587, 0.114], np.float32)
META = ["similarity", "hash", "punsafe", "pwatermark", "aesthetic", "WIDTH", "HEIGHT", "caption", "url",
        "status", "width", "height", "original_width", "original_height", "exif", "sha256"]

def band(std, mean, tol=6.0):
    """Consecutive near-uniform lines from the edge, and their mean grey."""
    over = np.nonzero(std >= tol)[0]
    n = int(over[0]) if over.size else len(std)
    return n, (float(mean[:n].mean()) if n else -1.0)

def stats(a):
    out = {}
    g = a @ LUMA
    rs, rm, cs, cm = g.std(1), g.mean(1), g.std(0), g.mean(0)
    for name, std, mean in (("top", rs, rm), ("bottom", rs[::-1], rm[::-1]), ("left", cs, cm), ("right", cs[::-1], cm[::-1])):
        out[f"b_{name}"], out[f"b_{name}_mean"] = band(std, mean)
    mx, mn = a.max(2), a.min(2)
    out["white_frac"] = float((mn > 240).mean()); out["black_frac"] = float((mx < 15).mean())
    out["ring_white"] = float((mn[RING] > 240).mean()); out["ring_black"] = float((mx[RING] < 15).mean())
    out["sat_mean"] = float(((mx - mn) / np.maximum(mx, 1)).mean())
    rg = a[..., 0] - a[..., 1]; yb = 0.5 * (a[..., 0] + a[..., 1]) - a[..., 2]
    out["colorfulness"] = float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))
    out["chroma_mean"] = float((mx - mn).mean())
    out["lum_mean"] = float(g.mean()); out["lum_std"] = float(g.std())
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1]
    out["lap_var"] = float(lap.var())
    out["edge_frac"] = float(((np.abs(np.diff(g, axis=1))[:-1] + np.abs(np.diff(g, axis=0))[:, :-1]) > 40).mean())
    p = np.abs(np.fft.rfft2((g - g.mean()) * HANN)) ** 2
    tot = p[NONDC].sum() + 1e-9
    for b, m in zip(BANDS, MASKS):
        out[f"hf{int(b * 10)}"] = float(p[m].sum() / tot)
    q = a.astype(np.uint8) >> 3
    idx = (q[..., 0].astype(np.int32) << 10) | (q[..., 1].astype(np.int32) << 5) | q[..., 2]
    out["n_colors"] = int(np.count_nonzero(np.bincount(idx.ravel(), minlength=32768)))
    hist = np.bincount(g.astype(np.uint8).ravel(), minlength=256) / g.size
    nz = hist[hist > 0]; out["entropy"] = float(-(nz * np.log2(nz)).sum())
    small = np.asarray(Image.fromarray(g.astype(np.uint8)).resize((9, 8), Image.BILINEAR), np.int16)
    out["dhash"] = int(np.packbits((small[:, 1:] > small[:, :-1]).ravel()).view(">u8")[0].astype(np.int64))
    return out

def one(args):
    path, out_dir = args
    shard = int(os.path.basename(path)[:5])
    dst = os.path.join(out_dir, f"{shard:05d}.parquet")
    if os.path.exists(dst):
        return shard, -1
    samples, err = {}, None
    try:
        with tarfile.open(path, "r:") as tf:
            for m in tf:
                stem, ext = m.name.rsplit(".", 1)
                s = samples.setdefault(stem, {"shard": shard, "key": stem})
                data = tf.extractfile(m).read()
                if ext == "json":
                    j = json.loads(data)
                    for k in META:
                        s[k] = j.get(k)
                    s["hash"] = str(s["hash"])
                elif ext == "txt":
                    s["txt"] = data.decode("utf-8", "replace")
                elif ext == "jpg":
                    s["jpg_bytes"] = len(data)
                    try:
                        im = Image.open(io.BytesIO(data)); s["mode"] = im.mode; s["w"], s["h"] = im.size
                        im = im.convert("RGB")
                        if im.size != (N, N):
                            im = im.resize((N, N))
                        s.update(stats(np.asarray(im, dtype=np.float32))); s["decode_ok"] = True
                    except Exception as ex:
                        s["decode_ok"] = False; s["mode"] = repr(ex)[:80]
    except Exception as ex:
        err = repr(ex)[:200]
    rows = list(samples.values())
    for r in rows:
        r["tar_error"] = err
    pq.write_table(pa.Table.from_pylist(rows), dst + ".tmp")
    os.replace(dst + ".tmp", dst)
    return shard, len(rows)

if __name__ == "__main__":
    out_dir = sys.argv[1]; os.makedirs(out_dir, exist_ok=True)
    tars = sorted(glob.glob(os.path.join(D, "*.tar")))
    if len(sys.argv) > 2:
        tars = tars[: int(sys.argv[2])]
    t = time.time(); n = 0
    with Pool(int(os.environ.get("PROCS", 30))) as p:
        for i, (shard, k) in enumerate(p.imap_unordered(one, [(x, out_dir) for x in tars])):
            n += max(k, 0)
            if i % 25 == 0:
                print(f"{i+1}/{len(tars)} shards, {n} samples, {time.time()-t:.0f}s, {n/(time.time()-t):.0f}/s", flush=True)
    print("done", n, time.time() - t)

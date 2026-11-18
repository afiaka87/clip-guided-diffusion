r"""Stage B: one GPU pass over the stage-A survivors, streamed from the shard tars.

Per image: a SigLIP2-B/16-256 embedding (fp16, unit norm), the BMD YOLO11n watermark detector's top box confidence
(at 320px, on the same decoded batch), and strict edge-band features computed in the loader workers.
Results go to D:\laion2b-filter\stageB\part_XXXX.npz holding only completed shards, so a restart skips them.
"""
import glob, io, os, sys, tarfile, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

SRC = os.environ.get("CGD_SRC", r"D:\laion2b-en-aesthetic")  # The CGD_ overrides are how imagenet_filter reuses this.
OUT = os.environ.get("CGD_STAGEB", r"D:\laion2b-filter\stageB")
SEL = os.environ.get("CGD_STAGE_A", r"D:\laion2b-filter\selection\stage_a.parquet")
LUMA = np.array([0.299, 0.587, 0.114], np.float32)
EDGES = ("top", "bottom", "left", "right")
PADCOLS = [f"{e}_{k}" for e in EDGES for k in ("n", "ref", "step", "straight")]

def sid(shard, key):
    return int(shard) * 10**9 + int(key)

def edge(lines):
    """lines: (L, W) grey, ordered from the edge inwards. Returns the flat band's length, its grey level, the step
    into the next line, and how much of that next line differs from the band (1.0 = a straight edge, as padding has)."""
    std = lines.std(1); mean = lines.mean(1); ref = mean[0]; n = 0
    while n < len(std) and std[n] < 2.0 and abs(mean[n] - ref) < 3.0:
        n += 1
    if n == 0 or n == len(std):
        return n, ref, 0.0, 0.0
    nxt = lines[n]
    return n, ref, abs(mean[n] - ref) + std[n], float((np.abs(nxt - ref) > 10).mean())

def pad_features(a):
    g = a.astype(np.float32) @ LUMA
    out = []
    for lines in (g, g[::-1], g.T, g.T[::-1]):
        out += edge(lines)
    return np.array(out, np.float32)

class Shards(IterableDataset):
    def __init__(self, shards, wanted):
        self.shards, self.wanted = shards, wanted
    def __iter__(self):
        wi = get_worker_info()
        mine = self.shards if wi is None else self.shards[wi.id::wi.num_workers]
        for s in mine:
            want = self.wanted[s]
            with tarfile.open(os.path.join(SRC, f"{s:05d}.tar")) as t:
                for m in t:
                    if not m.name.endswith(".jpg"):
                        continue
                    i = sid(s, m.name[:-4])
                    if i not in want:
                        continue
                    a = np.asarray(Image.open(io.BytesIO(t.extractfile(m).read())).convert("RGB"))
                    if a.shape[:2] != (256, 256):
                        a = np.asarray(Image.fromarray(a).resize((256, 256), Image.BICUBIC))
                    yield i, torch.from_numpy(a.copy()).permute(2, 0, 1), torch.from_numpy(pad_features(a))

def main():
    import open_clip
    from ultralytics import YOLO
    os.makedirs(OUT, exist_ok=True)
    sel = pd.read_parquet(SEL, columns=["shard", "key"])
    sel["id"] = [sid(s, k) for s, k in zip(sel.shard, sel.key)]
    wanted = {s: set(g.id) for s, g in sel.groupby("shard")}
    done = set()
    for f in glob.glob(os.path.join(OUT, "part_*.npz")):
        done |= set(np.load(f)["shards"].tolist())
    todo = [s for s in sorted(wanted) if s not in done]
    expected = {s: len(wanted[s]) for s in todo}
    print(f"{len(sel)} survivors in {len(wanted)} shards; {len(done)} shards done, {len(todo)} to go", flush=True)
    if not todo:
        return
    model, _, _ = open_clip.create_model_and_transforms("hf-hub:timm/ViT-B-16-SigLIP2-256", device="cuda")
    model.eval()
    yolo = YOLO(glob.glob(os.path.expanduser(r"~\.cache\huggingface\hub\models--BitcrushedHeart--bmd_watermark_n\snapshots\*\*.pt"))[0])
    loader = DataLoader(Shards(todo, {s: wanted[s] for s in todo}), batch_size=256, num_workers=7, pin_memory=True, prefetch_factor=4)
    buf = {s: [] for s in todo}; got = {s: 0 for s in todo}; ready = []
    part = len(glob.glob(os.path.join(OUT, "part_*.npz"))); t0 = time.time(); n = 0

    def flush():
        nonlocal part, ready
        if not ready:
            return
        rows = [r for s in ready for r in buf.pop(s)]
        np.savez(os.path.join(OUT, f"part_{part:04d}.npz"), shards=np.array(ready),
                 ids=np.concatenate([r[0] for r in rows]), emb=np.concatenate([r[1] for r in rows]),
                 wm=np.concatenate([r[2] for r in rows]), pad=np.concatenate([r[3] for r in rows]))
        part += 1; ready = []

    with torch.no_grad():
        for ids, x, pad in loader:
            xg = x.cuda(non_blocking=True).float() / 255
            with torch.autocast("cuda", dtype=torch.float16):
                e = model.encode_image((xg - 0.5) / 0.5)
            e = F.normalize(e.float(), dim=-1).half().cpu().numpy()
            res = yolo.predict(F.interpolate(xg, size=(320, 320), mode="bilinear", align_corners=False), imgsz=320, conf=0.05, half=True, verbose=False)
            wm = np.array([float(r.boxes.conf.max()) if len(r.boxes) else 0.0 for r in res], np.float32)
            ids = ids.numpy(); shards = ids // 10**9
            for s in np.unique(shards):
                k = shards == s
                buf[int(s)].append((ids[k], e[k], wm[k], pad.numpy()[k]))
                got[int(s)] += int(k.sum())
                if got[int(s)] == expected[int(s)]:
                    ready.append(int(s))
            n += len(ids)
            if len(ready) >= 20:
                flush()
                el = time.time() - t0
                print(f"{n} images, {n/el:.0f} img/s, {part} parts, eta {(sum(expected.values()) - n) / (n / el) / 60:.0f} min", flush=True)
    flush()
    missing = [s for s in todo if got[s] != expected[s]]
    print("done", n, "images in", f"{(time.time()-t0)/60:.1f} min;", "incomplete shards:", missing, flush=True)

if __name__ == "__main__":
    main()

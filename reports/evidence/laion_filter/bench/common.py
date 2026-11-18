import os, time, torch, numpy as np, pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
SAMPLE = r"D:\laion2b-filter\bench_sample"
OUT = os.path.join(SAMPLE, "out"); os.makedirs(OUT, exist_ok=True)

def sample():
    return pd.read_parquet(os.path.join(SAMPLE, "sample.parquet"))

class Images(Dataset):
    def __init__(self, paths, tf): self.p, self.tf = list(paths), tf
    def __len__(self): return len(self.p)
    def __getitem__(self, i): return self.tf(Image.open(self.p[i]).convert("RGB"))

def loader(paths, tf, bs, workers=8):
    return DataLoader(Images(paths, tf), batch_size=bs, num_workers=workers, pin_memory=True, prefetch_factor=4)

class Timer:
    """Images per second after the first `skip` batches, with the GPU synchronised."""
    def __init__(self, skip=2): self.skip, self.i, self.n, self.t = skip, 0, 0, None
    def step(self, k):
        self.i += 1
        if self.i == self.skip:
            torch.cuda.synchronize(); self.t = time.time()
        elif self.i > self.skip:
            self.n += k
    def rate(self):
        torch.cuda.synchronize(); return self.n / (time.time() - self.t)

def gpu_only(fn, x, iters=10):
    """Throughput with the input already on the GPU: the model's own ceiling."""
    with torch.no_grad():
        for _ in range(2): fn(x)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(iters): fn(x)
        torch.cuda.synchronize()
    return iters * x.shape[0] / (time.time() - t)

def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True)
    with open(os.path.join(OUT, "bench_log.txt"), "a") as f: f.write(s + "\n")

import torchvision.transforms as T

def split(tf):
    """Split a CLIP preprocessing Compose into a worker part that ends in a uint8 tensor, and the normalisation,
    done on the GPU. Shipping float32 batches between processes is what limits Windows DataLoaders."""
    cpu, mean, std = [], (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    for t in tf.transforms:
        n = type(t).__name__
        if n in ("Resize", "CenterCrop"):
            cpu.append(t)
        elif n == "Normalize":
            mean, std = t.mean, t.std
    cpu.append(T.PILToTensor())
    m = torch.tensor(mean, device="cuda").view(1, 3, 1, 1); s = torch.tensor(std, device="cuda").view(1, 3, 1, 1)
    return T.Compose(cpu), (lambda x: (x.cuda(non_blocking=True).float() / 255 - m) / s)

import io, tarfile
from torch.utils.data import IterableDataset, get_worker_info

def sid(shard, key):
    return int(shard) * 10**9 + int(key)

class TarImages(IterableDataset):
    """Stream the sample's images straight from the shard tars, one worker per shard, as the real pipeline would.
    Yields (id, image tensor); `ids` restricts it to a subset."""
    def __init__(self, shards, tf, ids=None):
        self.shards, self.tf, self.ids = sorted(set(int(s) for s in shards)), tf, (set(ids) if ids is not None else None)
    def __iter__(self):
        wi = get_worker_info()
        mine = self.shards if wi is None else self.shards[wi.id::wi.num_workers]
        for s in mine:
            with tarfile.open(rf"D:\laion2b-en-aesthetic\{s:05d}.tar") as t:
                for m in t:
                    if not m.name.endswith(".jpg"):
                        continue
                    i = sid(s, m.name[:-4])
                    if self.ids is not None and i not in self.ids:
                        continue
                    yield i, self.tf(Image.open(io.BytesIO(t.extractfile(m).read())).convert("RGB"))

def tar_loader(df, tf, bs, ids=None, workers=7):
    return DataLoader(TarImages(df.shard.unique(), tf, ids), batch_size=bs, num_workers=workers, pin_memory=True, prefetch_factor=4)

def aligned(df, ids, values):
    """Put per-id results back in the sample's row order (NaN for rows not scored)."""
    pos = {i: k for k, i in enumerate(sid(s, k) for s, k in zip(df.shard, df.key))}
    out = np.full((len(df),) + values.shape[1:], np.nan, dtype=np.float32)
    out[[pos[i] for i in ids]] = values
    return out

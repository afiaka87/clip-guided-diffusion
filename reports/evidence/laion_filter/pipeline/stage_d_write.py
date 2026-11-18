r"""Stage D: write a selected tier as webdataset shards, optionally captioning each image with Florence-2-base.

    python stage_d_write.py best   # D:\laion2b-filter\selection\best1m.parquet -> F:\laion-aesthetic-best1m-256, captioned
    python stage_d_write.py clean  # D:\laion2b-filter\selection\clean.parquet  -> F:\laion-aesthetic-clean-256, alt-text only

Each sample keeps its original JPEG bytes (no re-encode) and its original JSON, plus a "cgd_filter" object with every
score the filter computed. In the best tier, .txt is the generated caption and the JSON adds "generated_caption"; the
alt-text stays in the JSON's "caption". In the clean tier, .txt is the alt-text, as in the source.
Shards hold 5,000 samples, are written to a .tmp name and renamed when complete, and each has an index parquet
beside it, so a restart skips everything already written.
"""
import glob
import io
import json
import os
import sys
import tarfile
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

SRC = os.environ.get("CGD_SRC", r"D:\laion2b-en-aesthetic")  # As in stage B.
SEL = r"D:\laion2b-filter\selection"
TIERS = {
    "best": (os.path.join(SEL, "best1m.parquet"), r"F:\laion-aesthetic-best1m-256", True),
    "clean": (os.path.join(SEL, "clean.parquet"), r"F:\laion-aesthetic-clean-256", False),
}
PER_SHARD = 5000
CAPTION_SIZE = 512
CAPTION_TASK = "<DETAILED_CAPTION>"
CAPTION_MODEL = "florence-community/Florence-2-base"
SCORE_COLS = ["aes_v25_probe", "aes_v2_probe", "quality", "wm_yolo", "zs_text", "zs_collage", "zs_screenshot",
              "zs_logo", "zs_nsfw", "illus", "leaf", "parent", "hf3", "hf4", "lap_var", "ring_white", "white_frac",
              "colorfulness", "lum_mean"]


def sid(s, k):
    return int(s) * 10**9 + int(k)


class Source(IterableDataset):
    """Stream the wanted samples from the source tars, one worker per source shard."""

    def __init__(self, wanted, caption):
        self.wanted, self.shards, self.caption = wanted, sorted(wanted), caption

    def __iter__(self):
        wi = get_worker_info()
        mine = self.shards if wi is None else self.shards[wi.id::wi.num_workers]
        for s in mine:
            want, have = self.wanted[s], {}
            with tarfile.open(os.path.join(SRC, f"{s:05d}.tar")) as t:
                for m in t:
                    stem, ext = m.name.rsplit(".", 1)
                    if ext not in ("jpg", "json") or sid(s, stem) not in want:
                        continue
                    have.setdefault(stem, {})[ext] = t.extractfile(m).read()
                    if len(have[stem]) == 2:
                        d = have.pop(stem)
                        img = None
                        if self.caption:
                            im = Image.open(io.BytesIO(d["jpg"])).convert("RGB").resize((CAPTION_SIZE, CAPTION_SIZE), Image.BICUBIC)
                            img = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)
                        yield sid(s, stem), d["jpg"], d["json"], img


def collate(batch):
    ids, jpgs, jsons, imgs = zip(*batch)
    return list(ids), list(jpgs), list(jsons), (torch.stack(imgs) if imgs[0] is not None else None)


class Captioner:
    def __init__(self):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(CAPTION_MODEL)
        self.proc.num_image_tokens = (CAPTION_SIZE // 32) ** 2 + 1  # 577 at 768; 257 at 512
        self.model = AutoModelForImageTextToText.from_pretrained(CAPTION_MODEL, dtype=torch.float16).cuda().eval()
        dummy = Image.new("RGB", (CAPTION_SIZE, CAPTION_SIZE))
        x = self.proc(text=[CAPTION_TASK], images=[dummy], size={"height": CAPTION_SIZE, "width": CAPTION_SIZE}, return_tensors="pt")
        self.ids, self.mask = x["input_ids"].cuda(), x["attention_mask"].cuda()
        ip = self.proc.image_processor
        self.mean = torch.tensor(ip.image_mean, device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor(ip.image_std, device="cuda").view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, imgs):
        px = ((imgs.cuda(non_blocking=True).float() / 255 - self.mean) / self.std).half()
        b = len(px)
        g = self.model.generate(input_ids=self.ids.repeat(b, 1), attention_mask=self.mask.repeat(b, 1),
                                pixel_values=px, max_new_tokens=96, num_beams=1, do_sample=False)
        return [c.strip() for c in self.proc.batch_decode(g, skip_special_tokens=True)]


def jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else round(float(v), 5)
    return v


def main(tier):
    sel_path, out_dir, caption = TIERS[tier]
    sel_path = os.environ.get("SEL_PATH", sel_path); out_dir = os.environ.get("OUT_DIR", out_dir)
    os.makedirs(out_dir, exist_ok=True)
    sel = pd.read_parquet(sel_path)
    sel["id"] = [sid(s, k) for s, k in zip(sel.shard, sel.key)]
    score_arrays = {c: sel[c].to_numpy() for c in SCORE_COLS if c in sel.columns}  # per column, so ints stay ints
    row_of = dict(zip(sel.id, range(len(sel))))
    for f in glob.glob(os.path.join(out_dir, "*.tmp")):
        os.remove(f)  # a shard cut off by a stop; it is rewritten
    done = set()
    idx_files = sorted(glob.glob(os.path.join(out_dir, "*.parquet")))
    for f in idx_files:
        done |= set(pd.read_parquet(f, columns=["source_id"]).source_id)
    next_shard = len(idx_files)
    todo = sel[~sel.id.isin(done)]
    wanted = {int(s): set(g.id) for s, g in todo.groupby("shard")}
    print(f"{tier}: {len(sel)} selected, {len(done)} already written in {next_shard} shards, {len(todo)} to go", flush=True)
    if not len(todo):
        return
    cap = Captioner() if caption else None
    loader = DataLoader(Source(wanted, caption), batch_size=64 if caption else 512, num_workers=6, collate_fn=collate,
                        prefetch_factor=4, pin_memory=caption)
    pending, t0, n = [], time.time(), 0

    def write(rows):
        nonlocal next_shard
        name = f"{next_shard:05d}"
        tmp = os.path.join(out_dir, name + ".tar.tmp")
        index = []
        with tarfile.open(tmp, "w") as t:
            for j, (i, jpg, js, text) in enumerate(rows):
                key = f"{next_shard * PER_SHARD + j:09d}"
                meta = json.loads(js)
                meta["source_shard"], meta["source_key"] = int(i // 10**9), meta.get("key")
                meta["key"] = key
                meta["cgd_filter"] = {c: jsonable(v[row_of[i]]) for c, v in score_arrays.items()}
                if caption:
                    meta["generated_caption"], meta["caption_model"] = text, f"{CAPTION_MODEL} {CAPTION_TASK} @{CAPTION_SIZE}px"
                for ext, data in (("jpg", jpg), ("json", json.dumps(meta, ensure_ascii=False).encode()),
                                  ("txt", (text if caption else (meta.get("caption") or "")).encode())):
                    ti = tarfile.TarInfo(f"{key}.{ext}"); ti.size = len(data); ti.mtime = int(time.time())
                    t.addfile(ti, io.BytesIO(data))
                index.append({"key": key, "source_id": i, "source_shard": meta["source_shard"],
                              "source_key": meta["source_key"], "txt": text if caption else meta.get("caption")})
        os.replace(tmp, os.path.join(out_dir, name + ".tar"))
        pd.DataFrame(index).to_parquet(os.path.join(out_dir, name + ".parquet"))
        next_shard += 1

    for ids, jpgs, jsons, imgs in loader:
        texts = cap(imgs) if caption else [None] * len(ids)
        pending += list(zip(ids, jpgs, jsons, texts))
        n += len(ids)
        while len(pending) >= PER_SHARD:
            write(pending[:PER_SHARD]); pending = pending[PER_SHARD:]
            el = time.time() - t0
            print(f"{tier}: shard {next_shard - 1:05d} written, {n} this run, {n / el:.1f} img/s, "
                  f"eta {(len(todo) - n) / (n / el) / 3600:.1f} h", flush=True)
    if pending:
        write(pending)
    print(f"{tier}: done, {n} written this run in {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])

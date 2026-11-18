r"""Stage C: from the stage-A table and the stage-B embeddings to the clean tier and the best million.

1. Drop straight-edged padding/letterbox/frames, YOLO watermarks, and zero-shot text, collage, screenshot/chart/map,
   logo/icon and nudity (thresholds set by eye on calibration sheets of the 82k sample, in calibration.json).
2. Score aesthetics with ridge probes on the SigLIP2 embedding, trained on the sample's real judge outputs:
   LAION aesthetic v2 (L/14 MLP, 82k labels) and Aesthetic Predictor v2.5 (so400m, 20k labels).
3. k-means on the GPU (20,000 leaves, then 1,000 parents over the leaf centroids); SemDeDup inside each leaf:
   cosine > 0.95 is a near-duplicate, and the best-scored copy stays.
4. Clean tier = everything left. Best tier = 1M by water-filling the parents, then the leaves, highest quality
   first, above a quality floor, with illustration-like content capped at 20%.
Writes D:\laion2b-filter\selection\{scored,clean,best1m}.parquet.
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SEL = r"D:\laion2b-filter\selection"
STAGEB = os.environ.get("CGD_STAGEB", r"D:\laion2b-filter\stageB")
SAMPLE = r"D:\laion2b-filter\bench_sample\out"
DEV = os.environ.get("STAGEC_DEVICE", "cuda")
OUTSEL = os.environ.get("STAGEC_OUT", SEL)
CFG = {
    "pad_min": 3, "pad_step": 12, "pad_straight": 0.6,
    "wm_yolo": 0.35,
    "zs": {"text": 9.0, "collage": 9.0, "screenshot": 9.0, "logo": 9.0, "nsfw": 9.0},  # overridden by calibration.json
    "dedup_cos": 0.95, "leaves": 20000, "parents": 1000, "best_n": 1_000_000, "floor_quantile": 0.25,
    "illus_cap": 0.20, "seed": 0,
}
JUNK = {
    "text": ["an image with a lot of text", "a poster with text", "a quote in large letters", "a flyer",
             "a page of text", "an infographic", "a book cover with a title"],
    "collage": ["a collage of several photos", "a grid of multiple images", "a before and after comparison",
                "a photo collage with many pictures"],
    "screenshot": ["a screenshot of a website", "a screenshot of a video game", "a user interface",
                   "a chart or graph", "a map", "a diagram"],
    "logo": ["a logo", "an icon", "a badge or emblem design", "a t-shirt graphic design on white"],
    "nsfw": ["a nude photo", "an erotic photo", "a person in underwear"],
}
NEUTRAL = ["a photo", "a picture", "an image", "a photograph of a scene", "a painting"]
ILLUS = ["a painting", "an oil painting", "a watercolor painting", "a drawing", "a cartoon illustration", "clipart",
         "a vector illustration", "an anime drawing", "a comic book illustration", "a 3d render"]
PHOTO = ["a photo", "a photograph", "a snapshot taken with a camera"]
EDGES = ("top", "bottom", "left", "right")


def sid(s, k):
    return int(s) * 10**9 + int(k)


def kmeans(x, k, iters=12, seed=0, bs=16384):
    """Spherical k-means on the GPU. x: (N, D) float16 unit vectors on the GPU."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    c = x[torch.randperm(len(x), generator=g)[:k].to(x.device)].float()
    a = torch.empty(len(x), dtype=torch.long, device=x.device)
    for _ in range(iters):
        s = torch.zeros_like(c)
        for i in range(0, len(x), bs):  # in batches: a float32 copy of all of x does not fit beside it
            xb = x[i:i + bs].float()
            a[i:i + bs] = (xb @ c.T).argmax(1)
            s.index_add_(0, a[i:i + bs], xb)
        cnt = torch.bincount(a, minlength=k)
        c = F.normalize(torch.where((cnt == 0)[:, None], c, s / cnt.clamp(min=1)[:, None]), dim=-1)
    for i in range(0, len(x), bs):
        a[i:i + bs] = (x[i:i + bs].float() @ c.T).argmax(1)
    return a, c


def near_duplicates(e, thr, cells=2**26):
    """e: (n, D) unit vectors, best first. True where an earlier (better) row has cosine > thr.
    In blocks of rows, so a large leaf never needs its whole n x n similarity matrix at once."""
    n = len(e)
    dup = torch.zeros(n, dtype=torch.bool, device=e.device)
    b = max(1, cells // n)
    col = torch.arange(n, device=e.device)
    for r0 in range(0, n, b):
        r1 = min(n, r0 + b)
        later = col[None, :] > torch.arange(r0, r1, device=e.device)[:, None]
        dup |= ((e[r0:r1] @ e.T > thr) & later).any(0)
    return dup


def waterfill(sizes, total):
    """Per-group quotas: the largest cap c with sum(min(size, c)) <= total, the remainder to the largest groups."""
    sizes = np.asarray(sizes)
    if sizes.sum() <= total:
        return sizes.copy()
    lo, hi = 0, int(sizes.max())
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if np.minimum(sizes, mid).sum() <= total:
            lo = mid
        else:
            hi = mid - 1
    q = np.minimum(sizes, lo)
    rest = int(total - q.sum())
    if rest > 0:
        room = np.where(sizes > q)[0]
        room = room[np.argsort(-sizes[room])][:rest]
        q[room] += 1
    return q


def main():
    t0 = time.time()

    def log(*a):
        print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)

    cfg = dict(CFG)
    cal = os.path.join(SEL, "calibration.json")
    if os.path.exists(cal):
        cfg.update(json.load(open(cal)))
    cfg.update(json.loads(os.environ.get("STAGEC_CFG", "{}")))
    log("config", cfg)
    a = pd.read_parquet(os.environ.get("CGD_STAGE_A", os.path.join(SEL, "stage_a.parquet")))
    a["id"] = [sid(s, k) for s, k in zip(a.shard, a.key)]
    ids, emb, wm, pad = [], [], [], []
    for f in sorted(glob.glob(os.path.join(STAGEB, "part_*.npz"))):
        d = np.load(f)
        ids.append(d["ids"]); emb.append(d["emb"]); wm.append(d["wm"]); pad.append(d["pad"])
    ids = np.concatenate(ids); emb = np.concatenate(emb); wm = np.concatenate(wm); pad = np.concatenate(pad)
    pos = pd.Series(np.arange(len(ids)), index=ids)
    a = a[a.id.isin(pos.index)].reset_index(drop=True)
    order = pos.loc[a.id].values
    emb = emb[order]; wm = wm[order]; pad = pad[order]
    log(f"{len(a)} samples with embeddings")
    from stage_b_embed import PADCOLS
    for j, c in enumerate(PADCOLS):
        a[c] = pad[:, j]
    a["wm_yolo"] = wm

    E = torch.from_numpy(emb).to(DEV)
    import open_clip
    m, _, _ = open_clip.create_model_and_transforms("hf-hub:timm/ViT-B-16-SigLIP2-256")
    m.eval()
    tok = open_clip.get_tokenizer("hf-hub:timm/ViT-B-16-SigLIP2-256")
    with torch.no_grad():
        def enc(ps):
            return F.normalize(m.encode_text(tok(ps)).float(), dim=-1).to(DEV)

        def sims(ps):
            t = enc(ps)
            return torch.cat([(E[i:i + 262144].float() @ t.T).max(1).values for i in range(0, len(E), 262144)])

        neu = sims(NEUTRAL)
        for k, ps in JUNK.items():
            a[f"zs_{k}"] = (sims(ps) - neu).cpu().numpy()
        a["illus"] = (sims(ILLUS) - sims(PHOTO)).cpu().numpy()
    del m

    # 1. Junk.
    n = a[[f"{e}_n" for e in EDGES]].values
    st = a[[f"{e}_step" for e in EDGES]].values
    sr = a[[f"{e}_straight" for e in EDGES]].values
    straight = ((n >= cfg["pad_min"]) & (n < 256) & (st > cfg["pad_step"]) & (sr >= cfg["pad_straight"])).any(1)
    reason = np.array([""] * len(a), dtype=object)
    tests = [("padding_or_frame", straight), ("watermark_yolo", a.wm_yolo.values >= cfg["wm_yolo"])]
    tests += [(f"zeroshot_{k}", a[f"zs_{k}"].values >= v) for k, v in cfg["zs"].items()]
    for name, t in tests:
        reason[np.asarray(t, bool) & (reason == "")] = name
    log("junk drops:", pd.Series(reason[reason != ""]).value_counts().to_dict())

    # 2. Aesthetic probes trained on the sample's judges.
    from sklearn.linear_model import Ridge
    s = pd.read_parquet(os.path.join(SAMPLE, "sample_scored.parquet"))
    se = np.load(os.path.join(SAMPLE, "emb_siglip2_b16_256.npy")).astype(np.float32)
    v2 = np.load(os.path.join(SAMPLE, "aes_v2.npy"))
    v25 = np.load(os.path.join(SAMPLE, "aes_v25.npy"))
    clean_s = ((s.sha_n < 3) & ~s.changed).values
    p2 = Ridge(alpha=10).fit(se[clean_s], v2[clean_s])
    h = clean_s & ~np.isnan(v25)
    p25 = Ridge(alpha=10).fit(se[h], v25[h])
    for name, p in (("aes_v2_probe", p2), ("aes_v25_probe", p25)):
        a[name] = np.concatenate([p.predict(emb[i:i + 262144].astype(np.float32)) for i in range(0, len(emb), 262144)])

    def z(v):
        v = np.asarray(v, np.float64)
        return (v - v.mean()) / v.std()

    ss = np.minimum(a.original_width, a.original_height)
    a["quality"] = (0.45 * z(a.aes_v25_probe) + 0.30 * z(a.aes_v2_probe) + 0.10 * z(np.log(a.hf4 + 1e-4))
                    + 0.15 * z(np.log(np.minimum(ss, 768))) - 1.0 * (a.wm_yolo.values >= 0.2))
    log("quality computed")

    # 3. Clusters and near duplicates.
    leaf, cent = kmeans(E, cfg["leaves"], iters=cfg.get("iters", 12), seed=cfg["seed"])
    par, _ = kmeans(cent.half() if DEV == "cuda" else cent, cfg["parents"], iters=cfg.get("iters", 12), seed=cfg["seed"] + 1)
    a["leaf"] = leaf.cpu().numpy()
    a["parent"] = par.cpu().numpy()[a["leaf"].values]
    log("clustered")
    dup = np.zeros(len(a), bool)
    q = a.quality.values
    L = a.leaf.values
    order_l = np.argsort(L, kind="stable")
    bounds = np.searchsorted(L[order_l], np.arange(cfg["leaves"] + 1))
    with torch.no_grad():
        for c in range(cfg["leaves"]):
            idx = order_l[bounds[c]:bounds[c + 1]]
            if len(idx) < 2:
                continue
            idx = idx[np.argsort(-q[idx])]
            e = E[torch.from_numpy(idx).to(DEV)].float()
            dup[idx[near_duplicates(e, cfg["dedup_cos"]).cpu().numpy()]] = True
    t = dup & (reason == "")
    reason[t] = "near_duplicate"
    log("near duplicates:", int(t.sum()))
    a["stage_c_reason"] = reason
    a.to_parquet(os.path.join(OUTSEL, "scored.parquet"))
    clean = a[reason == ""].copy()
    log("clean tier:", len(clean))

    clean.sort_values("id").to_parquet(os.path.join(OUTSEL, "clean.parquet"))
    if not cfg["best_n"]:  # imagenet_filter keeps the whole clean tier
        json.dump(cfg, open(os.path.join(OUTSEL, "stage_c_config_used.json"), "w"), indent=1)
        return

    # 4. The best million.
    floor = clean.quality.quantile(cfg["floor_quantile"])
    pool = clean[clean.quality >= floor].copy()
    pool["is_illus"] = pool.illus > 0
    sizes = pool.groupby("parent").size()
    quota = pd.Series(waterfill(sizes.values, cfg["best_n"]), index=sizes.index)
    chosen = []
    for p, g in pool.groupby("parent"):
        ls = g.groupby("leaf").size()
        lq = pd.Series(waterfill(ls.values, int(quota[p])), index=ls.index)
        for leaf_id, gl in g.groupby("leaf"):
            chosen.append(gl.nlargest(int(lq[leaf_id]), "quality").index.values)
    best = pool.loc[np.concatenate(chosen)].copy()
    over = int(best.is_illus.sum() - cfg["illus_cap"] * len(best))
    if over > 0:
        drop = best[best.is_illus].nsmallest(over, "quality").index
        rest = pool.drop(best.index)
        add = rest[~rest.is_illus].nlargest(over, "quality").index
        best = pd.concat([best.drop(drop), pool.loc[add]])
    best = best.sort_values("id")
    log(f"best tier: {len(best)}, illustration share {best.is_illus.mean():.3f}, parents used "
        f"{best.parent.nunique()}, quality floor {floor:.3f}")
    clean.sort_values("id").to_parquet(os.path.join(OUTSEL, "clean.parquet"))
    best.to_parquet(os.path.join(OUTSEL, "best1m.parquet"))
    json.dump(cfg, open(os.path.join(OUTSEL, "stage_c_config_used.json"), "w"), indent=1)
    log("written")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()

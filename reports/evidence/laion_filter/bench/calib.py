"""Calibration sheets on the 82k sample: strict padding with straightness, YOLO watermark bins, zero-shot junk scores."""
import os, sys, numpy as np, pandas as pd, torch, open_clip
from PIL import Image
from multiprocessing import Pool
sys.path.insert(0, sys.argv[1])
from stage_b_embed import pad_features, PADCOLS
from sheet_keys import sheet
OUT = r"D:\laion2b-filter\bench_sample\out"; SH = r"D:\laion2b-filter\analysis\sheets"
df = pd.read_parquet(os.path.join(OUT, "sample_scored.parquet"))
def pf(p): return pad_features(np.asarray(Image.open(p).convert("RGB")))
if __name__ == "__main__":
    if "top_n" not in df:
        with Pool(12) as pool: P = np.stack(pool.map(pf, df.path.tolist(), chunksize=256))
        for j, c in enumerate(PADCOLS): df[c] = P[:, j]
        df.to_parquet(os.path.join(OUT, "sample_scored.parquet"))
    n = np.stack([df[f"{e}_n"] for e in ("top","bottom","left","right")], 1); st = np.stack([df[f"{e}_step"] for e in ("top","bottom","left","right")], 1); sr = np.stack([df[f"{e}_straight"] for e in ("top","bottom","left","right")], 1)
    band = (n >= 3) & (n < 256) & (st > 12)
    straight = (band & (sr >= .6)).any(1); curvy = (band & (sr < .6)).any(1) & ~straight
    ss = np.minimum(df.original_width, df.original_height); arr = np.maximum(df.original_width, df.original_height) / ss
    surv = (df.lum_std >= 4) & (df.sha_n < 3) & ~df.changed & (ss >= 256) & (arr <= 1.8) & (df.ring_white < .5) & (df.chroma_mean >= 4) & (df.n_colors >= 256) & (df.pwatermark < .5) & (df.punsafe < .1)
    print("survivors", surv.sum(), " straight band:", (straight & surv).mean() / surv.mean(), " curvy band only:", (curvy & surv).mean() / surv.mean())
    x = df[straight & surv].sample(32, random_state=0); sheet(list(zip(x.shard, x.key)), os.path.join(SH, "c_straight_pad.png"), labels=[f"t{a:.0f} b{b:.0f} l{c:.0f} r{d:.0f}" for a,b,c,d in x[["top_n","bottom_n","left_n","right_n"]].values])
    x = df[curvy & surv].sample(32, random_state=0); sheet(list(zip(x.shard, x.key)), os.path.join(SH, "c_curvy_band.png"), labels=[f"t{a:.0f} b{b:.0f} l{c:.0f} r{d:.0f}" for a,b,c,d in x[["top_n","bottom_n","left_n","right_n"]].values])
    w = np.load(os.path.join(OUT, "wm_yolo_320.npy")); df["wm320"] = w
    for lo, hi in ((.2, .35), (.35, .5), (.5, 1.01)):
        m = (df.wm320 >= lo) & (df.wm320 < hi)
        x = df[m].sample(min(24, m.sum()), random_state=0); print(f"wm320 in [{lo},{hi}): {m.sum()} of {(~np.isnan(w)).sum()}")
        sheet(list(zip(x.shard, x.key)), os.path.join(SH, f"c_wm_{lo}.png"), labels=[f"{v:.2f}" for v in x.wm320])
    # zero-shot junk scores
    sig = torch.from_numpy(np.load(os.path.join(OUT, "emb_siglip2_b16_256.npy")).astype(np.float32))
    m, _, _ = open_clip.create_model_and_transforms("hf-hub:timm/ViT-B-16-SigLIP2-256"); m.eval(); tok = open_clip.get_tokenizer("hf-hub:timm/ViT-B-16-SigLIP2-256")
    JUNK = {
      "text": ["an image with a lot of text", "a poster with text", "a quote in large letters", "a flyer", "a page of text", "an infographic", "a book cover with a title"],
      "collage": ["a collage of several photos", "a grid of multiple images", "a before and after comparison", "a photo collage with many pictures"],
      "screenshot": ["a screenshot of a website", "a screenshot of a video game", "a user interface", "a chart or graph", "a map", "a diagram"],
      "logo": ["a logo", "an icon", "a badge or emblem design", "a t-shirt graphic design on white"],
      "watermark": ["a stock photo with a watermark across it", "an image with a watermark"],
      "nsfw": ["a nude photo", "an erotic photo", "a person in underwear"],
    }
    NEUTRAL = ["a photo", "a picture", "an image", "a photograph of a scene", "a painting"]
    with torch.no_grad():
        enc = lambda ps: torch.nn.functional.normalize(m.encode_text(tok(ps)).float(), dim=-1)
        neu = (sig @ enc(NEUTRAL).T).max(1).values
        for k, ps in JUNK.items():
            s = (sig @ enc(ps).T).max(1).values - neu; df[f"zs_{k}"] = s.numpy()
    for k in JUNK:
        c = f"zs_{k}"; q = df.loc[surv, c].quantile([.9, .97, .99, .995]).round(3).tolist(); print(c, "survivor quantiles 90/97/99/99.5:", q)
        for qq in (.97, .99):
            t = df.loc[surv, c].quantile(qq); band_ = surv & (df[c] >= t) & (df[c] < df.loc[surv, c].quantile(min(qq + .02, .999)))
            x = df[band_].sample(min(24, band_.sum()), random_state=0)
            sheet(list(zip(x.shard, x.key)), os.path.join(SH, f"c_zs_{k}_{qq}.png"), labels=[f"{v:.3f}" for v in x[c]])
    df.to_parquet(os.path.join(OUT, "sample_scored.parquet"))

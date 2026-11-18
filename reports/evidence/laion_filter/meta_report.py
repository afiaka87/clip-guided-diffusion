import sys, numpy as np, pandas as pd
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
from load import load
m = load()
m["has_jpg"] = m.jpg_bytes.notna(); m["error"] = m.tar_error
print("rows", len(m), "has_jpg", m.has_jpg.sum(), "tar errors", m.error.notna().sum(), m.error.dropna().unique()[:3])
print("status", m.status.value_counts().to_dict())
print("width/height", m.groupby(["width","height"]).size().sort_values().tail(5).to_dict())
q = [0, .001, .01, .05, .1, .25, .5, .75, .9, .95, .99, .999, 1]
for c in ["aesthetic","similarity","punsafe","pwatermark","original_width","original_height","jpg_bytes"]:
    print(c.ljust(16), " ".join(f"{v:.4g}" for v in m[c].quantile(q).values))
mn = np.minimum(m.original_width, m.original_height); mx = np.maximum(m.original_width, m.original_height)
ar = mx / mn
print("short side quantiles", " ".join(f"{v:.0f}" for v in mn.quantile(q)))
for t in [64, 128, 192, 224, 256, 320, 384, 512, 1024]:
    print(f"  short side < {t}: {(mn < t).mean()*100:.2f}%  ({(mn<t).sum()})")
print("aspect (long/short) quantiles", " ".join(f"{v:.3f}" for v in ar.quantile(q)))
for t in [1.0001, 1.2, 1.34, 1.5, 1.78, 2, 3]:
    print(f"  aspect > {t}: {(ar > t).mean()*100:.2f}%  -> crop keeps < {100/t:.0f}% of the long side")
print("portrait share", (m.original_height > m.original_width).mean(), "square", (m.original_height == m.original_width).mean())
print("WIDTH==original_width", (m.WIDTH == m.original_width).mean(), "HEIGHT==original_height", (m.HEIGHT == m.original_height).mean())
print("caption==txt", (m.caption.fillna("") == m.txt).mean())
L = m.caption.fillna("").str.len(); W = m.caption.fillna("").str.split().str.len()
print("caption chars", " ".join(f"{v:.0f}" for v in L.quantile(q)))
print("caption words", " ".join(f"{v:.0f}" for v in W.quantile(q)))
cap = m.caption.fillna("")
for name, pat in [("url/www", r"https?://|www\.|\.com\b"), ("file-ish", r"\.(jpg|jpeg|png|gif|webp)\b|^(img|dsc|image)[ _-]?\d+"),
                  ("shutterstock/stock", r"(?i)shutterstock|stock photo|stock image|istock|dreamstime|alamy|123rf|depositphotos|getty"),
                  ("royalty free", r"(?i)royalty[- ]free|vector|clipart|clip art|illustration"),
                  ("printable/pattern", r"(?i)printable|pattern|template|coloring page|svg"),
                  ("for sale/$", r"(?i)\$\d|for sale|buy |price|free shipping|discount"),
                  ("non-ascii", r"[^\x00-\x7f]"), ("<= 3 words", None)]:
    s = (W <= 3) if pat is None else cap.str.contains(pat, regex=True)
    print(f"  {name:20s} {s.mean()*100:.2f}%")
print("exact dup sha256", m.sha256.duplicated().sum(), "dup url", m.url.duplicated().sum(), "dup caption", cap.duplicated().sum())
print("top captions", cap.value_counts().head(15).to_dict())
dom = m.url.fillna("").str.extract(r"https?://([^/]+)/")[0].str.replace(r"^www\.", "", regex=True)
print("top domains", dom.value_counts().head(40).to_dict())
print("n domains", dom.nunique())
ex = m.exif.fillna("{}")
print("exif nonempty", (ex != "{}").mean())
print("aesthetic by shard decile", m.groupby(m.shard // 100).aesthetic.mean().round(3).to_dict())
print("aesthetic >= thresholds", {t: int((m.aesthetic >= t).sum()) for t in [7, 7.25, 7.5, 7.75, 8, 8.5]})

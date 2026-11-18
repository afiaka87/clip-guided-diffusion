"""Content composition by SigLIP2-B/16-256 zero-shot over the 80k sample, for all images and for classical-filter survivors."""
import os, numpy as np, pandas as pd, torch, open_clip, collections
from common import sample, OUT
from zs_b32 import CATS
df = pd.read_parquet(os.path.join(OUT, "sample_scored.parquet")) if os.path.exists(os.path.join(OUT, "sample_scored.parquet")) else sample()
sig = torch.from_numpy(np.load(os.path.join(OUT, "emb_siglip2_b16_256.npy")).astype(np.float32))
m, _, _ = open_clip.create_model_and_transforms("hf-hub:timm/ViT-B-16-SigLIP2-256"); m.eval()
tok = open_clip.get_tokenizer("hf-hub:timm/ViT-B-16-SigLIP2-256")
names = list(CATS); prompts = [p for k in names for p in CATS[k]]; owner = np.array([i for i, k in enumerate(names) for _ in CATS[k]])
with torch.no_grad():
    t = torch.nn.functional.normalize(m.encode_text(tok(prompts)).float(), dim=-1)
sim = (sig @ t.T).numpy()
cat = np.full((len(sig), len(names)), -9.0)
for j, o in enumerate(owner): cat[:, o] = np.maximum(cat[:, o], sim[:, j])
df["zs_cat"] = [names[i] for i in cat.argmax(1)]
ss = np.minimum(df.original_width, df.original_height); arr = np.maximum(df.original_width, df.original_height) / ss
surv = (df.lum_std >= 4) & (df.sha_n < 3) & ~df.changed & (ss >= 256) & (arr <= 1.8) & (df.ring_white < .5) & (df.chroma_mean >= 4) & (df.n_colors >= 256) & (df.pwatermark < .5) & (df.punsafe < .1)
a = df.zs_cat.value_counts(normalize=True); b = df.zs_cat[surv].value_counts(normalize=True)
print(f"{'category':32s} {'all':>6s} {'survivors':>9s}   (n={len(df)}, survivors={surv.sum()})")
for k in a.index: print(f"{k:32s} {100*a[k]:5.1f}% {100*b.get(k, 0):8.1f}%")
df.to_parquet(os.path.join(OUT, "sample_scored.parquet"))

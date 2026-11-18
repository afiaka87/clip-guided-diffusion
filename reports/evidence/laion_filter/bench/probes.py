"""What the GPU outputs say: score drift since LAION, agreement between aesthetic judges, how well cheap probes on
SigLIP2 embeddings reproduce each judge, and SigLIP2 zero-shot composition."""
import os, glob, numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.metrics import roc_auc_score
from common import sample, OUT
df = sample(); N = len(df)
L = lambda n: np.load(os.path.join(OUT, n))
sig = L("emb_siglip2_b16_256.npy").astype(np.float32); txt = L("txt_siglip2_b16_256.npy").astype(np.float32)
v1, v2 = L("aes_v1.npy"), L("aes_v2.npy")
df["v1_now"], df["v2"] = v1, v2
df["sig_align"] = (sig * txt).sum(1)
# Global placeholder / duplicate counts from the full scan.
sha = pd.concat([pq.read_table(f, columns=["sha256"]).to_pandas() for f in glob.glob(r"D:\laion2b-filter\analysis\scan\*.parquet")]).sha256.value_counts()
df["sha_n"] = df.sha256.map(sha).fillna(1)
ar = np.abs(np.log((df.WIDTH / df.HEIGHT) / (df.original_width / df.original_height)))
df["changed"] = ar > 0.05
r = lambda a, b: f"pearson {pearsonr(a, b)[0]:.3f} spearman {spearmanr(a, b)[0]:.3f}"
print(f"N={N}")
print("== Drift: LAION's stored V1 against V1 recomputed on the downloaded file")
print("all:", r(df.aesthetic, df.v1_now), f" mean stored {df.aesthetic.mean():.2f}, recomputed {df.v1_now.mean():.2f}")
ok = (df.sha_n < 3) & ~df.changed
print("excluding placeholders and changed images:", r(df.aesthetic[ok], df.v1_now[ok]))
for name, m in [("placeholders (sha seen >= 3)", df.sha_n >= 3), ("aspect changed", df.changed), ("clean", ok)]:
    print(f"  {name:30s} n={m.sum():6d}  recomputed V1 median {df.v1_now[m].median():.2f}, share still > 7: {(df.v1_now[m] > 7).mean():.3f}")
print(f"share of all files whose recomputed V1 is <= 6: {(df.v1_now <= 6).mean():.3f}, <= 6.5: {(df.v1_now <= 6.5).mean():.3f}")
print("== Agreement between aesthetic judges (clean images)")
print("V1 now vs V2:", r(df.v1_now[ok], df.v2[ok]))
if os.path.exists(os.path.join(OUT, "aes_v25.npy")):
    v25all = L("aes_v25.npy"); has = ~np.isnan(v25all); df["v25"] = v25all; d = df[has]; okk = ok[has]; v25 = v25all[has]
    print("V2 vs V2.5:", r(d.v2[okk], d.v25[okk]), "| V1 now vs V2.5:", r(d.v1_now[okk], d.v25[okk]), "| stored V1 vs V2.5:", r(d.aesthetic[okk], d.v25[okk]))
    print("V2.5 quantiles:", np.round(np.quantile(v25, [.05, .25, .5, .75, .95]), 2), " V2 quantiles:", np.round(np.quantile(v2, [.05, .25, .5, .75, .95]), 2))
print("== Alt-text alignment: SigLIP2 vs LAION's B/32 similarity:", r(df.similarity.clip(0, 1), df.sig_align))

def probe(y, X, name, idx=None):
    idx = np.arange(len(y)) if idx is None else idx
    rng = np.random.default_rng(0); p = rng.permutation(idx); k = int(.8 * len(p)); tr, te = p[:k], p[k:]
    m = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(X[tr], y[tr]); pr = m.predict(X[te])
    top_true = y[te] >= np.quantile(y[te], .6); top_pred = pr >= np.quantile(pr, .6)
    print(f"probe {name:28s} train {len(tr):6d} test {len(te):6d}  {r(y[te], pr)}  top-40% overlap {np.mean(top_pred[top_true]):.3f}")
    return m
print("== Ridge probes on SigLIP2-B/16-256 embeddings")
probe(df.v1_now.values, sig, "V1 (L/14 linear)")
probe(df.v2.values, sig, "V2 (L/14 MLP)")
if os.path.exists(os.path.join(OUT, "aes_v25.npy")):
    probe(v25, sig[has], "V2.5 (so400m)")
    for nn in (2000, 5000, 10000):
        probe(v25[:nn], sig[has][:nn], f"V2.5 with {nn} labels")
for f in ("wm_yolo_640.npy", "wm_yolo_320.npy"):
    if os.path.exists(os.path.join(OUT, f)):
        wa = L(f); hw = ~np.isnan(wa); w = wa[hw]; y = (w >= .5).astype(int); X = sig[hw]
        rng = np.random.default_rng(0); p = rng.permutation(len(w)); k = int(.8 * len(p))
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(X[p[:k]], y[p[:k]]); pr = lr.predict_proba(X[p[k:]])[:, 1]
        print(f"watermark probe vs {f}: positives {y.mean():.3f}, AUC {roc_auc_score(y[p[k:]], pr):.3f}; LAION pwatermark AUC {roc_auc_score(y, df.pwatermark[hw]):.3f}")
df.to_parquet(os.path.join(OUT, "sample_scored.parquet"))

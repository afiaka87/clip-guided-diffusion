r"""Content composition of the stage-A survivors, the clean tier and the best tier: SigLIP2 zero-shot over the same 32
categories as the 82k sample, from the stage-B embeddings, on the CPU. Also how evenly each tier spreads over the
1,000 parent clusters, and the medians of its scores."""
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
from zs_b32 import CATS  # noqa: E402

SEL = r"D:\laion2b-filter\selection"
STAGEB = r"D:\laion2b-filter\stageB"


def main():
    import open_clip
    ids, emb = [], []
    for f in sorted(glob.glob(os.path.join(STAGEB, "part_*.npz"))):
        d = np.load(f)
        ids.append(d["ids"]); emb.append(d["emb"])
    ids = np.concatenate(ids); emb = np.concatenate(emb)
    m, _, _ = open_clip.create_model_and_transforms("hf-hub:timm/ViT-B-16-SigLIP2-256"); m.eval()
    tok = open_clip.get_tokenizer("hf-hub:timm/ViT-B-16-SigLIP2-256")
    names = list(CATS)
    prompts = [p for k in names for p in CATS[k]]
    owner = np.array([i for i, k in enumerate(names) for _ in CATS[k]])
    with torch.no_grad():
        t = F.normalize(m.encode_text(tok(prompts)).float(), dim=-1)
    cat = np.empty(len(emb), np.int16)
    for i in range(0, len(emb), 200_000):
        sim = (torch.from_numpy(emb[i:i + 200_000].astype(np.float32)) @ t.T).numpy()
        per = np.full((len(sim), len(names)), -9.0, np.float32)
        for j, o in enumerate(owner):
            per[:, o] = np.maximum(per[:, o], sim[:, j])
        cat[i:i + 200_000] = per.argmax(1)
    cat = pd.Series(np.array(names)[cat], index=ids)
    scored = pd.read_parquet(os.path.join(SEL, "scored.parquet"), columns=["id", "parent", "quality", "aes_v25_probe",
                                                                         "aes_v2_probe", "stage_c_reason"])
    best = pd.read_parquet(os.path.join(SEL, "best1m.parquet"), columns=["id"])
    tiers = {"stage A": scored, "clean": scored[scored.stage_c_reason == ""], "best": scored[scored.id.isin(best.id)]}
    shares = pd.DataFrame({k: cat.loc[v.id].value_counts(normalize=True) * 100 for k, v in tiers.items()}).fillna(0)
    print(shares.sort_values("best", ascending=False).round(1).to_string())
    print()
    for k, v in tiers.items():
        p = v.parent.value_counts()
        print(f"{k}: n={len(v):,}  parents {p.size}  largest parent {p.iloc[0] / len(v) * 100:.2f}%  "
              f"top 10 parents {p.iloc[:10].sum() / len(v) * 100:.1f}%  median quality {v.quality.median():.2f}  "
              f"median v2.5 probe {v.aes_v25_probe.median():.2f}  median v2 probe {v.aes_v2_probe.median():.2f}")


if __name__ == "__main__":
    main()

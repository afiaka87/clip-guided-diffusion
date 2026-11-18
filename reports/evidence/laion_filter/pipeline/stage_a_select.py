r"""Stage A: the classical hard filters over the full scan. Writes the survivors, with every scan column, to
D:\laion2b-filter\selection\stage_a.parquet, and the reason each other sample was dropped to stage_a_dropped.parquet."""
import glob, os, numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

SCAN = r"D:\laion2b-filter\analysis\scan"
OUT = r"D:\laion2b-filter\selection"

def main():
    tabs = [pq.read_table(f) for f in sorted(glob.glob(os.path.join(SCAN, "*.parquet")))]
    m = pa.concat_tables(tabs, promote_options="permissive").to_pandas()
    m = m[m.shard != 988].reset_index(drop=True)  # truncated tar
    ss = np.minimum(m.original_width, m.original_height); ar = np.maximum(m.original_width, m.original_height) / ss
    sha_n = m.sha256.map(m.sha256.value_counts())
    dh_n = m.dhash.map(m.dhash.value_counts()); dh_caps = m.dhash.map(m.groupby("dhash").caption.nunique())
    tests = [
        ("undecodable", m.decode_ok != True),
        ("blank", m.lum_std < 4),
        ("placeholder_sha", sha_n >= 3),
        ("placeholder_dhash", (dh_n >= 5) & (dh_caps >= 3)),
        ("exact_duplicate", m.sha256.duplicated()),
        ("image_changed_since_scored", np.abs(np.log((m.WIDTH / m.HEIGHT) / (m.original_width / m.original_height))) > 0.05),
        ("upscaled_source_under_256", ss < 256),
        ("aspect_over_1.8", ar > 1.8),
        ("white_ring_over_half", m.ring_white >= 0.5),
        ("greyscale", m.chroma_mean < 4),
        ("under_256_colours", m.n_colors < 256),
        ("laion_pwatermark_over_0.5", m.pwatermark >= 0.5),
        ("laion_punsafe_over_0.1", m.punsafe >= 0.1),
        ("bad_metadata", (m.aesthetic > 9.75) | (m.similarity > 1) | m.similarity.isna()),
    ]
    reason = pd.Series("", index=m.index)
    for name, t in tests:
        t = np.asarray(t, bool) & (reason.values == "")
        reason[t] = name
    keep = reason == ""
    m["stage_a_reason"] = reason
    m[keep].reset_index(drop=True).to_parquet(os.path.join(OUT, "stage_a.parquet"))
    m.loc[~keep, ["shard", "key", "stage_a_reason"]].to_parquet(os.path.join(OUT, "stage_a_dropped.parquet"))
    print("kept", int(keep.sum()), "of", len(m)); print(reason[~keep].value_counts().to_string())

if __name__ == "__main__":
    main()

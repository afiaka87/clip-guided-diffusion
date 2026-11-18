"""Copy every image of 14 random shards into one folder of JPEGs plus a parquet of their scan rows."""
import os, glob, random, tarfile
import pandas as pd, pyarrow.parquet as pq
OUT = r"D:\laion2b-filter\bench_sample"
os.makedirs(os.path.join(OUT, "img"), exist_ok=True)
tars = sorted(glob.glob(r"D:\laion2b-en-aesthetic\*.tar"))[17:-1]
random.seed(7); pick = sorted(random.sample(tars, 14))
rows = []
for t in pick:
    s = int(os.path.basename(t)[:5])
    with tarfile.open(t) as tf:
        for m in tf:
            if m.name.endswith(".jpg"):
                with open(os.path.join(OUT, "img", f"{s:05d}_{m.name}"), "wb") as f:
                    f.write(tf.extractfile(m).read())
    rows.append(pq.read_table(rf"D:\laion2b-filter\analysis\scan\{s:05d}.parquet").to_pandas())
    print(t, flush=True)
df = pd.concat(rows, ignore_index=True)
df["path"] = [os.path.join(OUT, "img", f"{s:05d}_{k}.jpg") for s, k in zip(df.shard, df.key)]
df = df[[os.path.exists(p) for p in df.path]].reset_index(drop=True)
df.to_parquet(os.path.join(OUT, "sample.parquet")); print(len(df), "samples")

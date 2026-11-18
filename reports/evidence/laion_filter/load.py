import glob, os, pyarrow as pa, pyarrow.parquet as pq
def load(d=r"D:\laion2b-filter\analysis\scan", columns=None):
    tabs = [pq.read_table(f, columns=columns) for f in sorted(glob.glob(os.path.join(d, "*.parquet")))]
    return pa.concat_tables(tabs, promote_options="permissive").to_pandas()

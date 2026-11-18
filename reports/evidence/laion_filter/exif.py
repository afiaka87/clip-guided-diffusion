import json, pandas as pd
from load import load
from sheet_keys import sheet
m = load(columns=["shard","key","exif","original_width","original_height","WIDTH","HEIGHT"])
o = m.exif.fillna("{}").map(lambda s: json.loads(s).get("Image Orientation"))
print(len(m), o.value_counts(dropna=False).head(10).to_dict())
rot = m[o.astype(str).str.contains("Rotated 90|Rotated 270|^6$|^8$", regex=True)]
print("rotated-90 exif", len(rot), "of which LAION dims swapped vs downloaded:", int(((rot.WIDTH == rot.original_height) & (rot.HEIGHT == rot.original_width) & (rot.WIDTH != rot.HEIGHT)).sum()), "same:", int(((rot.WIDTH == rot.original_width) & (rot.HEIGHT == rot.original_height)).sum()))
x = rot.sample(min(24, len(rot)), random_state=0)
sheet(list(zip(x.shard, x.key)), "exif_rot.png", labels=[str(v) for v in o[x.index]])

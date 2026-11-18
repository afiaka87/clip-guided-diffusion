import numpy as np, pandas as pd
from load import load
m = load(columns=["shard","key","url","caption","original_width","original_height","sha256","dhash","aesthetic","similarity","pwatermark","lum_std","jpg_bytes"])
n = len(m); pct = lambda s: f"{s.mean()*100:.2f}% ({int(s.sum()):,})"
dom = m.url.fillna("").str.extract(r"https?://([^/]+)/", expand=False).fillna("").str.replace(r"^www\.", "", regex=True)
groups = {
 "pinterest": r"pinimg\.com$",
 "stock photo / vector sites": r"(dreamstime|shutterstock|ftcdn|depositphotos|istockphoto|alamy|123rf|freepik|vectorstock|bigstockphoto|canstockphoto|gettyimages|stock\.adobe|pond5|colourbox|featurepics|fotolia)",
 "search-engine thumbnails": r"(gstatic\.com|mm\.bing\.net)$",
 "shops (shopify, etsy, ebay, amazon, aliexpress, wix, squarespace)": r"(shopify|etsystatic|ebayimg|amazon|alicdn|wixstatic|squarespace)",
 "blogspot / wordpress": r"(blogspot|wp\.com|wordpress)",
 "wikimedia": r"wikimedia",
 "flickr / smugmug": r"(staticflickr|smugmug)",
}
for k, pat in groups.items():
    print(f"{k:70s} {pct(dom.str.contains(pat, regex=True))}")
print("vectorstock placeholder size 30720x20562:", pct((m.original_width == 30720) & (m.original_height == 20562)))
sha_n = m.sha256.map(m.sha256.value_counts())
print("sha256 seen >= 3:", pct(sha_n >= 3), " exact duplicates beyond first:", pct(m.sha256.duplicated()))
print("unique sha256:", m.sha256.nunique())
cap = m.caption.fillna("")
bad = cap.str.contains(r"(?i)click (image|here)|larger version|image not (available|found)|no image|coming soon|placeholder|untitled|^image$|^photo$|^img_?\d+|^dsc_?\d+", regex=True)
print("junk-caption patterns:", pct(bad))
print("html tags in caption:", pct(cap.str.contains(r"<[a-z/][^>]*>", regex=True)))
print("aesthetic > 9.75 or similarity > 1 (bad values):", int((m.aesthetic > 9.75).sum()), int((m.similarity > 1).sum()), int((m.similarity < .2).sum()))

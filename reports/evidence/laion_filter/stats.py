import json, glob, collections, os
d = r"D:\laion2b-en-aesthetic"
tot = collections.Counter(); status = collections.Counter(); shards=[]
for f in sorted(glob.glob(os.path.join(d, "*_stats.json"))):
    s = json.load(open(f))
    for k in ("count","successes","failed_to_download","failed_to_resize"): tot[k]+=s[k]
    status.update(s["status_dict"]); shards.append((os.path.basename(f), s["count"], s["successes"]))
print(len(shards), dict(tot))
tars = sorted(glob.glob(os.path.join(d, "*.tar")))
print("tars", len(tars), "first", os.path.basename(tars[0]), "last", os.path.basename(tars[-1]))
have = {os.path.basename(t)[:5] for t in tars}; st = {s[0][:5] for s in shards}
print("tar without stats", sorted(have-st), "stats without tar", sorted(st-have))
for k,v in status.most_common(12): print(v, k[:90])
succ = [s[2] for s in shards]
import statistics
print("success per shard min/median/max", min(succ), statistics.median(succ), max(succ))
print("low shards", [s for s in shards if s[2] < 3000][:20])

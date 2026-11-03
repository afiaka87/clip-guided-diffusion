import json, sys
rows = {}
meta = {}
for name, path in (("cpu", sys.argv[1]),):
    for line in open(path):
        if not line.startswith("{"):
            continue
        r = json.loads(line)
        if "resident_GiB" in r:
            meta[(r["name"], r.get("depth", 1))] = r
        elif "peak_alloc" in r:
            rows.setdefault((r["name"], r["px"], r.get("depth", 1)), {})[r["mode"]] = r
        elif "error" in r and "mode" in r:
            rows.setdefault((r["name"], r["px"], r.get("depth", 1) if "depth" in r else "?"), {})[r["mode"]] = dict(peak_alloc="ERR", extra="ERR")
modes = ["forward", "full", "ckpt", "tiny", "tiny_ckpt", "x0_vae", "x0_tiny"]
print("name px depth resident | " + " ".join(modes) + "  (extra GiB above resident)")
for key in sorted(rows, key=str):
    row = rows[key]
    res = next((v.get("resident") for v in row.values() if "resident" in v), "?")
    print(*key, res, "|", *[row.get(m, {}).get("extra", "-") for m in modes])

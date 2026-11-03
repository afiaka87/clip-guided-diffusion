import json, sys, urllib.request, collections
OUT = sys.argv[1]
REPOS = ["stabilityai/sd-turbo", "stable-diffusion-v1-5/stable-diffusion-v1-5", "sd2-community/stable-diffusion-2-1-base",
         "stabilityai/stable-diffusion-xl-base-1.0", "stabilityai/sdxl-turbo", "segmind/Segmind-Vega", "segmind/SSD-1B",
         "PixArt-alpha/PixArt-Sigma-XL-2-512-MS", "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
         "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers", "black-forest-labs/FLUX.2-klein-4B",
         "Tongyi-MAI/Z-Image-Turbo", "kandinsky-community/kandinsky-2-2-decoder", "kandinsky-community/kandinsky-2-2-prior"]
report = {}
for repo in REPOS:
    url = f"https://huggingface.co/api/models/{repo}?blobs=true"
    try:
        data = json.load(urllib.request.urlopen(url, timeout=30))
    except Exception as error:
        report[repo] = str(error)
        continue
    parts = collections.defaultdict(dict)
    for f in data.get("siblings", []):
        name, size = f["rfilename"], f.get("size") or 0
        if name.endswith((".safetensors", ".bin", ".ckpt")) and size > 1e6:
            folder = name.split("/")[0] if "/" in name else "."
            parts[folder][name.split("/")[-1]] = round(size / 1e9, 3)
    report[repo] = {k: v for k, v in parts.items()}
json.dump(report, open(OUT, "w"), indent=0)
for repo, parts in report.items():
    print(repo)
    if isinstance(parts, str):
        print("  ", parts); continue
    for folder, files in parts.items():
        if folder == "." and len(files) > 6:
            files = dict(list(files.items())[:6])
        print("  ", folder, files)

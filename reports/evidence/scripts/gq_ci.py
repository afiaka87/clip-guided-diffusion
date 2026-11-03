"""Per-pair gradient statistics for five towers, for interval widths the main run did not keep. CPU only."""
import math, os, sys
src = open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "grad_quality.py")).read()
head = src.split("TX = {k:")[0].replace('print("loading", flush=True)',
    'SPECS = {k: SPECS[k] for k in ("openai/ViT-B-32", "PE-Core-B-16", "laion/ConvNeXt-B-w", "FARE4/ViT-B-32", "FARE4/ConvNeXt-B-w")}\nprint("loading", flush=True)')
exec(head)
TX = {k: {p: m.text([p]) for p in PROMPTS} for k, m in M.items()}
for k in M:
    rows = []
    for i_n in IMAGES:
        x = load_img(i_n)
        for p in PROMPTS:
            t = TX[k][p]
            g = grad(k, x, t, 1)
            rows.append((cos(g, grad(k, (x + th.randn_like(x) / 2048).clamp(0, 1), t, 1)), cos(g, grad(k, x, t, 2)), hf_share(g)))
    out = []
    for i, name in enumerate(("noise", "cut", "hf")):
        v = [r[i] for r in rows]
        mean = sum(v) / len(v)
        sd = (sum((a - mean) ** 2 for a in v) / (len(v) - 1)) ** 0.5
        out.append(f"{name} {mean:.3f} +-{2.571 * sd / len(v) ** 0.5:.3f}")
    print(k, " ".join(out), flush=True)

"""What does the blend do, and where does gradient power sit?

x_in = sigma * x0 + (1 - sigma) * x, so dL/dx = sigma * J^T g + (1 - sigma) * g, where g
is CLIP's raw gradient and J the U-Net's Jacobian. The second term reaches x without
going through the U-Net at all. Measure each path at real states of a real run.
"""
import torch as th
import torch.nn.functional as tf

from harness import Config, Pipeline, PROMPTS, Views, aggregate, per_cut_gradients, resample, save_json, wait_for_gpu, device

STEPS = (8, 11, 13, 15, 16, 17, 18, 19)


def high_frequency(g):
    """Share of spectral power above 0.25 cycles per pixel. White noise is about 0.80."""
    side = g.shape[-1]
    power = th.fft.rfft2(g).abs() ** 2
    fy = th.fft.fftfreq(side, device=g.device)[:, None]
    fx = th.fft.rfftfreq(side, device=g.device)[None]
    mask = (fy**2 + fx**2).sqrt() > 0.25
    return (power[..., mask].sum() / power.sum()).item()


def cosine(a, b):
    return tf.cosine_similarity(a.flatten()[None], b.flatten()[None]).item()


print("waiting for gpu:", wait_for_gpu(), flush=True)
pipe = Pipeline(128, "ViT-B/32")
rows = []
for prompt in PROMPTS[:6]:
    text, _ = pipe.target(prompt)

    def hook(step, x, x0, x_in, fac, views):
        if step + 6 not in STEPS:
            return
        x01 = x_in.detach().add(1).div(2)
        cfg = Config(n=4)
        gen = th.Generator().manual_seed(step)
        estimate = Views(pipe.clip_size, cfg, gen)
        gs = []
        for _ in range(2):
            plan = estimate.draw(x.shape[-1], fac)
            grads, _ = per_cut_gradients(pipe.clip, text, estimate, x01, plan)
            gs.append(aggregate(grads, cfg))
        g1, g2 = gs
        through = [th.autograd.grad(x0, x, grad_outputs=g, retain_graph=True)[0] for g in (g1, g2)]
        blended = [fac * j + (1 - fac) * g for j, g in zip(through, (g1, g2))]
        identity_share = ((1 - fac) * g1).norm().item() / (fac * through[0]).norm().item()
        row = dict(prompt=prompt, step=step + 6, sigma=float(fac), noise_share=float((1 - fac) * fac),
                   pair_raw=cosine(g1, g2), pair_unet=cosine(*through), pair_blend=cosine(*blended),
                   hf_raw=high_frequency(g1), hf_unet=high_frequency(through[0]), hf_blend=high_frequency(blended[0]),
                   identity_over_unet=identity_share, cos_blend_vs_unet=cosine(blended[0], through[0]),
                   rms_raw=g1.square().mean().sqrt().item(), rms_unet=through[0].square().mean().sqrt().item())
        rows.append(row)

    pipe.sample(Config(), prompt, 0, hook=hook)
    print("done", prompt, flush=True)
save_json("e5_blend", rows)
print(f"{'step':>4s} {'sigma':>6s} {'eps in x_in':>11s} | {'pair raw':>8s} {'pair unet':>9s} {'pair blend':>10s} | {'hf raw':>6s} {'hf unet':>7s} {'hf blend':>8s} | {'id/unet':>7s} {'cos(b,u)':>8s}")
for step in STEPS:
    part = [r for r in rows if r["step"] == step]
    mean = lambda key: sum(r[key] for r in part) / len(part)  # noqa: E731
    print(f"{step:4d} {mean('sigma'):6.3f} {mean('noise_share'):11.3f} | {mean('pair_raw'):8.3f} {mean('pair_unet'):9.3f} {mean('pair_blend'):10.3f} | "
          f"{mean('hf_raw'):6.3f} {mean('hf_unet'):7.3f} {mean('hf_blend'):8.3f} | {mean('identity_over_unet'):7.3f} {mean('cos_blend_vs_unet'):8.3f}")

print("\n== resampler: gradient spectrum, repeatability and what CLIP reads, at the last guided step ==")
results = {}
for prompt in PROMPTS[:6]:
    text, _ = pipe.target(prompt)
    store = {}

    def hook2(step, x, x0, x_in, fac, views):
        if step + 6 == 19:
            store["x01"] = x0.detach().add(1).div(2).clamp(0, 1)

    pipe.sample(Config(), prompt, 0, hook=hook2)
    x01 = store["x01"]
    for name in ("area", "bilinear", "bicubic", "lanczos3"):
        cfg = Config(n=16, resample=name)
        gen = th.Generator().manual_seed(1)
        views = Views(pipe.clip_size, cfg, gen)
        estimates = []
        for _ in range(4):
            plan = views.draw(128, 0.3)
            grads, _ = per_cut_gradients(pipe.clip, text, views, x01, plan)
            estimates.append(grads.mean(0))
        pairs = [cosine(estimates[i], estimates[j]) for i in range(4) for j in range(i)]
        with th.no_grad():
            gen = th.Generator().manual_seed(2)
            views = Views(pipe.clip_size, Config(n=64, resample=name), gen)
            plan = views.draw(128, 0.3)
            cuts = views.cut(x01.repeat(64, 1, 1, 1), plan)
            from harness import clip_util, losses
            embeds = tf.normalize(pipe.clip.encode_image(clip_util.preprocess(pipe.clip, cuts)).float(), dim=-1)
            similarity = (embeds @ tf.normalize(text, dim=-1).T).mean().item()
        row = results.setdefault(name, dict(pair=[], hf_grad=[], hf_input=[], similarity=[]))
        row["pair"].append(sum(pairs) / len(pairs))
        row["hf_grad"].append(high_frequency(estimates[0]))
        row["hf_input"].append(high_frequency(cuts - cuts.mean((2, 3), keepdim=True)))
        row["similarity"].append(similarity)
save_json("e5_resampler", results)
for name, row in results.items():
    print(f"{name:10s} " + " ".join(f"{key}={sum(values) / len(values):.4f}" for key, values in row.items()))

p = "/home/sam/GitHub/clip-guided-diffusion/research_notes/CLIP guided diffusion quality improvements/diffusion_backbones.md"
s = open(p).read()

effect = """**Does guiding through the cheap paths change the result?** Real SD 1.5 at fp32 on the CPU, DDIM 20 steps, CFG 7.5, guided on steps 3 to 14 at one relative strength (rate 0.2 of the prediction's RMS), 4 cutouts through ViT-B/32. Images decoded with the full VAE, then scored by the guiding CLIP and by held-out PE-Core-B-16. 2 prompts, 1 seed (seed 0) each, 10 images. Deltas are guided minus unguided on the same prompt and noise. Intervals are 95% t-intervals on the 2 paired deltas (df = 1), mean ± half-width:

| Path | Prompt 0 guide | Prompt 1 guide | Guide delta, mean ± half-width | Prompt 0 held-out | Prompt 1 held-out | Held-out delta, mean ± half-width |
|---|---|---|---|---|---|---|
| none (absolute) | 0.3467 | 0.3598 | n/a | 0.3179 | 0.3247 | n/a |
| `xt_vae` | +0.0219 | +0.0457 | +0.034 ± 0.151 | -0.0148 | +0.0095 | -0.003 ± 0.154 |
| `xt_tiny` | +0.0070 | +0.0260 | +0.017 ± 0.121 | -0.0233 | +0.0083 | -0.008 ± 0.201 |
| `x0_vae` | +0.0059 | +0.0118 | +0.009 ± 0.038 | +0.0159 | -0.0033 | +0.006 ± 0.122 |
| `x0_tiny` | +0.0071 | +0.0140 | +0.011 ± 0.044 | +0.0270 | -0.0050 | +0.011 ± 0.203 |

Every interval includes zero. What the data do show is limited: all four paths raised the guiding score on both prompts, and held-out deltas changed sign between prompts on every path. In the 3 prompt 0 images I looked at (`none`, `xt_vae`, `x0_tiny`) neither guided result had adversarial artifacts. That is an anecdote.
"""

edits = [
    ("<!-- TAKEAWAY-EFFECT -->",
     "In a small test (2 prompts, 1 seed each, every 95% interval including zero), all four cheap and full paths raised the guiding CLIP score on both prompts, so TAESD and x0 gradients can steer, but how well is not established."),
    ("<!-- EFFECT-RESULTS -->", effect.rstrip()),
    ("| SD 1.5 | 512 | 2.33 | 2.93 | 5.92 | 4.14 | 3.43 | 2.85 | 5.20 | 2.71 | G |\n",
     "| SD 1.5 | 512 | 2.33 | 2.93 | 5.92 | 4.14 | 3.43 | 2.85 | 5.20 | 2.71 | G |\n| SD 2.1, SD-Turbo's architecture | 512 | 2.34 | 2.81 | ~5.9 | | 3.36 | 2.79 | | 2.69 | C, P for full from SD 1.5 |\n"),
    ("| Sana Sprint 1.6B | 1024 | 4.14 |",
     "| Sana Sprint 1.6B | 512 | 4.14 | 4.43 | | | none | none | | none | C |\n| Sana Sprint 1.6B | 1024 | 4.14 |"),
    ("| FLUX.2 klein 4B, bf16 | 512 |",
     "| Kandinsky 2.2 decoder | 512 | 3.02 | 3.65 | | | none | none | | none | C |\n| FLUX.2 klein 4B, bf16 | 512 |"),
    ("Checks on the method:\n",
     "Each cell is one run at seed 0, so no seed count above 1 and no interval applies. Peak allocation depends on shapes, not weights, so repeats would not add much. The only error estimates are the checks below, and projected cells have none of their own.\n\nChecks on the method:\n"),
    ("`x0_tiny` 0.11 s.\n", "`x0_tiny` 0.11 s. Each is one timed run, so these have no interval.\n"),
    ("for 2 prompts at each of 3 noise levels. A first pass",
     "for 2 prompts at seed 0 at each of 3 noise levels, 6 gradient draws per point. A first pass"),
    ("Two draws of cutouts through the same decoder barely agree either:\n",
     "Two draws of cutouts through the same decoder barely agree either. Cells are mean ± 95% half-width over the 2 prompts (df = 1), so a narrow interval can simply mean the two prompts happened to land close together:\n"),
    ("| 751 | 0.16 (low-pass 0.36) | 0.16 (0.41) | 0.24 (0.32) | 0.12 (0.24) |",
     "| 751 | 0.16 ± 0.61 (low-pass 0.36 ± 0.91) | 0.16 ± 0.16 (0.41 ± 0.50) | 0.24 ± 0.62 (0.32 ± 0.08) | 0.12 ± 0.01 (0.24 ± 0.34) |"),
    ("| 451 | 0.10 (0.28) | 0.10 (0.47) | 0.28 (0.53) | 0.10 (0.29) |",
     "| 451 | 0.10 ± 0.13 (0.28 ± 0.31) | 0.10 ± 0.39 (0.47 ± 0.84) | 0.28 ± 0.49 (0.53 ± 0.02) | 0.10 ± 0.31 (0.29 ± 0.03) |"),
    ("| 151 | 0.08 (0.16) | 0.15 (0.60) | 0.20 (0.53) | 0.10 (0.42) |",
     "| 151 | 0.08 ± 0.43 (0.16 ± 1.98) | 0.15 ± 0.33 (0.60 ± 0.99) | 0.20 ± 0.05 (0.53 ± 1.28) | 0.10 ± 0.02 (0.42 ± 0.32) |"),
    ("In the 4 points from the first pass, the gradient that stops at x0 agreed with the full gradient through the U-Net at cosine 0.30 to 0.64 raw, rising as noise fell.",
     "In the 4 points from the first pass (2 prompts, seed 0, one gradient each, range only, no interval), the gradient that stops at x0 agreed with the full gradient through the U-Net at cosine 0.30 to 0.64 raw, rising as noise fell."),
    ("There TAESD falls below the same-decoder figure, so it has a systematic difference from the full VAE as well as sampling noise. Cosines on this few samples are indicative only.",
     "There TAESD's point estimate falls below the same-decoder figure at all 3 noise levels, which hints at a systematic difference from the full VAE on top of sampling noise. The intervals mostly overlap, so that is a hint and not a finding."),
    ("- Only SD 1.5 and the ADM baselines were measured on the card. Other GPU jobs held it for the whole of this work, so I had to use CPU counting. The queued GPU sweep did not get a slot, and no model other than SD 1.5 has a guided-step time.",
     "- Only SD 1.5 and the ADM baselines were measured on the card. Other GPU jobs held it for the whole of this work, so I had to use CPU counting. The queued GPU sweep never got a slot and was stopped at wrap-up, along with the on-card gradient agreement run and the fp16 against bf16 timing. No model other than SD 1.5 has a guided-step time.\n- The Part B effect test stopped at 2 of 4 prompts when time ran out, at 1 seed and 1 rate. Its intervals are too wide to rank the paths or to show any effect on the held-out score. The rate was not tuned per path, so paths with different gradient directions may not be compared at their best.\n- Kandinsky 2.2's `full`, `ckpt` and `x0_vae` rows, and Sana Sprint 1.6B's `full`, `ckpt` and `x0_vae` rows, failed on my 18 GB address-space cap and were not rerun. The SD 3.5 depth rows and klein 384 rows were cancelled, so klein stays projected from 512."),
    ("| 1 to 4 | [Stability AI Community]", "| 1 to 4 | [Stability AI Community]"),
    ("| 512 | 866M | 2.6 GB | ~2.9 | ~2.9 / ~5.9 |", "| 512 | 866M | 2.6 GB | 2.81 | 2.79 / ~5.9 |"),
    ("| not measured, weights alone 3.0 | no tiny decoder / not measured |", "| 3.65 at 512 | no tiny decoder / run failed, not estimated |"),
    ("SD-Turbo's memory figures are SD 1.5's, measured on the card, because the two U-Nets are within 1% in parameters. SD-Turbo itself was not run.",
     "SD-Turbo's forward and `tiny_ckpt` figures are counted on SD 2.1's U-Net, which has the same 865.9M parameters. Its `full` figure is projected from SD 1.5 on the card. All are one run at seed 0, with no interval."),
    ("It pays inference VRAM only, about 2.9 GiB,", "It pays inference VRAM only, about 2.8 GiB,"),
    ("- Noise optimisation with CLIP alone, on these cards, is unmeasured. ReNO used an ensemble of rewards.\n",
     "- Noise optimisation with CLIP alone, on these cards, is unmeasured. ReNO used an ensemble of rewards.\n- The tiny-decoder routes (Vega at 1024, SDXL at 1024, klein) rest on a 2 prompt, 1 seed effect test whose intervals all include zero, on SD 1.5 only. Run that test at more prompts and seeds before committing to any of them.\n"),
]
for old, new in edits:
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)
open(p, "w").write(s)

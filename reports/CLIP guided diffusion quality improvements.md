# Change the prior, not the cutouts

CLIP-guided ADM falls short of modern models because its prior is a 2021 ImageNet U-Net, and better cutouts cannot supply what that prior never learned. Swapping the prior for SD-Turbo with CLIP noise optimisation scored +.096 ± .029 on 12 held-out prompts and improved all 12, against +.0188 ± .015 for the best change that stays on ADM. Nearly 60 cutout arms on ADM and SD-Turbo beat nothing, because the gradient is capped and the model discards most of it, so its variance is not the bottleneck. Most of what looked strong on one seed then failed on more: dynamic thresholding, cross-entropy, PAG, a sine schedule and ER-SDE alone all lost or tied, and 128 + Real-ESRGAN fell from ahead of direct 512 on two judges to a tie. What shipped is `-bb` with SD-Turbo, Vega and SDXL-Lightning in 8 GB, the ViT-B/32 + PE-Core-B-16 ensemble, `-best N` and `-up`, plus an aesthetic term rated over cutouts, which at .05 on SD-Turbo raises a held-out rating from 6.35 to 6.90 at no measurable score cost. These limits stand: no human rated an image, the judges resize to about 224px, nothing ran on the RTX 4070, Vega and Lightning have one seed of scores, and FLUX.2 klein 4B stayed unrun.

## How to read the evidence in this report

Each claim below is labelled **Published**, **Measured here** or **Untested**, and estimates are called estimates. Most local numbers come from one of two designs, named in each table so the sample size travels with the number.

| Label | Sample and interval | What it cannot show |
|---|---|---|
| Design A | 128px, ViT-B/32, 30 calls, rate .08. 12 prompts × 2 seeds, 24 images per arm, paired against base on shared sampler noise. Change in CLIP similarity to the prompt, pooled over 3 judges that did not guide (ViT-B/16, ViT-H-14-quickgelu, ViT-L-16-SigLIP2-256). 95% CI across prompts ([script][e4], [log][e4log]) | Not corrected for 27 comparisons. Effects under about .01 are unresolved |
| Design B | 12 prompts, seed 0 only, same 3 judges reported separately, guidance ViT-B/32. 95% bootstrap CI on the paired mean across prompts ([scripts and scores][cas]) | Seed variance is not in the interval. Seed 0 is unusually saturated (.56 against .19 to .48 for seeds 1 to 7, at 64px) |
| Design C | The post-notes pilot: 128px, 12 prompts, seed 0, one image per prompt, 27 arms against base. Each arm's change on each of the 3 judges, with ±1.96 standard errors across prompts ([arms][pilot], [scoring][pjudge], [scores][pscores]) | No seed variance. 81 uncorrected intervals, so about 4 should exclude zero by chance |
| Design D | The confirmation runs on the code as shipped: ADM 128px, 30 calls, rate .08, 12 prompts × 4 seeds (1 to 4), 48 images per arm unless a row says 2 seeds. Pooled change on the same 3 judges, paired on prompt and seed, 95% CI across prompts ([script][admexp], [scores][admb], [cutouts][admcut], [counts][admcutn], [aesthetic][admaes]) | About 55 uncorrected arms, so 2 or 3 should exclude zero by chance. Effects under about .01 are unresolved |
| Design E | SD-Turbo 512 through `-bb`: 50 steps of noise optimisation, ViT-B/32 + PE-Core-B-16, 4 cutouts, tiny decoder. 12 prompts × 3 seeds, or seed 1 alone where stated. Same pooled score and CI ([script][turbotune], [scores][turbotune3], [cutouts][cutturbo], [ideas][novel], [aesthetic][aesturbo]) | One-seed rows are leads, not results |

Judge similarity is not image quality. The judges resize to about 224px, so they cannot see fine detail, and no human study was run. Anything described "by eye" is one researcher's reading of contact sheets, an anecdote. Scripts, logs, scores and contact sheets are in [evidence](evidence/). The generated images were not kept.

## CLIP guidance moves scores by .1; cutout changes move them by under .01

The size of each lever explains where to spend effort. **Measured here,** turning guidance off at 256px costs about .1 on every judge, so CLIP guidance works. Everything in the cutout family is an order of magnitude smaller than that, while choosing a better seed or guide is several times larger than any cutout change.

| Lever | Sample | Change |
|---|---|---|
| Guidance off, 256px, against default | Design B, n=12 | ViT-B/16 −.106 [−.134, −.079]; ViT-H −.117 [−.155, −.081]; SigLIP2 −.093 [−.123, −.062] |
| Best of 8 seeds at 64px, two judges pick, third scores, against the 8-seed mean (reanalysis of saved scores for this report) | 12 prompts, 8 seeds, bootstrap over prompts | Held-out ViT-B/16 +.020 [+.011, +.031]; ViT-H +.047 [+.031, +.063]; SigLIP2 +.029 [+.015, +.043] |
| ADM 128 replaced by SD-Turbo 512 with CLIP noise optimisation | 12 prompts, seed 0, pooled ([table][gap]) | +.096 ± .029 (12 of 12) |
| ADM 128 replaced by plain one-step SD-Turbo 512 | Same | +.081 ± .023 (12 of 12) |
| ADM 128 replaced by SD 1.5, CFG 7.5, 20 steps | Same | +.071 ± .031 (10 of 12) |
| ViT-B/32 + PE-Core-B-16 ensemble as guide | Design D, pooled | +.0188 ± .015 (10 of 12) |
| Best of 4 ADM seeds, picked by two judges, scored by ViT-H | Design D seeds, against the candidates' mean | +.0154 ± .0070 (11 of 12) |
| CLIP correction on SD-Turbo, against the plain model | Design E, 3 seeds, pooled | +.0110 ± .005 (12 of 12) |
| Guide swapped to PE-Core-B-16, 128px | Design D, pooled | +.0008 ± .015 (6 of 12) |
| Best cutout variant, Disco-style schedule | Design A, then fresh seeds | +.0102 ± .0141, then +.0001 ± .0170 |
| 16 cutouts instead of 4 | Design D, pooled | +.0052 ± .011 (6 of 12) |
| 32 cutouts instead of 4 | Design D, pooled | +.0018 ± .007 (8 of 12) |

**Published,** the ceiling looks structural. GLIDE's human raters preferred classifier-free guidance to CLIP guidance on both photorealism and caption match, at Elo −73.2 against 82.7 and 29.3 against 110.9, and the authors note that humans disagree with CLIP score ([GLIDE](https://arxiv.org/abs/2112.10741)). CLIPAG shows how far score and realism separate: CLIPScore 34.7 at FID 42.3, against GLIDE's FID 12.2 ([CLIPAG](https://ar5iv.labs.arxiv.org/html/2306.16805)). ADM itself is unguided FID 23.24 at ImageNet 512, against 2.56 for EDM2-S at half the parameters ([EDM2](https://arxiv.org/pdf/2312.02696)), though an ImageNet FID gap is direction, not a prediction for CLIP-guided output.

**Measured here,** prompt fit is set by the base. "A pencil sketch of an old man with a beard" came out as a pen in 64 and 128px images (one prompt, seed 0), and no upscaler recovered it ([cascade scripts][cas]). **Measured later,** that gap is now a number, and it is the largest lever in this table. On the same prompts, seed and judges, ADM 128 scored .289, .277 and .087; SD 1.5 scored .325, .364 and .178; plain SD-Turbo .340, .373 and .184; and SD-Turbo with CLIP noise optimisation .359, .386 and .196 ([table][gap]). Size explains little of it: on seed 1, the same 12 prompts, ADM 256 gains +.0188 ± .0164 over ADM 128 and ADM 512 +.0249 ± .0289, about a quarter of the swap. The held-out aesthetic predictor agrees, 4.7 for ADM against 6.5 for SD-Turbo. Nothing tried on ADM closes that: the best change that stays on ADM reaches +.0188, about a fifth of the swap. CLIP is still useful on the new prior, but as a correction of about +.011, not as the source of the image.

## Neither more nor different cutouts help, because variance is not the bottleneck

**Published,** cutout practice is almost entirely habit. Counts run from 4 in diffusers to 16 in v-diffusion, 32 to 40 in most notebooks and 64 per step in Disco, set by memory rather than measurement ([Sloan](https://www.robinsloan.com/lab/cutouts/), [Disco](https://raw.githubusercontent.com/alembics/disco-diffusion/main/disco.py)). Disco's only stated rationale for overview cuts is one unmeasured sentence, "good for early structure". No paper or notebook found measures gradient variance against cutout count, and no study compares these schemes on a held-out metric. Crowson's own later sampler dropped cutouts for a single view with a one-patch translation ([k-diffusion](https://github.com/crowsonkb/k-diffusion/blob/master/sample_clip_guided.py)). The one numeric cutout ablation found is one prompt in Text2Mesh (.36 full, .30 without crops, .20 without any augmentation) ([Text2Mesh](https://ar5iv.labs.arxiv.org/html/2112.03221)).

**Measured here,** the end-to-end sweep is the main evidence, and it is flat. Only the two clearest losses, EMA .8 and blur σ 2, survive a correction for 27 comparisons. The leaders were selected from 27 and shrank on seeds they had not seen, so any true effect is below about .01.

| Variant, Design A | Seeds 0, 1 (prompts improved of 12) | Fresh seeds 2, 3 (prompts improved) |
|---|---|---|
| Disco-style schedule, 3 of 4 cuts whole until σ < .988, then 1 of 4 | +.0102 ± .0141 (6) | +.0001 ± .0170 (4) |
| Pure x0 instead of the `fac` blend | +.0076 ± .0121 (8) | +.0043 ± .0149 (5) |
| 16 cuts | +.0064 ± .0090 (8) | +.0055 ± .0125 (6) |
| Pure x0 with Disco-style schedule | Not run | −.0011 ± .0171 (5) |
| 8 cuts | +.0034 ± .0122 (8) | Not run |
| 32 cuts | +.0006 ± .0159 (4) | Not run |
| Grey .25; hflip .5 | +.0050 ± .0114 (5); −.0035 ± .0126 (4) | Not run |
| Whole frames with jitter ±9 px | +.0035 ± .0098 (6) | Not run |
| Latin hypercube layout; R3 layout | −.0020 ± .0069 (5); −.0009 ± .0070 (7) | Not run |
| Bilinear; bicubic; lanczos3 | −.0030 ± .0130 (6); −.0120 ± .0125 (3); −.0137 ± .0127 (4) | Not run |
| Noise per cut .03; .1 | −.0099 ± .0144 (3); −.0149 ± .0135 (2) | Not run |
| `min_cut_fraction` .4 | −.0174 ± .0149 (3) | Not run |
| Median of 3 means at n8; drop least aligned quarter at n8 | −.0157 ± .0160 (3); −.0071 ± .0163 (7) | Not run |
| Blur σ 1; σ 2 on the gradient | −.0055 ± .0075 (5); −.0205 ± .0097 (0) | Not run |
| Gradient EMA .5; .8 | −.0123 ± .0099 (2); −.0461 ± .0119 (0) | Not run |

**Measured later,** a second sweep tried the schemes the first one skipped, on the code as shipped, and it is flat in the same place (Design D, [counts][admcutn], [layouts][admcut]). The first row is a calibration: the same 4 random crops, drawn by the sweep's harness instead of the repo's. It lands at −.0039 ± .016, so a two-seed arm inside about ±.015 means nothing.

| Scheme, ADM 128 | Seeds | Pooled change (prompts improved of 12) |
|---|---|---|
| 4 random crops, redrawn | 2 | −.0039 ± .016 (6) |
| 8; 16; 32; 64 cuts | 4 | −.0019 ± .009 (5); +.0052 ± .011 (6); +.0018 ± .007 (8); −.0002 ± .010 (4) |
| 16 cuts, ViT-B/32 + PE-Core-B-16 | 4 | +.0161 ± .009 (10), against +.0188 ± .015 for that ensemble at 4 cuts |
| Stratified position and size, 4; 16 cuts | 2; 4 | +.0049 ± .015 (7); +.0002 ± .010 (5) |
| 2 crops and their mirror images | 2 | +.0003 ± .009 (4) |
| Whole frame + 3 crops of 70 to 100% | 2 | −.0042 ± .009 (4) |
| Crop size shrinking through sampling; growing | 2 | −.0067 ± .010 (5); −.0123 ± .016 (4) |
| Whole frame + 7 crops of 44 to 70% | 2 | −.0137 ± .016 (3) |
| Whole frame + 3 crops of 44 to 70% | 2 | −.0174 ± .010 (1) |
| Whole frame only | 2 | −.0172 ± .010 (0) |
| Reflect padding; with stratifying | 2 | −.0223 ± .023 (2); −.0246 ± .015 (2) |

Two things come out of that table. The two-seed lead of 16 cuts, +.0131 ± .017, halved on four seeds, the same way the first sweep's leader shrank, and 16 cuts on the ensemble added nothing to the ensemble. And overview cuts did not earn their place. The whole frame alone loses, the usual mix of whole frame and small crops loses, and adding a whole frame to the repo's own crops only ties.

SD-Turbo at 512 was the real test of the premise, since a 224px crop there is 44% of the side and the guide sees native detail. The answer is the same (Design E, [scores][cutturboall]). Eleven layouts on seed 1 all sit inside ±.007 of the repo's crops, from +.0002 ± .005 for crops of 90 to 100% to −.0069 ± .011 for the whole frame plus 3 crops of 30 to 50%, and the Disco-style mix of whole frame and small crops is −.0032 ± .003 on 3 seeds (3 of 12). Cut count is flat too: −.0038 at n1, −.0017 at n16, and cut size .3 is −.0039 ([scores][turbotune]). Noise optimisation normalises its gradient too, so the capped-step explanation below may apply here as well, though that was not tested.

The paradox is that cutout gradients are noisy, and removing the noise still does not help. Two independent 4-cut gradients have a cosine of about .21, and repeatability tracks Spearman-Brown closely with single-cut agreement c = .060 (16 cells, 4 prompts × 4 guided steps, 1 sampler seed, 8 estimates per cell, no per-count interval) ([log][e3b]). Going from 4 to 32 cuts raises gradient transfer to other models about 1.7× (ViT-B/16 .080 to .138; PE-Core-B-16 .056 to .089), and the image score does not move. The likely explanation, not a proven one, has two parts. The sampler's cap fixes step size, so extra cuts only change direction. The U-Net's Jacobian then discards most raw gradient power: .60 to .69 of raw CLIP gradient power sits above .25 cycles per pixel, against .03 after the Jacobian at σ .997 and .35 at σ .33 (6 prompts, seed 0, 95% CI over prompts) ([e5 log][e5]). Extrapolating repeatability to infinite cuts still leaves transfer of only about .07 to .25, so most of the gradient is specific to the model. Crop averaging cannot remove that part, and that is the argument for changing the tower rather than the crops.

One mechanism did show up clearly. The repo's `x_in` blend adds a direct path that bypasses the U-Net, and late in the window it dominates (6 prompts, seed 0, 95% CI over prompts) ([e5 log][e5]):

| σ | Repeatability, through U-Net | Repeatability, blend | Direct path / U-Net path |
|---|---|---|---|
| .997 | .833 ± .094 | .830 ± .093 | .035 ± .005 |
| .673 | .505 ± .164 | .301 ± .117 | 1.91 ± .48 |
| .327 | .300 ± .107 | .186 ± .059 | 8.39 ± 1.05 |

Removing that path is the best-motivated single change here, yet pure x0 stayed unproven: both estimates are positive and both intervals include zero. It was also never tuned, and `-rate` may want to differ for it.

So the cutout algorithm should be changed in only a few ways, and most of the candidates can be dropped. Keep the repo's reproducible area pool, because no interpolator beat it. The smoother kernels raise gradient repeatability (.387 ± .116 for area, .507 ± .149 bicubic, 6 prompts, seed 0), but that gain did not reach the image. Offer `-cutn 16` as safe rather than better, at about 13 ms per image per cut for ViT-B/32 (RTX 2070 SUPER, median of 5 repeats, no interval) ([log][e2]). Do not add EMA across steps, gradient blur, heavy per-cut noise, smaller minimum crops, robust aggregation or low-discrepancy layouts. Latin hypercube raised repeatability at k ≥ 16 (+.032 ± .008 at k=16) but moved neither transfer nor images. Any warp would also need new work, since every bilinear warp, `grid_sample` and reflect pad tested has a nondeterministic CUDA backward and would break seed reproducibility without an index or fixed-point replacement ([script][e1]).

**Still untested,** the result now covers ADM 128 and SD-Turbo 512, with ViT-B/32 and its ensemble with PE-Core-B-16. ADM at 256 and 512 and PE-Core-B-16 alone stay unrun, and PE-Core-B-16 repeats far worse than ViT-B/32 (.120 against .213 at k=4, 16 cells each, 1 seed), so cut count could still matter there, though the ensemble's flat result at 16 cuts counts against it. Robust towers invert the premise, since their cuts already agree, though the pilot below found no image gain from them at n4. Disco's split backward, which accumulates CLIP gradients on `x_in` and then runs one U-Net backward, makes cut count a time cost and not a VRAM cost. On ViT-B/32 the batch limit is already about 330 cuts at 128px and 150 at 256px on 8 GB, and about 560 and 390 on 12 GB (estimates from per-cut allocation and README peaks; the VRAM column carries to the 4070, its time does not). The split is worth building for ensembles more than for cut count. Saliency-placed crops, tiled coverage and patch-token losses have no precedent in guidance and no local run. Latin hypercube's flat result suggests placement has little to work with at 128, so they stay speculative.

## Of four early wins, seed selection held up and dynamic thresholding did not

**Dynamic thresholding fixes clipping and loses score.** **Published,** Imagen's rule gives "significantly better photorealism and alignment" under strong guidance ([Imagen](https://arxiv.org/abs/2205.11487)). **Measured here** (256px, Design B, n=12), replacing the 1.2 clamp with quantile .995 took clipped pixels from 4.04% to .03% and mean saturation from .453 to .180. The judges did not object, but they did not reward it either: +.010 [−.011, +.036], +.015 [−.016, +.046] and +.008 [−.022, +.039], with 5, 7 and 7 of 12 prompts improved. It costs about four lines in `cgd/sampling.py`. By eye, a fox sat in a plausible field where the default drew hard graphic shapes, while a dragonfly went grey. The saved scores give a target for tuning: the unguided run has saturation .346, so .995 pushes colour well below what the U-Net makes on its own. The sweep should look for a quantile that lands near that value. This was run only at 256 and only on seed 0.

**Measured later,** the early tie did not hold (Design D, 128, [scores][admb]). Quantiles .995, .999 and .9995 scored −.0189 ± .014, −.0190 ± .016 and −.0255 ± .015, improving 2, 2 and 1 of 12 prompts. Clipped pixels fell from 2.8% to 0, but saturation landed at .20, .14 and .13, so no quantile reached the unguided target. Stretching contrast afterwards brought saturation back to .21 to .26 and did not bring the score back (−.0176 to −.0256). A new variant built for this, `-gl`, holds back only the part of CLIP's update that would push pixels out of range. At .99 it took clipped pixels to .6% with saturation .36, near the unguided level, and scored −.0162 ± .012 alone and +.0004 ± .016 on the ensemble (5 of 12, .5% clipped). So `-dt` and `-gl` ship as options, off by default. The judges still prefer the saturated image, and whether a person does is untested.

**A cascade is the cheap route to 512, and it only ties.** **Published,** ADM's cascade halves FID at 512 (ADM-G 7.72, ADM-G+ADM-U 3.85), and ADM guides only the low-resolution model ([ADM](https://arxiv.org/abs/2105.05233)). **Measured here,** that holds for CLIP guidance with a GAN doing the upscale ([method and data][cas]):

| Pipeline, 512px, Design B, n=12 | Time, s | Peak GiB per stage | ViT-B/16 vs direct | ViT-H vs direct | SigLIP2 vs direct | Clipped % |
|---|---|---|---|---|---|---|
| Direct 512 | 18.45 | 4.08 | .295 (mean) | .282 (mean) | .107 (mean) | 6.5 |
| 128 + Real-ESRGAN x4plus | 3.47 + .06 | 2.33, .16 | −.002 [−.020, +.015] | +.029 [+.003, +.053] | +.020 [+.004, +.034] | 9.5 |
| 128 + HAT (ImageNet pretrain) | 3.47 + .38 | 2.33, .70 | +.016 [+.002, +.031] | +.021 [−.003, +.043] | +.018 [+.001, +.034] | 5.0 |
| 128 + ADM upsampler, unguided | 3.47 + 11.82 | 2.33, 2.13 | +.010 [−.005, +.025] | +.020 [−.012, +.046] | +.014 [−.008, +.034] | 4.9 |
| 128 + ADM upsampler, guided | 3.47 + 23.03 | 2.33, 4.06 | +.012 [−.003, +.026] | +.021 [−.010, +.046] | +.015 [−.007, +.034] | Not recorded |

Times are sampling only, so they exclude model load, and the second load in a cascade was not measured. Guiding the upsampler doubled its time for +.001 to +.002 at 512 (guided minus unguided, intervals at or spanning zero), so skip it. At 256 the same term gave +.016 [+.008, +.025], +.013 [+.007, +.019] and +.007 [+.002, +.013], which is real but small. The cascade does not work from 64: 64→256 loses to direct 256 by −.068 [−.098, −.038], −.058 [−.097, −.018] and −.043 [−.072, −.012], because the base image sets the score. These judges cannot see 512px detail, so whether the ADM upsampler's extra 11.8 s buys visible detail needs a human look. x4plus is BSD-3-Clause and 67 MB, and at 512→2048 it took .85 s and 2.03 GiB without tiling,, so it shipped as `-up`. It does raise clipped pixels from 6.5% to 9.5%.

**Measured later,** the lead did not repeat on seed 1 through the shipped `-up` ([scores][casc1], [script][cascpy]). Against direct 512, 128 + x4plus scored −.0135 ± .028 pooled (6 of 12), with ViT-B/16 .290 against .322, ViT-H .303 against .304 and SigLIP2 .114 against .121. The held-out rating fell from 5.32 to 4.83 and clipped pixels rose from 4.1% to 10.4%. So it is a tie at about a sixth of the cost, 2.9 + .17 s against 18.8 s and 2.33 against 4.08 GiB, and `-up` stays optional rather than replacing direct 512.

**Seed selection works on the CPU and generalises to judges that did not pick.** **Published,** reranking has a long record: DALL-E reports clear gains up to 32 samples and diminishing returns after that ([DALL-E](https://arxiv.org/abs/2102.12092)). Ma et al. warn of "verifier hacking", where search converges to one verifier's biases, and find ensembles generalise best ([Ma et al.](https://arxiv.org/abs/2501.09732)). **Measured here** (12 prompts, 8 seeds, 64px, ViT-H as selector), best of 8 gained +.021 [+.004, +.039] on ViT-B/16 and +.022 [+.007, +.039] on SigLIP2 over seed 0. Best of 8 by ViT-B/16 then the unguided upsampler gained +.037 [+.011, +.063] on ViT-H and +.017 [+.002, +.032] on SigLIP2. It still sat below direct 256 by −.021 [−.051, +.010] and −.026 [−.055, +.004].

Seed 0's saturation raised a concern that those gains were an artifact of a bad baseline, so this report recomputed them from the saved scores against the mean of the candidate pool. The concern does not hold: seed 0's scores are ordinary (−.001 to +.012, all intervals span zero) even though its saturation is .559 against .334 across all 8 seeds. Against a typical seed, the rank average of two judges scored by the third gains +.020 [+.011, +.031] to +.047 [+.031, +.063] from a pool of 8, improving 10 to 12 of 12 prompts, and +.015 [+.005, +.025] to +.029 [+.014, +.043] from a pool of 4. Judges correlate only .68 to .81 across seeds, which is why a two-judge ensemble is the safer selector. It costs about 1.25 s per 64px candidate and no VRAM for judging. Best of 4 at 128 would take about 14 s, an estimate, since that size was not run. **Published,** Progressive Seed Pruning goes further, dropping seeds on x̂₀ at 25% and 50% of sampling at no added model calls, in latent CFG models only ([PSP](https://arxiv.org/abs/2607.21591)). Whether x̂₀ predicts final score on this sampler is **untested**.

**Measured later,** the shipped `-best N` repeats this at 128 and on SD-Turbo, with ViT-B/16 and SigLIP2 picking and ViT-H scoring alone, against the mean of the candidates ([script][bestpy], [scores][bestscores]). ADM best of 4 gained +.0154 ± .0070 (11 of 12 prompts), and best of 4 on the ensemble +.0245 ± .0114 (12 of 12). On SD-Turbo best of 3 gained +.0053 ± .0052 (8 of 12), and +.0061 ± .0057 (7 of 12) with the aesthetic term, so reranking matters less on a model whose seeds are already good. It costs N times the run.

**The secondary model is the strongest VRAM lever.** **Measured here,** running the U-Net without a gradient and taking guidance through Crowson's 13.8M-parameter secondary model cut guided 256 from 8.02 s and 5.33 GiB to 5.80 s and 1.92 GiB, at −.011 [−.026, +.004], +.008 [−.013, +.030] and +.005 [−.008, +.019] (Design B, n=12) ([method][cas]). At 512 it saved 1.5 GiB and 8 s but trended lower, −.012 [−.025, −.001], −.015 [−.058, +.022] and −.012 [−.036, +.010], so it belongs at 256. On 8 GB that leaves about 5.7 GiB free at 256 for heavier towers, by arithmetic rather than measurement. **Measured later,** on more seeds it is a tie where it matters and a loss where it does not: −.0048 ± .023 (6 of 12, 2 seeds) at 256 with 1.91 GiB and 5.8 s, −.0090 ± .022 (6 of 12, 1 seed) at 512 with 2.64 GiB and 11.2 s, and −.0191 ± .022 (2 of 12, 4 seeds) at 128, where it saves only 1 GiB ([256][s256], [512][s512]). Disco's own weight hosts have rotted. The working mirror is a personal [HF Space](https://huggingface.co/spaces/huggi/secondary_model_imagenet_2.pth/resolve/main/secondary_model_imagenet_2.pth) whose sha256 matches Disco's pin, so the weights need re-hosting or pinning by hash.

## Robust towers failed their first real test; the ViT-B/32 + PE-Core-B-16 ensemble replicated and cross-entropy did not

**Published,** no paper shows a large gain from ensembles, and the closest statement is that three ViT-B/32 models "(marginally) improve" quality while "a single model is sufficient" ([DAS](https://arxiv.org/abs/2502.07753)). Robust towers have stronger evidence, though only off diffusion: in CLIPDraw, R-precision was 1% with plain CLIP without augmentation, 70% with augmentation, and 72% for the adversarially fine-tuned model with it ([CLIPAG](https://ar5iv.labs.arxiv.org/html/2306.16805)). No paper uses FARE or TeCoA for diffusion guidance, and off-manifold robustness work warns that excessive robustness makes gradients worse again ([Srinivas et al.](https://arxiv.org/html/2305.19101v2)).

**Measured here,** robust towers change gradient quality more than anything else in these notes, and they do it before any image exists (2 images × 3 prompts, 6 pairs, 95% half-width across pairs, CPU, fp32; 1 noise draw and 2 cutout seeds per pair) ([script][gq]):

| Tower | Cross-cutout cosine | High-frequency share |
|---|---|---|
| OpenAI ViT-B/32 (default) | .091 ± .019 | .611 ± .041 |
| PE-Core-B-16 | .094 ± .071 | .591 ± .015 |
| laion ConvNeXt-B-w | .551 ± .027 | .467 ± .128 |
| FARE4 ViT-B/32 | .743 ± .042 | .071 ± .030 |
| FARE4 ConvNeXt-B-w | .858 ± .027 | .160 ± .055 |

Two details matter. Clean ConvNeXt already agrees across cutouts at .551, so architecture accounts for part of the effect without any adversarial training. And clean towers' gradients barely overlap (cosine −.021 to .160, all 28 pairs, 1 image and 1 prompt, no interval), which is why averaging towers might remove what averaging crops cannot. In a transfer proxy with no diffusion and no U-Net (4 pairs, 1 seed sequence, no interval, so ignore differences of .01), the held-out rise was .024 for the default, .044 for PE-Core-B-16, .050 for FARE4 ConvNeXt and .062 for ViT-B/32 + ViT-B/16 + FARE4 ConvNeXt, and 12 of 16 planned guides finished. This proxy can disagree with real sampling, and the cutout results show exactly that kind of gap between gradient metrics and images.

A warning sits in the data. FARE4's low high-frequency share looks like the gradient blur that lost at 128 (−.0205 ± .0097, 0 of 12), so "smooth gradient" is not automatically "good gradient". Blur discards signal, whereas a robust tower may concentrate it, but only a pipeline run can separate the two. The pilot below ran that test. FARE repos load through open_clip with `hf-hub:` names and were confirmed by a 4-step CPU smoke test on 1 prompt, which proves only the code path ([losses notes](../research_notes/CLIP%20guided%20diffusion%20quality%20improvements/losses_and_ensembles.md)). ConvNeXt has no attention to patch for determinism. OpenAI RN towers need fp16 for the same reason, since their attention pool was nondeterministic at fp32 on 5 repeats.

**Measured here, after the notes closed,** the real sampler reversed the prediction (Design C). The proxy's best guide, ViT-B/32 + ViT-B/16 + FARE4 ConvNeXt, was not itself an arm, but its two nearest relatives were, and FARE4 ConvNeXt alone was the weakest of the FARE arms on two judges.

| Arm, 128px, against base | ViT-B/16 | ViT-H | SigLIP2 | Output high-frequency share (base .0425) |
|---|---|---|---|---|
| ViT-B/32 + PE-Core-B-16 | +.012 ± .015 (7) | +.022 ± .023 (8) | +.037 ± .024 (10) | .0418 |
| ViT-B/32, cross-entropy against ImageNet text | +.007 ± .017 (7) | +.022 ± .022 (9) | +.040 ± .024 (10) | .0539 |
| Disco's ViT-B/32 + ViT-B/16 + RN50 | +.008 ± .013 (7) | +.011 ± .032 (8) | +.030 ± .019 (10) | .0548 |
| PE-Core-B-16 alone | −.006 ± .017 (4) | +.025 ± .025 (8) | +.025 ± .022 (8) | .0316 |
| ViT-B/32, TV 50 instead of 150 | +.003 ± .015 (8) | +.002 ± .026 (8) | +.015 ± .009 (10) | .0503 |
| ViT-B/32, TV 0 | −.013 ± .016 (3) | −.018 ± .033 (4) | −.008 ± .018 (8) | .0400 |
| CLIPAG ViT-B-32 | −.008 ± .016 (5) | +.010 ± .021 (8) | +.013 ± .020 (7) | .0242 |
| laion ConvNeXt-B | −.008 ± .021 (5) | +.008 ± .031 (6) | +.018 ± .027 (8) | .0334 |
| FARE4 ViT-B-32 | −.006 ± .013 (5) | +.004 ± .020 (4) | +.008 ± .023 (8) | .0161 |
| FARE4 ViT-B-16 | −.017 ± .019 (2) | −.015 ± .027 (4) | +.006 ± .023 (7) | .0159 |
| FARE4 ConvNeXt-B | −.021 ± .026 (6) | −.020 ± .042 (7) | +.006 ± .023 (7) | .0208 |
| ViT-B/32 + FARE4 ConvNeXt | −.005 ± .017 (5) | −.004 ± .023 (8) | +.015 ± .012 (10) | .0255 |
| FARE4 ViT-B/32 + ViT-B/16 + ConvNeXt | −.010 ± .019 (3) | +.006 ± .027 (6) | +.013 ± .020 (7) | .0162 |
| ADM classifier scale 1 with CLIP | +.001 ± .012 (7) | +.001 ± .041 (6) | +.027 ± .023 (8) | .0456 |
| ImageNet class only, no CLIP | −.060 ± .021 (0) | −.078 ± .032 (1) | −.035 ± .024 (4) | .0293 |

Parenthesised counts are prompts improved of 12. Treat the top three as leads, not results: only SigLIP2 clearly separates them from base, ViT-H only touches zero on cross-entropy, ViT-B/16 never does, and SigLIP2 favours most arms in this table. Robust towers lose nothing dramatic, but they gain nothing either, and they halve output high-frequency share. That matches the warning about blur: a low-frequency gradient is not a better gradient. Whether those smoother images look better to a person is **untested**, since the judges cannot see it. FARE4 ConvNeXt's .858 cross-cut agreement therefore predicted nothing about the image, the same gap between gradient metric and output that the cutout sweep showed.

**Measured later,** fresh seeds decided the two leaders in opposite directions (Design D, seeds 1 to 4, [scores][admb]).

| Arm, against base | Pooled change (prompts improved of 12) | ViT-B/16, ViT-H, SigLIP2 (base .293, .289, .109) |
|---|---|---|
| ViT-B/32 + PE-Core-B-16 | +.0188 ± .015 (10) | .308, .313, .127 |
| Ensemble + ER-SDE | +.0169 ± .008 (12) | .307, .311, .125 |
| Ensemble + cross-entropy | +.0061 ± .018 (4) | .295, .297, .117 |
| PE-Core-B-16 alone | +.0008 ± .015 (6) | .288, .296, .110 |
| Cross-entropy alone | −.0102 ± .015 (3) | .283, .275, .102 |

The ensemble is the only guide change that replicated, and only as a pair: PE-Core-B-16 alone ties. Cross-entropy, the pilot's surprise, trended the wrong way on its own and drags the ensemble down, so `-loss contrastive` stays an option and nothing more. It also failed on SD-Turbo, −.0085 ± .006 on seed 1. Those failures make the point of the rerun: it was picked from 27 arms, and selection alone can produce a +.02. The ensemble's 5.27 GiB peak in the pilot was **not valid**, because the harness kept earlier arms' towers on the card. Measured alone on the card, it peaks at 2.69 GiB against 2.33 at 128, 5.64 against 5.28 at 256, and 4.43 against 4.08 at 512, and adds about 3 s per run at every size, load included (5.9 s against 2.9 s, 10.7 against 7.8, 21.6 against 18.8). The gain holds up at the larger sizes on fewer seeds: +.0150 ± .014 (7 of 12, 2 seeds) at 256 and +.0124 ± .015 (9 of 12, 1 seed) at 512 ([256][s256], [512][s512]). One run took 25.5 s per image at 128 because DNS was down, so open_clip's Hub check retried for about 20 s on each load before using the cache. That is the one known cost of running these towers offline.

Three smaller results are useful. TV 50 ties the default, and TV 0 trends lower, so `-tvs` is not a lever. Templates, negative text, mean embedding and multiscale TV all tie within noise.

**The aesthetic term trades score for rating, and rating cutouts makes the trade better.** **Measured later,** the pilot's free aesthetic term did not hold on fresh seeds. On ADM the whole frame at .1 raised the held-out ViT-B/16 rating from 4.79 to 5.18 and scored −.0410 ± .015 (0 of 12, [scores][admaes]). Rating 4 random cutouts instead, a variant built for this repo, did better on both axes: .05, .1 and .2 reached 5.27, 5.52 and 5.67 at −.0125 ± .008, −.0215 ± .009 and −.0466 ± .012, while clipped pixels rose from 2.8% to 3.8%, 6.6% and 6.7% ([scores][aescut]). SD-Turbo shows the same shape from a much higher base (Design E, [whole frame][aesturbo], [cutouts][aespc]):

| SD-Turbo 512, 3 seeds | Held-out rating | Guide's own rating | Pooled change | Saturation |
|---|---|---|---|---|
| No aesthetic term | 6.35 | 6.30 | | .36 |
| Whole frame .05 | 6.93 | 7.96 | −.0048 ± .003 (2) | .39 |
| Whole frame .1 | 6.93 | 8.17 | −.0065 ± .006 (4) | .41 |
| Whole frame .2 | 7.01 | 8.33 | −.0134 ± .007 (2) | .41 |
| Cutouts .05 | 6.90 | 7.30 | −.0022 ± .004 (3) | .42 |
| Cutouts .1 | 7.12 | 7.65 | −.0106 ± .008 (1) | .45 |
| Cutouts .2 | 7.24 | 7.82 | −.0124 ± .006 (2) | .50 |

The whole frame pushes the guide's own ViT-B/32 rating much further while the held-out ViT-B/16 rating gains less, so part of that gain looks specific to the guide, which is the failure cutouts were invented to prevent. That reading is an inference from these two columns. The intervals overlap between neighbouring rows, so the case rests on the pattern across scales and on ADM agreeing, not on any single comparison. Cutouts at .05 became the `-bb` default, with its score cost indistinguishable from zero, and ADM stays at 0 because every scale there loses score and adds clipping. By eye (seeds 1 and 2, 6 prompts) .1 and .2 often show better light and composition than .05, and at .2 one wizard hat shrank to a party hat, so the scale is a real choice and only a human comparison can set it.

One arm is **invalid**: laion ViT-B-32 scored −.060, −.083 and −.052, as bad as no CLIP. Its name and OpenAI's `ViT-B/32` both sanitise to the same `imagenet_text_features_ViT-B-32.pt` in `imagenet_text_features()`, and the pilot's cache directory holds only that one file, so it very likely chose classes with the wrong model's features. This was read from code and file listings and not rerun. The default repo is not affected because `ViT-B-32` is not in `OPEN_CLIP_MODELS`, but the collision needs fixing before anyone adds that entry.

**Published,** realism guidance on ADM has the best numbers on these exact checkpoints, none of them under CLIP or at this step count. ADM's classifier reaches FID 5.44 at 256 with 25 DDIM steps ([ADM](https://arxiv.org/abs/2105.05233)). PAG takes conditional ADM 256 from 10.94 to 6.32 and unconditional from 26.21 to 16.23, at 250 steps with no training ([PAG](https://arxiv.org/abs/2403.17377)). ER-SDE takes classifier-guided ADM 256 from 5.47 to 4.62 at 30 NFE against the repo's solver family ([ER-SDE](https://arxiv.org/abs/2309.06169)). A sine-shaped guidance schedule moves FID from 5.57 to 5.24, and its CLIP improvement appears only as a plot over a few prompts ([EluCD](https://arxiv.org/abs/2310.11311)). ICG suits ADM because it needs no null token ([ICG](https://arxiv.org/abs/2407.02687)).

**Measured later,** the free swaps and realism guidance were all built and none moved the score on ADM (Design D, [scores][admb]). PAG reproduces at 30 calls with CLIP as a steady loss: −.0127 ± .018 at scale .5, −.0126 ± .009 at 1 and −.0202 ± .011 at 3. ER-SDE alone is a tie at +.0036 ± .014 (8 of 12), though it is the steadiest partner for the ensemble (12 of 12 above). A sine schedule scored −.0064 ± .009 and a rising one −.0111 ± .012. Showing CLIP only x̂₀ scored +.0024 ± .009, and removing just the gradient through the noisy image −.0041 ± .011, which closes the direct-path idea the cutout section left open. Conditioning on the second-closest class was −.0131 ± .035, and class contrast −.0072 ± .011 at .5. The arms named `pag` and `contrast` in the score file are invalid, from a `no_grad` bug in that first version, and the figures here are the `pag2` and `contrast2` reruns after the fix.

Other guidance formulas are a weaker bet. No method has evidence at 30 calls, the published ImageNet CLIP runs behind FreeDoM and Shen spend about 379 and 559 calls, and on the repo's arithmetic its cap already pushes 2 to 4 times harder per logSNR than those methods, so magnitude is not the open lever. Splitting methods independently support the repo's choice to keep guidance first-order and out of the multistep history ([Wizadwongsa et al.](https://arxiv.org/abs/2301.11558)).

**Measured here,** the classifier ran and the judges did not reward it. `create_classifier` is broken in the crowsonkb fork the repo depends on and needs two fixes; with those patched at runtime, `128x128_classifier.pt` loaded strictly and ran. Added to CLIP on the fast sampler, at 3.6 s per image, scale 1 tied on two judges (table above) and lowered the held-out aesthetic score by −.383 ± .340 (3 of 12 prompts up). With CLIP removed, it added nothing to class conditioning on the judges (−.061 ± .024, −.081 ± .039, −.046 ± .028). Those judges measure caption match, not ADM's claimed realism gain, so no FID-style conclusion follows, and a human look is still owed. APG's orthogonal update, the free direction filter that looked promising on paper, doubled clipped pixels at 256 (4.04% to 8.41%; +.001 [−.010, +.014], +.006 [−.009, +.021], −.004 [−.020, +.009]; Design B, n=12).

**Untested,** text-side changes are cheap and have only indirect support. Template ensembling evidence is recognition accuracy, not generation ([CLIP](https://arxiv.org/pdf/2103.00020)), and CLIPDraw found negatives "often negligible" ([CLIPDraw](https://ar5iv.labs.arxiv.org/html/2106.14843)). The most interesting is replacing the text target with a prior-sampled image embedding. In unCLIP, the text embedding passed zero-shot gave FID 16.55 against 7.99 for the full stack ([unCLIP](https://ar5iv.labs.arxiv.org/html/2204.06125)), and MIT-licensed [nousr/conditioned-prior](https://huggingface.co/nousr/conditioned-prior) has a ViT-B/32 checkpoint that matches this repo's default model. That number concerns a decoder, not guidance, and the prior has not been tested here.

## Noise optimisation on one-step models works and fits 8 GB; FLUX.2 klein 4B stayed unrun

Attaching CLIP to a text-conditional model changes its job. **Published,** GLIDE's result still applies, so CLIP guidance is not a better conditioner than CFG. Its evidenced role is correcting attributes through noise optimisation: ReNO optimises initial noise for 50 steps through a one-step model and raises GenEval on SD-Turbo from .49 to .62, SDXL-Turbo .54 to .65, HyperSDXL .56 to .65 and FLUX-schnell .64 to .72, and SD-Turbo + ReNO wins up to 77% of pairwise comparisons against plain SD-Turbo ([ReNO](https://ar5iv.labs.arxiv.org/html/2406.04312)). ReNO used several rewards including CLIPScore, not CLIP alone. Its A100 peaks of 15 to 50 GB involve larger rewards and are not comparable with a single small CLIP at fp16. UPainting reports that CLIP guidance combined with CFG gives higher quality, but its FID compares different models ([UPainting](https://arxiv.org/abs/2210.16031)). Universal Guidance's .2632 against .2818 is often misread: it compares CLIP-guided unconditional SD with text-conditional SD, not CLIP plus CFG against CFG ([Universal Guidance](https://proceedings.iclr.cc/paper_files/paper/2024/file/dfbb3d1807b21dadee735eb75069ada4-Paper-Conference.pdf)). Per-step CLIP guidance on FLUX.2, Z-Image or Sana has no published evaluation.

**Measured later,** ReNO's recipe ported to CLIP, built as `-bb`, is the best thing this repo now makes. It follows the CLIP gradient on one-step noise for 50 steps through the tiny decoder, then renders the best noise once with the full VAE. These are on-card numbers from the RTX 2070 SUPER for the ensemble guide and the aesthetic term, alone on the card, with memory peaks valid for the whole run ([scores][backbones], [script][aes]).

| Backbone, size | Plain, `-iters 0` | With CLIP and aesthetic | Plain against the reference | With CLIP against the reference |
|---|---|---|---|---|
| SD-Turbo 512 | 6.4 s, 2.47 GiB | 27.2 s, 3.21 GiB | −.0147 | −.0057 |
| Vega 768 | 9.5 s, 2.90 GiB | 36.0 s, 3.60 GiB | −.0198 | −.0102 |
| Vega 1024 | 9.9 s, 3.93 GiB | 49.8 s, 4.65 GiB | Not scored | Not scored |
| Lightning 768 | 10.4 s, 6.29 GiB | 57.6 s, 6.99 GiB | −.0146 | −.0177 |
| Lightning 1024 | 10.2 s, 7.33 GiB | Out of memory on 8 GB | Not scored | Not run |

Times are the wall clock of one `cgd` run, startup and loading included, with the weights already on disk ([times][wall], [script][wallsh]). Peaks are from the same code without the CLI ([plain][plainlog]; with CLIP, [Vega 768][pcv768], [Vega 1024][pcv1024], [Lightning 768][pcl768]). Scores are seed 1 only, pooled, against SD-Turbo 512 with CLIP and no aesthetic term, and were taken before the rating moved to cutouts, with the whole frame at .1. Moving it changed no peak.

Three points came out of building it. **The tiny decoder is as good as the full VAE:** +.0010 ± .003 (8 of 12, Design E) for half the time and about 1 GiB less, so it became the default and the 2 prompt test below is superseded. **CLIP correction helps SD-Turbo and little else:** +.0110 ± .005 on Turbo (12 of 12 prompts better than plain), but on seed 1 at 768 Vega gains about +.010 and Lightning loses −.003 ± .01, where its plain output is already the sharpest by eye. **Settings barely matter:** 100 steps adds +.0024 ± .002 (34.7 s), 200 steps +.0057 ± .004 on one seed with a visibly adversarial look, and everything else tried on seed 1 ties or loses (learning rate 2.5, penalty .1, 2 or 4 render steps, one to 16 cuts). Six ideas invented for this path also tie on seed 1: optimising the text embedding too (+.0041 ± .008), picking the best of 8 or 32 noises first (+.0021 ± .007, −.0001 ± .006), a multi-scale noise step (+.0017 ± .005, −.0010 ± .009), and optimising through 2 model steps (−.0041 ± .010) ([scores][novel]).

The rest of this section is the estimate made before anything ran, kept because SD 1.5, PixArt-Sigma, Sana, Z-Image and klein have no better numbers. **Estimated, with limits.** The table gives peak GiB for one guided step of four ViT-B/32 cutouts, as `tiny_ckpt` (through the denoiser, tiny decoder, checkpointed) and `full` (denoiser and full VAE). Each cell is one run at seed 0, so no interval applies. G is on the RTX 2070 SUPER, C is counted from a CPU allocator timeline, and P is projected from a fit. Checks: CPU counting matched the card within 5% on SD 1.5, and the harness read about .2 GiB high on ADM. Only SD 1.5 is G. Usable memory is 7.62 GiB on the 2070 SUPER (measured) and about 11.6 GiB on the 4070 (inferred; nothing ran there). Fused attention is not in these numbers, and the plain attention needed for seed reproducibility would raise them by an unmeasured amount ([benchmark files][bench]).

| Rank | Model | px | Guided peak GiB, `tiny_ckpt` / `full` | Source | 8 GB | 12 GB | License, gating | Attachment |
|---|---|---|---|---|---|---|---|---|
| 1 | [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo) | 512 | 2.79 / ~5.9 | C, P | Yes | Yes | Stability Community, ungated. Commercial use needs registration; grant ends above USD 1M revenue | Noise optimisation through one step, or per step on its own timesteps |
| 2 | [Segmind-Vega](https://huggingface.co/segmind/Segmind-Vega) + [VegaRT](https://huggingface.co/segmind/Segmind-VegaRT) | 1024 | ~3.6 / over 12 | P | Tiny decoder only | Tiny decoder only | Apache-2.0, ungated | eps, maps to `sampling.py` |
| 3 | [FLUX.2 klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | 1024 | ~6.4 fp8 storage, ~10 bf16 / not estimated | P | fp8 only, experiment | Yes, bf16 | Apache-2.0, ungated | Flow. 4 steps, so noise optimisation or one corrective step |
| 4 | [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) + [Lightning](https://huggingface.co/ByteDance/SDXL-Lightning) | 1024 | ~7.1 / over 12 | P | Under .5 GiB margin; treat 768 as safe | Yes | OpenRAIL++, ungated | Tied to the checkpoint's timesteps |
| 5 | [SD 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) + [LCM-LoRA](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5) | 512 | 2.85 / 5.92 | G | Yes | Yes | OpenRAIL-M, ungated | eps. Cached here |
| 6 | [PixArt-Sigma](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-512-MS) | 512 | 2.28 / ~5.5 | C, P | Yes | Yes | OpenRAIL++, ungated | eps with learned sigma, T5 on CPU |
| 7 | [Sana Sprint](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers) 0.6B, 1.6B | 1024 | No tiny decoder / ~7.5, ~9.4 with a custom checkpoint wrapper | P | 0.6B only, thin | Yes | Apache-2.0 tag, ungated | `trigflow`, a third parameterisation |
| 8 | [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | 1024 | ~8.6 fp8 storage (bf16 over 12) | P | 512 only, ~7.5 | fp8 only | Apache-2.0, ungated | Flow, list inputs |
| 9 | [EDM2-S](https://raw.githubusercontent.com/NVlabs/edm2/main/README.md) | 512 | Not measured | None | Unknown | Unknown | CC BY-NC-SA | Same class + CLIP recipe, needs a decoder |
| 10 | [cc12m_1_cfg](https://huggingface.co/multimodalart/crowsonkb-v-diffusion-cc12m-1-cfg) | 256 | ~5.5, no decoder | Estimate | Likely | Yes | MIT tag, third-party mirror, hash unchecked | v prediction, conditions on CLIP embeddings |

Three findings change how to build this. **The full VAE, not the U-Net, is the expensive backward pass:** on SD 1.5 at 512, adding U-Net backward to a tiny decoder adds .7 GiB, while swapping TAESD for the full VAE adds 2.5 GiB. **Weights then set the limit:** on the tiny paths the peak is weights plus 1.5 to 2 GiB at 1024, so denoiser weights under about 5.2 GiB fit 1024 on 8 GB, and fp8 storage is what brings the 4B and 6B transformers into range. **bf16 is emulated on the RTX 2070 SUPER,** whose native check returns False, so U-Nets should run fp16 there and klein 4B on that card stacks three untested risks. Gated or non-commercial exclusions include SDXL-Turbo, official SD 3.5 and FLUX.1 repos, FLUX.3 klein, Stable Cascade and DeepFloyd IF. Qwen-Image, HiDream and AuraFlow do not fit.

An earlier and much smaller test of whether cheap gradients steer ran on real SD 1.5, TAESD and ViT-B/32 at fp32 on CPU, DDIM 20, CFG 7.5, guided steps 3 to 14, rate .2 (2 prompts, 1 seed each, 10 images, df = 1, the run stopped at 2 of 4 prompts):

| Path | Guide delta | Held-out PE-Core-B-16 delta |
|---|---|---|
| Through U-Net, full VAE | +.034 ± .151 | −.003 ± .154 |
| Through U-Net, TAESD | +.017 ± .121 | −.008 ± .201 |
| x0 only, full VAE | +.009 ± .038 | +.006 ± .122 |
| x0 only, TAESD | +.011 ± .044 | +.011 ± .203 |

Every interval includes zero. All four paths raised the guiding score on both prompts, and held-out deltas changed sign between prompts on all four. The Design E comparison above replaces this test for noise optimisation. Per-step guidance through a tiny decoder stays unmeasured.

Integration is not a pipeline callback, since `__call__` runs under `no_grad` in diffusers. It needs a bare denoiser and decoder plus a per-family adapter returning x0, and `sampling.py` lines 142-143, the `X0_CLAMP`, `NoiseLevels` and timestep scaling all assume ADM ([backbone notes](../research_notes/CLIP%20guided%20diffusion%20quality%20improvements/diffusion_backbones.md)). Per-step guidance would need `-rate` and the window retuned per family. Noise optimisation avoided all of that, because it has no solver, and that is why it was built first. Seed reproducibility needed plain attention processors on the U-Net and decoder, and cuDNN's deterministic flag on top ([tests](../test.py)): the two together give byte-identical images. The estimate of 25 s for 50 iterations turned out to be 17.5 s with the tiny decoder.

Pixel-space upgrades keep the project's character but not much else. EDM2-S is the one direct replacement inside the same class-conditional recipe, with a license cost, and it stays limited to ImageNet classes. cc12m_1_cfg replaces class selection with real CLIP conditioning at about ADM 256's cost, but it is an unevaluated 2021 model. The repo's ADM path should stay as the option with no new dependency.

## Thirteen expected results failed here, and most planned runs happened later

These are the plain contradictions. Most concern long, free-running or latent settings that differ from a capped 14-step pixel window, so the literature is not wrong so much as not transferable. The judges also measure caption match, not realism, so the last four rows say the claimed gain does not reach this score, not that the published FID is wrong.

| Published or community claim | Local result | Reading |
|---|---|---|
| More cutouts help; notebooks use 32 to 64 ([Disco](https://raw.githubusercontent.com/alembics/disco-diffusion/main/disco.py)) | n32 +.0006 ± .0159, Design A, 128 | Disco's split backward is not the "most valuable missing piece" for score at 128 |
| Overview cuts early, inner cuts late ([Disco](https://raw.githubusercontent.com/alembics/disco-diffusion/main/disco.py)) | +.0102 ± .0141, then +.0001 ± .0170 | Whole frame + small crops also lost at Turbo 512 (−.0032 ± .003, Design E), where the crops do span 44 to 100% of the side |
| Augmentation is essential: 1% to 70% ([CLIPAG](https://ar5iv.labs.arxiv.org/html/2306.16805)) | Flip −.0035 ± .0126, grey +.0050 ± .0114, jitter +.0035 ± .0098, Design A | A stroke optimiser with thousands of free steps is not this sampler. Warps remain unrun |
| Momentum helps guidance: accuracy 55.67 to 99.99 at β .99 ([momentum study](https://ar5iv.labs.arxiv.org/html/2507.00687)), AdamCG 51.0 to 68.1 ([AdamCG](https://arxiv.org/html/2603.16797v2)) | EMA .5 −.0123 ± .0099; EMA .8 −.0461 ± .0119, 0 of 12 | Fits APG's finding that negative momentum wins on CFG ([APG](https://ar5iv.labs.arxiv.org/html/2410.02416)). Over 14 fast-changing steps an EMA goes stale. Adam itself was not run |
| Blurring gradients improves transfer: 13.0 to 37.4% ([TI](https://ar5iv.labs.arxiv.org/html/1904.02884)); a low-pass is the suggested fix for adversarial detail | Blur σ 2 −.0205 ± .0097, 0 of 12; output high-frequency share halved | The U-Net Jacobian is already the low-pass. Result is for 128 only |
| Noise around x0 smooths guidance ([LGD](https://proceedings.mlr.press/v202/song23k.html), [Shen](https://arxiv.org/abs/2403.12404)) | Noise .03 −.0099 ± .0144; .1 −.0149 ± .0135 | The blend's shared noise, .12 peak on [0, 1], is already larger. LogMeanExp aggregation was not run |
| Adversarially robust CLIP gives far better guidance: CLIPDraw R-precision 1% to 72% ([CLIPAG](https://ar5iv.labs.arxiv.org/html/2306.16805)), and local cross-cut cosine .091 to .858 | FARE4 ConvNeXt −.021 ± .026, −.020 ± .042, +.006 ± .023; three-FARE ensemble −.010 ± .019, +.006 ± .027, +.013 ± .020, Design C | The local proxy ranked this ensemble family first, and images did not follow. Excessive robustness may be hurting ([Srinivas et al.](https://arxiv.org/html/2305.19101v2)); FARE at lower epsilon was not run |
| Smoother resampling into CLIP should beat box upsampling (v-diffusion and k-diffusion practice) | Bicubic −.0120 ± .0125, lanczos3 −.0137 ± .0127 | Keep the area pool at 128. Untested with PE-Core, whose own preprocessor is bilinear |
| APG's projection cuts saturation, SD2.1 .36 to .27 ([APG](https://ar5iv.labs.arxiv.org/html/2410.02416)) | Clipped 4.04% to 8.41% at 256, Design B | A CFG model difference is not a CLIP gradient. Rescaling and momentum parts were not run |
| Dynamic thresholding gives "significantly better photorealism and alignment" ([Imagen](https://arxiv.org/abs/2205.11487)) | .995 −.0189 ± .014, .999 −.0190 ± .016, Design D | Clipping goes, and so does saturation and score. A blind human check is the missing test |
| PAG takes ADM 256 from FID 10.94 to 6.32 ([PAG](https://arxiv.org/abs/2403.17377)) | Scale .5 −.0127 ± .018, 1 −.0126 ± .009, 3 −.0202 ± .011, Design D | At 250 steps with no CLIP in the source. Here it spends model calls and pulls away from the caption |
| ER-SDE takes guided ADM 256 from 5.47 to 4.62 at 30 NFE ([ER-SDE](https://arxiv.org/abs/2309.06169)) | +.0036 ± .014 (8 of 12), Design D | A tie alone. It does steady the ensemble, +.0169 ± .008, 12 of 12 |
| A sine schedule moves FID from 5.57 to 5.24 ([EluCD](https://arxiv.org/abs/2310.11311)) | Sine −.0064 ± .009, rising −.0111 ± .012, Design D | Keeping total guidance fixed, a flat rate stays best inside this window |

FreeU belongs with these as the sharpest failure of a latent result: at SD settings clipped pixels went from 4.04% to 42.39% and ViT-B/16 fell −.037 [−.053, −.022], and the mild setting also failed at 38.27% (Design B, n=12). The community claim that the secondary model hurts quality also failed at 256, tying the judges as above, though it trended lower at 512. Where the two agree: dynamic thresholding, cascades, reranking, guiding only the base, and CARV's finding that cutting gradient variance tenfold need not improve FID ([CARV](https://research.nvidia.com/labs/sil/projects/CARV/)).

Two literature conflicts need checking before anyone builds on them. Shen's paper presents Polyak steps, but its published code applies them only to faces, so the ImageNet 27.7 may not involve Polyak at all ([code](https://github.com/BIGKnight/Understanding-Training-free-Diffusion-Guidance/blob/master/CelebA%20and%20ImageNet/functions/denoising.py)). The research notes also disagree on PAG's unconditional layers: one lists `middle_block.1`, the other adds `input_blocks.14.1`, `16.1` and `17.1`. Both agree on `input_blocks.14.1` for the conditional model.

Planned experiments that did not produce a result:

| Experiment | State | Consequence |
|---|---|---|
| Real-pipeline pilot | The notes record it as never run. It ran after they closed, 17:43 to 18:05 on 2026-11-01, and all 27 arms sampled. The GPU judge ran out of memory loading four fp32 models, so a CPU copy scored the saved images unchanged apart from caching text features | Its leaders were rerun as Design D. laion ViT-B-32 stays invalid. No human look |
| Cutout confirmations with PE-Core-B-16 and at 256 | The first attempt stopped mid-way. Design D and E replaced it, with the ensemble at 16 cuts and with SD-Turbo at 512 | ADM at 256 and 512, and PE-Core-B-16 alone, still have no cutout data |
| Repeatability rerun at 256 (`e3b_variance.py "ViT-B/32" 256`) | Never started | No gradient data at 256 |
| Pure x0 at 256 and with PE-Core, and any `-rate` tuning for it | 128 ran on 4 seeds as `-view x0`, +.0024 ± .009. The rest did not run | Pure x0 is a tie at 128, shipped as an option |
| Warps, `(avg + max) / 2` pooling, noise-dependent blur, Laplacian normalisation, LGD-MC, Adam across steps | Not run | Remain untested on this sampler |
| GPU backbone sweep, on-card gradient agreement, fp16 against bf16 timing | SD-Turbo, Vega and Lightning then ran for real. The rest of the sweep did not | PixArt-Sigma, Sana, Z-Image and klein remain estimates |
| Part B effect test | Stopped at 2 of 4 prompts, 1 seed, 1 rate | Replaced by tiny against full decoder on 3 seeds, +.0010 ± .003. Per-step guidance through a tiny decoder is still unmeasured |
| Kandinsky and Sana 1.6B `full`, `ckpt`, `x0_vae`; SD 3.5 depth; klein at 384 | Failed on allocation or were cancelled | Those rows are absent or projected |
| Anything on the RTX 4070 | No such card available | 12 GB capacity is inferred, and per-cut time is unmeasured |
| GLIDE noised CLIP through the fast sampler | Written, refused by the session's permission classifier because it would execute code cloned from `openai/glide-text2im`. Nothing was executed or loaded | Needs that repo allowed as a dependency |
| Native-scale crop judge, and upsampler sensitivity to steps and blurred `low_res` | Stopped unfinished | Nothing measures detail quality at 512 |
| fp16 against fp32 gradients on robust towers; `convnext_large_d_320` cost | Not run; out of memory while a sibling held 5.6 GiB | Keep fp32 for FARE until checked |
| SD 1.5 on the repo's prompts and judges | Ran on seed 0, +.071 ± .031, [table][gap] | The gap to a modern model is measured, on one seed |
| Multi-seed rerun of Design B results | Dynamic thresholding reran on 4 seeds, the secondary model on 4 at 128, 2 at 256 and 1 at 512, and the cascade on seed 1 | The cascade fell to a tie, on two seeds in all |

Of the 22 practitioner ideas tested on paper, local runs have now pruned or settled most. Smooth resampling, per-cut noise, the APG projection, PAG and class contrast lost, and 16 cuts and the detached x̂₀ gradient tied. Dynamic thresholding fixed clipping and lost score, so it awaits a blind human check. Seed selection held up on judges, on repeated seeds. The cascade fell to a tie on seed 1. The secondary model covered the Jacobian-free question and ties at 256. The ADM classifier tied the judges in the pilot. ICG, TSG, DAS grafts, noise pre-optimisation on ADM, noise-aware CLIP and score composition remain unrun.

## Untested ideas that survive the local results

Everything in this table is hypothesis. The falsifiers should use at least 4 seeds, because the two-seed leaders of Design A and Design D did not replicate. Two ideas left this table once they ran: attenuating the direct path (detached view −.0041 ± .011) and class contrast (−.0072 ± .011), both Design D.

| Idea | Why it survives | Test | Falsifier |
|---|---|---|---|
| Fewer cuts on a robust tower, for speed only | Cross-cutout cosine .858 means 1 or 2 cuts may equal 4. The tower itself gained nothing in the pilot, so this is worth it only if a human prefers its smoother images | FARE4 ConvNeXt at n1, n2, n4 | n1 loses to n4, or humans do not prefer the tower |
| Compose a text score with ADM: `ε_ADM + w(ε_text − ε_text,∅)` using cc12m_1_cfg at 256 | The only idea that breaks the 1000-class limit without leaving ADM; precedent in [composition](https://arxiv.org/abs/2206.01714) and V-Majesty's parallel models | Timestep map by logSNR, second U-Net in VRAM | Mush, even with an MCMC corrector ([Du et al.](https://arxiv.org/abs/2302.11552)) |
| Prior-sampled image target | Fixes text-to-image embedding mismatch at nil runtime VRAM | Sample on CPU, pass as `target_embeds` | No judge gain against the bare prompt |
| x̂₀ seed pruning on this sampler | Selection works; pruning would drop its cost | 8 seeds on 12 prompts, Spearman between score at each step and final score | Rank correlation under .3 before halfway |
| DAS as an init ([DAS](https://arxiv.org/abs/2502.07753) lists merging with diffusion as future work) | Needs no repo change: 100 DAS steps, then `-init` at skip 500 to 700 | One script | U-Net does not remove the artifacts |
| Noise pre-optimisation with no U-Net gradient | ReNO-style loop on the existing model at about 5 s | 20 CLIP steps on x_T, then sample unguided | Gain under .01; x̂₀ at max noise may be too close to a class mean |

The `multires` gradient preconditioner derived from DAS tied at power 1 (−.000 ± .019, +.000 ± .031, +.014 ± .019, Design C), with output high-frequency share cut from .0425 to .0167, and lost at power 2 (−.017 ± .016, −.030 ± .038, −.004 ± .021). Power 1 is a free smoothing control that blur was not, so it deserves one human comparison and no more until then.

## Conclusion: what shipped, what lost and what is left

The surprise in this evidence is what the cutout was never doing. For five years the community tuned crops as if gradient variance were the problem, and the measurements show the opposite on both ADM and SD-Turbo: variance is high and removing it changes nothing, because the gradient step is capped and the model discards most of it. Effort belongs on what makes the image, then on what produces the gradient and which sample to keep. Seed selection and the ensemble held up on repeated seeds, and the prior swap beat all of it. The ranking in the first draft of this report did not hold: its first entry lost, half of its fourth entry lost, and the entry it expected least from locally, SD-Turbo noise optimisation at rank 7, became the best thing the repo makes. Ranking on seed 0 was the error, and the fix is in the design of every later run.

| First-draft rank | Change | Result on more seeds | State |
|---|---|---|---|
| 1 | Dynamic thresholding | −.0189 ± .014 at .995, 4 seeds; clipped 2.8% to 0, saturation .46 to .20 | `-dt`, off by default. `-gl`, built after it, keeps colour and ties on the ensemble |
| 2 | 128 + Real-ESRGAN x4plus as the route to 512 | Tie on seed 1: −.0135 ± .028 (6 of 12), rating 4.83 against 5.32, clipped 10.4% against 4.1%, at about a sixth of the time | `-up`, off by default |
| 3 | Keep the best of N by judges | ADM best of 4 +.0154 ± .0070 (11 of 12), with the ensemble +.0245 ± .0114 (12), held-out ViT-H | `-best N` |
| 4 | Ensemble, and cross-entropy | Ensemble +.0188 ± .015 (10 of 12), repeated at 256 and 512. Cross-entropy −.0102 ± .015 | `-clip ViT-B/32+PE-Core-B-16`, default with `-bb`. `-loss contrastive`, off by default |
| 5 | Secondary model | Tie at 256 and 512, loss at 128 | `-sec`, off by default, 5.3 to 1.9 GiB at 256 |
| 6 | SD 1.5 baseline, SD-Turbo img2img over a CGD image | SD 1.5 +.071 ± .031. Img2img +.023 to +.049 by strength, and it inherits ADM's errors: the pencil sketch stays a pen | Baseline measured, seed 0. Img2img not built |
| 7 | SD-Turbo noise optimisation with CLIP | +.096 ± .029 over ADM (12 of 12), seed 0. +.0110 ± .005 over plain SD-Turbo (12 of 12), 3 seeds | `-bb sd-turbo` |
| 8 | Cutouts where the premise may differ | Flat on SD-Turbo 512, and with 16 cuts on the ensemble | No change. 4 random crops stay |
| 9 | Realism on ADM: PAG, ICG, classifier | PAG −.0126 ± .009 at scale 1. ICG not built. Classifier tied in the pilot and not built | `-pag`, off by default |
| 10 | Sine schedule, ER-SDE | Sine −.0064 ± .009. ER-SDE +.0036 ± .014, and steadier on the ensemble | `-gs`, `-solver`, off by default |
| 11 | 1024 on tiny decoders | Vega fits at 4.65 GiB. Lightning with CLIP does not fit 8 GB at 1024 | `-bb vega`, `-bb sdxl-lightning`. Use 768 for Lightning on 8 GB |
| 12 | FLUX.2 klein 4B on the RTX 4070 | Not run | Not built |

Options beyond the first draft that shipped after being tested: `-view`, `-class` and `-contrast` on ADM, all off by default because none beat the default, and the aesthetic term over cutouts, default .05 with `-bb` and 0 on ADM. Noise-preselection, multi-scale noise steps, optimising the text embedding and rendering through 2 steps all tied on one seed and did not ship.

Do not build: FARE-only guides as a quality default, gradient EMA, gradient blur, per-cut noise, `min_cut_fraction` .4, low-discrepancy crop layouts, reflect padding, whole-frame-only cutouts, crop sizes that change during sampling, APG's orthogonal update alone, FreeU at SD settings, CLIP guidance on upsamplers at 512, 64 as a base for 256, or more steps of noise optimisation than 100, where the gain turns adversarial by eye.

What this does not settle: judge similarity is not image quality, so every option that lost on score and looks better by eye is an open question that only a blind human comparison can close, and no such study was run. The rating predictor is a model too. Nothing ran on the RTX 4070, and the Vega and Lightning results at 768 rest on one seed.

[e1]: evidence/exp/e1_determinism.py
[e2]: evidence/exp/e2.log
[e3b]: evidence/exp/e3b_vitb32_128.log
[e4]: evidence/exp/e4_end_to_end.py
[e4log]: evidence/exp/e4_ofat_128.log
[e5]: evidence/exp/e5.log
[cas]: evidence/cascade_exp/
[gq]: evidence/scripts/grad_quality.py
[bench]: evidence/bench/
[pilot]: evidence/scripts/pilot.py
[pjudge]: evidence/scripts/pilot_judge_cpu.py
[pscores]: evidence/scores/pilot_scores.json
[admexp]: evidence/scripts/adm_exp.py
[admb]: evidence/scores/adm128_b.txt
[admcut]: evidence/scores/admcut.txt
[admcutn]: evidence/scores/admcut_n.txt
[admaes]: evidence/scores/admaes.txt
[turbotune]: evidence/scores/tune.txt
[turbotune3]: evidence/scores/tune3.txt
[cutturbo]: evidence/scores/cut_turbo3.txt
[cutturboall]: evidence/scores/cut_turbo_all.txt
[novel]: evidence/scores/novel.txt
[aesturbo]: evidence/scores/aes_turbo.txt
[aes]: evidence/scripts/aes.py
[backbones]: evidence/scores/backbones.txt
[bestpy]: evidence/scripts/best.py
[bestscores]: evidence/scores/best.txt
[gap]: evidence/scores/gap_seed0.txt
[s256]: evidence/scores/sizes256.txt
[s512]: evidence/scores/sizes512.txt
[casc1]: evidence/scores/cascade_seed1.txt
[cascpy]: evidence/scripts/cascade.py
[aescut]: evidence/scores/aescut.txt
[aespc]: evidence/scores/aes_cutouts_turbo.txt
[wall]: evidence/scores/cli_wall.txt
[wallsh]: evidence/scripts/cli_wall.sh
[plainlog]: evidence/scores/plain_sizes.log
[pcv768]: evidence/scores/pc_vega768.log
[pcv1024]: evidence/scores/pc_vega1024.log
[pcl768]: evidence/scores/pc_light768.log

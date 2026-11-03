import sys
p = sys.argv[1]
s = open(p).read()
M = {}
M["Q1-TAKEAWAY"] = """A cascade from 128 is the clear win at 512: every 128-based route tied or beat direct 512 on three independent CLIP judges (n=12, one seed), at 3.5 s and 2.3 GiB with a GAN against 18.5 s and 4.1 GiB. At 256 the picture is mixed: 64→256 costs 4.4 s and 1.6 GiB against 7.6 s and 5.3 GiB, but it inherits the 64 px image's lower score and loses to direct 256 by .04 to .07, so 128 is the better base. CLIP guidance on the upsampler adds a small gain at 256 and nothing measurable at 512, which matches ADM's practice of guiding only the base."""
M["Q1-MEASURED"] = """Method: 12 prompts, seed 0, repo default settings, each pipeline against direct generation at the same final size. Judges were ViT-B/16, ViT-H-14-quickgelu (dfn5b) and ViT-L-16-SigLIP2-256 on the CPU, none of which steered: guidance is ViT-B/32. Intervals are 95% bootstrap on the paired mean across prompts. They do not capture seed variance, and judges resize to about 224 px, so they cannot see fine detail, and they are not human ratings. Times are medians of sampling only, excluding model load (the README's 256 wall time is 11.4 s against 7.6 s here, and a cascade loads two models). VRAM is peak allocated in this process, on the RTX 2070 SUPER. The GPU was shared with other agents during `sel64_*` and part of `sr256_u`, the later stages were quiet. Scripts and images are in `/tmp/claude-1000/-home-sam-GitHub-clip-guided-diffusion/34d680cf-d9a7-412c-9f94-d042cb1cd642/scratchpad/cascade_exp/`.

- Both upsamplers loaded strictly with the README flags: 311,795,910 parameters for 64→256 and 308,835,270 for 128→512. So the README's attention 32,16 is right for 128→512, and the paper's table is not. — Local measurement
- `low_res` passed straight through `sampling.sample_loop_progressive` via `model_kwargs`, with no sampler change. — Local measurement

| Final size | Pipeline | Time (s) | Peak GiB per stage | ViT-B/16 vs direct | ViT-H vs direct | SigLIP2 vs direct |
|---|---|---|---|---|---|---|
| 256 | direct `d256` | 7.59 | 5.28 | .315 (mean) | .294 (mean) | .114 (mean) |
| 256 | 64 + upsampler unguided | 1.25 + 3.16 | 1.62, 1.27 | −.068 [−.098,−.038] | −.058 [−.097,−.018] | −.043 [−.072,−.012] |
| 256 | 64 + upsampler guided | 1.25 + 5.34 | 1.62, 3.77 | −.052 [−.083,−.021] | −.045 [−.082,−.006] | −.035 [−.062,−.006] |
| 256 | 64 + x4plus | 1.25 + .03 | 1.62, .06 | −.065 [−.091,−.038] | −.041 [−.076,−.004] | −.035 [−.058,−.010] |
| 256 | 128 alone, scored as is | 3.47 | 2.33 | −.020 [−.039,−.002] | +.004 [−.038,+.041] | −.010 [−.041,+.020] |
| 512 | direct `d512` | 18.45 | 4.08 | .295 (mean) | .282 (mean) | .107 (mean) |
| 512 | 128 + upsampler unguided | 3.47 + 11.82 | 2.33, 2.13 | +.010 [−.005,+.025] | +.020 [−.012,+.046] | +.014 [−.008,+.034] |
| 512 | 128 + upsampler guided | 3.47 + 23.03 | 2.33, 4.06 | +.012 [−.003,+.026] | +.021 [−.010,+.046] | +.015 [−.007,+.034] |
| 512 | 128 + x4plus | 3.47 + .06 | 2.33, .16 | −.002 [−.020,+.015] | +.029 [+.003,+.053] | +.020 [+.004,+.034] |
| 512 | 128 + HAT | 3.47 + .38 | 2.33, .70 | +.016 [+.002,+.031] | +.021 [−.003,+.043] | +.018 [+.001,+.034] |

- Guidance on the upsampler, guided minus unguided: at 256 +.016 [+.008,+.025], +.013 [+.007,+.019], +.007 [+.002,+.013]; at 512 +.002 [+.000,+.003], +.001 [−.000,+.002], +.001 [−.001,+.003]. — Local measurement
- Diffusion against GAN upscaling, upsampler minus x4plus: at 256 −.003 [−.017,+.009], −.017 [−.032,−.003], −.008 [−.023,+.007]; at 512 +.012 [+.001,+.025], −.009 [−.021,+.001], −.006 [−.021,+.008]. — Local measurement
- My own viewing of contact sheets, not a human study: the upsampler outputs look like clean photographs with detail the 64 and 128 inputs lacked, and they lack the hard adversarial texture of `d256`. Direct 512 often zooms in and loses composition, for example a fox reduced to a blurred head. — Local measurement
- Prompt fit is set by the base. "A pencil sketch of an old man with a beard" came out as a pen in the 64 and 128 images, and no upscaler fixed that. — Local measurement"""
M["Q1-INFER"] = """- Default recommendation to test: `-size 512` becomes 128 plus an upscaler, and direct 512 stops being the route to 512. A ×4 GAN is enough on these judges. Whether the ADM upsampler's extra 11.8 s buys visible detail is a question these judges cannot answer, because they resize to about 224 px, so it needs a human look.
- Do not make 64→256 the default for 256. Base resolution decides the score. Use 128 as the base; for a 256 output, 128 plus a GAN or `-init -skip` both reach direct 256's score band.
- Skip CLIP guidance on upsamplers. At 512 it doubles time and VRAM for +.001 to +.002.
- The ADM upsamplers saw only clean area-downsampled photographs, and CLIP-guided bases are not that, so input mismatch is a real risk that the outputs here did not visibly suffer from.
- Plug-in: add an `'sr'` table to `DIFFUSION_LOOKUP` with the two URLs and flags above, and a loader in `cgd/script_util.py` built on `sr_create_model_and_diffusion` that keeps the meta device, mmap and assign path. Pass `{"y": class_index, "low_res": image_in_minus1_1}` as `model_kwargs`. `load_guided_diffusion`'s `lru_cache(maxsize=1)` and the generator's references must release the base model before the upsampler loads, or an 8 GB card holds both. A `--upscale` stage can then run after `clip_guided_diffusion()` and write its own output.
- Both upsamplers are about 1.2 GB downloads on top of the base, and the GAN route is 67 MB, which also favours the GAN as the first thing to ship."""
M["Q2-TAKEAWAY"] = """The secondary model can replace U-Net backpropagation at 256 at no measurable loss on the judges (−.011 to +.008, all intervals span zero) while cutting peak VRAM from 5.3 to 1.9 GiB and time from 8.0 to 5.8 s; at 512 it saves 1.5 GiB and 8 s but trends lower (−.012 to −.015). The remaining Disco knobs are unmeasured folklore, though `clamp_grad` and `sat_scale` overlap with what this repo already does."""
M["Q2-MEASURED"] = """- Harness check: a copy of the sampler with these hooks reproduced `d256` and `d512` with paired differences of exactly 0 on every judge, so the variant numbers are comparable to the repo's default. — Local measurement
- The checkpoint loaded strictly from the HF mirror: 13,841,547 parameters, 22× fewer than the ADM upsamplers. — Local measurement

| Size | Guidance through | Time (s) | Peak GiB | ViT-B/16 | ViT-H | SigLIP2 |
|---|---|---|---|---|---|---|
| 256 | U-Net (default) | 8.02 | 5.33 | baseline | baseline | baseline |
| 256 | Secondary model | 5.80 | 1.92 | −.011 [−.026,+.004] | +.008 [−.013,+.030] | +.005 [−.008,+.019] |
| 256 | Unguided, for scale | 5.17 | 1.91 | −.106 [−.134,−.079] | −.117 [−.155,−.081] | −.093 [−.123,−.062] |
| 512 | U-Net with checkpointing | 18.73 | 4.13 | baseline | baseline | baseline |
| 512 | Secondary model | 10.80 | 2.64 | −.012 [−.025,−.001] | −.015 [−.058,+.022] | −.012 [−.036,+.010] |

- Guidance here kept the repo's rate cap, window and losses, with x0 for CLIP coming from the secondary model. The U-Net still supplied the sampler's x0. — Local measurement"""
M["Q2-INFER"] = """- The secondary model is the strongest VRAM lever found: it turns guided 256 into a 2 GiB job, which would leave room for a heavier CLIP or more cutouts on 8 GB.
- Plug-in: add a `loss_gradient` path in `cgd/sampling.py` where the U-Net runs under `no_grad`, then `x.requires_grad_()`, `pred = secondary(x, cos_t).pred`, with `cos_t = atan2(sigma, alpha) * 2 / pi`, feeding `loss_fn(x, pred, sigma)`. The class must be vendored from Disco, since it is not in the repo's dependencies, and the hash above should be checked on download.
- Because the mirror is a personal Space, the weights should be re-hosted or pinned by hash before the repo relies on them.
- `clamp_max` is Disco's absolute version of the repo's relative `guidance_rate`, and `sat_scale` is close to `range_loss`, so neither is a new idea here."""
M["Q3-TAKEAWAY"] = """Img2img through the repo's existing `-init -skip` is the only technique here measured on this U-Net: it ties direct generation on the judges at equal or lower time but keeps the base image's saturated look. Latent upscaler, SDXL refiner and TAESD do not apply; tiling recipes do apply but are untested in pixel space, and they cost one U-Net call per tile."""
M["Q3-MEASURED"] = """| Pipeline | Time (s) | Peak GiB | ViT-B/16 vs direct | ViT-H vs direct | SigLIP2 vs direct |
|---|---|---|---|---|---|
| 128 → 256 `-skip 500` | 3.47 + 5.07 | 2.33, 5.28 | −.011 [−.031,+.010] | +.008 [−.027,+.042] | +.006 [−.020,+.029] |
| 128 → 256 `-skip 700` | 3.47 + 3.57 | 2.33, 5.28 | −.010 [−.028,+.006] | +.002 [−.036,+.036] | +.006 [−.024,+.033] |
| 256 → 512 `-skip 700` | 7.59 + 8.32 | 5.28, 4.08 | +.016 [−.002,+.033] | +.021 [−.002,+.045] | +.013 [−.004,+.030] |

- Laplacian variance at 256 was 9092 direct, 1245 at `-skip 500` and 287 at `-skip 700`, so `-skip 700` on a resized 128 stays soft. Direct's figure is inflated by adversarial texture, so higher is not better. — Local measurement
- By eye, 256→512 at `-skip 700` keeps the base's saturated, posterised look, including its failures, so it adds resolution and does not refine. — Local measurement
- An unguided U-Net call at 256 cost 0.17 s (5.17 s for 30 calls). — Local measurement"""
M["Q3-INFER"] = """- A 256→512 pass at `-skip 700` costs 15.9 s and 5.3 GiB against 3.5 s and 2.3 GiB for 128 plus a GAN with no better score, so img2img upscaling is not worth shipping as the 512 route.
- A tiled pass with 256 px tiles at stride 128 over 512 is 9 tiles per step, so about 46 s for 30 unguided calls, or fewer at high `-skip`. Keeping to low noise (`-skip 700` or above) would protect composition from the class-in-every-tile problem. This is an estimate, not a measurement.
- Plug-in for tiling: in `sample_loop_progressive`, replace the single `model()` call with a loop over overlapping crops, averaging eps with Gaussian weights as in Mixture of Diffusers. Only MIT-licensed code should be consulted: not Ultimate SD Upscale (GPL-3.0) or the Tiled Diffusion extension (CC BY-NC-SA).
- Deep Shrink, ScaleCrafter and HiDiffusion all fit as a `UNetModel.forward` override, which worked for FreeU. FreeU's failure below is a warning that latent-tuned U-Net patches can fail badly in pixel space."""
M["Q4-TAKEAWAY"] = """Real-ESRGAN x4plus (BSD-3-Clause, 67 MB) is the right first upscaler: 0.06 s and 0.16 GiB for 128→512, 0.85 s and 2.0 GiB for 512→2048 with no tiling. HAT and SwinIR-L scored about the same at much higher cost, and all SR diffusion models need an extra SD-class checkpoint of several GB, most under non-commercial licenses, so none is justified yet."""
M["Q4-MEASURED"] = """Measured through `spandrel` on the RTX 2070 SUPER, GPU quiet. x4plus, UltraSharp and Compact ran at fp16, SwinIR-L and HAT at fp32 because spandrel marks them as not supporting half. Peak includes weights. Compact is `realesr-general-x4v3.pth`, [v0.2.5.0](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth), 4,885,111 bytes, BSD-3-Clause. HAT is the ImageNet pretrain from the mirror, which is not GAN-trained.

| Model | Params | 64→256 | 128→512 | 256→1024 | 512→2048 |
|---|---|---|---|---|---|
| x4plus | 16.7M | .03 s, .06 GiB | .06 s, .16 GiB | .22 s, .53 GiB | .85 s, 2.03 GiB |
| 4x-UltraSharp | 16.7M | .04 s, .06 GiB | .06 s, .16 GiB | .22 s, .53 GiB | .84 s, 2.03 GiB |
| Compact | 1.2M | under .01 s, .01 GiB | under .01 s, .02 GiB | .02 s, .04 GiB | .05 s, .14 GiB |
| SwinIR-L | 28.0M | .08 s, .19 GiB | .46 s, .36 GiB | 2.26 s, .99 GiB | 8.53 s, 3.55 GiB |
| HAT | 20.8M | .13 s, .24 GiB | .38 s, .70 GiB | 1.90 s, 2.52 GiB | Out of memory on 8 GB with no tiling |

| 128→512, vs direct `d512` | ViT-B/16 | ViT-H | SigLIP2 | Clipped pixels (direct 6.5%) |
|---|---|---|---|---|
| x4plus | −.002 [−.020,+.015] | +.029 [+.003,+.053] | +.020 [+.004,+.034] | 9.5% |
| 4x-UltraSharp | +.000 [−.013,+.014] | +.023 [−.008,+.049] | +.021 [+.003,+.037] | 8.6% |
| Compact | −.001 [−.018,+.017] | +.022 [−.004,+.044] | +.020 [+.004,+.035] | 12.4% |
| SwinIR-L | +.005 [−.011,+.021] | +.023 [−.009,+.051] | +.018 [−.000,+.036] | 12.1% |
| HAT | +.016 [+.002,+.031] | +.021 [−.003,+.043] | +.018 [+.001,+.034] | 5.0% |

- These numbers settle the conflict above for this card: x4plus at fp16 needs about 2 GiB at 512 input, not 4–6 GB. — Local measurement"""
M["Q4-INFER"] = """- Ship x4plus first, with Compact as the option for speed. UltraSharp's CC BY-NC-SA license makes it a poor default in an MIT repo, and it scored no better.
- HAT's lower clipping (5.0% vs 9.5%) and its one significant B/16 gain are worth a look if a GAN's edges prove too hard, but it should get tiling, and its Google Drive weights are awkward for `script_util.download()`.
- Plug-in: `spandrel` loads all of these and could be a lazy optional import, like `open_clip`, or the RRDB network can be vendored in about 60 lines. Fetching goes through `script_util.download()`.
- SR diffusion models were tuned on real photo degradations, and their latent VAE round trip is a further mismatch here, so their published scores do not carry over. If one is ever tried, ResShift or InvSR are the small ones, but both are S-Lab 1.0, so non-commercial. Of the Apache-2.0 models, OSEDiff has no public weight URL I could verify, and SeeSR and DiffBIR are 4.5 to 6.9 GB."""
M["Q6-TAKEAWAY"] = """Dynamic thresholding is the one postprocess-style change that visibly removes the CLIP look: at no cost it took clipped pixels from 4.0% to 0.03% and mean saturation from .45 to .18, with judges unchanged or slightly up (+.008 to +.015, intervals span zero). By eye the images look far more natural, if sometimes grey, so the 0.995 quantile needs tuning. Orthogonal guidance and after-the-fact colour matching did little."""
M["Q6-MEASURED"] = """All at 256, against the default sampler. Mean saturation is the HSV S channel.

| Change | Clipped % | Saturation | ViT-B/16 | ViT-H | SigLIP2 |
|---|---|---|---|---|---|
| Default | 4.04 | .453 | baseline | baseline | baseline |
| Dynamic threshold, quantile .995, replacing clamp 1.2 | .03 | .180 | +.010 [−.011,+.036] | +.015 [−.016,+.046] | +.008 [−.022,+.039] |
| Orthogonal guidance update, APG with η=0 against x0 | 8.41 | .469 | +.001 [−.010,+.014] | +.006 [−.009,+.021] | −.004 [−.020,+.009] |
| Histogram match to the unguided image, 100% | 1.85 | .400 | −.004 [−.010,+.002] | −.009 [−.020,+.001] | −.007 [−.017,+.003] |
| Same, 50% blend | 1.60 | .412 | −.002 [−.006,+.002] | −.005 [−.012,+.002] | −.003 [−.010,+.004] |
| Saturation × .85 after sampling | 3.30 | .386 | −.001 [−.004,+.000] | −.000 [−.004,+.003] | −.001 [−.003,+.001] |

- By eye, dynamic thresholding produced a fox in a plausible field and a lighthouse on a rocky coast in place of the default's hard graphic shapes, though the dragonfly came out rather grey. Histogram matching kept the same adversarial shapes with milder colour. — Local measurement
- Cost of histogram matching was .01 s on the CPU, and that of dynamic thresholding was within timing noise. The unguided reference used for it costs a further full unguided run (5.2 s), though. — Local measurement"""
M["Q6-INFER"] = """- Dynamic thresholding works inside the trajectory, where the U-Net can respond to it, and postprocessing can only recolour what is already there. That fits the difference seen here.
- Plug-in: in `cgd/sampling.py`, replace `guided_x0.clamp(-x0_clamp, x0_clamp)` with a per-image `th.quantile` of `|x0|`, clamped below at 1, then clamp and divide. It is about four lines and could be a `--dynamic_threshold` option. Test quantiles .995 to .9995 and a blend with the current clamp, and rerun the repo's quality protocol, since saturation .18 may be overshooting.
- APG's η=0 projection did not transfer: it doubled clipped pixels. APG was built for CFG, where the update is a model difference. Its other components, rescaling and momentum, were not tried.
- Sharpening is not needed on top of a GAN, whose output is already hard-edged."""
M["Q7-TAKEAWAY"] = """Selection works on CPU-only judges and transfers to other judges: best of 4 at 64 px gave most of the gain, and best of 8 then upsampled beat seed 0 upsampled on both held-out judges (+.037 and +.017). Independent judges agree only moderately on seeds (r about .7 to .8), and published work warns that a single verifier gets hacked, so an ensemble is the safer choice."""
M["Q7-MEASURED"] = """Eight seeds per prompt at 64 px. Gain is the picked image's score minus seed 0's. Judges that did not select are held out.

| Selected by | Pool | ViT-B/16 | ViT-H | SigLIP2 |
|---|---|---|---|---|
| ViT-B/16 | 2 | +.009 (selector) | −.000 [−.011,+.012] | +.002 [−.005,+.008] |
| ViT-B/16 | 4 | +.032 (selector) | +.016 [−.005,+.039] | +.012 [−.003,+.029] |
| ViT-B/16 | 8 | +.038 (selector) | +.022 [−.003,+.050] | +.014 [−.003,+.031] |
| ViT-H | 4 | +.016 [−.001,+.032] | +.030 (selector) | +.018 [+.006,+.032] |
| ViT-H | 8 | +.021 [+.004,+.039] | +.045 (selector) | +.022 [+.007,+.039] |
| SigLIP2 | 4 | +.020 [+.002,+.037] | +.022 [+.002,+.043] | +.022 (selector) |
| SigLIP2 | 8 | +.024 [+.006,+.043] | +.038 [+.014,+.066] | +.032 (selector) |

- End to end, best of 8 by ViT-B/16 and then the unguided upsampler, against seed 0 then the same upsampler: ViT-H +.037 [+.011,+.063], SigLIP2 +.017 [+.002,+.032]. It still sat below direct 256 by −.021 [−.051,+.010] on ViT-H and −.026 [−.055,+.004] on SigLIP2. — Local measurement
- Within-prompt seed correlation between judges: B/16 and ViT-H .683, B/16 and SigLIP2 .699, ViT-H and SigLIP2 .808. — Local measurement
- Cost: 1.25 s per candidate quiet, plus .36 s per image for the ViT-B/16 judge on a busy CPU, with no VRAM. — Local measurement
- Seed sets global colour: mean saturation at 64 was .56 for seed 0 and .19 to .48 for seeds 1 to 7, across the same prompts. — Local measurement"""
M["Q7-INFER"] = """- Best of 4 at 128 costs about 14 s, similar to one direct 512, and then a GAN finishes in under a second. That is a strong candidate for a quality preset, though it was not run at 128 here.
- SigLIP2 as selector generalised best here, but with n=12 the ordering among judges is not reliable. Ma et al.'s ensemble result argues for averaging ranks across two or three judges.
- The saturation finding means all seed-0 comparisons in these notes carry a seed effect that the intervals do not show. A rerun with more seeds should come before any default changes.
- Plug-in: a `--candidates N` loop around `clip_guided_diffusion()` at the base size with a batch or a seed offset, scoring on the CPU in `clip_util` with a model other than the guidance model. The image with the highest mean rank goes on to the upscale stage. x̂₀-based pruning fits `sample_loop_progressive`, since it already yields `pred_xstart`, but it is unproven on this sampler."""
M["Q8-TAKEAWAY"] = """FreeU at SD strengths is actively harmful on this pixel-space U-Net: clipped pixels jumped from 4% to 42% and ViT-B/16 fell .037, and even a mild setting failed. ADM's schedules already reach near-zero terminal SNR, so zero-SNR fixes matter little here, and offset and pyramid noise need retraining. No measured evidence exists for multi-scale CLIP guidance, though it is reasonable to try through the cutout schedule."""
M["Q8-MEASURED"] = """FreeU was ported by overriding `UNetModel.forward`, applied at the output stages with `model_channels*4` and `*2`, the analogues of SD's 1280 and 640.

| Setting (b1, b2, s1, s2) | Clipped % | ViT-B/16 | ViT-H | SigLIP2 |
|---|---|---|---|---|
| Default | 4.04 | baseline | baseline | baseline |
| 1.3, 1.4, .9, .2 (SD1.4) | 42.39 | −.037 [−.053,−.022] | −.019 [−.045,+.007] | −.020 [−.040,+.002] |
| 1.1, 1.2, .9, .2 (mild) | 38.27 | −.024 [−.038,−.008] | −.010 [−.040,+.020] | −.015 [−.032,+.001] |

- By eye, the SD setting gives posterised black and white graphic shapes over the whole frame. — Local measurement
- Time rose about 2%, from 8.02 to 8.24 s, and VRAM by .08 GiB. — Local measurement"""
M["Q8-INFER"] = """- FreeU's `s` of .2 on skip low frequencies is probably the problem, because in pixel space the skips carry colour and tone directly. Testing `s` near 1 with `b` alone would isolate that, but there is no evidence it would help.
- This is the clearest case in these notes of a latent result failing to carry to pixel space.
- Base resolution is already a form of multi-resolution guidance: the 64 and 128 stages do the coarse work, and the cascade results suggest detail needs no CLIP at all. That is a cheaper reading of "coarse to fine" than scheduling CLIP scales.
- The zero-SNR argument is the reason not to port offset noise: the schedule is not the flaw here, and the repo's bias toward one mean colour per seed comes from low-frequency content in the sampled noise and model, which is a separate question.
- Restart-style renoising is the only item in this section that needs no training, and `-init -skip` on a finished sample approximates it."""
for k, v in M.items():
    m = "[[" + k + "]]"
    assert s.count(m) == 1, k
    s = s.replace(m, v)
pairs = [
("- HAT was not run: its practical real-world GAN weights live on Google Drive and Baidu only.\n", "- Only HAT's ImageNet pretrain was run. Its real-world GAN weights live on Google Drive and Baidu only.\n"),
("- FreeU on pixel-space U-Nets has no published evidence, and pyramid and offset noise need training, so neither was measured as a training change here.\n", "- FreeU on pixel-space U-Nets has no published evidence beyond the local test above. Pyramid and offset noise need training and were not tested.\n- Multi-scale CLIP guidance itself was not run here.\n"),
("- No published result runs CLIP guidance on an upsampling stage", "- A sensitivity test of the 64→256 upsampler to step count (10, 60) and to blurring `low_res` was started but stopped unfinished, and a native-scale crop judge meant to see 512 px detail did not run, so nothing here measures detail quality at 512.\n- Cascade wall time including the second model load was not measured.\n- All local numbers are one seed. Seed 0 is unusually saturated (see question 7), so these need repeating with more seeds.\n- No published result runs CLIP guidance on an upsampling stage"),
]
for a, b in pairs:
    assert a in s, a[:40]
    s = s.replace(a, b)
assert "[[" not in s
open(p, "w").write(s)

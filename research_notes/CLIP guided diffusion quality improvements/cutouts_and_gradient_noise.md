# Better cutouts, augmentation and gradient noise reduction for CLIP guidance

> **Status, 2026-11-02:** These are the notes as written on 2026-11-01, kept as the record of what was known then. Later measurements overturned some of it, including its leading recommendations. [The report](../../reports/CLIP%20guided%20diffusion%20quality%20improvements.md) has the current results, and wins wherever the two disagree.

## Which cutout schemes exist, what does each change, and which is worth having?

### Takeaway
At 128 with ViT-B/32, no cutout scheme measurably beat the repo's 4 uniform crops: of 28 variants the best was +.010 +- .014 on judge similarity, it fell to +.000 +- .017 on fresh seeds, and 32 cuts scored +.001 +- .016. Do not port Disco's stack on this evidence; the settings that clearly lose are smaller crops (`min_cut_fraction` .4) and heavy per-cut noise.

### Cited Findings
- **Disco Diffusion `MakeCutoutsDango`** (2021-22, credited to Dango233). Constructor `(cut_size, Overview=4, InnerCrop=0, IC_Size_Pow=.5, IC_Grey_P=.2)`. Overview cuts are the whole image, padded square and resized. Up to 4 are plain, grey, hflip and grey hflip; beyond 4 they are identical tensors that only the augmentations separate. Inner cuts use `size = rand**pow * (max - min) + min`, with `min = min(sides, cut_size)`, and the first `int(p * n) + 1` are grey. Resizing is ResizeRight's default cubic. Augmentations: hflip .5, noise .01, RandomAffine (10 degrees, translate .05, bilinear), noise, RandomGrayscale .1, noise, ColorJitter .1 on all four channels. Defaults: `cut_overview "[12]*400+[4]*600"`, `cut_innercut "[4]*400+[12]*600"`, `cut_ic_pow "[1]*1000"`, `cut_icgray_p "[0.2]*400+[0]*600"`, `cutn_batches 4`. So 16 cuts per batch, 64 per step. [Source](https://raw.githubusercontent.com/alembics/disco-diffusion/main/disco.py)
- Disco's own guidance on the split is one sentence with no measurement: "Overview cuts see the entire image and are good for early structure, innercuts are your standard cutn." [Source](https://raw.githubusercontent.com/alembics/disco-diffusion/main/disco.py)
- The mmagic and discoart ports keep the same schedule and logic, so they are not independent evidence. [mmagic](https://mmagic.readthedocs.io/en/latest/_modules/mmagic/models/editors/disco_diffusion/guider.html), [discoart](https://raw.githubusercontent.com/jina-ai/discoart/main/discoart/nn/make_cutouts.py)
- **Katherine Crowson's later code dropped cutouts.** k-diffusion's `sample_clip_guided.py` uses one view, `KA.RandomAffine(0, (1/14, 1/14), p=1, padding_mode='border')`, a resize_right resize with reflect padding, and the pure `denoised` estimate, at ViT-B/16, guidance 500, DPM++ 2S ancestral with static thresholding. [Source](https://github.com/crowsonkb/k-diffusion/blob/master/sample_clip_guided.py) Her v-diffusion sampler still used cutn 16, cut_pow 1, `adaptive_avg_pool2d`, on the pure prediction. [Source](https://github.com/crowsonkb/v-diffusion-pytorch/blob/master/clip_sample.py)
- **nerdyrodent VQGAN-CLIP.** Latest cutouts average `avgpool` and `maxpool` outputs, then a kornia stack with per-cut noise factor U(0, .1). Defaults: cutn 32, cut_pow 1, augments Af, Pe, Ji, Er. Parameters: Ji p .7 strength .1; Sh .3; Gn 1.0 p .5; Pe .7 p .7; Ro 15; Af degrees 15, translate .1, shear 5, zeros padding; Er scale (.1, .4); Re scale (.1, 1). Earlier variants in the same file: original (Lanczos prefilter plus bicubic), Update (hflip, jitter .01, sharp .3, Affine 30/.1 border, Perspective .2 p .4), NRUpdate and PoolingUpdate. [Source](https://github.com/nerdyrodent/VQGAN-CLIP)
- **pixray** splits cuts by purpose: `cutn_zoom = int(2*cutn/3)` zoom cuts and the rest wide. Zoom: RandomPerspective .4 p .7, RandomResizedCrop scale (.1, .75) ratio (.85, 1.2) p .7, ColorJitter hue .1 sat .1 p .8. Wide: RandomAffine translate .025, scale .95, CenterCrop, RandomPerspective .2 p .7, the same jitter. It alternates reflection and border padding per iteration, uses `(avg + max) / 2`, and adds noise with factor U(0, .1). `cut_pow` is accepted but unused. Presets: draft 32, normal 48, better 32, best 10, supreme 8. [Source](https://raw.githubusercontent.com/dribnet/pixray/master/pixray.py)
- **pytti-core.** `size = max_size * N(.8, .3).clip(cut/max, 1) ** cut_pow`, with padding .25 and border modes clamp, mirror, smear, wrap and black, then `adaptive_avg_pool2d`. Augmentations: hflip .3, Affine 30/.1 p .8 border, Perspective .2 p .4, jitter .01 p .7, Erasing (.1, .4) p .7, noise_fac .1. Defaults: 40 cutouts, cut_pow 2 (yaml) or 1.5 (class), `cutout_border .25`. It credits Crowson, jbusted and dribnet. [Source](https://github.com/pytti-tools/pytti-core)
- **deep-daze** adds a Gaussian offset around the centre (`center_focus=2`, std = centre/focus), and Gaussian size sampling (mean .6, std .2, bounds .1 to 1), plus a loss on the mean embedding of all cuts weighted by `averaging_weight .3`. [Source](https://github.com/lucidrains/deep-daze) **big-sleep** uses 128 cutouts and EMA .99 on latents. [Source](https://github.com/lucidrains/big-sleep)
- **Aphantasia** adds alignment modes uniform, central, overscan and overmax with `macro .4`, and bicubic resampling with `align_corners=True`. [Source](https://github.com/eps696/aphantasia)
- **Hugging Face's community CLIP guided pipeline** defaults to `num_cutouts=4`, `cut_power=1.0`, `adaptive_avg_pool2d`, on the same `fac` blend this repo uses. [Source](https://raw.githubusercontent.com/huggingface/diffusers/main/examples/community/clip_guided_stable_diffusion.py)
- **CLIPDraw** uses D duplicates (8 in its comparison), each with RandomPerspective and RandomResizedCrop. Without augmentation it reports "nonsense"; the comparison is qualitative. [Source](https://ar5iv.labs.arxiv.org/html/2106.14843) CLIPDraw++ uses 4. [Source](https://arxiv.org/pdf/2312.02345) CLIPasso uses the same two transforms with a `NUM_AUGS` setting. [Source](https://github.com/yael-vinker/CLIPasso/blob/main/models/loss.py)
- **StyleCLIP** has no cutouts: it upsamples ×7, then `AvgPool2d(size // 32)`. [Source](https://github.com/orpatashnik/StyleCLIP/blob/main/criteria/clip_loss.py)
- **DALL-E Flow** has no cutout scheme. It uses CLIP to rank candidates from DALL-E and GLID-3 XL, then SwinIR. [Source](https://github.com/jina-ai/dalle-flow)
- **CLIPstyler** uses crop 128, n 64, RandomPerspective p 1 at .5, and zeroes patch losses below a threshold .7 (`loss_temp[loss_temp < thresh] = 0`). [Code](https://github.com/cyclomon/CLIPstyler/blob/main/train_CLIPstyler.py), [paper](https://arxiv.org/pdf/2112.00374)
- **FuseDream** replicates DiffAugment (colour, translation, resize, cutout) 50 times with bilinear interpolation. It gives no isolated ablation of the augmentation. [Paper](https://ar5iv.labs.arxiv.org/html/2112.01573), [code](https://github.com/gnobitab/FuseDream)
- **Blended Diffusion** runs CLIP on the x0 estimate with N extending augmentations: RandomAffine 15/.1 border, RandomPerspective .7, p .7. Evidence is qualitative. [Paper](https://ar5iv.labs.arxiv.org/html/2111.14818), [code](https://github.com/omriav/blended-diffusion)
- **Text2Mesh** gives the only numeric cutout ablation found, on one candle prompt: full method .36, without crops .30, without any 2D augmentation .20, and calls the latter "completely unrelated to the target text prompt". [Source](https://ar5iv.labs.arxiv.org/html/2112.03221) CLIP-Actor criticises those extreme close-ups for producing random, empty renders. [Source](https://arxiv.org/pdf/2206.04382)
- Practitioner count: "most notebooks use 32 to 40", limited by memory. Reducing 30 to 20 to 10 to 2 on one seed made images fuzzier, but the author would not call 10 to 30 better or worse. [Source](https://www.robinsloan.com/lab/cutouts/)
- **Local measurement, end to end, the main evidence in these notes.** 28 variants, each changing one thing from the repo's default (4 uniform crops, area pooling, `x_in` blend), through the real sampler at 128 with ViT-B/32, 30 calls, rate .08. 12 prompts × 2 seeds (0, 1), 24 images per variant, with the same sampler noise across variants so differences are paired. Score: CLIP similarity to the prompt under judges that did not guide, ViT-B-16 (openai), ViT-H-14-quickgelu (dfn5b) and ViT-L-16-SigLIP2-256 (webli). Numbers are the paired difference from base, pooled over judges, with a 95% CI across the 12 prompts, and how many of the 12 improved. None of these CIs is corrected for 28 comparisons. [e4_end_to_end.py](../../reports/evidence/exp/e4_end_to_end.py), log [e4_ofat_128.log](../../reports/evidence/exp/e4_ofat_128.log)

| Variant | Difference from base | Prompts improved |
|---|---|---|
| Disco style schedule (3 of 4 cuts whole until σ < .988, then 1 of 4) | +.0102 +- .0141 | 6 |
| Pure x0 | +.0076 +- .0121 | 8 |
| n16 | +.0064 +- .0090 | 8 |
| Grey .25 | +.0050 +- .0114 | 5 |
| Antithetic | +.0047 +- .0147 | 6 |
| Per-cut normalised gradient | +.0041 +- .0087 | 7 |
| 4 whole frames, jitter ±9 px | +.0035 +- .0098 | 6 |
| n8 | +.0034 +- .0122 | 8 |
| n32 | +.0006 +- .0159 | 4 |
| R3 layout | -.0009 +- .0070 | 7 |
| Latin hypercube layout | -.0020 +- .0069 | 5 |
| Bilinear | -.0030 +- .0130 | 6 |
| 1 whole overview + 3 crops | -.0030 +- .0135 | 5 |
| hflip .5 | -.0035 +- .0126 | 4 |
| Blur σ 1 px | -.0055 +- .0075 | 5 |
| 1 whole frame with jitter | -.0062 +- .0115 | 4 |
| n8, drop least aligned quarter | -.0071 +- .0163 | 7 |
| 1 whole frame | -.0075 +- .0124 | 4 |
| Noise .03 per cut | -.0099 +- .0144 | 3 |
| Bicubic | -.0120 +- .0125 | 3 |
| Gradient EMA .5 | -.0123 +- .0099 | 2 |
| Lanczos3 | -.0137 +- .0127 | 4 |
| Noise .1 per cut | -.0149 +- .0135 | 2 |
| n8, median of 3 means | -.0157 +- .0160 | 3 |
| `min_cut_fraction` .4 | -.0174 +- .0149 | 3 |
| Blur σ 2 px | -.0205 +- .0097 | 0 |
| Gradient EMA .8 | -.0461 +- .0119 | 0 |

- The base variant's own row is zero by construction. Measured on a GPU shared with other researchers at times, so seconds per image are indicative: 3.0 to 3.3 s for 4 cuts and 3.55 s for n32. Peak sampling VRAM across all variants, n32 included, was 2.78 GiB. Also local: [harness.py](../../reports/evidence/exp/harness.py).
- **Local measurement, confirmation on fresh seeds.** Same setup and judges, seeds 2 and 3, which the table above never saw, 12 prompts × 2 seeds, 95% CI over prompts, pooled judges. Pure x0 +.0043 +- .0149 (5 of 12 improved), Disco style schedule +.0001 +- .0170 (4), n16 +.0055 +- .0125 (6), pure x0 with Disco schedule -.0011 +- .0171 (5). Neither leader replicated. [e4_confirm_128.log](../../reports/evidence/exp/e4_confirm_128.log), [e4_sets.py](../../reports/evidence/exp/e4_sets.py)

### Inferences
- Only EMA .8 and blur 2 are clear enough to survive a correction for 28 comparisons. Noise .1, `min_cut_fraction` .4, lanczos3 and EMA .5 are nominally negative, and should be read as "no reason to adopt", not proven harm.
- The positive leaders were compatible with zero and selected from 28, and on fresh seeds they did shrink, the Disco schedule to nothing. Any true effect is under about .01.
- Disco's whole-image overview cuts add little at 128 because the repo's crops already span 70 to 100% of the frame. Overview cuts should matter more at 256 and 512, where crops are smaller relative to the frame, but that was not measured end to end.
- `min_cut_fraction` .4 hurts at 128 because a 51 px crop is upsampled 4.4×, so CLIP reads blocks.
- `-cutn` at 8 to 16 costs 56 to 168 ms per ViT-B/32 image. n16 was positive both times (+.0064 +- .0090, then +.0055 +- .0125), which is safe to offer, not proven better.

### Gaps
- Zoetrope's cutouts: no source found. Jbusted's: only a credit in pytti.
- No study compares these cutout schemes on a held-out metric. Everything from Disco, pytti, nerdyrodent and pixray is practice, not measurement.
- 256 and 512 end-to-end comparisons of the schemes were not run. The planned confirmations at 256 and with PE-Core-B-16 were stopped by the deadline, before any image was scored. Nothing here is measured end to end above 128 or with a guide other than ViT-B/32.
- Judge similarity is not image quality. No human or perceptual evaluation was done. The saved images were in session storage and are gone. Their scripts and logs are in `../../reports/evidence/exp/`.
- CIs of about +- .01 cannot resolve effects below roughly .01, and 2 seeds per prompt limit the resolution further.

## Which differentiable augmentations stop adversarial gradients, and which keep a seed reproducible?

### Takeaway
Augmentation is essential for free-running optimisers, with strong numbers, but that need does not carry over to this sampler: none of flips, grey, noise or jitter beat plain crops at 128. If a warp is ever wanted, note that every bilinear warp breaks seed reproducibility unless it is replaced with an index or matrix form.

### Cited Findings
- **Strongest measured evidence that augmentation matters:** in CLIPAG's CLIPDraw test, R-precision was 1% with plain CLIP without augmentation, 70% with augmentation, and 72% for the adversarially fine-tuned model with augmentation. Aesthetic scores were 3.70, 4.14 and 4.31. [Source](https://ar5iv.labs.arxiv.org/html/2306.16805)
- That result is for a stroke optimiser with thousands of free steps, so it speaks to the need for augmentation in general, not to a 14-step diffusion window. [Source](https://ar5iv.labs.arxiv.org/html/2306.16805)
- **Lucid `standard_transforms`:** pad 12, jitter 8, scale `1 + (i - 5) / 50` for i in 0 to 10, rotate ±10 with 5 extra zeros, jitter 4. [Source](https://github.com/tensorflow/lucid/blob/master/lucid/optvis/transform.py)
- **Feature Visualization** (2017): pad 16, jitter 16, scale {1, .975, 1.025, .95, 1.05}, rotate ±5, jitter 8, crop, over 2560 Adam steps with a decorrelated Fourier parameterisation. Results are qualitative. It treats gradient blurring as preconditioning. [Source](https://distill.pub/2017/feature-visualization/)
- **Differentiable Image Parameterizations** is also qualitative. It suspects checkerboard gradient artefacts and argues shared parameterisations enlarge basins of attraction. [Source](https://distill.pub/2018/differentiable-parameterizations/)
- **Direct Ascent Synthesis** (Feb 2025) sums image resolutions from 1 to 224, with 32 augmentations per step: noise std .2 and jitter ±56 (a 336 canvas, cropped to 224), SGD lr .2, 100 steps, ViT-B/32 ensemble. It gets a natural 1/f² spectrum, slope about -2. Comparisons are qualitative. [Source](https://arxiv.org/html/2502.07753v1)
- **Expectation over Transformation** is the formal framing: optimise the mean loss over a transform distribution. [Source](https://arxiv.org/abs/1707.07397) Sitawarin et al. analyse its convergence and note it is unbiased but high variance. [Source](https://arxiv.org/pdf/2207.03574)
- **Ensemble Everything Everywhere** (2024) claims multi-resolution input gives perceptually aligned gradients without adversarial training. [Source](https://arxiv.org/abs/2408.05446) Zhang, Tramèr et al. break it with adaptive attacks and attribute the robustness to gradient masking. [Source](https://arxiv.org/html/2411.14834v2) A note attributed to Fort disputes that attack; only its abstract was visible. [Source](https://www.catalyzex.com/author/Stanislav%20Fort) These conflict, and none is about generation quality.
- **PyTorch's reproducibility note** covers `ctc_loss`, `grid_sample`, `interpolate` and `pad`. It does not list `adaptive_avg_pool2d`, though this repo documents and I measured that it also differs. [Source](https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample) A forum thread reports the same for `grid_sample`. [Source](https://discuss.pytorch.org/t/f-grid-sample-non-deterministic-backward-results/27566)
- Kornia warps go through `F.grid_sample`, so they inherit that nondeterminism. [Source](https://kornia.readthedocs.io/en/v0.7.1/_modules/kornia/geometry/transform/imgwarp.html)
- **Local measurement, determinism of the CUDA backward.** Method: 30 backward passes on one input, one weight tensor, compared to the first with `torch.equal`. torch 2.9.1+cu128, torchvision 0.24.1, RTX 2070 SUPER, cudnn defaults. Scripts: [e1_determinism.py](../../reports/evidence/exp/e1_determinism.py), [e1b_determinism.py](../../reports/evidence/exp/e1b_determinism.py), [e1c_tv.py](../../reports/evidence/exp/e1c_tv.py). Differences are small, about 1e-6 to 1e-5 relative, but the sampler feeds them back through every later step.

| Operation | Backward on CUDA | Reproducible replacement |
|---|---|---|
| `tf.adaptive_avg_pool2d`, `tf.interpolate(area)` | Differs | `cgd.modules.adaptive_avg_pool` (`W @ x @ W.T`) |
| `tf.interpolate` bilinear, bicubic, with or without antialias | Differs at every ratio tested, except identity and bilinear downscales of 2× or more (448 to 224, 512 to 224). Bicubic 512 to 224 still differs, since its 4 taps overlap below 4× | `W @ x @ W.T`, ResizeRight (`cgd/ResizeRight`, cubic, lanczos3, by_convs) |
| `tf.interpolate` nearest, nearest-exact | Identical | None needed |
| `tf.grid_sample` bilinear, bicubic, and nearest with border padding | Differs | Fixed-point int64 bilinear warp with custom backward (`FixedPointBilinear` in e1_determinism.py), since int64 `scatter_add` is order independent |
| `tvf.affine`, `tvf.perspective` bilinear; `tvf.resized_crop` | Differs | `tvf.affine` nearest with zero padding is identical; otherwise slice crop plus a matrix resize |
| `tf.pad` reflect, replicate, at widths 1, 2, 4, 16 | Differs | Advanced index padding, identical for both modes at widths 2 and 16 |
| `tf.pad` constant, circular; `roll`, `flip`, slice crop | Identical | None needed |
| `tvf.gaussian_blur` | Differs, through its reflect pad; the conv itself is identical | `W @ x @ W.T`, or `conv2d` over advanced index reflect padding |
| `index_select` with repeated indices | Differs | Advanced indexing with repeats is identical |
| `avg_pool2d`, `max_pool2d`, `adaptive_max_pool2d`, `pixel_unshuffle`, `conv_transpose2d`, `conv2d` | Identical | None needed |
| `rgb_to_grayscale`, `adjust_brightness/contrast/saturation/hue/sharpness`, `rfft2` low-pass | Identical | None needed |

- **Local measurement, the repo's `tv_loss` is not exposed.** It uses `tf.pad(..., "replicate")`, but 200 backward passes were bit identical. The same pad under a random adjoint differed in 95 of 199 passes, only at the corner pixel. A replicated pixel minus its source is exactly zero, so the squared difference sends an exact zero back. Any change that puts nonzero gradient on padded pixels would lose this. [e1c_tv.py](../../reports/evidence/exp/e1c_tv.py)
- **Local measurement, flips, grey, noise and jitter end to end** (12 prompts × 2 seeds, 95% CI over prompts, pooled judges): hflip -.0035 +- .0126, grey .25 +.0050 +- .0114, noise .03 -.0099 +- .0144, noise .1 -.0149 +- .0135, 4 whole frames with jitter +.0035 +- .0098. Warps (affine, perspective) were not tested end to end. [e4_end_to_end.py](../../reports/evidence/exp/e4_end_to_end.py), log [e4_ofat_128.log](../../reports/evidence/exp/e4_ofat_128.log)
- **Local measurement, gradient transfer under noise** (ViT-B/32 at 128, 4 prompts × 4 guided steps, 1 seed, 8 independent estimates per cell, paired against iid bicubic with a 95% CI over the 16 cells). Per-cut noise .02 moved cosine with ViT-B/16's 256-cut gradient by +.028 at k=32, and with PE-Core-B-16's by -.019. Pooled, that is +.005 +- .014. Noise .05 pooled to -.017 +- .027. [e3_variance.py](../../reports/evidence/exp/e3_variance.py), [e3b_variance.py](../../reports/evidence/exp/e3b_variance.py), [e3_analyse.py](../../reports/evidence/exp/e3_analyse.py), [e3b_analyse.py](../../reports/evidence/exp/e3b_analyse.py)

### Inferences
- The contrast with CLIPAG's 1% to 70% has a likely cause. A stroke optimiser takes thousands of unconstrained steps and can reach an adversarial pattern. This sampler takes 14 capped steps, each through a denoiser that discards most of the raw gradient (see the x0 question). The denoiser is doing the augmentation's job.
- The determinism rule that fits all the data: backward gives order-dependent results when three or more adds land on one float. Two adds commute. That explains the corner-only difference on `tf.pad`, and why bilinear goes reproducible at 2× downscale and above. It is a hypothesis that fits these results, not something read from PyTorch's source.
- Colour operations and flips are safe for reproducibility, but earned nothing here. A warp would need the int64 fixed-point form in `e1_determinism.py`, which is not worth building without an end-to-end reason.

### Gaps
- Affine and perspective warps were not run end to end at all, so kornia-style stacks remain untested on this sampler.
- Determinism was tested on one card and one torch build. Other GPUs and drivers may differ, and the RTX 4070 was not available.
- No source was found that explains the CUDA backward order, so the three-add rule stays a hypothesis.
- FARE and other robust CLIP models could not be found in diffusion guidance work, and were not tried here.

## How many cutouts are enough, and which variance reduction actually helps?

### Takeaway
Single-cut gradients share only about 6 to 10% of their power, and repeatability follows Spearman-Brown closely, so cuts act like exchangeable draws and 4 cuts is far from converged. Even so, going from 4 to 32 raised gradient transfer about 1.7× and left the image score unchanged (+.001 +- .016), so gradient variance is not the bottleneck at 128 on ViT-B/32.

### Cited Findings
- No paper or notebook found measures gradient variance against cutout count for CLIP guidance. Counts are set by memory and habit: 4 in diffusers, 16 in v-diffusion, 32 to 40 in notebooks, 64 per step in Disco. [Sloan](https://www.robinsloan.com/lab/cutouts/), [diffusers](https://raw.githubusercontent.com/huggingface/diffusers/main/examples/community/clip_guided_stable_diffusion.py)
- **Adam across steps.** AdamCG reports ImageNet accuracy 68.1% against 51.0% for plain classifier guidance. AdamDPS improves inpainting LPIPS from .16 to .12 and FID from 27.79 to 23.42. On CIFAR-10 accuracy rises from 42.8 to 52.6 but FID worsens from 32.3 to 58.0. All at 100 DDPM steps. It reports that adjacent DPS gradients are often negatively correlated. [Source](https://arxiv.org/html/2603.16797v2)
- **Momentum on non-robust classifier guidance.** EMA with β .99, no debiasing, was best: CelebA accuracy rose from 55.67 to 99.99. Adam needed a lower scale. The x0 prediction reduced sensitivity. [Source](https://ar5iv.labs.arxiv.org/html/2507.00687)
- **Conflicting evidence from CFG.** APG found *negative* momentum best for classifier-free guidance: FID 6.49 at a negative β, 6.84 at 0, 7.10 at +.5, and 13.38 at -1.5. This is a different signal. [Source](https://ar5iv.labs.arxiv.org/html/2410.02416)
- **Monte Carlo over the x0 posterior.** LGD-MC replaces the loss with `log mean exp(-loss)` over `x ~ N(x̂0, r²I)`, `r_t = σ/√(1+σ²)`, and runs the U-Net once. On super-resolution, n=1 gave FID 5.42, n=10 5.32, n=100 5.21, against 74.11 for DPS. Gains from extra samples were small. [Source](https://proceedings.mlr.press/v202/song23k.html)
- **Variance is not always the bottleneck.** CARV's stratified estimator (R=8, K=16) cut DMD gradient variance about tenfold, yet FID stayed inside the noise band of the IID runs, and the authors conclude variance is not the bottleneck there. In SDS it needed 2.6 to 3.3 times less compute for the same variance, with about 2 times faster wall-clock progress on CLIP score. [Source](https://research.nvidia.com/labs/sil/projects/CARV/) SteinDreamer's control variates saved 14 to 22% of iterations in score distillation; its code was unreleased. [Source](https://arxiv.org/pdf/2401.00604)
- **Antithetic and QMC, general.** Ren et al. prove a no-free-lunch result: no antithetic strategy beats sampling without replacement for a natural class of Monte Carlo problems, so antithetic sampling has to be fitted to the function at hand. [Source](https://proceedings.mlr.press/v97/ren19b.html) In stochastic smoothing, antithetic sampling did worse than the plain counterpart except for vanilla MC without a covariate. The ranking was Cartesian RQMC, then Latin hypercube RQMC, then Latin QMC, then MC. Cartesian RQMC needs `s = k^n` samples. [Source](https://arxiv.org/pdf/2410.08125)
- RQMC is stated to improve variance rates to `O(1/s^(1+2/n))`, and that paper says QMC had not previously been considered for gradient estimation. It concerns non-differentiable algorithms. No QMC work on image crops was found. [Source](https://arxiv.org/pdf/2410.08125)
- R-sequences are additive recurrences from the root of `x^(d+1) = x + 1`, claimed to improve on Sobol and Halton. The primary blog failed to load over TLS, so this rests on search results. [Source](https://extremelearning.com.au/tag/golden-ratio)
- **Transfer attack analogues**, which also try to keep what is common across views. Variance tuning, N=20 and β=1.5, raised Inc-v3 to Inc-v4 transfer from 43.6 to 71.7%. [Source](https://ar5iv.labs.arxiv.org/html/2103.15571) Token gradient regularisation zeroes extreme token gradients for +8.8%. [Source](https://ar5iv.labs.arxiv.org/html/2303.15754) None of this measures image quality.
- **Rejecting poor crops.** SpectralCLIP shows that without the threshold (τ=0), CLIPstyler produces many more artefacts. It reports 74.44% user preference for its band-stop DCT filter over tokens, and runs 1.095 times slower. [Source](https://ar5iv.labs.arxiv.org/html/2303.09270)
- **Local measurement, repeatability against cutout count.** At real guided states, independent k-cut gradients were drawn and compared by cosine (`pair`: 1 is fully repeatable). The first run has ViT-B/32 at 128, 4 prompts × 4 guided steps (σ .997, .92, .67, .33), 1 sampler seed, 8 estimates per cell, 13 schemes. Repo scheme measured `pair` .065, .11, .207, .35, .498, .668 at k = 1, 2, 4, 8, 16, 32; a one-parameter Spearman-Brown fit with c = .060 gives .06, .114, .204, .339, .506, .672. Cuts needed for `pair` .5 are 11 to 19 by step, and 41 to 72 for .8. [e3_variance.py](../../reports/evidence/exp/e3_variance.py), [e3b_variance.py](../../reports/evidence/exp/e3b_variance.py), [e3_analyse.py](../../reports/evidence/exp/e3_analyse.py), [e3b_analyse.py](../../reports/evidence/exp/e3b_analyse.py)
- **Correction to that run:** it reused one crop seed in all 16 cells, so its CIs were too tight and its layout gains were inflated. It showed Latin hypercube as worth up to 1.3× the cuts. The rerun below, with a distinct seed per cell, is the one to trust.
- **Local measurement, layouts, corrected seeds** (paired on `pair` against iid bicubic, 95% CI over cells). ViT-B/32 at 128 has 16 cells. PE-Core-B-16 fp16 also has 16 cells. Both use 1 sampler seed and 8 estimates per cell.

| Layout | ViT-B/32 k=4 | ViT-B/32 k=16 | ViT-B/32 k=32 | PE-Core k=4 | PE-Core k=16 | PE-Core k=32 |
|---|---|---|---|---|---|---|
| Latin hypercube | +.001 +- .019 | +.032 +- .008 | +.025 +- .007 | -.002 +- .008 | +.007 +- .010 | +.012 +- .008 |
| R3 additive | +.006 +- .019 | +.017 +- .006 | +.014 +- .009 | -.003 +- .008 | -.002 +- .006 | -.006 +- .009 |
| Antithetic (mirrored positions) | +.016 +- .026 | +.013 +- .009 | -.002 +- .010 | -.004 +- .010 | +.006 +- .008 | -.000 +- .008 |

- None of those layouts changed transfer to another model: every ViT-B/32 cell lies within +- .006 of zero with CI about +- .005, and end to end they are also flat (Latin hypercube -.0020 +- .0069, R3 -.0009 +- .0070, antithetic +.0047 +- .0147, 12 prompts × 2 seeds). Cosines against a `self` target are not reported, because the target was bicubic and that biases them against the area pool.
- **Local measurement, transfer against count, repo scheme** (cosine with another model's 256-cut mean, first run, 16 cells, 1 seed, no interval computed per count). ViT-B/16: .042, .052, .080, .092, .120, .138 at k = 1 to 32, so 1.73× from k=4 to k=32. PE-Core-B-16: .030, .038, .056, .068, .081, .089, or 1.60×. The rerun agrees within .02 at every count. Dividing by `sqrt(pair)` gives a limit for infinite cuts of about .07 to .25, depending on step and judge.
- **Local measurement, PE-Core repeats worse.** With PE-Core-B-16 fp16 at 128, the repo scheme gives `pair` .041, .074, .120, .212, .347, .497 at k = 1 to 32, against .071, .132, .213, .344, .511, .672 for ViT-B/32 (16 cells each, 1 seed, corrected seeds). This supports the README's remark that ViT-B/32's gradient is the smoothest.
- **Local measurement, robust aggregation and momentum end to end** (12 prompts × 2 seeds, 95% CI): per-cut normalisation +.0041 +- .0087, median of 3 means at n8 -.0157 +- .0160, drop the least aligned quarter at n8 -.0071 +- .0163, EMA .5 -.0123 +- .0099, EMA .8 -.0461 +- .0119, the last with 0 of 12 prompts improved. [e4_end_to_end.py](../../reports/evidence/exp/e4_end_to_end.py), log [e4_ofat_128.log](../../reports/evidence/exp/e4_ofat_128.log)
- **Local measurement, cost per extra cutout.** Forward plus backward on random input at the model's input size, for 1 to 32 cuts, median of 5 timed repeats, linear fit, RTX 2070 SUPER. There is no interval; repeats were close, and the card was idle when the run began, though other researchers shared it at times. Per-image time multiplies by the 14 guided steps. [e2_cost.py](../../reports/evidence/exp/e2_cost.py), [e2.log](../../reports/evidence/exp/e2.log)

| Model | ms per cut | Extra time per image per cut | MiB per cut |
|---|---|---|---|
| ViT-B/32 | .9 | 13 ms | 17.5 |
| ViT-B/16 | 5.5 | 77 ms | 61.3 |
| PE-Core-B-16 fp32 | 18.7 | 262 ms | 20.8 |
| PE-Core-B-16 fp16 | 6.1 | 85 ms | 11.0 |
| ViT-L-16-SigLIP2-256 fp32 | 79.3 | 1.11 s | 47.2 |
| ViT-L-16-SigLIP2-256 fp16 | 23.0 | 322 ms | 24.4 |
| PE-Core-L-14-336 fp32 | 218.9 | 3.06 s | 150.5 |
| PE-Core-L-14-336 fp16 | 83.9 | 1.17 s | 76.6 |
| ViT-SO400M-16-SigLIP2-384 fp16 | 83.7 | 1.17 s | 83.9 |

- ViT-B/32's end-to-end check agrees: n32 added about .4 s over n4 for 28 extra cuts, or 14 ms per cut. For PE-Core-L-14-336 fp16 the logged fit of 33.8 MiB per cut is wrong, because the previous model had not been freed. 76.6 is the slope of the raw rows from 8 to 16 cuts.

### Inferences
- With guidance capped per step, more cuts keep the step size and only change its direction. The U-Net's Jacobian already removes much of the noise, so cleaner input to it gains little. The measurement shows the outcome; the explanation is the likely one, not a proven one.
- The transfer limit of .07 to .25 means that even infinite cuts leave most of the gradient specific to one model. Averaging crops cannot remove that part, whereas averaging models could. That is an argument for ensembles over more cuts, and it is the sibling research's territory.
- Latin hypercube is free and slightly raises repeatability at k >= 16 on ViT-B/32, but with no transfer or image gain it does not earn the code.
- Momentum across steps fails here despite its results in AdamCG and EMA classifier guidance. Those run 100 or more steps with a fixed class target; this sampler has 14 guided steps whose content changes quickly, so an EMA is stale. APG's negative momentum result is at least consistent with that.
- **VRAM on each card, ViT-B/32.** Allocation per cut does not depend on the card, so the table carries to the RTX 4070. From the README peaks, the upper bound on cuts in one batch is about 330 at 128 and 150 at 256 on 8 GB, and about 560 and 390 on 12 GB. Disco-style chunking, taking the gradient in `x_in` space as `harness.py` does, removes that limit and leaves only time.

### Gaps
- RTX 4070 time per cut is unmeasured: there was no such card here, and no scaling figure has a source. Only the VRAM column carries over.
- PE-Core-B-16 and larger models were not compared end to end by cut count. Their lower repeatability means cut count could matter more there, so the n32 result must not be generalised to them. This is the most useful unfinished experiment.
- The 256 repeatability rerun (`e3b_variance.py "ViT-B/32" 256`) never started, because the deadline came first.
- The 1 sampler seed in E3 means states are not sampled widely. CIs there cover cells, not seeds.
- QMC on crops, stratified crops and control variates: no literature found. R3's source could only be reached through search results.
- Adam across steps (AdamCG) and LGD-MC were not tried.

## Does low-pass or frequency filtering of the gradient help?

### Takeaway
Do not low-pass the gradient: blur σ 2 px was clearly worse (-.021 +- .010, 0 of 12 prompts improved), and σ 1 px gave nothing (-.006 +- .008), although both halved the output's high-frequency share. The real gradient low-pass already exists, and it is the U-Net's Jacobian.

### Cited Findings
- **Translation-invariant attacks** convolve the gradient with a Gaussian (15×15, or 7×7 for undefended models), shown to approximate an ensemble of translated images. DIM to TI-DIM raised transfer from 13.0 to 37.4%. This is the clearest result linking blur to translation averaging. [Source](https://ar5iv.labs.arxiv.org/html/1904.02884)
- **Feature Visualization** describes gradient blurring as a form of preconditioning, and suggests bilateral filtering for sharper edges. Qualitative. [Source](https://distill.pub/2017/feature-visualization/)
- **Direct Ascent Synthesis** parameterises the image as a sum over resolutions, which reweights the gradient toward low frequencies by construction. [Source](https://arxiv.org/html/2502.07753v1)
- **Aphantasia** optimises in FFT or wavelet space. [Source](https://github.com/eps696/aphantasia)
- **SmoothGrad** averages gradients over input noise, with diminishing returns near n=50 and noise at 10 to 20% of the range. It was built for saliency maps. [Source](https://ar5iv.labs.arxiv.org/html/1706.03825)
- **Laplacian gradient normalisation** is standard in DeepDream implementations. [Source](https://www.mathworks.com/help/deeplearning/ref/deepdreamimage.html)
- **Frequency work in diffusion is all on CFG, not classifier gradients:** FDG uses per-frequency guidance scales [Source](https://arxiv.org/pdf/2506.19713), FreSca edits the score in Fourier space [Source](https://alphaxiv.org/overview/2504.02154v3), and LF-CFG targets low-frequency artefacts. [Source](https://arxiv.org/html/2506.21452v1)
- **ViT gradients carry the patch grid.** Two recent papers document this and propose fixes. [Source](https://arxiv.org/pdf/2510.09135), [DAVE](https://arxiv.org/html/2602.06613v1)
- **Local measurement, blur end to end** (Gaussian on the `x_in` gradient through a reproducible `W @ g @ W.T` matrix; 128, ViT-B/32, 12 prompts × 2 seeds, 95% CI, pooled judges). Share of output spectral power above .25 cycles per pixel was .0364 for base, .0190 at σ 1 and .0173 at σ 2. [e4_end_to_end.py](../../reports/evidence/exp/e4_end_to_end.py), log [e4_ofat_128.log](../../reports/evidence/exp/e4_ofat_128.log)
- **Local measurement, where gradient power sits** (6 prompts, seed 0, 8 guided steps, 95% CI over prompts, 4-cut estimates). The raw CLIP gradient has .60 to .69 of its power above .25 cycles per pixel, against about .80 for white noise. Through the U-Net Jacobian it has .03 at σ .997 and .35 at σ .33, so `J` removes most of it. The `x_in` blend's share rises from .035 +- .012 at σ .997 to .657 +- .030 at σ .33. [e5_blend_spectrum.py](../../reports/evidence/exp/e5_blend_spectrum.py), [e5_ci.py](../../reports/evidence/exp/e5_ci.py)

### Inferences
- Blur removes output detail and costs score, so the high-frequency gradient is not plain noise as far as the judges can tell. A high-frequency share in this range may simply be fine detail.
- TI attacks gain transfer from blur, and translation jitter should approximate that. Here jitter also did nothing, which fits: the images do not need more transfer in that sense at 128.
- Blur at 128 is coarse relative to the image. A 2 px sigma on 128 is much more than on 512, so this result should not be carried to 512 without a test.
- If a blur is ever applied, apply it to the identity path only. That is close to what pure x0 does, and it is the more direct test.

### Gaps
- Noise-dependent blur (stronger at high σ), Laplacian pyramid normalisation and frequency-weighted guidance were not run. No paper applies them to CLIP guidance.
- Only 128 was tested. Patch-grid artefacts are a ViT-B/16 and PE-Core concern, and those were not tested end to end.
- SmoothGrad-style noise averaging is covered only through per-cut noise, not as a dedicated test.

## Which resampler should feed CLIP, and what does upsampling lose?

### Takeaway
Keep the repo's area pool, because it reproduces exactly and no interpolator beat it at 128: bicubic was -.012 +- .013, lanczos3 -.014 +- .013 and bilinear -.003 +- .013. Smoother resamplers raise gradient repeatability, but that gain did not reach the image.

### Cited Findings
- **Each model's own preprocessor differs** (open_clip 3.3.0 `get_pretrained_cfg`, read locally). ViT-B-32 and ViT-B-16 openai: bicubic, shortest edge, CLIP mean. PE-Core-B-16, L-14-336, bigG-14-448: bilinear, squash, mean and std .5. SigLIP2 L-16-256, SO400M, gopt: bicubic, squash, .5. ViT-H-14-quickgelu dfn5b: bicubic, squash, CLIP mean. [Source](https://github.com/mlfoundations/open_clip)
- Practice varies widely with no comparison found: nearest-acting area pooling on upsampling in this repo and diffusers, cubic ResizeRight in Disco, Lanczos prefilter plus bicubic in early VQGAN+CLIP, `(avg + max) / 2` in pixray and nerdyrodent, ×7 upsample then average pool in StyleCLIP. [Disco](https://raw.githubusercontent.com/alembics/disco-diffusion/main/disco.py), [nerdyrodent](https://github.com/nerdyrodent/VQGAN-CLIP), [StyleCLIP](https://github.com/orpatashnik/StyleCLIP/blob/main/criteria/clip_loss.py)
- k-diffusion resizes with resize_right using reflect padding. [Source](https://github.com/crowsonkb/k-diffusion/blob/master/sample_clip_guided.py)
- **Repo facts** (`cgd/modules.py`). `min_size = min(int(max_size * min_cut_fraction), cut_size)`, so at 512 with a 224 model the smallest crop is 224, or 44% of the side and not 70%. At 128 it is 89 px, and at 256 with ViT-B/32 it is 179 px. All cuts in a batch share coordinates. `adaptive_avg_pool` upsamples as nearest, so a CLIP model below its native resolution sees blocks.
- **Arithmetic for small images.** An 89 px crop into 448 is 5× upsampling, so a 14 px patch covers 2.8 source pixels. A whole 128 image into 448 is 3.5×, or 4 source pixels. For ViT-B/32 at 224, a 128 image gives a patch of 18.3 source pixels. Anything above 1/5 of Nyquist in the CLIP input is empty in the first case, whichever interpolator is used.
- **Local measurement, resamplers** at the last guided step (6 prompts, seed 0, 4 independent 16-cut estimates, 95% CI over prompts). All use reproducible matrix forms. `pair` was .387 +- .116 for area, .505 +- .144 bilinear, .507 +- .149 bicubic and .511 +- .152 lanczos3. CLIP's input carries almost no high frequency in any case, .004 to .006 of its power. [e5_blend_spectrum.py](../../reports/evidence/exp/e5_blend_spectrum.py), [e5_ci.py](../../reports/evidence/exp/e5_ci.py)
- **Repeatability, corrected seeds** (ViT-B/32 at 128, 16 cells, 1 seed): area below bicubic by .082 +- .018 at k=4 and .077 +- .018 at k=32. Transfer shows no consistent gain: at k=32, area is +.010 +- .024 to ViT-B/16 and -.014 +- .017 to PE-Core-B-16. PE-Core as guide, 16 cells: area -.016 +- .018 at k=32 to ViT-B/16. [e3_variance.py](../../reports/evidence/exp/e3_variance.py), [e3b_variance.py](../../reports/evidence/exp/e3b_variance.py), [e3_analyse.py](../../reports/evidence/exp/e3_analyse.py), [e3b_analyse.py](../../reports/evidence/exp/e3b_analyse.py)

### Inferences
- Part of the bicubic repeatability gain is a side effect, since a smoother adjoint raises cosine by itself. Do not compare `pair` across resamplers as if it were signal.
- Area's lead over bicubic is not significant, and the base images were all made with area, so any preference is weak. The firm conclusion is only that switching is not justified at 128.
- A model's own preprocessor (bilinear for PE-Core, bicubic for SigLIP2) might matter more with those models, since they were trained on it. PE-Core's transfer lean toward bicubic hints at that.
- Upsampling invents no information whichever kernel is used, so at 128 going to a 448 model mainly spends compute on empty bands.

### Gaps
- Resamplers were not tested end to end with PE-Core or SigLIP2 guides, or at 256 and 512 where cuts are downsampled and aliasing would matter.
- `(avg + max) / 2` from pixray and nerdyrodent was not tested.
- No published resampler comparison for CLIP guidance was found.

## Should guidance read x0, the blend, or a noised image?

### Takeaway
Late in the guided window the blend's direct path dominates, up to 8.4 times the U-Net path at σ .33, and it passes raw CLIP gradient to x. Removing that path looked promising but stayed unproven, at +.008 +- .012 and then +.004 +- .015 on fresh seeds, and per-cut noise is not worth adding.

### Cited Findings
- Disco and the diffusers pipeline use `x_in = x0 * fac + x * (1 - fac)`, with `fac = sqrt(1 - alpha_bar)`. Disco takes the CLIP gradient on `x_in`, then back-propagates it to `x` through `x_in`. [Disco](https://raw.githubusercontent.com/alembics/disco-diffusion/main/disco.py), [diffusers](https://raw.githubusercontent.com/huggingface/diffusers/main/examples/community/clip_guided_stable_diffusion.py)
- Crowson's later samplers use the pure denoised estimate. [k-diffusion](https://github.com/crowsonkb/k-diffusion/blob/master/sample_clip_guided.py), [v-diffusion](https://github.com/crowsonkb/v-diffusion-pytorch/blob/master/clip_sample.py)
- GLIDE trains CLIP on noised images and reports results equal to or better than engineered augmentation pipelines, without numbers. [Source](https://arxiv.org/pdf/2112.10741)
- Non-robust classifier guidance reports that x0 reduces sensitivity to the noise-level mismatch. [Source](https://ar5iv.labs.arxiv.org/html/2507.00687)
- MPGD skips the U-Net Jacobian by taking the gradient at x0, for up to 3.8 times speed. No CLIP evaluation was found. [Source](https://ar5iv.labs.arxiv.org/html/2311.16424)
- LGD-MC samples around x0 at `r_t = σ/√(1+σ²)`, with one U-Net call. [Source](https://proceedings.mlr.press/v202/song23k.html)
- DAS uses noise std .2 per augmentation. [Source](https://arxiv.org/html/2502.07753v1) Disco uses .01 three times. pixray and nerdyrodent use a per-cut factor U(0, .1). [Disco](https://raw.githubusercontent.com/alembics/disco-diffusion/main/disco.py), [pixray](https://raw.githubusercontent.com/dribnet/pixray/master/pixray.py)
- **Repo arithmetic** (`cgd/sampling.py`). Guidance runs on model calls 6 to 19 of 30, 14 steps, at σ .999, .998, .997, .993, .988, .977, .957, .922, .866, .782, .673, .55, .431, .327. With `x = αx0 + σε̂`, the blend is `[σ + (1-σ)α]x0 + (1-σ)σ ε̂`. The x0 coefficient never falls below .915. The noise coefficient peaks at .247 near σ .55, or about .12 on the [0, 1] scale. It is one shared noise sample, not per cut. The gradient is `σJᵀg + (1-σ)g`, so the second term reaches x without passing through the U-Net.
- **Local measurement, the two gradient paths** (128, ViT-B/32, 6 prompts, seed 0, 95% CI over prompts; `pair` is the cosine between two 4-cut estimates).

| σ | `pair` raw CLIP | `pair` through U-Net | `pair` blend | Direct path / U-Net path |
|---|---|---|---|---|
| .997 | .252 +- .062 | .833 +- .094 | .830 +- .093 | .035 +- .005 |
| .922 | .246 +- .065 | .719 +- .112 | .629 +- .124 | .479 +- .167 |
| .673 | .188 +- .046 | .505 +- .164 | .301 +- .117 | 1.91 +- .48 |
| .431 | .181 +- .031 | .355 +- .097 | .204 +- .038 | 5.39 +- .95 |
| .327 | .177 +- .058 | .300 +- .107 | .186 +- .059 | 8.39 +- 1.05 |

- [e5_blend_spectrum.py](../../reports/evidence/exp/e5_blend_spectrum.py), [e5_ci.py](../../reports/evidence/exp/e5_ci.py)
- **Local measurement, end to end** (12 prompts × 2 seeds, 95% CI, pooled judges). Pure x0 +.0076 +- .0121, 8 of 12 prompts improved, with high-frequency share .0433 against .0364. On fresh seeds it gave +.0043 +- .0149, 5 of 12 improved. Noise .03 per cut -.0099 +- .0144 and noise .1 -.0149 +- .0135. [e4_end_to_end.py](../../reports/evidence/exp/e4_end_to_end.py), log [e4_ofat_128.log](../../reports/evidence/exp/e4_ofat_128.log)

### Inferences
- At σ <= .67 the blend gives up most of the U-Net's repeatability, ending near raw CLIP's. That is the clearest mechanism for pure x0 to help. Its output is not smoother, though, at .0433 against .0364, so the effect is not simple denoising.
- The guidance cap rescales both forms, so pure x0 keeps its step size. No separate rate tuning was done, and pure x0 might want a different `-rate`.
- A cheap in-between exists: keep `fac` but weight the direct path down at low σ. Not tested.
- The blend's own noise, .12 peak on [0, 1], is already larger than what was added per cut, and shared noise is the larger influence. This may be why extra per-cut noise only costs score.

### Gaps
- Pure x0 was not tuned. Its 256 and PE-Core runs were stopped by the deadline. Its two 128 estimates are both positive and both include zero.
- LGD-MC and MPGD were not tried. MPGD's Jacobian-free step would drop the U-Net path this data favours.
- A noise-aware CLIP in the GLIDE sense has no pretrained weights for these models that I found.

## Are attention, saliency, tiled cutouts or patch token losses worth trying?

### Takeaway
None of these is established: no source uses saliency cutouts or patch-token losses for guidance, and I ran no test. Crops are not where the gain is on this evidence, so treat these as speculative research, not a next step.

### Cited Findings
- **Saliency or attention placed cutouts:** none found in CLIP guided generation. The community Stable Diffusion CLIP pipeline, for one, uses uniform random crops. [Source](https://raw.githubusercontent.com/huggingface/diffusers/main/examples/community/clip_guided_stable_diffusion.py)
- **Patch token maps are a segmentation technique so far.** MaskCLIP, SCLIP and SC-CLIP modify the last attention block to make tokens spatially meaningful. No guidance use was found. [MaskCLIP](https://ar5iv.labs.arxiv.org/html/2112.01071), [SCLIP](https://arxiv.org/pdf/2312.01597), [SC-CLIP](https://arxiv.org/html/2411.15869v2)
- **CLIPVG** discusses the limits of patch-wise CLIP losses. [Source](https://arxiv.org/pdf/2212.02122)
- **Local patch losses that do work** are CLIPstyler's thresholded crops, for style transfer, where each patch should match the same target. [Source](https://arxiv.org/pdf/2112.00374)
- **Tiled cutouts:** pixray's split, Aphantasia's `overscan` and `overmax` alignments and Disco's overview plus inner split are the nearest practice, and none is a true tiling with a coverage guarantee. [pixray](https://raw.githubusercontent.com/dribnet/pixray/master/pixray.py), [Aphantasia](https://github.com/eps696/aphantasia)
- No local measurement was made for saliency cutouts, tiling or patch tokens. The nearest result is that better spatial coverage changed nothing: Latin hypercube, whose purpose is even coverage, was -.0020 +- .0069 end to end (12 prompts × 2 seeds).

### Inferences
- Since even coverage and more cuts both gave nothing at 128, smarter placement has little to work with at that size. It could matter at 512, where crops can be subject-sized.
- A patch-token loss would need per-token targets, or it reduces to a pooled loss with more noise. CLIPstyler's approach fits style, where all patches share one target, and not subject prompts.
- Saliency needs an extra backward pass per step, which breaks the 14-step budget for large models.

### Gaps
- Everything in this question is untested here.
- SGOOL's saliency work was seen only through a secondary article, so it is not cited.
- Whether MaskCLIP-style attention edits keep useful gradients is unknown. They were built for inference, and they would also interact with the repo's plain attention path.

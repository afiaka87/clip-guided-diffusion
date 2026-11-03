# Practitioner tricks and novel ideas for CLIP-guided diffusion

> **Status, 2026-11-02:** These are the notes as written on 2026-11-01, kept as the record of what was known then. Later measurements overturned some of it, including its leading recommendations. [The report](../../reports/CLIP%20guided%20diffusion%20quality%20improvements.md) has the current results, and wins wherever the two disagree.

Evidence tags used throughout: **[code]** read directly from the source file. **[paper]** an author's reported result. **[community]** a practitioner claim with no measurement behind it. **[here]** measured or read in this repo. **[derived]** my own calculation. Anything tagged [community] should not be treated as measured.

Local paths are relative to the repo root. Research date: November 2026.

## 1. What did the 2021 to 2023 CLIP art community settle on?

### Takeaway
The community converged on four things this repo lacks: cutouts scheduled from global to local across the run, many distinct cutouts accumulated at no U-Net cost, light augmentation, and multi-model CLIP ensembles. The repo already has its own version of gradient capping, the x0 blend and the range loss. Most community numbers are tied to 250 to 1000 step runs at 512px or larger, so they do not transfer directly.

### Cited Findings

**Lineage**
- Original notebook by Katherine Crowson. Daniel Russell then tuned it for "quick generations in 15-100 timesteps rather than 1000, as well as more robust augmentations", and Dango233 and nshepperd improved quality "especially so for shorter runs". The DangoCutn cutout method is Dango233's. [code] — [Disco Diffusion credits](https://github.com/alembics/disco-diffusion/blob/main/disco.py)
- Disco's own settings table is headed "Defaults are heavily outdated" and disagrees with its code: table `eta` 0.5, `rand_mag` 0.1, `clip_guidance_scale` 1000, `tv_scale` 150, against code 0.8, 0.05, 5000 and 0. Use the code values below. [code] — [disco.py](https://github.com/alembics/disco-diffusion/blob/main/disco.py)

**Cutouts**
- Disco's final defaults: `cut_overview = [12]*400+[4]*600`, `cut_innercut = [4]*400+[12]*600`, `cut_ic_pow = [1]*1000`, `cut_icgray_p = [0.2]*400+[0]*600`, `cutn_batches = 4`. Schedules are indexed by `1000 - t`, so index 0 is the noisy start: global views early, local crops late. That is 16 cuts per batch and 64 per model per step. [code] — [disco.py](https://github.com/alembics/disco-diffusion/blob/main/disco.py)
- Crop size is `rand()**cut_ic_pow * (max - min) + min`, so a power above 1 favours small crops and below 1 favours large ones. This repo uses the same formula as `cut_pow`. [code] — [disco.py](https://github.com/alembics/disco-diffusion/blob/main/disco.py); local `cgd/modules.py`
- With `Overview <= 4` Disco adds the frame, its grayscale, its flip, then the flipped grayscale. Above 4 it repeats one frame and relies on the augmentations to differ. Inner crops are gray while `i <= int(IC_Grey_P * InnerCrop)`, so the first crop is always gray, even at p=0. [code] — [disco.py](https://github.com/alembics/disco-diffusion/blob/main/disco.py)
- Latent Majesty's defaults: `cut_overview = [8]*500+[4]*500`, `cut_innercut = [0]*500+[4]*500`, `cut_ic_pow = .2`, `cut_icgray_p = [.1]*300+[0]*1000`, `cutn_batches = 1`, plus a Gaussian-blurred subset of cuts (`cut_blur_n`, kernel 3). These run on Latent Diffusion, so they are not a direct match for pixel space. [code] — [Majesty latent.ipynb](https://github.com/multimodalart/majesty-diffusion/blob/main/latent.ipynb)
- A shared Majesty config uses `cut_overview = [8]*1000`, `cut_innercut = [2]*1000`, `cut_ic_pow = .2`, `clamp_index = [.45]*1000`. [community] — [the_other_zippy_defaults.cfg](https://huggingface.co/datasets/multimodalart/latent-majesty-diffusion-settings/blob/9cedbcdb65c859184a9236f23c609a054c8281fb/the_other_zippy_defaults.cfg)
- A recipe for a recognisable Disco look at 768x1024: guidance 18000+, 200-250 steps, TV 4000+, range 80, saturation 2000+, no secondary model ("important"), eta 0.9 ("important"), `cut_ic_pow` 10, and cut schedules split into 200s of 10/8/6/2/0 overview with the opposite for inner crops. [community] — [Disco Diffusion Tips gist](https://gist.github.com/lxe/ec08c4ecc1f79a084d47e1b82d3a65ff)

**Augmentation**
- Disco's still-image set: flip p=0.5, noise 0.01, affine 10 degrees with translate 0.05, noise, grayscale p=0.1, noise, ColorJitter 0.1 on all channels. Video mode used the stronger set with perspective, and that stronger set is what `cgd/modules.py` has. [code] — [disco.py](https://github.com/alembics/disco-diffusion/blob/main/disco.py); local `cgd/modules.py`

**Gradient handling**
- Disco splits guidance in two. It accumulates CLIP gradients on `x_in` across all cut batches, then backpropagates once with `torch.autograd.grad(x_in, x, x_in_grad)`. Extra cutouts therefore cost CLIP time only, never another U-Net backward. A NaN anywhere zeroes the step. [code] — [disco.py](https://github.com/alembics/disco-diffusion/blob/main/disco.py)
- `clamp_max` defaults to 0.05 as an RMS cap, which equals this repo's legacy `use_magnitude` path. Shared v5.x settings files use 0.5 to 0.57, an order of magnitude looser. [code, community] — [disco.py](https://github.com/alembics/disco-diffusion/blob/main/disco.py); [HuemanInstrument settings](https://gist.github.com/HuemanInstrument/76234a35784c2c22a65c3de50b617567)
- Majesty normalises by the peak, then clamps on a schedule: `grad=grad/grad.abs().max()*opt.mag_mul` with `opt_mag_mul = 20`, then RMS clamp from `clamp_index = [2.4, 2.1]` interpolated linearly across the run. Its `grad_scale = 0.25` carries the comment "Lower value result in more coherent and detailed result". [code, community] — [Majesty latent.ipynb](https://github.com/multimodalart/majesty-diffusion/blob/main/latent.ipynb)
- Secondary Model Method: Crowson's small x0 predictor replaces the U-Net inside `cond_fn`, as `x_in = out * fac + x * (1 - fac)`, so the U-Net is out of the gradient graph. Disco turns it on by default, while the recipe above calls avoiding it "important". That conflict was never measured in what I found. [code, community] — [disco.py](https://github.com/alembics/disco-diffusion/blob/main/disco.py); [gist](https://gist.github.com/lxe/ec08c4ecc1f79a084d47e1b82d3a65ff)

**Losses**
- Disco defaults: `tv_scale` 0, `range_scale` 150, `sat_scale` 0. Shared configs range from `range_scale` 1110 to 1666 and `tv_scale` 0 to 151. This repo: TV 150, range 50, saturation 0. [code, community, here] — [disco.py](https://github.com/alembics/disco-diffusion/blob/main/disco.py); [v5.2 settings](https://gist.github.com/HuemanInstrument/060a25dc32618e5deafaffaf462ba9b6); local `cgd/cgd.py`
- `sat_scale` is from nshepperd's JAX notebook, as `abs(x_in - x_in.clamp(-1,1)).mean()`. The repo's version is identical. [code] — [disco.py](https://github.com/alembics/disco-diffusion/blob/main/disco.py); local `cgd/cgd.py`
- Majesty v1.5 added variance and mean losses, an improved range scale and "experimental compression and punishment settings" (`compress_steps = 200`, `compress_factor = 0.1`, `punish_steps = 200`, `punish_factor = 0.5`). [code] — [Majesty README](https://github.com/multimodalart/majesty-diffusion/blob/main/README.md); [latent.ipynb](https://github.com/multimodalart/majesty-diffusion/blob/main/latent.ipynb)

**Sampling**
- Disco's `eta` defaults to 0.8 and one recipe says 0.9. Majesty scheduled it linearly (`opt_ddim_eta` to `opt_eta_end`). This repo uses 0.5 on a different solver. [code, community, here] — [disco.py](https://github.com/alembics/disco-diffusion/blob/main/disco.py); [Majesty README](https://github.com/multimodalart/majesty-diffusion/blob/main/README.md); local `cgd/sampling.py`
- Majesty has customised dynamic thresholding (`threshold_percentile = .85`, `threshold = 1`), and notes that PLMS "works but does not lookg good". This repo has a static clamp of 1.2. [code] — [Majesty latent.ipynb](https://github.com/multimodalart/majesty-diffusion/blob/main/latent.ipynb); local `cgd/sampling.py`

**Model choice**
- Disco defaults to an ensemble of ViT-B/32, ViT-B/16 and RN50. Majesty defaults to ViT-B/16, ViT-L/14 and laion2b ViT-B/32. The recipe says add ViT-L/14 "or if have ram, 336 (very nice)". [code, community] — [disco.py](https://github.com/alembics/disco-diffusion/blob/main/disco.py); [latent.ipynb](https://github.com/multimodalart/majesty-diffusion/blob/main/latent.ipynb); [gist](https://gist.github.com/lxe/ec08c4ecc1f79a084d47e1b82d3a65ff)
- The community also moved to fine-tuned unconditional ADM checkpoints: pixel art, watercolor, PulpSciFi, and a portrait generator. [code] — [disco.py](https://github.com/alembics/disco-diffusion/blob/main/disco.py)

**Init**
- Crowson's notebook: `skip_timesteps` "needs to be between approx. 200 and 500 when using an init image". This repo defaults to half. [code, here] — [EleutherAI app.py](https://huggingface.co/spaces/EleutherAI/clip-guided-diffusion/blob/main/app.py); local `cgd/cgd.py`
- Disco's `perlin_init` starts from mixed colour and grayscale multi-octave Perlin noise. Its table suggests `init_scale` 1000. [code] — [disco.py](https://github.com/alembics/disco-diffusion/blob/main/disco.py)
- Centipede Diffusion feeds a fast Latent Diffusion output into Disco as an init: "LL creates a cohesive overall shape. Then DD adds the details". Its author pauses between stages to reject bad layouts, because LD is cheap to reroll. [community] — [Centipede README](https://github.com/Zalring/Centipede_Diffusion/blob/main/README.md)

**Post-processing**
- "ALWAYS use partial saves and take the 90%-ish partial save as your final and ALWAYS run it through" Real-ESRGAN. [community] — [gist](https://gist.github.com/lxe/ec08c4ecc1f79a084d47e1b82d3a65ff)

**Afiaka87's repo history**
- July 2021: augmentations disabled and "remove sharpen", RN50x16 as default. August 2021: `cutn` lowered and ViT-B/32 made default for VRAM, CLIP class search added and then disabled by default. The current fast sampler makes that class search its core idea. [here] — local `git log`

### Inferences
- **Most valuable missing piece:** Disco's split backward. It allows 16 or more distinct cutouts without touching U-Net VRAM.
- At 128px the original sampler's cutouts can only be the whole frame, so `distinct_cutouts()` in `cgd/modules.py` collapses them to one. The README's 128px tie is therefore 4 distinct cutouts against a single view, and the README reports no test of many distinct cutouts at 128px or 256px.
- `cgd/modules.py` resizes with `adaptive_avg_pool`, which is a box filter. At 128px every cutout is enlarged to 224, so that filter duplicates pixels. v-diffusion, by contrast, upsamples `min(shape) < 256` bilinearly before cutting, and k-diffusion and Disco use `resize()`. See N1.
- `clamp_max` values do not port to this repo, because `-rate` caps guidance relative to the noise and already removes the dependence on step count that those values needed.
- The cut schedules are quoted against 1000-index tables at 250 steps, so the schedule *shape* is what ports, not the counts.

### Gaps
- Zippy's Disco Diffusion Cheatsheet returned HTTP 410 on both URL forms, and web.archive.org could not be fetched. Its recommended ranges are not in these notes, only the notebook it documented.
- "A Traveler's Guide to the Latent Space" rendered empty. The EZ Charts and the "Disco Diffusion Modifiers" guide were not reached. Reddit and Discord archives were not reachable either.
- nshepperd's `jax-guided-diffusion` README returned 404 on both `main` and `master`. I could not read the JAX notebook directly, only what Disco credits to it. I found nothing attributable to jarrodmiller.
- I could not find a controlled comparison of the secondary model against the full U-Net.

## 2. Which prompting tricks are specific to CLIP, and is prompt expansion useful?

### Takeaway
Rewording alone showed no measurable effect in the one controlled study. Negative prompts against failure modes, stacked style tokens and several phrasings of the same idea have the strongest practitioner support, but nobody seems to have measured their effect on scores. Any test must judge against the bare prompt, or the score only measures how far the target moved.

### Cited Findings
- Controlled study on VQGAN+CLIP: nine permutations of the same keywords gave "no significant difference" over 143 annotated grids. Seed did matter (p<0.01). More iterations did not correlate with satisfaction: 100 to 500 sufficed, with 200 preferred most. [paper, measured] — [Liu and Chilton, CHI 2022](https://arxiv.org/abs/2109.06977)
- DAS uses weighted text as a corrective: -0.3 "Optical Character Recognition" to stop CLIP writing text, +0.3 "octane render, unreal engine, ray tracing, volumetric lighting", -0.3 "multiple exposure" against stacked images, and -0.6 on OCR in its Figure 7. [paper] — [Direct Ascent Synthesis](https://arxiv.org/abs/2502.07753)
- Negative examples from practice: "text:-.9:-1", and "dof:-1" with "blur:-1". "Beeple" is said to add blur, orbs and colour, and "kinkade" colour. [community] — [pytti Grimoire](https://github.com/pytti-tools/pytti-book/blob/main/Grimoire.md); [gist](https://gist.github.com/lxe/ec08c4ecc1f79a084d47e1b82d3a65ff)
- This repo's own SigLIP2 run "wrote nonsense text across" a sushi prompt, which is the failure the OCR negative targets. [here] — local `README.md`
- "Say the same thing in multiple different ways", for example nine pipe-separated phrasings of "queen". Also name forums and sites whose images have the look you want, and name a medium. [community] — [pytti Grimoire](https://github.com/pytti-tools/pytti-book/blob/main/Grimoire.md)
- "Trending on Artstation" gave a painterly look "with lots of cyan, red and purple" in a 126-modifier visual comparison. No scores were recorded. [community] — [NightCafe experiment, archived](https://gwern.net/doc/www/creator.nightcafe.studio/c2b3f7f8f7d2e84dba6f058ccb3d4813d6d3a061.html)
- `fuzzy_prompt` adds several noisy copies of each embedding with `rand_mag` 0.05, then clamps to 0-1. It is off by default. [code] — [disco.py](https://github.com/alembics/disco-diffusion/blob/main/disco.py)
- Latent Majesty adds LAION aesthetic embeddings to the text target, `target + aesthetic_embedding * weight` then renormalises, and also descends an aesthetic predictor at `aesthetic_loss_scale = 400`. [code] — [Majesty latent.ipynb](https://github.com/multimodalart/majesty-diffusion/blob/main/latent.ipynb)
- Prompt schedules: pytti moves between scenes by "linearly interpolating", with 200 `interpolation_steps` given as a good default. [code] — [pytti Settings](https://github.com/pytti-tools/pytti-book/blob/main/Settings.md). MagicMix is the published relative, blending two semantics on a diffusion model. [paper] — [MagicMix](https://arxiv.org/abs/2210.16056)
- Template ensembling: OpenAI uses 80 ImageNet templates, and forward selection stopped at 7. [code] — [CLIP prompt notebook](https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb). This repo picks its class with one template, `a photo of a {class}`. [here] — local `cgd/clip_util.py`
- Practitioners found GAN outputs out-scoring the best human drawing on CLIP similarity, which the author attributed to the GAN being able to "cheat" by colluding with the judge. [community] — [Roboflow](https://blog.roboflow.com/ai-generated-art/)

### Inferences
- An LLM that only rewrites a prompt is unlikely to help, given the permutation result. If expansion helps, it will be by adding new concrete tokens or by averaging over phrasings, and a template list can do both without a model.
- Because guidance text and judge text must differ, score only on the bare prompt, never on the expanded one.
- Style tokens probably trade fidelity for look, and a CLIP judge cannot see that trade (Section 7). Eyes-on checks are required here.
- Prompt schedules were an animation feature. Using them inside one 30-step run, with style only in the later steps, is a small untested idea (N9).

### Gaps
- I found no controlled measurement of what "trending on ArtStation" does to CLIP scores, despite targeted searching.
- I found no measured study of LLM prompt expansion under CLIP guidance.
- Liu and Chilton studied VQGAN+CLIP, so those results may not transfer to diffusion.

## 3. What were nshepperd's and crowsonkb's own later methods, and how do they relate to this repo?

### Takeaway
This repo's fast sampler is close to Crowson's later design, with its own capped guidance and second-order rule. Two differences stand out: her later code dropped the `fac` blend entirely, and it replaced cutouts with one full-frame view plus a one-patch translation. Text-conditional v-diffusion models also exist with public weights, and Majesty ran two models in parallel.

### Cited Findings
- `k-diffusion/sample_clip_guided.py`: guidance on `denoised` alone with no blend, `cond_denoised = denoised + grad * sigma**2`, then a static clamp to ±1 after guidance. Defaults are ViT-B/16, scale 500, 100 steps, Karras sigmas with rho 7, and `sample_dpmpp_2s_ancestral` with eta 1. There are no TV or range losses. [code] — [sample_clip_guided.py](https://github.com/crowsonkb/k-diffusion/blob/master/sample_clip_guided.py)
- Its only augmentation is `KA.RandomAffine(0, (1/14, 1/14), p=1, padding_mode='border')`, one view per sample, resized with reflect padding. Note 224/14 = 16, one ViT-B/16 patch. [code] — [sample_clip_guided.py](https://github.com/crowsonkb/k-diffusion/blob/master/sample_clip_guided.py)
- `v-diffusion-pytorch/clip_sample.py`: guidance on the x0 estimate `pred`, again no blend, 16 cutouts by default, and images under 256 are bilinearly upsampled 2x before cutting. [code] — [clip_sample.py](https://github.com/crowsonkb/v-diffusion-pytorch/blob/master/clip_sample.py)
- v-diffusion public models: `cc12m_1_cfg`, a 602M parameter 256px CLIP-conditioned model with CFG, "the recommended model"; `yfcc_1` and `yfcc_2` unconditional at 512px; `wikiart_256`, `imagenet_128`, `danbooru_128`. CFG sampling is labelled "best". [code] — [v-diffusion README](https://github.com/crowsonkb/v-diffusion-pytorch/blob/master/README.md)
- V-Majesty ran "parallel multi-model diffusion (e.g.: run `cc12m_1` and `yfcc_2` at the same time - with or without lerping)". This is direct practitioner precedent for mixing models at sampling time. [code] — [Majesty README](https://github.com/multimodalart/majesty-diffusion/blob/main/README.md)
- `sample_dpmpp_2m_sde` takes `solver_type` `'heun'` or `'midpoint'`, default midpoint, and `sample_dpmpp_3m_sde` also exists. [code] — [k_diffusion/sampling.py](https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py)
- Repo sampler: DPM-Solver++(2M) SDE at `eta` 0.5 on a uniform logSNR grid, with the second-order correction taken from the unguided x0 only. [here] — local `cgd/sampling.py`
- Min-SNR is by Hang et al., not Crowson. [paper] — [Min-SNR](https://arxiv.org/abs/2303.09556). k-diffusion implements a `soft-min-snr` weighting. [code] — [k_diffusion/layers.py](https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/layers.py). Her later pixel-space work is the Hourglass DiT. [paper] — [HDiT](https://arxiv.org/abs/2401.11605)
- `k_diffusion/sampling.py` contains no mention of Anderson acceleration, and a search for it alongside k-diffusion and CLIP guidance returned nothing. [code] — [k_diffusion/sampling.py](https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py)

### Inferences
- The repo's update `sigma**2 / alpha * gradient` is the VP form of k-diffusion's `sigma**2`. The repo's `0.5 * x0_weight * (h / previous_h) * (x0 - previous_x0)` is the midpoint variant. It therefore already matches `sample_dpmpp_2m_sde(solver_type='midpoint')` apart from the deliberate guidance handling.
- Min-SNR, soft-min-SNR and HDiT are training-time work, so they only matter if this project trains a model.
- The 1/14 translation looks like deliberate patch-grid jitter, since k-diffusion defaults to ViT-B/16, but no source says so. The equivalent for ViT-B/32 would be 1/7.
- Crowson dropping `fac` is a signal rather than proof. Her later models were also trained differently, so it should be tested here (N3).

### Gaps
- I could not connect Crowson to Anderson acceleration. An arXiv query for it with "diffusion models" returned no entries, and that query may have been too narrow.
- I found no written explanation from either author of why they dropped the blend or reduced cutouts.

## 4. What is the evidence on noise-conditional gradients, and can noise or dropout replace the fac blend?

### Takeaway
The claim is well supported: stock CLIP was not trained on noisy images, and a noise-aware CLIP does better per GLIDE. A public noise-aware CLIP exists but only at 64px on filtered data. The cheaper path is LGD-MC, which adds noise around x0 at CLIP level with no extra U-Net calls. The `fac` blend already does something like this by accident, and only near the end of the guidance window.

### Cited Findings
- `fac` comes from Crowson's notebook: `fac = sqrt_one_minus_alphas_cumprod[cur_t]`, `x_in = pred_xstart * fac + x * (1 - fac)`. Disco kept it unchanged, and `clip_denoised` defaults to False. [code] — [EleutherAI app.py](https://huggingface.co/spaces/EleutherAI/clip-guided-diffusion/blob/main/app.py); [disco.py](https://github.com/alembics/disco-diffusion/blob/main/disco.py)
- GLIDE hypothesised that poor CLIP guidance results come from CLIP not being trained on noised images. It trained a noised ViT and reported its simpler approach produced "images of equal or higher quality" than the notebook. CFG still beat CLIP guidance overall. [paper] — [GLIDE](https://arxiv.org/abs/2112.10741)
- GLIDE's public checkpoints include `clip/image-enc` and `clip/text-enc`, and the repo describes "a filtered noise-aware CLIP model". [code] — [glide-text2im README](https://github.com/openai/glide-text2im/blob/main/README.md); [download.py](https://github.com/openai/glide-text2im/blob/main/glide_text2im/download.py)
- UPainting avoided noisy CLIP as too costly to train, fed CLIP the predicted clean image, and reported that CLIP guidance combined with CFG gives higher quality than either. [paper] — [UPainting](https://arxiv.org/abs/2210.16031)
- Noise smooths a classifier: for a loss bounded by C, the noise-smoothed loss is C·√(2/(πσ²))-Lipschitz, so "Adding Gaussian noise transforms non-Lipschitz classifiers into Lipschitz ones". Training-free guidance is more exposed to adversarial gradients than noise-trained classifiers. [paper] — [Shen et al., NeurIPS 2024](https://arxiv.org/abs/2403.12404)
- LGD-MC replaces the point estimate with `∇ log (1/n) Σ exp(-ℓ(x⁽ⁱ⁾))`, with `x⁽ⁱ⁾ ~ N(x̂₀, r_t²I)` and `r_t = σ_t / √(1+σ_t²)`. The authors stress "the diffusion model is only called once". [paper] — [Song et al., ICML 2023](https://proceedings.mlr.press/v202/song23k.html)
- LGD-MC scored lowest of the methods on CLIP-guided ImageNet in Shen et al. (24.3, against 25.1 to 27.7). A newer paper adds a backward step before Monte Carlo sampling (ABMS). [paper] — [Shen et al.](https://arxiv.org/abs/2403.12404); [ABMS, ICLR 2026](https://arxiv.org/abs/2603.06685)
- PPAP guides GLIDE with several specialists, one per noise range, using data-free knowledge transfer, so off-the-shelf classifiers become usable with few trainable parameters. [paper] — [Go et al., CVPR 2023](https://arxiv.org/abs/2212.05973)

**What the blend does in this repo** [derived, ADM linear schedule, local calculation]. Writing `x = αx̂₀ + σε̂` gives `x_in = (σ + (1-σ)α) x̂₀ + (1-σ)σ ε̂`:

| t | σ | x̂₀ coefficient | noise coefficient | direct path weight 1-σ |
|---|---|---|---|---|
| 800 | 0.999 | 0.999 | 0.001 | 0.1% |
| 700 | 0.997 | 0.997 | 0.003 | 0.3% |
| 500 | 0.960 | 0.971 | 0.038 | 4% |
| 300 | 0.778 | 0.918 | 0.172 | 22% |
| 100 (window end) | 0.324 | 0.964 | 0.219 | 68% |

### Inferences
- The blend is effectively pure x̂₀ for most of the window, then adds up to 0.22 of *the model's own noise estimate* near the end, not fresh noise.
- At the end of the window, about two thirds of the Jacobian weight is a direct path bypassing the U-Net. That may explain why low-noise guidance reads as adversarial detail, as `cgd/sampling.py` observes. This is a hypothesis, since Jacobian weight is not the same as gradient share.
- TV loss is also computed on `x_in`, so near the window's end it partly penalises that noise term rather than the image.
- In VP terms the LGD formula simplifies. With σ_VE = σ/α, `r_t = σ` exactly, for unit-variance data. For images in [-1, 1] the variance is nearer 0.25, so `r_t` should scale by roughly 0.5, and in practice a tunable `k·r_t` is needed. This is my derivation, not the paper's.
- "x0 with deterministic dropout" is feasible in principle: seeded token dropout on the ViT would keep a seed reproducible. Nothing I found tests this. It belongs after N3, since noise has theory behind it and dropout has none.
- Any new randomness must use seeded generators, and new resamplers should keep the weight-matrix form, given the determinism work in `cgd/modules.py`.

### Gaps
- I found no public noise-aware CLIP at 224px, and no paper with FID for noise-aware against plain CLIP guidance on the same model.
- I found no source explaining why `fac` was chosen.
- I found no work on token dropout for CLIP guidance.

## 5. Which recent directions could graft onto this pipeline?

### Takeaway
DAS is the most interesting graft because its parameterisation can become a gradient filter or an x0 inner loop at near-zero cost. Noise optimisation works in the literature but costs far more than this sampler, unless a one-step proxy or a cheap prescreen is used. Pixel optimisation with SDS is likely the wrong way round for this project, and robust CLIP models have not been tried here at all.

### Cited Findings

**Direct Ascent Synthesis**
- An image is a tanh-squashed sum of components from 1x1 to 224x224, optimised directly on CLIP with no generator. Output follows a 1/f² spectrum (slope about -2). [paper] — [Fort and Whitaker, 2025](https://arxiv.org/abs/2502.07753)
- Settings: 100 steps of SGD at lr 0.2, noise std 0.2, jitter ±56 (a 336 canvas centre-cropped to 224), 32 augmentations per step, three ViT-B/32 CLIPs. The two augmentations "prove crucial" and "neither is sufficient alone". Ensembling helps "marginally" and "a single model is sufficient", but "some CLIP models were particularly bad at being turned into generators without any obvious reason why". It takes seconds on one GPU. [paper] — [DAS](https://arxiv.org/abs/2502.07753)
- The authors list "Merging DAS with diffusion models" as future work. [paper] — [DAS](https://arxiv.org/abs/2502.07753)

**Robust CLIP**
- CLIPAG adversarially fine-tunes CLIP's vision tower (OpenCLIP ViT-B/32, L2 ε=1.5) and finds robust vision-language models have perceptually aligned gradients. Swapping it into VQGAN+CLIP raised CLIPScore on 100 prompts from 33.4 to 34.3, and its generator-free mode reached 36.8 after 1000 steps. Pretrained weights were promised, and I could not confirm release. [paper] — [CLIPAG, WACV 2024](https://arxiv.org/abs/2306.16805)
- FARE fine-tunes CLIP without labels, for about 0.2% of the original training cost. Public `hf-hub:chs20/fare2-clip` and `chs20/fare4-clip` are ViT-L/14, radius 2/255 and 4/255, and load through open_clip. [paper, code] — [Robust CLIP](https://arxiv.org/abs/2402.12336); [fare2-clip](https://huggingface.co/chs20/fare2-clip); [fare4-clip](https://huggingface.co/chs20/fare4-clip)
- CLIP-JEM follows on from this line and reports better guidance for CLIP-based generators. The later paper also notes CLIPAG's outputs are often non-realistic and depend heavily on augmentations. [paper] — [CLIP-JEM](https://arxiv.org/abs/2408.17046)
- Robust classifier guidance on diffusion was preferred by raters in a human preference survey. This comes from a search summary of the PDF, and I have no FID figure for it. [paper] — [Kawar et al.](https://arxiv.org/abs/2208.08664)

**Noise optimisation**
- DOODL optimises the initial noise through the whole denoising chain, using invertibility for memory, and beats one-step classifier guidance on computational and human metrics, including CLIP guidance on DrawBench. [paper] — [DOODL](https://arxiv.org/abs/2303.13703)
- ReNO runs 50 gradient ascent iterations on noise through one-step models in 20-50 s, and was preferred "nearly twice as often" to SDXL in user studies. [paper] — [ReNO](https://arxiv.org/abs/2406.04312)
- DNO treats sampling as a noise-to-sample map and can work with or without gradients. [code] — [Direct Noise Optimization](https://github.com/TZW1998/Direct-Noise-Optimization)
- Newer work avoids per-image optimisation: a trained noise-refining network, and gradient-free noise selection. [paper] — [A Noise Is Worth Diffusion Guidance, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/985786d06c1e45e9e8c65f7aca3547e4-Paper-Conference.pdf); [The Silent Prompt](https://arxiv.org/html/2412.05101v1)
- Seed choice alone matters: "the choice of initializing seed can significantly vary the quality". [paper] — [Liu and Chilton](https://arxiv.org/abs/2109.06977)
- FuseDream initialises from the top-scoring of many random BigGAN samples, scored with an augmented CLIP score that resists an FGSM attack which fools plain CLIP. [paper] — [FuseDream](https://arxiv.org/abs/2112.01573)

**Training-free guidance family**
- Direct relatives of this repo's guidance: [DPS](https://arxiv.org/abs/2209.14687), [Universal Guidance](https://arxiv.org/abs/2302.07121), [FreeDoM](https://arxiv.org/abs/2303.09833).
- MPGD differentiates with respect to x̂₀ rather than x_t, and projects with an autoencoder. It reports speedups up to 3.8x. [paper] — [MPGD](https://arxiv.org/abs/2311.16424); [Shen et al.](https://arxiv.org/abs/2403.12404)
- On unconditional ImageNet with CLIP text guidance over 100 DDIM steps, CLIP Scores were LGD-MC 24.3, MPGD-Z 25.1, UG 25.7, FreeDoM 25.9, and FreeDoM plus random augmentation and Polyak step size 27.7. [paper] — [Shen et al.](https://arxiv.org/abs/2403.12404)
- That augmentation is DiffAugment-style translation, resizing, colour and cutout with 10 transforms. The step size scales by `‖ε‖ / ‖g‖²`. The authors admit it still needs more model calls than trained methods. [paper] — [Shen et al.](https://arxiv.org/abs/2403.12404)

**Score distillation and pixel-space priors**
- SDS optimises parameters with a diffusion prior. The LGD paper criticises this family, which it calls D-PnP: it converges at a local minimum and may take many iterations, for example 15,000 in DreamFusion. [paper] — [DreamFusion](https://arxiv.org/abs/2209.14988); [Song et al.](https://proceedings.mlr.press/v202/song23k.html); [Graikos et al.](https://arxiv.org/abs/2206.09012)
- SDS is commonly reported to oversaturate and oversmooth and is sensitive to CFG scale. VSD addresses this by fine-tuning an extra U-Net during optimisation. This comes from a search summary that did not tie the SDS claim to one paper, and I did not read these papers directly. [paper] — [Rethinking Score Distillation](https://arxiv.org/html/2406.09417v2); [ProlificDreamer summary](https://deepai.org/publication/prolificdreamer-high-fidelity-and-diverse-text-to-3d-generation-with-variational-score-distillation)
- One-step routes: [Diff-Instruct](https://arxiv.org/abs/2305.18455) transfers a pretrained diffusion model into a one-step generator, and [Consistency Models](https://arxiv.org/abs/2303.01469) generate in one step by design. [paper]

**Composition**
- Diffusion models can be composed as energy models to bind concepts one model cannot. [paper] — [Liu et al., ECCV 2022](https://arxiv.org/abs/2206.01714)
- Existing composition is flawed, the sampler rather than the model is to blame, and MCMC-style samplers fix it. [paper] — [Du et al., ICML 2023](https://arxiv.org/abs/2302.11552)

**Class handling**
- `cgd/sampling.py` records that averaging dissimilar class embeddings makes the network output noise. [here] — local `cgd/sampling.py`
- `y` is a single class chosen once from a weighted prompt query, by nearest text feature. [here] — local `cgd/cgd.py`, `cgd/clip_util.py`

### Inferences
- DAS's gradient on each `P_r` is a downsampled gradient. Summing the resulting updates amounts to a spectral reweighting of one pixel gradient. That can be applied to the existing guidance gradient with no extra CLIP calls (N5).
- ReNO's 20-50 s depends on one-step models. A 128px run here takes 6.5 s, so 50 iterations through the 30-step chain would take over 5 minutes, before the added cost of backpropagating through every step. Only a proxy is sensible here.
- Averaging class *embeddings* fails, but composition happens in score space, which is a different operation. Averaging predictions across classes is grounded in Liu and Du, even though the embedding mean was not.
- Score distillation is mostly a way around a lack of sampler for non-image parameters. For images, sampling already exists. It is worth one control run at most (N22).

### Gaps
- A search summary put DOODL at about 10 minutes and D-Flow at about 40 minutes per image, but I could not pin those figures to a specific paper. The ICCV PDF of DOODL returned 403, so I have its abstract only and no CLIP-score numbers.
- The Universal Guidance forward, backward and self-recurrence details are from memory plus Shen's description of time travel. I only retrieved the abstract.
- I found nothing on interpolating ADM class embeddings with any guard, and nothing on optimising them directly.
- No published work I found runs DAS with diffusion, CLIP noise optimisation with no U-Net gradient, or a sequential class schedule on ADM.

## 6. Which 2024 to 2026 work helps old U-Nets, and what is the state of the art in model-free guidance?

### Takeaway
The strongest finding is that ADM itself gained a lot from self-guidance: PAG took ADM ImageNet 256 from FID 10.94 to 6.32 with no training and one extra forward pass. ICG gives class-conditional models a CFG-like signal with no null token, which suits ADM exactly. Beyond this, the state of the art is autoguidance on newer networks, and this repo's guidance window has independent support from the limited-interval paper.

### Cited Findings
- **PAG on ADM ImageNet 256:** conditional FID 10.94 to 6.32 (SAG 9.41), unconditional 26.21 to 16.23 (SAG 20.08), scale 1.0 for FID, 3.0 for qualitative results. Perturbed layers: conditional `input_blocks.14.1`; unconditional `input_blocks.14.1`, `16.1`, `17.1`, `middle_block.1`. Run with 250 step DDPM. [paper] — [PAG](https://arxiv.org/abs/2403.17377)
- The PAG authors note FID worsens at higher scales while images look better, and attribute this to FID misalignment with perception. [paper] — [PAG](https://arxiv.org/abs/2403.17377). SAG is the earlier blur-based method. [paper] — [SAG](https://arxiv.org/abs/2210.00939)
- **ICG:** replaces the unconditional branch with an independent condition: `D(z,t,ŷ) + w(D(z,t,y) - D(z,t,ŷ))`, with ŷ a random class or scaled Gaussian. FID matches CFG: DiT-XL/2 5.50 against 5.56, EDM2-XS 3.35 against 3.36, SD 20.05 against 20.13. Scales: 1.25 on EDM2, 1.4 on DiT. [paper] — [Sadat et al., ICLR 2025](https://arxiv.org/abs/2407.02687)
- **TSG:** perturbs the time embedding: `t̃ = t + s·tᵅ·n`, with `D(z,t̃) + w(D(z,t) - D(z,t̃))`. DiT unconditional FID 48.67 to 29.03, conditional 15.49 to 6.39. Settings on DiT conditional: w 2.5, s 2, α 1. Unconditional: w 5.0, s 1.0, constant, T in [200, 800]. ADM was not evaluated. [paper] — [Sadat et al.](https://arxiv.org/abs/2407.02687)
- **Limited interval:** guidance is "clearly harmful toward the beginning of the chain, largely unnecessary toward the end, and only beneficial in the middle". Record ImageNet-512 FID 1.81 to 1.40, and EDM2-XXL FD-DINOv2 33.09 to 29.16. [paper] — [Kynkäänniemi et al., NeurIPS 2024](https://arxiv.org/abs/2404.07724)
- **Autoguidance:** guide with a smaller, worse network. EDM2-S FID 2.56 to 1.34, record 1.25 at 512, 1.01 at 64, unconditional EDM2-S 11.67 to 3.86, all with public networks. [paper] — [Karras et al., NeurIPS 2024](https://arxiv.org/abs/2406.02507)
- **Saturation:** APG splits the guidance update against the prediction. The parallel component mainly raises saturation and the orthogonal one carries the quality gain, so APG down-weights the parallel part. [paper] — [Sadat et al.](https://arxiv.org/abs/2410.02416). Frequency-decoupled guidance weights low and high frequencies separately. [paper] — [FDG](https://arxiv.org/abs/2506.19713)
- PixelAsParam treats guidance terms as competing update directions on pixels and resolves conflicts with a projection. [paper] — [Dinh et al., ICML 2023](https://proceedings.mlr.press/v202/dinh23a.html)
- Noise-aware classifier guidance is still public for ADM: `64x64`, `128x128`, `256x256` and `512x512_classifier.pt`, at scales 1.0, 0.5, 1.0 and 4.0, or 10.0 on 256 unconditional. Upsamplers `64_256_upsampler.pt` and `128_512_upsampler.pt` are also public. [code] — [guided-diffusion README](https://github.com/openai/guided-diffusion/blob/main/README.md)
- ADM-G FID: 2.97 at 128, 4.59 at 256, 7.72 at 512. With upsampling, 3.94 at 256 and 3.85 at 512. [paper] — [Dhariwal and Nichol](https://arxiv.org/abs/2105.05233)
- Dynamic CFG scheduling via online feedback uses CLIP evaluators during sampling, on a CFG base. [paper] — [Dynamic CFG](https://arxiv.org/abs/2509.16131). Training-time removal of CFG reached FID 1.34 at ImageNet 256. [paper] — [Tang et al.](https://arxiv.org/abs/2502.12154)

### Inferences
- **The repo's window (10% to 80%) agrees with the limited-interval paper** even though that paper studies CFG, not CLIP. The `--uncond` exception also fits, because without a class CLIP has to set composition at high noise.
- PAG, ICG, TSG and ADM's classifier all act on the U-Net's *realism*, and none need text. That complements CLIP, which carries the prompt. They should raise quality rather than CLIP score, so a tie on score does not falsify them.
- The cascade halves FID at 512 (7.72 to 3.85), a gain of the same order as PAG's at 256 (10.94 to 6.32). Unlike PAG, it needs no change to guidance, because the upsampler runs after it. The two numbers come from different papers and samplers, so they are not directly comparable.
- `-rate` is structurally similar to Polyak scaling, since both normalise by the prediction's norm. APG's projection is the missing step.
- PAG's layer names are for the 256 network, so 128 needs its own layer search.

### Gaps
- I found no paper reporting ICG, TSG or PAG on ADM at 30 steps, or combined with CLIP guidance. All FIDs above are on long samplers.
- PAG's reported IS for conditional ADM (338.02 from 100.98) looked out of line with the rest of the table, so I left it out.
- I did not retrieve the guidance-interval bounds used on EDM2.
- I found no 2024 to 2026 paper reporting strongly better *CLIP* guidance on an old U-Net specifically.

## 7. Has anyone benchmarked CLIP-guided diffusion against modern methods?

### Takeaway
There is no direct benchmark of this pipeline against a modern model. Partial numbers suggest a gap of about 6 to 7 CLIP score points on the same judge family, but with different prompts and resolutions. The more important result is that CLIP score does not capture realism: generator-free CLIP optimisation can score above text-conditional models while having far worse FID.

### Cited Findings
- Stable Diffusion on ViT-B/16 CLIP score (×100): v1-4 34.91, v1-5 36.21, over 6 prompts, which the page itself warns is too few. [paper] — [Diffusers evaluation docs](https://huggingface.co/docs/diffusers/main/en/conceptual/evaluation)
- This repo on ViT-B/16: 0.292 on 12 held-out prompts at 128px, and 0.316 on 6 prompts at 256px `--uncond`. The 95% interval on a difference is about ±0.03. [here] — local `README.md`
- CLIPAG's table on COCO: CLIPAG IS 18.7, FID 42.3, CLIPScore 34.7, against CogView 33.2, AttnGAN 27.7, and GLIDE FID 12.2. The authors admit it does not rival the state of the art. [paper] — [CLIPAG](https://arxiv.org/abs/2306.16805)
- An FGSM attack raises plain CLIP score with no visible change. An augmented CLIP score resists this. [paper] — [FuseDream](https://arxiv.org/abs/2112.01573)
- GLIDE: CFG produced higher quality than CLIP guidance. [paper] — [GLIDE](https://arxiv.org/abs/2112.10741)
- Quantitative metrics "don't necessarily correspond to image quality", and CLIP-based ones can be biased. [paper] — [Diffusers evaluation docs](https://huggingface.co/docs/diffusers/main/en/conceptual/evaluation)

### Inferences
- Scoring by held-out CLIP models is the right test for *prompt fit*, and the wrong sole test for the user's actual complaint, which is about looking like a modern model. CLIPAG (FID 42.3, CLIPScore 34.7) shows the two can separate widely.
- CLIP models share weaknesses, so adversarial gain may transfer to held-out judges. Judges should average over random shifts and crops, as AugCLIP does, or the metric will reward exactly what is being removed.
- Comparing 0.292 with 0.349 is indicative only. To close that question, run SD 1.5 on the same 12 prompts and judges, which costs minutes and makes the gap measurable.

### Gaps
- I found no published benchmark of CLIP-guided ADM or Disco Diffusion against modern models with concrete numbers.
- The only CLIP-guided ImageNet numbers found (24.3 to 27.7) use different prompts from this repo, and I did not establish which judge model they used.

## 8. Novel ideas to test, each with a hypothesis, a cheap test and a falsifier

### Takeaway
Twenty-two tests follow, ordered by cost. The first nine need no new checkpoints, though N8(d) needs the LAION embedding file. For visible realism, the best-supported bets are the ADM upsampler cascade (N14), PAG or ICG self-guidance (N10, N11) and the ADM classifier as a second guide (N12). No single idea here is likely to reach modern quality, because the prior knows only 1000 ImageNet classes. Only N20 and N21 change that.

### Cited Findings
The evidence behind each idea is cited in Sections 1 to 7, and each entry below points to it. All ideas are mine unless a paper is named, and none has been run. Budgets reference local `README.md`: 128px runs in 6.5 s and 2.4 GiB, 256px in 11.4 s and 5.3 GiB, 512px in 22.4 s and 4.1 GiB, all on the RTX 2070 SUPER (8 GB). Anything that fits there also fits the RTX 4070 (12 GB).

### Inferences

#### Shared test protocol
- **Prompts and baseline:** the 12 held-out prompts, seeds 0, 1 and 2, against `uv run cgd` defaults at 128px unless noted. The prompt is the unit of analysis, and differences are paired.
- **Judges:** `ViT-B/16`, `ViT-H-14-quickgelu`, `ViT-L-16-SigLIP2-256`. A model never grades its own guidance, and `ViT-B/16` is discounted when guidance uses ViT-B/32 kin.
- **Scoring:** always against the *bare* prompt. Average each judge over 8 random shifts and crops, so adversarial gain does not count (Section 7).
- **Also record:** clipped-pixel share, spectrum slope, seconds, peak VRAM, and a contact sheet.
- **Fidelity ideas pass** if the mean paired change is positive on at least 2 of 3 judges, at least 9 of 12 prompts improve, and clipped pixels rise by no more than 1 point. **They are falsified** if the best arm's change is at or below +0.005 on at least 2 of 3 judges. Between the two is "no evidence", since 12 prompts resolve about ±0.03.
- **Realism ideas** (marked R) are not expected to move the score. **They are falsified** if the change is at or below -0.01 on at least 2 judges, *or* the user prefers the new image in fewer than 9 of 12 shuffled, blind A/B pairs. By chance, 9 or more of 12 happens about 7% of the time.

#### Tier 1: code changes only, no new checkpoints

**N1. Smooth resampling into CLIP below 224px.** Grounding: `cgd/modules.py` box-filters upward; v-diffusion and k-diffusion do not (Sections 1, 3).
- Hypothesis: box upsampling gives CLIP stair-stepped input and grid-aligned gradients, so a triangle or bicubic kernel raises scores at 64 and 128 but not at 256 and up.
- Test: swap the bins for a triangle kernel when `size > side`, keeping the weight-matrix form for determinism. Under an hour.
- Falsifier: shared rule at 128, or no change in the spectrum at the upsampling grid frequency. Any gain at 256 means the mechanism is wrong.

**N2. Stop differentiating through the U-Net.** Grounding: MPGD, Disco's secondary model, and Shen's table, where MPGD-Z sits about 3% below FreeDoM (Sections 1, 5).
- Hypothesis: a gradient on x̂₀ scaled by 1/α keeps the score at under 60% of guided-step time and memory, freeing budget for N4.
- Test: compute `loss` on detached `x0`, take the gradient on `x0`, divide by `alpha`. Measure at equal wall-clock after reinvesting the saving in cutouts.
- Falsifier: change at or below -0.01 on 2 judges at equal wall-clock. Conflict to watch: the community says the secondary model hurts quality.

**N3. Replace the fac blend with per-cutout noise around x̂₀.** Grounding: LGD-MC with one model call, Shen's Lipschitz result, Crowson dropping the blend (Sections 3, 4).
- Hypothesis: fresh noise at `k·r_t` per cutout smooths CLIP better than the model's own noise estimate, especially near the end of the window, and may let the window run below 10%.
- Test: arms are current blend, pure x̂₀, and `x̂₀ + k·r_t·n` for k in {0.1, 0.25, 0.5}, with a LogMeanExp aggregate across cutouts. Then sweep the window's low end. No extra U-Net cost.
- Falsifier: shared rule. LGD-MC's weak row in Shen's table makes this a real risk.

**N4. Minimal augmentation plus accumulated cutouts.** Grounding: k-diffusion's translation, DAS's shift and noise, Shen's augmentation, Disco's split backward (Sections 1, 3, 5).
- Hypothesis: translation by ±1 ViT patch with small noise beats no augmentation, while the repo's strong set with perspective does not, and 16 distinct cutouts beat 4.
- Test: split `guidance_loss` into the two stages Disco uses. Arms: {4, 8, 16} cutouts crossed with {none, shift, shift + noise}, at 128 and 256. Uses gradient accumulation so VRAM stays flat.
- Falsifier: shared rule. If 16 ties 4 with augmentation on, then the 2021 cutout budget was not doing what the community thought.

**N5. Multi-resolution preconditioning of the guidance gradient.** Grounding: DAS, and FDG as a cousin (Sections 5, 6).
- Hypothesis: replacing `g` with `Σ_r w_r up(down(g, r))`, renormalised to the same RMS, removes adversarial high frequencies, so `-rate` can rise and `-tvs` can fall to 0.
- Test: about 15 lines. Sweep `-rate` {0.08, 0.12, 0.16} with `-tvs` {150, 0}.
- Falsifier: shared rule, or gradient spectrum unchanged at equal rate. Note this would be a new use of DAS, as the paper only proposes merging with diffusion.

**N6. Project out the saturating component.** Grounding: APG on CFG, PixelAsParam (Section 6).
- Hypothesis: the part of the CLIP update parallel to x̂₀ drives clipping, so down-weighting it cuts clipped pixels at equal score and allows a higher rate.
- Test: weight η in {0, 0.25, 0.5, 1} on the parallel part, then try momentum on the orthogonal part.
- Falsifier: clipped share does not fall by at least 1 point at equal score. APG was shown on CFG only, so it may not carry over to CLIP.

**N7. Dynamic thresholding.** Grounding: Majesty (Section 1).
- Hypothesis: percentile thresholding beats the static 1.2 clamp on clipped pixels and on tone, without changing the prompt fit.
- Test: percentile in {0.95, 0.99, 0.995}, applied to `guided_x0`.
- Falsifier: R rule, plus clipped pixels must fall.

**N8. Text-side changes.** Grounding: DAS weights, pytti, Majesty, OpenAI templates (Section 2).
- Hypothesis (a): adding "Optical Character Recognition:-0.3" stops text on models prone to it, without lowering the bare-prompt score.
- Hypothesis (b): averaging embeddings from 8 phrasings or templates raises the bare-prompt score by reducing target-specific adversarial gradient.
- Hypothesis (c): class selection using the 7 selected templates picks a better class than the single template.
- Hypothesis (d): mixing the LAION rating-9 embedding at 0.1 to 0.3 improves look at little cost.
- Test: (a) and (c) need under a minute each per prompt. For (c), also count how often the class changes.
- Falsifier: shared rule for (b) and (c), R rule for (a) and (d). If (c) changes the class on fewer than 2 of 12 prompts, it is moot.

**N9. Schedules and same-architecture ensembles.** Grounding: Disco, Majesty and DAS (Sections 1, 2).
- Hypothesis: overview early and crops late beats fixed cuts. An `eta` of 0.8 falling to 0.2 beats a fixed 0.5. A guidance rate that declines through the window beats a fixed one. Style tokens only after halfway keep more prompt fit. OpenAI plus laion2b ViT-B/32 beats one model, as a cheap ensemble.
- Test: one factor at a time, at 256 for cuts because crops at 128 are heavily upscaled.
- Falsifier: shared rule, per factor. DAS already reports only a "marginal" ensemble gain, so expect a tie there.

#### Tier 2: extra forward passes, public checkpoints

**N10. ICG on ADM. (R)** Grounding: Section 6. ADM has no null token, which is what ICG was designed for.
- Hypothesis: `eps(x, ŷ) + w(eps(x, y) - eps(x, ŷ))` with a random class per step and w from 1.25 to 3 sharpens subjects at 30 steps.
- Test: one extra forward with no gradient, so about 50% more time and little more VRAM. Sweep w, window on and off.
- Falsifier: R rule. Watch for class-specific saturation, which would mean w is too high.

**N11. PAG on class-conditional ADM, and TSG for `--uncond`. (R)** Grounding: Section 6. This is the only one with an ADM result, FID 10.94 to 6.32.
- Hypothesis: the gain survives a 30-step sampler and CLIP guidance.
- Test: forward hook replacing attention weights with identity at `input_blocks.14.1` at 256, then a layer search at 128. For `--uncond`, TSG with `s` near 1, `w` near 3 to 5.
- Falsifier: R rule. If PAG works alone but not with CLIP, try applying it only outside the CLIP window.

**N12. ADM classifier as a second guide. (R)** Grounding: Section 6, FID 10.94 to 4.59 at 256.
- Hypothesis: the classifier supplies class-level realism while CLIP steers, and it adds no U-Net backward.
- Test: download `128x128_classifier.pt`, add `sigma**2 / alpha * scale * ∇log p(y|x)` with `scale` from 0.25 to 0.5, capped by `-rate` alongside CLIP.
- Falsifier: R rule, or a score fall above 0.01, meaning the class overrides the prompt. Prompts far from any class are the most likely failures.

**N13. Class handling.** Grounding: Section 5, plus Liu and Du for composition.
- (a) Composed classes. Hypothesis: `Σ w_i eps(x, c_i)` for the top 2 classes improves two-subject prompts, unlike the averaged embedding, at one extra forward per class. Falsifier: shared rule on two-subject prompts; if outputs turn to mush, add MCMC corrector steps per Du before rejecting.
- (b) Slerp with a guard. Hypothesis: the mean embedding fails partly because averaging shrinks its norm, so slerping two class embeddings stays on-manifold. Test: measure eps RMS and sample sanity across λ from 0 to 1, with a guard rejecting any λ where it leaves the single-class range. Falsifier: any λ from 0.2 to 0.8 gives noise.
- (c) Class schedule. Hypothesis: one class for layout, then re-ranking by CLIP on x̂₀ at the midpoint, improves on a fixed class. Falsifier: shared rule.
- (d) Direct embedding optimisation. Hypothesis: a few CLIP gradient steps on `y`'s embedding, projected to the top-k convex hull with renormalisation, outperform top-1. Falsifier: shared rule, or instability as in (b).

**N14. ADM upsampler cascade. (R)** Grounding: FID 7.72 to 3.85 at 512, and Centipede in practice (Sections 1, 6).
- Hypothesis: CLIP-guided 128px then `128_512_upsampler.pt` beats direct 512px on look, with similar score, for less time.
- Test: no gradient in the upsampler, so VRAM stays near an unguided pass. Also 64 to 256. Try low-strength CLIP guidance inside the upsampler as a second arm.
- Falsifier: R rule, comparing with direct 512 at equal wall-clock.

**N15. Seed triage.** Grounding: Liu's seed result, FuseDream, Centipede's manual rejection (Sections 1, 2, 5).
- Hypothesis: AugCLIP on x̂₀ partway down the schedule predicts the final score, so sampling K seeds and finishing only the best beats one seed.
- Test: 8 seeds on 12 prompts, then Spearman between the score at each step and the final score. This is a pure analysis of saved runs.
- Falsifier: rank correlation below 0.3 at every step before halfway. The expected-gain estimate then follows directly from the seed spread.

#### Tier 3: larger builds

**N16. Noise pre-optimisation without U-Net gradients.** Grounding: DOODL, ReNO and DNO all backprop, and a search found no no-gradient version (Section 5).
- Hypothesis: treating `x̂₀(x_T)`'s Jacobian as identity, then renormalising to the Gaussian shell, still moves x_T toward prompts, and even unguided sampling from it scores higher.
- Test: 20 CLIP steps on x_T with 4 cutouts, about 5 s, then sample with and without guidance. Control: a true one-step Jacobian.
- Falsifier: unguided score gain below 0.01. The risk is real: x̂₀ at maximum noise is close to a class mean, so this may do nothing.

**N17. DAS as composition, and as an x0 inner loop.** Grounding: DAS and Centipede. Universal Guidance's backward step, as I recall it, is the nearest precedent (Sections 1, 5).
- (a) Hypothesis: 100 DAS steps at 128, then `-init das.png` at a skip of 500 to 700, gives CLIP's composition with the U-Net's realism. The test needs no repo changes. Falsifier: shared rule, plus the U-Net must visibly remove artifacts.
- (b) Hypothesis: 3 to 5 DAS-parameterised steps on x̂₀ inside each guided step replace the U-Net backward at equal or better score. Falsifier: shared rule at equal wall-clock.

**N18. Robust guide. (R and fidelity)** Grounding: CLIPAG and FARE (Section 5). Not tried here, and the repo's larger-model results do not cover it.
- Hypothesis: perceptually aligned gradients reduce adversarial detail, so TV can drop and the window can run later. The prediction is that FARE beats plain ViT-L/14. The README does not score plain ViT-L/14, so it needs its own baseline run.
- Test: load `hf-hub:chs20/fare2-clip` through open_clip with `-fp16`. Use FARE2 and FARE4, and ViT-L/14 as the direct control. Judges exclude ViT-L/14 kin.
- Falsifier: shared rule against plain ViT-L/14. Also fails if clipped share is unchanged and the gradient spectrum is not more 1/f-shaped. Note FARE was trained for classification robustness at 224px, and CLIP-JEM warns these outputs can be non-realistic.

**N19. Noise-aware CLIP.** Grounding: GLIDE's public encoder and PPAP (Section 4).
- Hypothesis: guiding on x_t directly works at high noise, where x̂₀ is uninformative, so `--uncond` can drop its doubled step count, with no U-Net backward.
- Test: on 64px, or downsampled from 128, use `clip/image-enc` with the timestep. Check whether its schedule matches via logSNR.
- Falsifier: shared rule on `--uncond`. Filtered training data means people and some concepts will be weak, so check per prompt category before rejecting.

**N20. Compose a text-conditional score with ADM.** Grounding: Liu, Du, V-Majesty's parallel models (Sections 3, 5). Fixes the 1000-class limit.
- Hypothesis: `eps_ADM(x, c) + w(eps_text(x, prompt) - eps_text(x, ∅))` gives the text semantics of a weaker text model on ADM's realism.
- Test: start with GLIDE base at 64 against ADM 64, which needs only a logSNR-based timestep map. Then `cc12m_1_cfg` at 256 against ADM 256. An estimated 1.2 GiB of fp16 weights should fit, and this needs checking. Mirror availability of the v-diffusion weights also needs checking.
- Falsifier: shared rule at 64. If it fails, try one MCMC corrector step per Du before rejecting.

**N21. Swap ADM for EDM2 with autoguidance.** Grounding: Section 6. EDM2-S at FID 1.34 against ADM-G's 7.72 at 512.
- Hypothesis: the same CLIP guidance, on a far better prior, gives the largest single jump in realism.
- Test: EDM2-XS guiding EDM2-S, with guidance through the VAE decoder at 512. VRAM for decoder backward is unknown and must be measured first on 8 GB, with checkpointing.
- Falsifier: R rule, or not fitting 8 GB. This is the costliest option and takes the project off ADM.

**N22. Score distillation with CLIP as a control.** Grounding: Section 5.
- Hypothesis: optimising pixels with an SDS term plus a CLIP term does *not* beat the sampler at equal U-Net calls. This is included to close the question.
- Test: 128px, class-conditional, 150 iterations, seeded from N17(a).
- Falsifier of the expectation: it beats the sampler on the shared rule. Oversaturation and oversmoothing are the expected failures.

#### Suggested order
1. Run SD 1.5 on the same prompts and judges, to put a number on the gap (Section 7).
2. N1, N3, N4: cheap, and they change what CLIP sees.
3. N2: if it holds, everything later gets cheaper.
4. N14, N11, N10, N12: the highest expected visible gain.
5. N5, N6, N7: use what N3 and N4 leave behind.
6. N18, N20 and N21, if the realistic ceiling above still is not enough.

### Gaps
- No idea above has been run, and every time and VRAM cost given for a new arm is an estimate.
- Twelve prompts cannot resolve effects under about ±0.03. Small true gains will read as ties, and promising ties need a larger prompt set.
- The R rule depends on one viewer's preference over 12 pairs, which is weak but cheap evidence.
- Whether the idea interactions compound or cancel is unknown. Disco-era evidence does not separate them, so one factor at a time is the only safe plan.

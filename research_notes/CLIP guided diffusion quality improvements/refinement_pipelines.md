# Super-resolution, refinement and multi-stage pipelines for CLIP guided diffusion

> **Status, 2026-11-02:** These are the notes as written on 2026-11-01, kept as the record of what was known then. Later measurements overturned some of it, including its leading recommendations. [The report](../../reports/CLIP%20guided%20diffusion%20quality%20improvements.md) has the current results, and wins wherever the two disagree.

Status as of 2026-11-01. Every URL listed under "weights" answered an HTTP HEAD with a 200 and the stated size on that date, unless the line says otherwise. Lines labelled "Local measurement" were produced in this session against this repo's own code and the real checkpoints; the method is described once, under the first question. Latent-space results are flagged, because they may not carry to the pixel-space ADM U-Nets this repo uses.

## 1. Cascaded generation: ADM upsamplers, GLIDE upsampler, CDM, and how CLIP guidance should run in each stage

### Takeaway
A cascade from 128 is the clear win at 512: every 128-based route tied or beat direct 512 on three independent CLIP judges (n=12, one seed), at 3.5 s and 2.3 GiB with a GAN against 18.5 s and 4.1 GiB. At 256 the picture is mixed: 64→256 costs 4.4 s and 1.6 GiB against 7.6 s and 5.3 GiB, but it inherits the 64 px image's lower score and loses to direct 256 by .04 to .07, so 128 is the better base. CLIP guidance on the upsampler adds a small gain at 256 and nothing measurable at 512, which matches ADM's practice of guiding only the base.

### Cited Findings

#### Weights and parameters
- ADM 64→256 upsampler: `https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64_256_upsampler.pt`, 1,247,310,648 bytes. Flags: `--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --large_size 256 --small_size 64 --learn_sigma True --noise_schedule linear --num_channels 192 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True`. MIT. — [guided-diffusion README](https://github.com/openai/guided-diffusion)
- ADM 128→512 upsampler: `https://openaipublic.blob.core.windows.net/diffusion/jul-2021/128_512_upsampler.pt`, 1,235,462,529 bytes. Same flags except `--attention_resolutions 32,16 --num_head_channels 64 --large_size 512 --small_size 128`. — [guided-diffusion README](https://github.com/openai/guided-diffusion)
- The ADM paper lists 312M parameters for 64→256 and 309M for 128→512, 192 channels, depth 2, channel multipliers 1,1,2,2,4,4, dropout 0, linear schedule, batch size 256, 500K and 1050K iterations. Its table says attention at 32,16,8 for both, while the README flags use 32,16 for 128→512; the README flags are the ones that match the checkpoint (see local measurement below). — [ADM paper, table 13](https://arxiv.org/abs/2105.05233)
- Both are class-conditional, and the low-resolution image is concatenated to the input after bilinear interpolation. The upsamplers ship without any classifier scale. The official sampler uses 250 respaced steps through `super_res_sample.py --base_samples`. — [ADM paper](https://arxiv.org/abs/2105.05233); [guided-diffusion README](https://github.com/openai/guided-diffusion)
- GLIDE's upsampler: `https://openaipublic.blob.core.windows.net/diffusion/dec-2021/upsample.pt`, 1,593,693,639 bytes, MIT, repo archived. It is text-conditional and 64→256 only, the smaller filtered release (`super_res=True`, 192 channels, linear schedule), where the paper's own upsampler had 1.5B parameters and 1.6M training iterations. The paper sampled it with 27 strided steps (segments 10,10,3,2,2; only 2 timesteps in (800,1000], 10 in (0,200]). — [GLIDE paper](https://arxiv.org/abs/2112.10741); [glide-text2im](https://github.com/openai/glide-text2im)
- The noised CLIP models are also still up: `clip_image_enc.pt` 345,831,113 bytes and `clip_text_enc.pt` 286,773,071 bytes, same folder. — [glide-text2im](https://github.com/openai/glide-text2im)
- No 256→512 checkpoint exists in the ADM or GLIDE releases; the published pairs are 64→256 and 128→512. — [guided-diffusion README](https://github.com/openai/guided-diffusion)

#### Published evidence that cascades win
- ADM ImageNet 256 FID: ADM 10.94, ADM-U (upsampled) 7.49, ADM-G (guided) 4.59, ADM-G+ADM-U 3.94 at 250/250 steps (IS 215.84, precision .83, recall .53). ImageNet 512: ADM 23.24, ADM-U 9.96, ADM-G 7.72, ADM-G+ADM-U 3.85 at 250/250, 4.11 at 250/25, 5.96 at 25/25. — [ADM paper, table 6](https://arxiv.org/abs/2105.05233)
- That table's 25/25 row has a sFID of 12.10 against 5.86 at 250/250, so spatial structure pays for the short upsampling schedule. — [ADM paper, table 6](https://arxiv.org/abs/2105.05233)
- "When combining classifier guidance with upsampling, we only guide the lower resolution model." — [ADM paper](https://arxiv.org/abs/2105.05233)
- CDM reaches ImageNet FID 4.88 at 256 (IS 158.71), 3.52 at 128 and 1.48 at 64, with no classifier guidance, via a 32→64→256 cascade, and calls conditioning augmentation "crucial": Gaussian noise for low-resolution upsampling, Gaussian blur for 128→256. Truncated and non-truncated variants performed equally. — [CDM paper](https://arxiv.org/abs/2106.15282)
- Imagen found noise conditioning augmentation critical. `aug_level=0` gave the best FID but in practice it used `aug_level` 0.1 to 0.3. Guidance weights up to 7 to 10 on the super-resolution stages improved FID. — [Imagen paper](https://arxiv.org/abs/2205.11487)
- DreamBooth reports that Imagen's training noise level blurs high-frequency patterns, and lowered it from 1e-3 to 1e-5 to keep detail. — [DreamBooth paper](https://arxiv.org/pdf/2208.12242)
- The ADM upsamplers were not trained with noise conditioning augmentation. Their training loader is only `model_kwargs["low_res"] = F.interpolate(large_batch, small_size, mode="area")`, so they only ever saw clean area-downsampled photographs. — [super_res_train.py](https://raw.githubusercontent.com/openai/guided-diffusion/main/scripts/super_res_train.py)
- unCLIP used two upsamplers (64→256 with Gaussian blur corruption, 256→1024 with BSR degradation), 27 and 15 DDIM steps, reported "no benefit from conditioning the upsamplers on the caption", and ran no guidance on them. — [unCLIP paper](https://arxiv.org/abs/2204.06125)
- GLIDE's noised 64×64 ViT-L CLIP guiding the pretrained ImageNet 64 model (gradient scale 15.0, random class per step) followed by a diffusion upsampler gave results "equal or higher quality" than the community 256/512 notebook approach. The same paper found that humans prefer classifier-free guidance to CLIP guidance (Elo photorealism −73.2 vs 82.7, caption 29.3 vs 110.9) and that humans disagree with CLIP score. — [GLIDE paper](https://arxiv.org/abs/2112.10741)
- Relay Diffusion's table puts ADM at 10.94, CDM at 4.88, ADM-U,G at 3.94, and its own continuation of a low-resolution sample via blurring diffusion plus block noise at 5.27 unguided and 1.99/1.87 with CFG at ImageNet 256. — [Relay Diffusion](https://arxiv.org/abs/2309.03350)

#### What Disco Diffusion did
- `timestep_respacing = f'ddim{steps}'` and `diffusion_steps = (1000//steps)*steps if steps<1000 else steps`, default `steps` 250, `skip_steps` 10. — [disco.py](https://raw.githubusercontent.com/alembics/disco-diffusion/main/disco.py)
- Its "superres" was not a cascade. v4.1 added a post-pass, `do_superres`, using LDSR for "DDIM sharpening", with presets Faster (25 steps, half size first), Fast (100, half), Slow (25, no downsample), Very Slow (100, no downsample), `sr_post_downsample='Original Size'`, `sr_eta 1.0`, Lanczos. That is downsample, 4× LDSR, and back to the original size, so it refined without changing the resolution. v5.1 removed it along with SLIP. — [Disco commit removing superres](https://github.com/alembics/disco-diffusion/commit/a1f25c79e); [Disco README changelog](https://github.com/alembics/disco-diffusion)

#### Local measurement
Method: 12 prompts, seed 0, repo default settings, each pipeline against direct generation at the same final size. Judges were ViT-B/16, ViT-H-14-quickgelu (dfn5b) and ViT-L-16-SigLIP2-256 on the CPU, none of which steered: guidance is ViT-B/32. Intervals are 95% bootstrap on the paired mean across prompts. They do not capture seed variance, and judges resize to about 224 px, so they cannot see fine detail, and they are not human ratings. Times are medians of sampling only, excluding model load (the README's 256 wall time is 11.4 s against 7.6 s here, and a cascade loads two models). VRAM is peak allocated in this process, on the RTX 2070 SUPER. The GPU was shared with other agents during `sel64_*` and part of `sr256_u`, the later stages were quiet. Scripts and images are in `../../reports/evidence/cascade_exp/`.

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
- Prompt fit is set by the base. "A pencil sketch of an old man with a beard" came out as a pen in the 64 and 128 images, and no upscaler fixed that. — Local measurement

### Inferences
- Default recommendation to test: `-size 512` becomes 128 plus an upscaler, and direct 512 stops being the route to 512. A ×4 GAN is enough on these judges. Whether the ADM upsampler's extra 11.8 s buys visible detail is a question these judges cannot answer, because they resize to about 224 px, so it needs a human look.
- Do not make 64→256 the default for 256. Base resolution decides the score. Use 128 as the base; for a 256 output, 128 plus a GAN or `-init -skip` both reach direct 256's score band.
- Skip CLIP guidance on upsamplers. At 512 it doubles time and VRAM for +.001 to +.002.
- The ADM upsamplers saw only clean area-downsampled photographs, and CLIP-guided bases are not that, so input mismatch is a real risk that the outputs here did not visibly suffer from.
- Plug-in: add an `'sr'` table to `DIFFUSION_LOOKUP` with the two URLs and flags above, and a loader in `cgd/script_util.py` built on `sr_create_model_and_diffusion` that keeps the meta device, mmap and assign path. Pass `{"y": class_index, "low_res": image_in_minus1_1}` as `model_kwargs`. `load_guided_diffusion`'s `lru_cache(maxsize=1)` and the generator's references must release the base model before the upsampler loads, or an 8 GB card holds both. A `--upscale` stage can then run after `clip_guided_diffusion()` and write its own output.
- Both upsamplers are about 1.2 GB downloads on top of the base, and the GAN route is 67 MB, which also favours the GAN as the first thing to ship.

### Gaps
- A sensitivity test of the 64→256 upsampler to step count (10, 60) and to blurring `low_res` was started but stopped unfinished, and a native-scale crop judge meant to see 512 px detail did not run, so nothing here measures detail quality at 512.
- Cascade wall time including the second model load was not measured.
- All local numbers are one seed. Seed 0 is unusually saturated (see question 7), so these need repeating with more seeds.
- No published result runs CLIP guidance on an upsampling stage of a pixel-space cascade; the closest are ADM (guides only the base), unCLIP (no caption conditioning, no guidance on the upsamplers) and Imagen (guides SR with CFG, which needs training with text).
- Nobody appears to have tested blurring or noising `low_res` at sampling time for the ADM upsamplers. CDM and Imagen trained with that augmentation, the ADM upsamplers did not, so it cannot be assumed to help.
- GLIDE's upsampler was not run here. Its 1.59 GB model is in reach of 8 GB by size, but no VRAM or time figure was measured for it, and it would need the archived `glide-text2im` package plus a separate tokenizer path.
- CDM's own weights were never released, as far as I could find.

## 2. Disco Diffusion secondary model and tricks

### Takeaway
The secondary model can replace U-Net backpropagation at 256 at no measurable loss on the judges (−.011 to +.008, all intervals span zero) while cutting peak VRAM from 5.3 to 1.9 GiB and time from 8.0 to 5.8 s; at 512 it saves 1.5 GiB and 8 s but trends lower (−.012 to −.015). The remaining Disco knobs are unmeasured folklore, though `clamp_grad` and `sat_scale` overlap with what this repo already does.

### Cited Findings

#### Defaults and exact behaviour
| Knob | Default | What the code does | Source |
|---|---|---|---|
| `use_secondary_model` | True | The U-Net runs without a gradient. `out = secondary_model(x, cosine_t).pred`, `x_in = out*fac + x*(1-fac)`, `fac = sqrt_one_minus_alphas_cumprod[cur_t]`, gradient taken only through the small model | [disco.py](https://raw.githubusercontent.com/alembics/disco-diffusion/main/disco.py) |
| `clip_guidance_scale` | 5000 | Multiplies the summed CLIP loss | [disco.py](https://raw.githubusercontent.com/alembics/disco-diffusion/main/disco.py) |
| `tv_scale`, `range_scale`, `sat_scale` | 0, 150, 0 | `range_scale` limits how far RGB leaves range. `sat_scale` adds `abs(x_in - x_in.clamp(-1,1)).mean()`, documented as "how much saturation allowed", from nshepperd's JAX notebook | [disco.py](https://raw.githubusercontent.com/alembics/disco-diffusion/main/disco.py) |
| `cutn_batches` | 4 | Accumulates the CLIP gradient over several batches of cuts, `x_in_grad += grad/cutn_batches` | [disco.py](https://raw.githubusercontent.com/alembics/disco-diffusion/main/disco.py) |
| `clamp_grad`, `clamp_max` | True, 0.05 | `magnitude = grad.square().mean().sqrt(); grad*magnitude.clamp(max=clamp_max)/magnitude`, then NaN to zero | [disco.py](https://raw.githubusercontent.com/alembics/disco-diffusion/main/disco.py) |
| `skip_augs` | False | Skips the torchvision augmentations on cuts | [disco.py](https://raw.githubusercontent.com/alembics/disco-diffusion/main/disco.py) |
| `eta` | 0.8 | DDIM eta | [disco.py](https://raw.githubusercontent.com/alembics/disco-diffusion/main/disco.py) |
| `perlin_init`, `perlin_mode` | False, `mixed` | Replaces the init with an average of two perlin noise images | [disco.py](https://raw.githubusercontent.com/alembics/disco-diffusion/main/disco.py) |
| `randomize_class` | True | Re-randomises the ImageNet class each step | [disco.py](https://raw.githubusercontent.com/alembics/disco-diffusion/main/disco.py) |
| `clip_denoised` | False | Whether CLIP sees denoised or noisy output | [disco.py](https://raw.githubusercontent.com/alembics/disco-diffusion/main/disco.py) |
| `cut_overview`, `cut_innercut` | `[12]*400+[4]*600`, `[4]*400+[12]*600` | "Overview cuts see the entire image and are good for early structure, innercuts are your standard cutn" | [disco.py](https://raw.githubusercontent.com/alembics/disco-diffusion/main/disco.py) |

- The secondary model is a plain convolutional U-Net with no attention: `c=64`, channels 64,128,128,256,256,512, Fourier time features, predicting v, with `pred = input*alphas - v*sigmas`, time `t = atan2(sigma, alpha)*2/pi`. MMagic's documentation calls it a "smaller secondary diffusion model trained by Katherine Crowson to remove noise from intermediate timesteps to prepare them for CLIP". — [disco.py](https://raw.githubusercontent.com/alembics/disco-diffusion/main/disco.py); [MMagic docs](https://mmagic.readthedocs.io/zh-cn/latest/_modules/mmagic/models/editors/disco_diffusion/secondary_model.html)
- Weights: Disco's own hosts have rotted. `the-eye.eu` and `ipfs.pollinations.ai` did not connect and the `v-diffusion` S3 bucket answered 403/301 on 2026-11-01. A working mirror is `https://huggingface.co/spaces/huggi/secondary_model_imagenet_2.pth/resolve/main/secondary_model_imagenet_2.pth`, 55,380,626 bytes, whose sha256 `983e3de6f95c88c81b2ca7ebb2c217933be1973b1ff058776b970f901584613a` equals the hash pinned in Disco. License: disco-diffusion is MIT by its LICENSE text (Crowson 2021); the mirror itself states no license. — [disco.py](https://raw.githubusercontent.com/alembics/disco-diffusion/main/disco.py)
- The same-origin `v-diffusion-pytorch` repo is MIT. — [v-diffusion-pytorch](https://github.com/crowsonkb/v-diffusion-pytorch)
- Community Disco configs push `clamp_max` to 0.26–0.7, `sat_scale` to 15111–90000, `range_scale` to 195–1666 and `cutn_batches` down to 1–2. These are settings people posted, with no measurement behind them. — [HuemanInstrument config 1](https://gist.github.com/HuemanInstrument/c2ab7b821d78d071df5f443be9e95d6d), [config 2](https://gist.github.com/HuemanInstrument/76234a35784c2c22a65c3de50b617567), [config 3](https://gist.github.com/HuemanInstrument/71bbfce885b9c3a3cc58cd7d2047fc4c)

#### Local measurement
- Harness check: a copy of the sampler with these hooks reproduced `d256` and `d512` with paired differences of exactly 0 on every judge, so the variant numbers are comparable to the repo's default. — Local measurement
- The checkpoint loaded strictly from the HF mirror: 13,841,547 parameters, 22× fewer than the ADM upsamplers. — Local measurement

| Size | Guidance through | Time (s) | Peak GiB | ViT-B/16 | ViT-H | SigLIP2 |
|---|---|---|---|---|---|---|
| 256 | U-Net (default) | 8.02 | 5.33 | baseline | baseline | baseline |
| 256 | Secondary model | 5.80 | 1.92 | −.011 [−.026,+.004] | +.008 [−.013,+.030] | +.005 [−.008,+.019] |
| 256 | Unguided, for scale | 5.17 | 1.91 | −.106 [−.134,−.079] | −.117 [−.155,−.081] | −.093 [−.123,−.062] |
| 512 | U-Net with checkpointing | 18.73 | 4.13 | baseline | baseline | baseline |
| 512 | Secondary model | 10.80 | 2.64 | −.012 [−.025,−.001] | −.015 [−.058,+.022] | −.012 [−.036,+.010] |

- Guidance here kept the repo's rate cap, window and losses, with x0 for CLIP coming from the secondary model. The U-Net still supplied the sampler's x0. — Local measurement

### Inferences
- The secondary model is the strongest VRAM lever found: it turns guided 256 into a 2 GiB job, which would leave room for a heavier CLIP or more cutouts on 8 GB.
- Plug-in: add a `loss_gradient` path in `cgd/sampling.py` where the U-Net runs under `no_grad`, then `x.requires_grad_()`, `pred = secondary(x, cos_t).pred`, with `cos_t = atan2(sigma, alpha) * 2 / pi`, feeding `loss_fn(x, pred, sigma)`. The class must be vendored from Disco, since it is not in the repo's dependencies, and the hash above should be checked on download.
- Because the mirror is a personal Space, the weights should be re-hosted or pinned by hash before the repo relies on them.
- `clamp_max` is Disco's absolute version of the repo's relative `guidance_rate`, and `sat_scale` is close to `range_loss`, so neither is a new idea here.

### Gaps
- I found no paper or controlled comparison of any Disco knob. All of it is folklore, including the commonly repeated speed-up figure for the secondary model, which I could not find in a primary source.
- `perlin_init`, `sat_scale`, `cutn_batches` and `skip_augs` were not measured here. This repo's fast sampler already has `range_loss`, a capped guidance rate, and no augmentations by default, and cutouts are the adjacent cutout researcher's scope.

## 3. Hires fix, iterative upscaling and tiling: what applies to a pixel-space guided-diffusion U-Net

### Takeaway
Img2img through the repo's existing `-init -skip` is the only technique here measured on this U-Net: it ties direct generation on the judges at equal or lower time but keeps the base image's saturated look. Latent upscaler, SDXL refiner and TAESD do not apply; tiling recipes do apply but are untested in pixel space, and they cost one U-Net call per tile.

### Cited Findings

| Technique | Applies to pixel U-Net? | Cost | License | Weights | Evidence | Plug-in |
|---|---|---|---|---|---|---|
| Hires fix, img2img (SDEdit) | Yes, it is what `-init -skip` already does | See local measurement | n/a | Repo's | [SDEdit](https://arxiv.org/abs/2108.01073): more noise gives more realism and less faithfulness | Generate at low size, then rerun with `-init`, `-skip` |
| Crowson latent upscaler | No: SD latents only | 20 steps, guidance 0 | CreativeML Open RAIL++-M | [sd-x2-latent-upscaler](https://huggingface.co/stabilityai/sd-x2-latent-upscaler) | Model card only | Would need an SD VAE round trip |
| SDXL refiner | No: latent, needs SDXL | Paper calls it one of "two large models" | CreativeML Open RAIL++-M | [refiner-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0) | [SDXL](https://arxiv.org/abs/2307.01952): trained on the first 200 noise scales, user study preference | Idea carries, weights do not. A low-noise-only U-Net would have to be trained |
| MultiDiffusion | Yes in principle | One U-Net call per crop per step | Repo has no license | None | [MultiDiffusion](https://arxiv.org/abs/2302.08113): pixel-level averaging of crops, shown in SD latents, stride 8 latent/64 RGB | Average tiled `x0` in `sample_loop_progressive` |
| Mixture of Diffusers | Yes: states "any pre-trained diffusion model" | Memory about the largest region | MIT | None | [Mixture of Diffusers](https://arxiv.org/abs/2302.02412) | Same, with Gaussian tile weights |
| Ultimate SD Upscale | Yes as a recipe | 512 px tiles, denoise about 0.35 (0.15–0.20 minimal), mask blur 12–16, seams fix | GPL-3.0: do not copy code into this MIT repo | None | [FAQ](https://github.com/Coyote-A/ultimate-upscale-for-automatic1111/wiki/FAQ) practitioner advice only | GAN pre-upscale, then tiled low-noise pass |
| Tiled Diffusion extension | Recipe yes | n/a | CC BY-NC-SA 4.0: do not copy | None | [multidiffusion-upscaler](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111) | As MultiDiffusion |
| DemoFusion | Partly, skip residual and dilated sampling are model-agnostic | RTX 3090, longer runtime | MIT | SDXL | [DemoFusion](https://arxiv.org/abs/2311.16973) latent only | Progressive 256→512→1024 with residual injection |
| ScaleCrafter | Plausibly: acts on convolutions in the U-Net | Same calls, higher resolution | Original repo has no license; `-ptl` port MIT | None | [ScaleCrafter](https://arxiv.org/abs/2310.07702) latent SD only | Re-dilate ADM convolutions at sampling |
| HiDiffusion | Plausibly | Claims 1.5–6× faster | Apache-2.0 | None | [HiDiffusion](https://arxiv.org/abs/2311.17528) latent only | RAU-Net patch of the U-Net |
| Kohya Deep Shrink | Plausibly: downscale features at block 3 early, upscale later | `block_number` 3, `downscale` 2.0, start/end percent, `downscale_after_skip` | ComfyUI node docs | None | [ComfyUI docs](https://docs.comfy.org/built-in-nodes/PatchModelAddDownscale), no paper. The kohya repo URL 404s | Override `UNetModel.forward`, as the FreeU test here did |

- In AUTOMATIC1111, latent upscaling needs denoise above about 0.5–0.6 or it comes out blurred, while the ESRGAN path does not blur. — [issue 6357](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/6357); [issue 3630](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/3630)
- Schedule mapping for this repo's linear schedule, computed here from its betas: `-skip 250` starts at t=749 (α .058, σ .998), `-skip 500` at t=499 (α .28, σ .96), `-skip 700` at t=299 (α .63, σ .78), `-skip 750` at t=249 (α .72, σ .69), `-skip 900` at t=99 (α .947, σ .32). SD's "denoise" does not map one to one, because its schedule differs. — [Local computation from cgd/sampling.py](https://github.com/afiaka87/clip-guided-diffusion/blob/main/cgd/sampling.py); the schedule difference is in [Lin et al.](https://arxiv.org/abs/2305.08891)
- The U-Net accepts other sizes in multiples of size/8, as the repo's `-ht`/`-wd` offsets rely on. — [cgd/script_util.py](https://github.com/afiaka87/clip-guided-diffusion/blob/main/cgd/script_util.py)

#### Local measurement
| Pipeline | Time (s) | Peak GiB | ViT-B/16 vs direct | ViT-H vs direct | SigLIP2 vs direct |
|---|---|---|---|---|---|
| 128 → 256 `-skip 500` | 3.47 + 5.07 | 2.33, 5.28 | −.011 [−.031,+.010] | +.008 [−.027,+.042] | +.006 [−.020,+.029] |
| 128 → 256 `-skip 700` | 3.47 + 3.57 | 2.33, 5.28 | −.010 [−.028,+.006] | +.002 [−.036,+.036] | +.006 [−.024,+.033] |
| 256 → 512 `-skip 700` | 7.59 + 8.32 | 5.28, 4.08 | +.016 [−.002,+.033] | +.021 [−.002,+.045] | +.013 [−.004,+.030] |

- Laplacian variance at 256 was 9092 direct, 1245 at `-skip 500` and 287 at `-skip 700`, so `-skip 700` on a resized 128 stays soft. Direct's figure is inflated by adversarial texture, so higher is not better. — Local measurement
- By eye, 256→512 at `-skip 700` keeps the base's saturated, posterised look, including its failures, so it adds resolution and does not refine. — Local measurement
- An unguided U-Net call at 256 cost 0.17 s (5.17 s for 30 calls). — Local measurement

### Inferences
- A 256→512 pass at `-skip 700` costs 15.9 s and 5.3 GiB against 3.5 s and 2.3 GiB for 128 plus a GAN with no better score, so img2img upscaling is not worth shipping as the 512 route.
- A tiled pass with 256 px tiles at stride 128 over 512 is 9 tiles per step, so about 46 s for 30 unguided calls, or fewer at high `-skip`. Keeping to low noise (`-skip 700` or above) would protect composition from the class-in-every-tile problem. This is an estimate, not a measurement.
- Plug-in for tiling: in `sample_loop_progressive`, replace the single `model()` call with a loop over overlapping crops, averaging eps with Gaussian weights as in Mixture of Diffusers. Only MIT-licensed code should be consulted: not Ultimate SD Upscale (GPL-3.0) or the Tiled Diffusion extension (CC BY-NC-SA).
- Deep Shrink, ScaleCrafter and HiDiffusion all fit as a `UNetModel.forward` override, which worked for FreeU. FreeU's failure below is a warning that latent-tuned U-Net patches can fail badly in pixel space.

### Gaps
- No pixel-space evaluation of MultiDiffusion, Mixture of Diffusers, DemoFusion, ScaleCrafter, HiDiffusion or Deep Shrink was found, and none was run here. The class-conditional ADM U-Net would likely try to draw the class in every tile at high noise, which is untested.
- Published claims for these methods are mostly qualitative or latent-space; I found no independent head-to-head.

## 4. Fast upscalers and SR diffusion models

### Takeaway
Real-ESRGAN x4plus (BSD-3-Clause, 67 MB) is the right first upscaler: 0.06 s and 0.16 GiB for 128→512, 0.85 s and 2.0 GiB for 512→2048 with no tiling. HAT and SwinIR-L scored about the same at much higher cost, and all SR diffusion models need an extra SD-class checkpoint of several GB, most under non-commercial licenses, so none is justified yet.

### Cited Findings

#### GAN and transformer upscalers
| Model | Weights | Size | License | Published cost | Notes |
|---|---|---|---|---|---|
| Real-ESRGAN x4plus | [v0.1.0](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth) | 67,040,989 B | BSD-3-Clause, [repo](https://github.com/xinntao/Real-ESRGAN) | Sources conflict, see below | `--tile`, fp16 default, 589 open issues, last push 2024-08 |
| Real-ESRGAN x2plus | [v0.2.1](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth) | n/a | BSD-3-Clause | | |
| x4plus_anime_6B | [v0.2.2.4](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth) | n/a | BSD-3-Clause | | Illustration domain |
| 4x-UltraSharp | [Kim2091](https://huggingface.co/Kim2091/UltraSharp) | 66,961,958 B | CC BY-NC-SA 4.0: non-commercial, bundling this in an MIT repo needs care | | ESRGAN architecture, 150k iterations |
| SwinIR-L ×4 GAN | [v0.0](https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth) | 142,473,939 B | Apache-2.0 | [11.8M params for classic SwinIR](https://arxiv.org/abs/2108.10257) | |
| HAT | [Acly mirror](https://huggingface.co/Acly/hat), official on Google Drive and Baidu | 85,137,601 B | Apache-2.0, [repo](https://github.com/XPixelGroup/HAT) | [20.8M params, 103.7G MACs](https://arxiv.org/abs/2205.04437) | Mirror weights are the ImageNet pretrain. `Real_HAT_GAN_SRx4` and `_sharper` are Google Drive only |
| LDSR | [heibox](https://heibox.uni-heidelberg.de/f/578df07c8fc04ffbadf3/?dl=1), [CompVis HF](https://huggingface.co/CompVis/ldm-super-resolution-4x-openimages) | U-Net 454,611,821 B | CompVis | About 2 GB checkpoint | Reported "very good with textures on generated images, poor on real-world", [Replicate listing](https://replicate.com/nightmareai/latent-sr) |
| TAESD | [repo](https://github.com/madebyollin/taesd) | n/a | MIT | | Latent decoder. Not applicable: this repo has no latents |

- VRAM figures for these upscalers conflict by a factor of 2 to 3: one model card lists x4plus at about 2 GB, HAT about 4 GB, SwinIR-L about 3 GB and compact models about 1 GB, while another guide says 4–6 GB for 512 input. — [AEmotionStudio README](https://huggingface.co/AEmotionStudio/ai-upscale-models/blob/main/README.md); contradicted by [sbcode.net](https://sbcode.net/genai/esrgan)
- Real-ESRGAN on CPU, 512→2048: 19.9 s in PyTorch. — [Real-ESRGAN-OpenVINO README](https://huggingface.co/ibrhr/Real-ESRGAN-OpenVINO)

#### Super-resolution diffusion models
| Model | License | Weights | Size | Published speed | Requires |
|---|---|---|---|---|---|
| SR3 | Unofficial repo Apache-2.0 | None official | n/a | [paper](https://arxiv.org/abs/2104.07636) | Would need training |
| StableSR | S-Lab 1.0: non-commercial | [Iceclear/StableSR](https://huggingface.co/Iceclear/StableSR), `stablesr_turbo.ckpt` | 6,481,632,291 B | 50 steps 3459 ms | SD 2.1 latent |
| ResShift | S-Lab 1.0 | [v2.0](https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_realsrx4_s4_v3.pth) | 477,917,058 B | 4 steps 319 ms | VQ latent, `--chop_size 512` |
| SinSR | CC BY-NC-SA 4.0 | [v1.0](https://github.com/wyf0912/SinSR/releases/download/v1.0/SinSR_v1.pth) | 477,910,858 B | 1 step 138 ms | Same VQ |
| SeeSR | Apache-2.0 | [mirror](https://huggingface.co/alexnasa/SeeSR) | 4,507,181,640 B | n/a | SD 2 |
| SUPIR | Non-commercial, commercial by written permission of Dr. Jinjin Gu | [mirror](https://huggingface.co/camenduru/SUPIR), official on Baidu and Google Drive | 5,329,810,432 B | Low-VRAM mode: "12G for Diffusion, 16G for LLaVA" | SDXL, CLIP encoders, LLaVA |
| OSEDiff | Apache-2.0 | `huggingface.co/cswry/OSEDiff` answered 401; README ships weights in the repo's preset folder | n/a | 1 step 176 ms; 0.1046 s at 512 on an A100 | SD 2.1, RAM, DAPE |
| InvSR | S-Lab 1.0 | [OAOA/InvSR](https://huggingface.co/OAOA/InvSR), `noise_predictor_sd_turbo_v5.pth` | 135,417,810 B | 1 step 117 ms, 5 steps 244 ms | sd-turbo, `--chopping_size 256` |
| DiffBIR | Apache-2.0 | [lxq007/DiffBIR](https://huggingface.co/lxq007/DiffBIR), `general_full_v1.ckpt` | 6,894,632,360 B | 50 steps 7937 ms | SD 2.1 |

- The speed column comes from one comparison in the InvSR paper, on a single unspecified GPU, not independent measurement. — [InvSR](https://github.com/zsyOAOA/InvSR)
- 2026 frontier work is self-reported and latent: TADSR and ODTSR at CVPR 2026, RCOD at AAAI 2026, FluxSR. NTIRE 2026 ×4 had 194 registered entrants and 31 valid entries, with restoration and perceptual tracks. — [NTIRE 2026 report](https://arxiv.org/abs/2604.14558)

#### Local measurement
Measured through `spandrel` on the RTX 2070 SUPER, GPU quiet. x4plus, UltraSharp and Compact ran at fp16, SwinIR-L and HAT at fp32 because spandrel marks them as not supporting half. Peak includes weights. Compact is `realesr-general-x4v3.pth`, [v0.2.5.0](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth), 4,885,111 bytes, BSD-3-Clause. HAT is the ImageNet pretrain from the mirror, which is not GAN-trained.

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

- These numbers settle the conflict above for this card: x4plus at fp16 needs about 2 GiB at 512 input, not 4–6 GB. — Local measurement

### Inferences
- Ship x4plus first, with Compact as the option for speed. UltraSharp's CC BY-NC-SA license makes it a poor default in an MIT repo, and it scored no better.
- HAT's lower clipping (5.0% vs 9.5%) and its one significant B/16 gain are worth a look if a GAN's edges prove too hard, but it should get tiling, and its Google Drive weights are awkward for `script_util.download()`.
- Plug-in: `spandrel` loads all of these and could be a lazy optional import, like `open_clip`, or the RRDB network can be vendored in about 60 lines. Fetching goes through `script_util.download()`.
- SR diffusion models were tuned on real photo degradations, and their latent VAE round trip is a further mismatch here, so their published scores do not carry over. If one is ever tried, ResShift or InvSR are the small ones, but both are S-Lab 1.0, so non-commercial. Of the Apache-2.0 models, OSEDiff has no public weight URL I could verify, and SeeSR and DiffBIR are 4.5 to 6.9 GB.

### Gaps
- SR diffusion models were not run here: none could be tested without pulling a further SD checkpoint, and most of them are trained on photographic degradations, not on CLIP-guided samples.
- Only HAT's ImageNet pretrain was run. Its real-world GAN weights live on Google Drive and Baidu only.
- NTIRE 2026 winners and scores were not retrieved.
- Most SR papers report MUSIQ, CLIP-IQA or LPIPS on photos, and I did not extract them, because none of it is about generated inputs.

## 5. Face restoration

### Takeaway
Face restoration is a cheap optional post-pass with only weak relevance here: ImageNet U-Nets rarely produce convincing faces, and the only commercially clean option is RestoreFormer++ with Apache-2.0, though GFPGAN's own terms should be read before redistribution. No evidence exists for generated pixel-space faces, and nothing here was measured.

### Cited Findings
| Model | Weights | Size | License | Notes |
|---|---|---|---|---|
| GFPGAN 1.4 | [v1.3.0](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth) (also v1.3.4) | 348,632,874 B | Tencent custom, repo NOASSERTION, with third-party components that need checking | [Paper](https://arxiv.org/abs/2101.04061). Wired into Real-ESRGAN as `--face_enhance` |
| CodeFormer | [v0.1.0](https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth) | 376,637,898 B | S-Lab 1.0: non-commercial, commercial by contact | [Paper](https://arxiv.org/abs/2206.11253) |
| RestoreFormer++ | [v1.0.0](https://github.com/wzhouxiff/RestoreFormerPlusPlus/releases/download/v1.0.0/RestoreFormer++.ckpt) | 294,028,815 B | Apache-2.0 | [Paper](https://arxiv.org/abs/2201.06374) |

### Inferences
- If faces become a goal, these would run as a detached post-step that needs face detection first, and it should sit behind an opt-in flag: a blind restorer invents a face identity and detail that CLIP never scored, and on ImageNet samples most images contain no face at all.

### Gaps
- No VRAM, time or quality number was collected for these on generated inputs.
- No study was found on restoring faces from pixel-space ImageNet diffusion samples.

## 6. Postprocessing and the oversaturated CLIP guidance look

### Takeaway
Dynamic thresholding is the one postprocess-style change that visibly removes the CLIP look: at no cost it took clipped pixels from 4.0% to 0.03% and mean saturation from .45 to .18, with judges unchanged or slightly up (+.008 to +.015, intervals span zero). By eye the images look far more natural, if sometimes grey, so the 0.995 quantile needs tuning. Orthogonal guidance and after-the-fact colour matching did little.

### Cited Findings
- Static thresholding is essential at high guidance but oversaturates. Dynamic thresholding sets s to a percentile of |x0|, and when s>1 clamps to [−s,s] and divides by s, giving "significantly better photorealism and alignment". This is an image-space method. — [Imagen paper](https://arxiv.org/abs/2205.11487); [Lin et al.](https://arxiv.org/abs/2305.08891) ("designed only for image-space models")
- Adaptive Projected Guidance splits the guidance update against the denoised prediction D: the parallel component drives oversaturation. It applies ΔD⊥ + ηΔD∥ (η default 0), plus rescaling r and a reverse momentum β in [−0.75,−0.25], and it must project on the denoised estimate, not eps. FID and saturation, CFG→APG: EDM2-S 10.42→6.49, .46→.33; EDM2-XXL 8.65→4.94, .37→.31; DiT-XL/2 19.14→9.34, .37→.30; SD2.1 27.53→22.21, .36→.27; SDXL 26.29→25.35, .28→.18. All are latent-space and all are CFG, not gradient guidance. — [APG, ICLR 2025](https://arxiv.org/abs/2410.02416)
- Frequency-decoupled guidance: low-frequency guidance sets structure, high-frequency sets fidelity, with a separate scale each. — [FDG](https://arxiv.org/abs/2506.19713)
- Oversaturation is tied to the low-frequency part of guidance. — [arXiv 2506.21452](https://arxiv.org/abs/2506.21452)
- CFG rescale works for image and latent models, φ=0.7 at w=7.5. — [Lin et al.](https://arxiv.org/abs/2305.08891)
- StableSR's colour fix offers AdaIN (global mean and variance) and wavelet (transplant low frequencies from the reference). — [StableSR](https://github.com/IceClear/StableSR)
- sd-dynamic-thresholding is MIT. — [repo](https://github.com/mcmonkeyprojects/sd-dynamic-thresholding)
- FidelityFX CAS takes linear input and a sharpness in 0 to 1. An unofficial PyTorch port exists. — [Jamy-L port](https://github.com/Jamy-L/Pytorch-Contrast-Adaptive-Sharpening)
- This repo already bounds x0 at 1.2 and has `range_loss`. Its README reports 6% clipped pixels for the default at 128 and 7% for `-respace 1000`. — [cgd/sampling.py](https://github.com/afiaka87/clip-guided-diffusion/blob/main/cgd/sampling.py)

#### Local measurement
All at 256, against the default sampler. Mean saturation is the HSV S channel.

| Change | Clipped % | Saturation | ViT-B/16 | ViT-H | SigLIP2 |
|---|---|---|---|---|---|
| Default | 4.04 | .453 | baseline | baseline | baseline |
| Dynamic threshold, quantile .995, replacing clamp 1.2 | .03 | .180 | +.010 [−.011,+.036] | +.015 [−.016,+.046] | +.008 [−.022,+.039] |
| Orthogonal guidance update, APG with η=0 against x0 | 8.41 | .469 | +.001 [−.010,+.014] | +.006 [−.009,+.021] | −.004 [−.020,+.009] |
| Histogram match to the unguided image, 100% | 1.85 | .400 | −.004 [−.010,+.002] | −.009 [−.020,+.001] | −.007 [−.017,+.003] |
| Same, 50% blend | 1.60 | .412 | −.002 [−.006,+.002] | −.005 [−.012,+.002] | −.003 [−.010,+.004] |
| Saturation × .85 after sampling | 3.30 | .386 | −.001 [−.004,+.000] | −.000 [−.004,+.003] | −.001 [−.003,+.001] |

- By eye, dynamic thresholding produced a fox in a plausible field and a lighthouse on a rocky coast in place of the default's hard graphic shapes, though the dragonfly came out rather grey. Histogram matching kept the same adversarial shapes with milder colour. — Local measurement
- Cost of histogram matching was .01 s on the CPU, and that of dynamic thresholding was within timing noise. The unguided reference used for it costs a further full unguided run (5.2 s), though. — Local measurement

### Inferences
- Dynamic thresholding works inside the trajectory, where the U-Net can respond to it, and postprocessing can only recolour what is already there. That fits the difference seen here.
- Plug-in: in `cgd/sampling.py`, replace `guided_x0.clamp(-x0_clamp, x0_clamp)` with a per-image `th.quantile` of `|x0|`, clamped below at 1, then clamp and divide. It is about four lines and could be a `--dynamic_threshold` option. Test quantiles .995 to .9995 and a blend with the current clamp, and rerun the repo's quality protocol, since saturation .18 may be overshooting.
- APG's η=0 projection did not transfer: it doubled clipped pixels. APG was built for CFG, where the update is a model difference. Its other components, rescaling and momentum, were not tried.
- Sharpening is not needed on top of a GAN, whose output is already hard-edged.

### Gaps
- No published measurement of oversaturation in CLIP gradient guidance specifically was found; all of the above is CFG. The clipped-pixel and mean-saturation numbers here are the only data, and they do not measure perceived colour.
- The license of AMD's CAS code was not verified.
- Tone mapping and sharpening were not measured: the CLIP judges cannot rate them.

## 7. Candidate selection: N low-res seeds, independent scoring, refine the winner

### Takeaway
Selection works on CPU-only judges and transfers to other judges: best of 4 at 64 px gave most of the gain, and best of 8 then upsampled beat seed 0 upsampled on both held-out judges (+.037 and +.017). Independent judges agree only moderately on seeds (r about .7 to .8), and published work warns that a single verifier gets hacked, so an ensemble is the safer choice.

### Cited Findings
- DALL-E 1 reranks with CLIP and used N=512. Figure 9c shows clear FID and IS gains on MS-COCO up to a sample size of 32 and diminishing returns after. — [DALL-E paper](https://arxiv.org/abs/2102.12092)
- unCLIP draws two prior samples and keeps the one with the higher dot product with the text embedding. — [unCLIP paper](https://arxiv.org/abs/2204.06125)
- FLUX.1-dev on DrawBench, no search at 30 NFE vs random search at 2880 NFE (Aesthetic, CLIPScore, ImageReward, LLM Grader): none 5.79/0.71/0.97/84.29; Aesthetic 6.38/0.69/0.99/86.04; CLIPScore 5.68/0.82/1.22/86.15; ImageReward 5.81/0.74/1.58/87.09; ensemble of the three 6.06/0.77/1.41/88.18. — [Ma et al.](https://arxiv.org/abs/2501.09732)
- The same paper warns of "verifier hacking": random search converges to a verifier's biases. On ImageNet, DINO or CLIP logits raised IS steadily while FID eventually rose and recall fell. Local search reduced this, a wide neighbourhood brought it back. Single verifiers improved their own metric and could degrade others, and CLIPScore fell about 2% under Aesthetic search. Ensembles and ImageReward generalised best. — [Ma et al.](https://arxiv.org/abs/2501.09732). A later summary reads this as local search failing to beat random search overall, a stronger statement than Ma et al.'s. — [Riise et al.](https://arxiv.org/html/2510.16751v2)
- Progressive Seed Pruning starts 8 seeds for the budget of 4, scores x̂₀ with ImageReward, prunes to 4 at 25% of inference and to 2 at 50%, at no added model calls because it reuses denoiser output. Best-of-4 vs PSP: SD 1.5 IR .655→.827, GenEval .542→.574, human .594→.624; SDXL 1.098→1.224, .629→.645, .682→.713; SD 3.5 1.336→1.380, .747→.747, .831→.841, on 249 annotators and 8,295 images. The eventual best seed survives pruning about 80% of the time from 8 at 25%, and about 90% from 4 at 50%. Latent-space, trained reward model, CFG. — [PSP](https://arxiv.org/abs/2607.21591)
- Noise selection through inversion stability on SDXL-turbo: human win rate up to 57%, optimisation 72.5%, HPS v2 67%/88%, with small absolute means (HPS 27.27→27.49, AES 5.93→6.00). — [Noise selection](https://arxiv.org/abs/2407.14041)
- DiffRS rejects per timestep with a time-dependent discriminator, needing a trained discriminator. CIFAR-10 FID 1.59/1.52, ImageNet 64 1.26 at about 274 NFE, from a third-party summary. — [DiffRS](https://arxiv.org/abs/2405.17880)
- Aesthetic scoring: `sac+logos+ava1-l14-linearMSE.pth`, 3,714,759 bytes, Apache-2.0, a linear head on OpenAI ViT-L/14 embeddings. — [improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor). LAION's own `sa_0_4_vit_l_14_linear.pth` is 4,071 bytes, MIT. — [aesthetic-predictor](https://github.com/LAION-AI/aesthetic-predictor)
- An audit of the improved predictor's data counted 428,632 images (60% AVA, 34% SAC, 6% Logos) and found absolute and relative labels mixed. — [audit](https://arxiv.org/abs/2601.09896)
- NPNet learns "golden noise" at 3% time and memory, SD-based. Particle guidance diversifies a batch. — [NPNet](https://arxiv.org/abs/2411.09502); [Particle guidance](https://arxiv.org/abs/2310.13102)
- GLIDE says "humans disagree with CLIP score". — [GLIDE paper](https://arxiv.org/abs/2112.10741)

#### Local measurement
Eight seeds per prompt at 64 px. Gain is the picked image's score minus seed 0's. Judges that did not select are held out.

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
- Seed sets global colour: mean saturation at 64 was .56 for seed 0 and .19 to .48 for seeds 1 to 7, across the same prompts. — Local measurement

### Inferences
- Best of 4 at 128 costs about 14 s, similar to one direct 512, and then a GAN finishes in under a second. That is a strong candidate for a quality preset, though it was not run at 128 here.
- SigLIP2 as selector generalised best here, but with n=12 the ordering among judges is not reliable. Ma et al.'s ensemble result argues for averaging ranks across two or three judges.
- The saturation finding means all seed-0 comparisons in these notes carry a seed effect that the intervals do not show. A rerun with more seeds should come before any default changes.
- Plug-in: a `--candidates N` loop around `clip_guided_diffusion()` at the base size with a batch or a seed offset, scoring on the CPU in `clip_util` with a model other than the guidance model. The image with the highest mean rank goes on to the upscale stage. x̂₀-based pruning fits `sample_loop_progressive`, since it already yields `pred_xstart`, but it is unproven on this sampler.

### Gaps
- No selection or pruning result was found for pixel-space CLIP guidance. Everything published is CFG or latent.
- Whether x̂₀ at 25% of this fast sampler predicts final score was not measured here.
- No rank correlation for the aesthetic predictor was found, and no aesthetic score was run here.
- The DiffRS numbers are from a summary, not the paper's table.

## 8. Multi-resolution guidance and noise

### Takeaway
FreeU at SD strengths is actively harmful on this pixel-space U-Net: clipped pixels jumped from 4% to 42% and ViT-B/16 fell .037, and even a mild setting failed. ADM's schedules already reach near-zero terminal SNR, so zero-SNR fixes matter little here, and offset and pyramid noise need retraining. No measured evidence exists for multi-scale CLIP guidance, though it is reasonable to try through the cutout schedule.

### Cited Findings
- Zero terminal SNR table: Linear SNR(T)=4.035993e-05, √ᾱT=.006353; cosine 2.428735e-09, 4.928e-05; SD .004682, .068265. ADM's linear and cosine schedules are the near-zero ones; SD's is the flawed one by about 100×. Recomputing from betas reproduces the linear and cosine rows. — [Lin et al.](https://arxiv.org/abs/2305.08891)
- Offset noise is `randn + 0.1*randn(B,C,1,1)`, needs training, and moved mean brightness from .301 to .032 on a dark prompt. — [Crosslabs](https://www.crosslabs.org/blog/diffusion-with-offset-noise)
- Multi-resolution (pyramid) noise stacks noise from several scales with a discount, for example 0.8, in training. The report is blog-style, with no peer-reviewed paper. — [W&B report](https://wandb.ai/johnowhitaker/multires_noise/reports/Multi-Resolution-Noise-for-Diffusion-Model-Training--VmlldzozNjYyOTU2)
- Simple diffusion shifts the schedule by 2·log(64/d) for high resolutions. — [Simple diffusion](https://arxiv.org/abs/2301.11093)
- Pyramidal DDPM uses a single score network with coordinate conditioning across scales. — [Pyramidal DDPM](https://arxiv.org/abs/2208.01864)
- Restart sampling adds noise forward and re-denoises: EDM ImageNet 64 FID 1.36, and a better CLIP and Aesthetic trade-off on SD. — [Restart](https://arxiv.org/abs/2306.14878)
- FreeU scales first-half backbone channels by a mean-weighted factor and attenuates skip low frequencies with `Fourier_filter(threshold=1)`. Settings: SD1.4 b1 1.3, b2 1.4, s1 .9, s2 .2; SD2.1 1.4,1.6,.9,.2; SDXL 1.3,1.4,.9,.2. MIT. Evaluated only on latent U-Nets. — [FreeU paper](https://arxiv.org/abs/2309.11497); [FreeU repo](https://github.com/ChenyangSi/FreeU)
- SinDDM starts CLIP guidance at a chosen scale: coarse scales change global structure, finer scales change detail, and at the coarsest scale the first half of the steps run with no guidance. Single-image diffusion only. — [SinDDM](https://arxiv.org/pdf/2211.16582)
- Disco schedules cut types by time: overview cuts early, inner cuts late. No measurement accompanies this. — [disco.py](https://raw.githubusercontent.com/alembics/disco-diffusion/main/disco.py)

#### Local measurement
FreeU was ported by overriding `UNetModel.forward`, applied at the output stages with `model_channels*4` and `*2`, the analogues of SD's 1280 and 640.

| Setting (b1, b2, s1, s2) | Clipped % | ViT-B/16 | ViT-H | SigLIP2 |
|---|---|---|---|---|
| Default | 4.04 | baseline | baseline | baseline |
| 1.3, 1.4, .9, .2 (SD1.4) | 42.39 | −.037 [−.053,−.022] | −.019 [−.045,+.007] | −.020 [−.040,+.002] |
| 1.1, 1.2, .9, .2 (mild) | 38.27 | −.024 [−.038,−.008] | −.010 [−.040,+.020] | −.015 [−.032,+.001] |

- By eye, the SD setting gives posterised black and white graphic shapes over the whole frame. — Local measurement
- Time rose about 2%, from 8.02 to 8.24 s, and VRAM by .08 GiB. — Local measurement

### Inferences
- FreeU's `s` of .2 on skip low frequencies is probably the problem, because in pixel space the skips carry colour and tone directly. Testing `s` near 1 with `b` alone would isolate that, but there is no evidence it would help.
- This is the clearest case in these notes of a latent result failing to carry to pixel space.
- Base resolution is already a form of multi-resolution guidance: the 64 and 128 stages do the coarse work, and the cascade results suggest detail needs no CLIP at all. That is a cheaper reading of "coarse to fine" than scheduling CLIP scales.
- The zero-SNR argument is the reason not to port offset noise: the schedule is not the flaw here. The per-seed colour effect in question 7 might come from low-frequency content in the sampled noise, but that was not tested.
- Restart-style renoising is the only item in this section that needs no training, and `-init -skip` on a finished sample approximates it.

### Gaps
- No paper was found on low-frequency-first filtering of CLIP gradients in a diffusion sampler, or on CLIP at several scales in a pixel-space sampler. The search returned only the adjacent work above.
- FreeU on pixel-space U-Nets has no published evidence beyond the local test above. Pyramid and offset noise need training and were not tested.
- Multi-scale CLIP guidance itself was not run here.

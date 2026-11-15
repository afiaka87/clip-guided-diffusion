# Unconditional 256: experiments

Experiments for [the unconditional report](<../../Unconditional ADM CLIP guidance quality.md>), against the code at `d3c6324`. `cgd`'s recurrence (`-recur`) came out of them, and `queue4.sh` is the first run against it. Images go to `outputs/eval/ARM/`, which git ignores. Each arm's `runs.jsonl` records every image's path relative to the repository, its time and its peak VRAM, so the records work on any checkout, on Linux or Windows. `scores.json` caches the judges' scores on the GPU and `scores_cpu.json` on the CPU (`SCORE_DEVICE=cpu`), keyed by path and modification time. The two agree to the third decimal, but compare arms only within one.

| File | What it does |
|---|---|
| `common.py` | The 12 held-out prompts behind the README's numbers, and how records store paths. |
| `run_arm.py` | Generates one arm, 12 prompts × seeds 0-2, with any `generate()` that takes `adm.generate`'s options. It resumes where it stopped. |
| `score.py` | Scores arms with ViT-B/16, DFN5B ViT-H-14 and SigLIP2 ViT-L-16-256, plus the two aesthetic predictors, and compares them with a baseline, paired on (prompt, seed). |
| `sheet.py` | Contact sheet of arms side by side. |
| `search.py` | Recurrence, Feynman-Kac steering, progressive pruning and the high-noise x0 rollout on the unconditional sampler. With none of them it makes exactly `adm.generate`'s image (`check_search.py`). |
| `conditioning.py` | The U-Net with a zero-initialised CLIP projection added to its timestep embedding. |
| `train_conditioning.py` | Fine-tunes that projection, the timestep MLP and every ResBlock's embedding layers on webdataset tars. |
| `cond_generate.py` | Samples a fine-tuned checkpoint with classifier-free guidance and CLIP guidance, for `run_arm.py`. |
| `queue1.sh` to `queue4.sh` | The runs below, in order, on the RTX 2070 SUPER. `queue4.sh` did not run there: the 2070 fell off the bus as it started. |
| `queue5.sh`, `queue6.sh`, `queue.sh`, `todo.txt` | The runs on the RTX 4070. `queue5.sh` regenerates the baselines there and starts queue4's arms. From `queue6.sh` on, `queue.sh` runs the commands listed in `todo.txt` one at a time, so the list can be reordered while it runs, and `queue6.log` has what ran. |
| `layout.py` | Starts the unconditional pipeline from another arm's image, shrunk to a tiny layout and noised partway down the schedule. |
| `thumbnail.py` | Thumbnail guidance and hand-off: a 64x64 text-conditioned model (GLIDE, LAIONIDE, Karlo's decoder, OpenAI's CLIP of noisy images or `layout_prior.py`'s) sets the composition. Also CLIP guidance towards the image embedding Karlo's prior draws (`target`), limits on the thumbnail model's push (`parallel`, `limit`, `cap`) and PAG on the U-Net's call (`pag_scale`). `check_thumbnail.py` checks its formula exactly on Gaussian data. |
| `karlo_alone.py` | Karlo without the unconditional U-Net, as the reference: its whole pipeline, or its bare thumbnail. |
| `push_stats.py` | Records at every model call how large a thumbnail model's push is, how far it moves the denoised thumbnail and how much of that the sampler clamps (`push_stats.jsonl`). |
| `check_seed.py` | Draws one image of an arm again in a new process and compares the pixels. |
| `add_arm.py`, `reorder.py` | Insert an arm into `todo.txt`, and move arms to its front. `todo_later.txt` holds arms set aside. |
| `time_cli.py` | Wall clock and peak VRAM of the `cgd` command, for the README's Speed table. |
| `sheet_rows.py` | `sheet.py` turned on its side, a row per arm with a label and only the prompts asked for. |
| `self_guidance.py` | Perturbed-attention guidance with the four layers its paper perturbs on this checkpoint. |
| `cache_layouts.py`, `build_layout_data.py` | Shrink the LAION shards to 64x64 thumbnails with captions, and embed the captions, for `train_layout_prior.py`. |
| `layout_prior.py`, `train_layout_prior.py` | A small transformer that draws 64x64 thumbnails from CLIP text embeddings, and its training. |
| `confirm_prompts.json` | 66 PartiPrompts nothing was tuned on, the `confirm` prompt set of `run_arm.py` and `score.py`. |
| `rebaseline.md`, `search.md` | What `score.py` printed for each queue. |

## Re-baseline

All at the defaults (`-cutn 16`), 12 held-out prompts × 3 seeds, on the RTX 2070 SUPER (`rebaseline.md`):

| Arm | ViT-B/16 | ViT-H | SigLIP2 | Pooled change, 95% CI, prompts improved | Median time |
|---|---|---|---|---|---|
| `-uncond` (ViT-B/32) | 0.299 | 0.274 | 0.097 | baseline | 17.5 s |
| `-uncond -clip ViT-B/32+PE-Core-B-16` | 0.323 | 0.314 | 0.124 | +0.030 ± 0.017, 10/12 | 32.0 s |
| Class-conditional `-size 256` | 0.321 | 0.308 | 0.121 | +0.027 ± 0.025, 10/12 | 7.8 s |
| `-bb sd-turbo` | 0.352 | 0.383 | 0.190 | +0.085 ± 0.016, 12/12 | 34.9 s |

## Search on the unconditional sampler

On the ensemble guide, `ViT-B/32+PE-Core-B-16`, against `-uncond` with it at 60 steps (`uncond_ens`, 32.0 s). Change in prompt agreement over the three judges, and in the `ViT-B/16` aesthetic rating, which nothing guided (`search.md`):

| Arm | Prompt agreement | Rating | Median time |
|---|---|---|---|
| Recurrence 2 above t 300 (`r2_e`) | +0.018 ± 0.017, 8/12 | +0.51 ± 0.34, 11/12 | 53.2 s |
| `-n 90` (`n90_e`) | +0.018 ± 0.017, 9/12 | +0.01 ± 0.20 | 46.8 s |
| Recurrence 2 and `-n 90` (`r2n90_e`) | +0.014 ± 0.012, 8/12 | +0.33 ± 0.36 | 77.9 s |
| Recurrence 3 (`r3_e`) | +0.023 ± 0.012, 10/12 | +0.36 ± 0.25, 11/12 | 74.0 s |
| Recurrence 2 on every guided step (`r2all_e`) | +0.020 ± 0.016, 9/12 | +0.46 ± 0.34, 10/12 | 58.3 s |
| 3-call x0 rollout above sigma 0.9 (`roll3_e`) | -0.011 ± 0.022, 6/12 | -0.12 | 45.3 s |
| Best of 4 by PE-Core-L-14-336 (`bo4_e`) | +0.024 ± 0.009, 11/12 | +0.24 ± 0.33 | 125.3 s |
| Feynman-Kac steering, 4 particles (`fk4_e`) | +0.025 ± 0.011, 11/12 | +0.34 ± 0.23, 11/12 | 126.7 s |
| Pruning 8, 4, 2 (`prune_e`) | +0.024 ± 0.008, 11/12 | +0.39 ± 0.41 | 153.3 s |

Recurrence 2 above t 300 ties `-n 90` on prompt agreement, -0.0001 ± 0.018, and beats it on the rating, +0.50 ± 0.39 (p 0.021, 9/12). Nothing beat it per second: a third run, recurring below t 300 and adding `-n 90` all tied it on agreement and rated a little lower. Feynman-Kac steering and pruning tied plain best of 4 (+0.001 ± 0.006 and +0.000 ± 0.004), which `-best` already does, and the rollout lost. Selection by PE-Core-L, which no judge uses, carried over to the judges, but clipped more pixels, 2.4-4.2% against 1.4%.

## Fine-tuning on the RTX 4070

`train_conditioning.py` trains on 256x256 webdataset tars with captions, such as the LAION-2B-en aesthetic set in `~/Data/laion2b-en-aesthetic` (989 shards, 5.59M images). It drops images LAION rates as likely unsafe (`punsafe` above 0.1) or watermarked (`pwatermark` above 0.5), which keeps 94% of them. Each image is conditioned on nothing 10% of the time, so classifier-free guidance works. Otherwise it is conditioned on its caption's PE-Core-B-16 text embedding 45% of the time, and on its own image embedding with LAFITE noise for the rest. Training starts from exactly the unconditional checkpoint, because the projection's last layer is zero.

On the RTX 2070 SUPER at batch 2 it trains 1.8 images per second at a 5.1-5.5 GiB peak. Each extra image adds about 1.1 GiB, so batch 4 should fit in 12 GB with room to spare. The log prints `peak_gib`, so raise the batch if there is room. The whole set is 195 GB, so copy as many shards as fit, then:

```sh
uv sync
uv run python reports/evidence/uncond/train_conditioning.py \
    --data '/path/to/laion2b-en-aesthetic/*.tar' --out runs/emb \
    --batch-size 4 --accumulate 4 --workers 8
```

It saves `runs/emb/last.pt` every 500 steps and resumes from it, and writes a preview of four held-out prompts every 2000 steps. To score a checkpoint against the arms above:

```sh
uv run python reports/evidence/uncond/run_arm.py cond_STEP cond_generate:generate \
    '{"checkpoint": "runs/emb/last.pt", "clip_model": "ViT-B/32+PE-Core-B-16", "cfg_scale": 3.0}'
uv run python reports/evidence/uncond/score.py uncond_ens cond_STEP
```

On Windows, give `--data` a forward-slash glob such as `D:/laion2b-en-aesthetic/*.tar`, and run the `queue*.sh` scripts from Git Bash or MSYS2 with `sh`. The data loader's workers are spawned there, not forked, which the script supports.

Images from a different GPU are not the same images, because the U-Net's arithmetic differs, though the seed's noise is the same. Arms made on the 4070 are comparable with the 2070's only statistically, so for paired comparisons regenerate the baseline there too: `run_arm.py uncond_ens4070 cgd.adm:generate '{"unconditional": true, "clip_model": "ViT-B/32+PE-Core-B-16", "recurrence": 1}'`.

## Composition from a thumbnail model (RTX 4070, 2026-11-14)

All on the RTX 4070, 12 held-out prompts × 3 seeds, the ensemble guide and the defaults (`-cutn 16`), scored on the CPU (`scores_cpu.json`). `cgd_uncond_e` is cgd's own `-uncond` default with recurrence 2, which the 2070 never ran: it gains +0.0145 ± 0.0125 (p 0.042, 9/12) over no recurrence (`uncond_ens4070`), as the prototype did (`r2_e4070`, +0.0127 ± 0.0126).

**What was wrong.** The unconditional pipeline matches the prompt's textures but not its composition: a fox's body without a head, bricks without the bicycle. Started from another arm's image shrunk to 32x32 and noised to timestep 600 (`layout.py`, `turbo32_t600`), the same U-Net and CLIP guidance draw a coherent picture of it in half the time.

**CLIP similarity cannot see this.** The three CLIP judges score that arm the same as the baseline (-0.0001 ± 0.019). They reward a collage of the prompt's textures as much as a picture of it, and CLIP guidance from pure noise is very good at collages. So `score.py` now also reports PickScore and HPS v2.1, trained on people's choices between images of one prompt, and `blind.py` makes shuffled left/right sheets for a judge who does not know which arm is which.

**Thumbnail guidance** (`thumbnail.py`). GLIDE's 64x64 text-conditioned base model is asked, at every call, what the prompt changes about the noisy image averaged over 4x4 blocks, which has 16 times the signal to noise ratio. Its classifier-free guidance direction, enlarged and divided by 4, is added to the U-Net's noise prediction. `check_thumbnail.py` shows the formula is exact on Gaussian data. Hand-off is the other use: GLIDE draws a whole thumbnail, and sampling starts from it at timestep 600.

| Arm | ViT-B/16 | ViT-H | SigLIP2 | PickScore | HPS v2.1 | Rating | Median time |
|---|---|---|---|---|---|---|---|
| `uncond_ens4070`, recurrence 1 | 0.328 | 0.326 | 0.138 | 20.03 | 0.2137 | 5.03 | 17.9 s |
| `cgd_uncond_e`, the `-uncond` default | 0.339 | 0.346 | 0.150 | 20.71 | 0.2239 | 5.14 | 30.2 s |
| `turbo32_t600`, from SD-Turbo's 32x32 layout | 0.336 | 0.350 | 0.149 | 21.00 | 0.2343 | 5.79 | 14.5 s |
| `glide_t600_e`, hand-off from GLIDE's thumbnail | 0.335 | 0.346 | 0.156 | 21.11 | 0.2245 | 5.58 | 16.7 s |
| `xres_glide_w3_e`, thumbnail guidance at scale 3 | 0.347 | 0.357 | 0.168 | 21.50 | 0.2509 | 5.55 | 32.1 s |
| `glide_alone`, GLIDE with its own upsampler, no CLIP | 0.313 | 0.334 | 0.138 | 20.88 | 0.2048 | 4.95 | 4.2 s |
| `glide_thumb`, GLIDE's thumbnail enlarged bicubically | 0.285 | 0.318 | 0.132 | 21.06 | 0.2145 | 4.56 | 2.1 s |
| `turbo`, `-bb sd-turbo` (made on the 2070) | 0.352 | 0.383 | 0.190 | 24.07 | 0.3119 | 7.07 | 34.9 s |

Against `cgd_uncond_e`, paired per prompt:

| Arm | CLIP judges | PickScore | HPS v2.1 | Rating | Blind pairs won, seeds 1 and 2 |
|---|---|---|---|---|---|
| `xres_glide_w3_e` | +0.0122 ± 0.0067, p 0.005, 11/12 | +0.79 ± 0.51, p 0.007, 10/12 | +0.0270 ± 0.0160, p 0.003, 10/12 | +0.41 ± 0.30, 9/12 | 20 of 24, p 0.0015 |
| `glide_t600_e` | +0.0004 ± 0.0096, 7/12 | +0.41 ± 0.48, p 0.092, 7/12 | +0.0006 ± 0.0134, 6/12 | +0.44 ± 0.23, 11/12 | 15 of 22 decided, 2 ties, p 0.13 |
| `turbo32_t600` | -0.0001 ± 0.0187, 7/12 | +0.30 ± 0.86, 8/12 | +0.0105 ± 0.0209, 8/12 | +0.65 ± 0.23, 12/12 | not judged |

The blind judge of `xres_glide_w3_e` was three fresh agents that only read the sheets; they gave the same 24 answers, so they count as one judge. The blind judge of `glide_t600_e` was the session that ran the experiments, before it had seen either arm's images at those seeds. The keys and answers are in `outputs/eval/blind_keys/`.

GLIDE's own pipeline scores below both uses of its thumbnail here, so the unconditional U-Net with CLIP guidance is the better upsampler of it. The rest of the distance to SD-Turbo on PickScore, HPS and the rating is rendering, which a thumbnail cannot bring. The closest published method is Upsample Guidance (Hwang et al. 2024, arXiv 2404.01709), which uses the same matching of signal to noise ratio to let one model guide itself at a higher resolution; ILVR conditions an unconditional model on a reference image's low frequencies.

GLIDE comes from the `research` dependency group (`uv run ...`): `glide-text2im`, pinned in `uv.lock`. `"prior": "laionide"` loads LAIONIDE v3, GLIDE base fine-tuned on LAION, in its place.

**LAIONIDE v3 gives what GLIDE gives.** `xres_laionide_w3_e`, the same thumbnail guidance at scale 3 from the fine-tune, against `cgd_uncond_e`: +0.0123 ± 0.0084 on the CLIP judges (p 0.016, 9/12), +0.76 ± 0.50 on PickScore (21.47, p 0.007, 10/12), +0.0260 ± 0.0126 on HPS (0.2499, p 0.001, 11/12) and +0.45 ± 0.25 on the rating (5.59, 10/12), at 32.6 s and 6.7 GiB. Paired against `xres_glide_w3_e` it is +0.0001 ± 0.0066, -0.03 ± 0.35, -0.0010 ± 0.0083 and +0.05 ± 0.24: a tie on everything. Only one of these 12 prompts is of a person, which is what GLIDE's filtered data lacks and LAION has, so they do not test where the fine-tune should differ most.

### Karlo as the thumbnail model

Karlo (`kakaobrain/karlo-v1-alpha`, CreativeML Open RAIL-M, in the Hugging Face cache) is an open reproduction of unCLIP: a prior that turns the prompt into a CLIP ViT-L/14 image embedding, a 64x64 decoder conditioned on the prompt and that embedding, and a 64 to 256 super-resolution model. Its decoder is a thumbnail model like GLIDE's, on a cosine schedule, so `thumbnail.py` matches the signal to noise ratio the same way (`matched_predictions()`). It needs no dependency beyond `diffusers`. A seed reproduces its image byte for byte (`check_seed.py`, `xres_karlo_w3_e` prompt 3 seed 1).

Same protocol as above, against `cgd_uncond_e`:

| Arm | ViT-B/16 | ViT-H | SigLIP2 | PickScore | HPS v2.1 | Rating | Median time | Peak VRAM |
|---|---|---|---|---|---|---|---|---|
| `cgd_uncond_e`, the `-uncond` default | 0.339 | 0.346 | 0.150 | 20.71 | 0.2239 | 5.14 | 30.2 s | 5.9 GiB |
| `xres_glide_w3_e`, GLIDE's thumbnail guidance | 0.347 | 0.357 | 0.168 | 21.50 | 0.2509 | 5.55 | 32.1 s | 6.7 GiB |
| `xres_karlo_w3_e`, Karlo's thumbnail guidance at scale 3 | 0.351 | 0.373 | 0.178 | 22.65 | 0.2745 | 5.81 | 34.1 s | 7.8 GiB |
| `xres_karlo_w3_noclip`, the same without CLIP guidance or recurrence | 0.330 | 0.354 | 0.164 | 22.19 | 0.2536 | 5.52 | 10.3 s | 5.3 GiB |
| `karlo_t600_e`, hand-off from Karlo's thumbnail | 0.341 | 0.356 | 0.157 | 21.67 | 0.2443 | 5.61 | 17.4 s | 7.8 GiB |
| `karlo_thumb`, Karlo's thumbnail enlarged bicubically | 0.298 | 0.350 | 0.159 | 22.17 | 0.2529 | 4.98 | 3.1 s | 3.8 GiB |
| `karlo_alone`, Karlo's own pipeline, no U-Net and no CLIP guidance | 0.336 | 0.366 | 0.177 | 22.49 | 0.2838 | 5.82 | 2.9 s | 7.1 GiB |
| `turbo`, `-bb sd-turbo` (made on the 2070) | 0.352 | 0.383 | 0.190 | 24.07 | 0.3119 | 7.07 | 34.9 s | 3.2 GiB |

| Arm, against `cgd_uncond_e` | CLIP judges | PickScore | HPS v2.1 | Rating |
|---|---|---|---|---|
| `xres_karlo_w3_e` | +0.0223 ± 0.0097, p < 0.001, 12/12 | +1.94 ± 0.59, p 0.001, 11/12 | +0.0506 ± 0.0133, p < 0.001, 12/12 | +0.67 ± 0.30, p 0.002, 11/12 |
| `xres_karlo_w3_noclip` | +0.0042 ± 0.0204, 7/12 | +1.48 ± 0.93, p 0.009, 10/12 | +0.0297 ± 0.0215, p 0.016, 9/12 | +0.38 ± 0.47, 9/12 |
| `karlo_t600_e` | +0.0064 ± 0.0157, 7/12 | +0.97 ± 0.71, p 0.009, 11/12 | +0.0205 ± 0.0168, p 0.021, 10/12 | +0.47 ± 0.35, p 0.021, 10/12 |
| `karlo_alone` | +0.0145 ± 0.0136, p 0.052, 8/12 | +1.78 ± 0.65, p < 0.001, 12/12 | +0.0600 ± 0.0158, p < 0.001, 12/12 | +0.68 ± 0.38, p 0.001, 11/12 |

A fresh agent that only read the sheets preferred `xres_karlo_w3_e` to `cgd_uncond_e` in 23 of 24 blind pairs at seeds 1 and 2 (`base_vs_karlo3`, sign test p < 0.0001).

**Confirmed on prompts nothing was tuned on.** The 66 PartiPrompts of `confirm_prompts.json`, one seed, `confirm_karlo_w3` against `confirm_base` (the `-uncond` default with the ensemble): +0.0300 ± 0.0080 on the CLIP judges (58/66), +1.26 ± 0.23 on PickScore (20.46 to 21.72, 58/66), +0.0338 ± 0.0090 on HPS (0.2161 to 0.2499, 55/66) and +0.28 ± 0.22 on the rating (41/66), all p < 0.001 but the rating's 0.027, at 32.9 s against 28.1 s. A fresh agent preferred it in 54 of the 66 blind pairs (`confirm_base_vs_karlo3`, sign test p < 0.0001). Its share of clipped pixels is higher there, 5.2% against 1.0%.

**What it does and does not show.** Karlo's thumbnail guidance is the largest gain this folder has measured on the unconditional pipeline, on every judge and by eye: about three quarters of the distance to SD-Turbo on the CLIP judges and over half of it on PickScore and HPS. But Karlo's own pipeline, which takes 3 seconds, scores the same: paired against `karlo_alone`, `xres_karlo_w3_e` is +0.0079 ± 0.0078 on the CLIP judges (p 0.052, 8/12), +0.16 ± 0.46 on PickScore, -0.0093 ± 0.0097 on HPS (p 0.052, 3/12) and -0.01 ± 0.32 on the rating. So the composition is Karlo's, and the unconditional U-Net with CLIP guidance is an upsampler of it that is as good as Karlo's own and twelve times slower. With GLIDE the U-Net and CLIP beat the thumbnail model's own pipeline; with Karlo they tie it. A fresh agent given the two blind chose `xres_karlo_w3_e` in 14 of 24 pairs (`karlo3_vs_alone`, sign test p 0.54). The 66 confirmation prompts say the same: `confirm_karlo_w3` against `confirm_karlo_alone` is +0.0128 ± 0.0057 on the CLIP judges (50/66), +0.08 ± 0.22 on PickScore, -0.0214 ± 0.0065 on HPS (13/66) and +0.10 ± 0.20 on the rating. CLIP guidance buys agreement with the prompt, and Karlo's own upsampler what HPS rewards.

**Recurrence carries half of the gain.** `xres_glide_w3_r1_e`, GLIDE's thumbnail guidance without recurrence, scores as the baseline with recurrence does (CLIP judges -0.0031 ± 0.0138, PickScore -0.28 ± 0.59, HPS -0.0001 ± 0.0186, 20.3 s). Each rerun gives the thumbnail model another say, as it gives CLIP. Choosing the best of 8 GLIDE thumbnails by the guide's CLIP score before a hand-off (`glide_c8_t600_e`) changes nothing over one thumbnail (CLIP judges +0.0005 ± 0.0124, PickScore +0.53 ± 0.37, HPS -0.0018 ± 0.0123, 23.2 s).

### CLIP guidance towards the image embedding Karlo's prior draws

A text embedding is not where images are in CLIP, and guidance towards it pays for anything that reads as the prompt. Karlo's prior turns the prompt into the ViT-L/14 embedding of one plausible picture of it, which Karlo's own thumbnails match at a cosine of 0.75 to 0.80 against 0.26 to 0.30 for the text. `thumbnail.py`'s `target` makes that embedding what CLIP guidance pulls towards, with ViT-L/14 as the guide: alone (`"embedding"`), or as one more model of the ensemble (`"both"`). All at 4 cutouts, because ViT-L/14 at 16 overflows 12 GB (`karlo_z`, stopped after two images at 11.2 GiB and 66 s each).

| Arm | Against | CLIP judges | PickScore | HPS v2.1 | Rating | Median time |
|---|---|---|---|---|---|---|
| `uncond_L14_c4`, ViT-L/14 guiding towards the text | `cgd_uncond_e` | -0.0584 ± 0.0215, 0/12 | -1.48 ± 0.56, 0/12 | -0.0348 ± 0.0187, 1/12 | -0.26 ± 0.40 | 17.7 s |
| `karlo_z_c4`, ViT-L/14 guiding towards the embedding | `cgd_uncond_e` | -0.0910 ± 0.0298, 0/12 | -1.43 ± 0.68, 2/12 | -0.0299 ± 0.0268, 3/12 | +0.27 ± 0.33, 11/12 | 18.2 s |
| `karlo_ez_c4`, the ensemble and the embedding | `uncond_e_c4` | -0.0152 ± 0.0155, 3/12 | -0.06 ± 0.60, 4/12 | -0.0022 ± 0.0168, 5/12 | +0.16 ± 0.23, 10/12 | 24.8 s |
| `xres_karlo_w3_z_c4`, Karlo's thumbnail guidance and the embedding alone | `cgd_uncond_e` | +0.0103 ± 0.0138, 9/12 | +1.72 ± 0.75, 11/12 | +0.0540 ± 0.0168, 12/12 | +0.74 ± 0.37, 10/12 | 22.3 s |

It does not help. Alone, the embedding keeps the scene and the style and loses the subject: a snowy field with a small fox in it, a coast without the lighthouse, wooden shelves for sushi on slate. It ties the same model guiding towards the text on PickScore and HPS, and ViT-L/14 is a poor guide here either way. Added to the ensemble it costs agreement and changes nothing else. Under Karlo's thumbnail guidance it scores as the text-guided ensemble does on PickScore, HPS and the rating, and lower on agreement (+0.0103 against +0.0223).

`uncond_e_c4` is the baseline at 4 cutouts. Against `cgd_uncond_e` at 16 it is a tie, -0.0044 ± 0.0087 on the CLIP judges, -0.03 on PickScore and +0.0022 on HPS, at 20.5 s against 30.2 s.

### A CLIP of noisy images as the thumbnail model

OpenAI released, with GLIDE, the CLIP it trained on noisy 64x64 images (`clip_image_enc.pt` and `clip_text_enc.pt`, MIT, 630 MB together; a ViT with 4x4 patches and a timestep embedding, trained on GLIDE's filtered data). It cannot draw, but it can read a thumbnail at the noise levels where the composition forms, which the usual CLIP cannot: it still tells a fox from a mushroom at timestep 800. The gradient of its logit is the classifier form of the thumbnail direction, -sigma * grad log p(prompt | y), so `"prior": "glide-clip"` sends it through the same code as a generator's eps(y | prompt) - eps(y): the same averaging over 4x4 blocks, the same matching of the signal to noise ratio, the same division by 4. `check_thumbnail.py` checks that form exactly as well, with the Gaussian case's own classifier. It is CLIP guidance twice over, with no text-to-image model anywhere: this CLIP sets the composition on the thumbnail, and the ensemble on cutouts of the denoised image does the rest. The earlier research notes proposed it as shortlist item 8, and it could not run until `glide-text2im` was a tracked dependency.

Seed 0 only, 12 images, to choose the scale (GLIDE's notebook uses 3 on its text-conditional model, and ADM's classifier needed 10 on this unconditional checkpoint):

| Arm, against `cgd_uncond_e` at seed 0 | CLIP judges | PickScore | HPS v2.1 | Rating | Clipped | Median time |
|---|---|---|---|---|---|---|
| `xres_nclip_w3_e`, scale 3 | +0.0100 ± 0.0213, 6/12 | +0.38 ± 0.59, 9/12 | +0.0229 ± 0.0213, 9/12 | +0.26 ± 0.47 | 3.0% | 29.7 s |
| `xres_nclip_w10_e`, scale 10 | +0.0176 ± 0.0186, 8/12 | +0.63 ± 0.88, 7/12 | +0.0243 ± 0.0209, 10/12 | +0.38 ± 0.43 | 2.6% | 29.7 s |
| `xres_nclip_w30_e`, scale 30 | +0.0080 ± 0.0263, 7/12 | -0.38 ± 1.23, 6/12 | +0.0006 ± 0.0306, 6/12 | -0.13 ± 0.81 | 3.5% | 32.0 s |
| `xres_nclip_w10_noclip`, scale 10 and no other guidance, so no recurrence and no backward pass | -0.0439 ± 0.0492, 3/12 | -1.12 ± 0.83, 3/12 | -0.0397 ± 0.0311, 3/12 | -0.58 ± 0.61 | 4.0% | 8.2 s, 2.8 GiB |

At scale 10 over all 3 seeds, `xres_nclip_w10_e` against `cgd_uncond_e`: +0.0114 ± 0.0082 on the CLIP judges (p 0.012, 11/12), +0.56 ± 0.54 on PickScore (20.71 to 21.26, p 0.042, 8/12), +0.0225 ± 0.0147 on HPS (0.2239 to 0.2463, p 0.016, 9/12) and +0.43 ± 0.42 on the rating (5.14 to 5.57, 8/12), at 30.3 s against 30.2 s and 6.5 GiB against 5.9. A seed reproduces its image byte for byte (`check_seed.py`, prompt 5 seed 0). That is GLIDE's own thumbnail guidance to within the noise (`xres_glide_w3_e`: +0.0122, +0.79, +0.0270, +0.41), from a classifier a quarter of the generator's size. A fresh agent preferred it to `cgd_uncond_e` in 18 of 24 blind pairs at seeds 1 and 2 (`base_vs_nclip10`, sign test p 0.023). On its own it is not enough: the thumbnail carries the layout, and without the ensemble on cutouts nothing draws the detail.

**It keeps its gain at 4 cutouts.** `xres_nclip_w10_e_c4` against `uncond_e_c4`, both at `-cutn 4`: +0.0173 ± 0.0116 on the CLIP judges (p 0.001, 11/12), +0.75 ± 0.40 on PickScore (20.68 to 21.43, p 0.001, 11/12), +0.0215 ± 0.0158 on HPS (0.2261 to 0.2476, p 0.012, 10/12) and +0.24 ± 0.28 on the rating, at 22.4 s against 20.5 s and 6.4 GiB against 5.7. Against the 16-cutout default `cgd_uncond_e` it is +0.0129 ± 0.0090, +0.73 ± 0.47, +0.0237 ± 0.0145 and +0.41 ± 0.29, all at p < 0.01 and 10/12, in three quarters of the time. So the noise-aware CLIP costs 2 seconds an image, and 4 cutouts with it beat 16 without.

**Confirmed on the 66 prompts nothing was tuned on.** `confirm_nclip_w10` against `confirm_base`, one seed: +0.0159 ± 0.0078 on the CLIP judges (p < 0.001, 47/66), +0.34 ± 0.21 on PickScore (20.46 to 20.80, p 0.003, 46/66), +0.0154 ± 0.0073 on HPS (0.2161 to 0.2315, p < 0.001, 47/66) and +0.18 ± 0.21 on the rating (35/66), at 30.5 s against 28.1 s. A fresh agent preferred it in 50 of the 66 blind pairs (`confirm_base_vs_nclip10`, sign test p < 0.0001). Its cost is colour: 7.1% of pixels clipped against 1.0%, and saturation 0.49 against 0.37. Nothing caps this push the way the guidance rate caps the ensemble's. Capping it was tried, below, and takes the gain with it. Karlo's thumbnail guidance gains about twice as much on the same prompts (`confirm_karlo_w3`: +0.0300, +1.26, +0.0338, 54 of 66 blind).

**Where the clipped pixels come from.** `push_stats.py` records, at every model call, what a thumbnail model adds to the noise prediction (prompts 3, 8 and 0 at seed 0, `push_stats.jsonl`; the recording changes nothing, so the images are the arms' own). Means over the calls in each range of timesteps:

| Timesteps | Noise-aware CLIP, scale 10 | Karlo, scale 3 | GLIDE, scale 3 |
|---|---|---|---|
| 900 to 999: push over the U-Net's noise prediction, RMS | 0.020 | 0.016 | 0.006 |
| 900 to 999: RMS it moves the denoised thumbnail by | 2.08 | 1.57 | 0.63 |
| 900 to 999: thumbnail pixels it takes past the sampler's clamp | 47% | 35% | 10% |
| 800 to 899: the same three | 0.028, 1.12, 26% | 0.025, 1.05, 21% | 0.011, 0.42, 1% |
| 700 to 799 | 0.037, 0.67, 14% | 0.027, 0.48, 4% | 0.020, 0.36, 2% |
| 600 to 699 | 0.032, 0.29, 4% | 0.035, 0.31, 1% | 0.019, 0.17, 0% |
| below 600 | under 0.02, under 0.08, 0% | under 0.05, under 0.10, 0% | under 0.02, under 0.07, 0% |

Next to the noise every push is small, 1 to 4% of the U-Net's prediction and under the 8% that caps CLIP guidance, so a cap of that kind would never bind. But above timestep 800 the noise is 25 to 150 times the signal, and there the two strong pushes move the denoised thumbnail, whose pixels span -1 to 1, by more than its whole range. The sampler's clamp at 1.2 then flattens a quarter to a half of it, and the composition is first laid down as a poster. That is the order of the clipped pixels on the 66 confirmation prompts: 7.1% with the noise-aware CLIP, 5.2% with Karlo, 1.0% without either. The part of the push along the denoised thumbnail, which adaptive projected guidance (Sadat et al. 2024, arXiv 2410.02416) removes because it only scales the image, is a third to a half of it above timestep 900 and under 5% below 800, and above 900 it points against the U-Net's blur, not along it.

**Holding the push back takes the gain with it.** `thumbnail.py`'s `Limits` hold a push back in three ways, all off by default: `parallel`, the share kept of the part along the denoised thumbnail; `limit`, the sampler's own `limit_guidance()` quantile (`-gl`) applied to the denoised thumbnail, so only as much of the push as keeps it in range; and `cap`, a bound on its RMS next to the noise prediction's, which the table above shows would never bind and was not run.

| Arm | Against | CLIP judges | PickScore | HPS v2.1 | Rating | Clipped | Images over 5% clipped |
|---|---|---|---|---|---|---|---|
| `xres_nclip_w10_p0_e`, `parallel` 0, seed 0 | `xres_nclip_w10_e` at seed 0 | -0.0065 ± 0.0079, 5/12 | +0.08 ± 0.79 | -0.0031 ± 0.0188 | +0.05 ± 0.42 | 3.9% against 2.6% | 3 against 2 of 12 |
| `xres_nclip_w10_gl99_e`, `limit` 0.99, seed 0 | `xres_nclip_w10_e` at seed 0 | -0.0024 ± 0.0106, 6/12 | -0.29 ± 0.53 | -0.0034 ± 0.0156 | -0.23 ± 0.51 | 1.6% against 2.6% | 0 against 2 of 12 |
| `confirm_nclip_w10_gl99`, `limit` 0.99, 66 prompts | `confirm_nclip_w10` | -0.0103 ± 0.0072, p 0.008, 24/66 | -0.33 ± 0.23, p 0.009, 23/66 | -0.0093 ± 0.0072, p 0.016, 25/66 | -0.05 ± 0.16 | 2.8% against 7.1% | 14 against 29 of 66 |
| the same | `confirm_base` | +0.0056 ± 0.0091, 34/66 | +0.01 ± 0.19, 30/66 | +0.0060 ± 0.0075, 36/66 | +0.13 ± 0.17 | 2.8% against 1.0% | 14 against 4 of 66 |

Removing the part along the image does not lower the clipping. The range limit does, by 60% on the confirmation prompts, and takes two thirds of the gain: PickScore is back at the default's, and a fresh agent that only read the sheets preferred the unlimited push in 43 of the 66 blind pairs (`confirm_nclip10_vs_gl99`, sign test p 0.019) and chose between the limited one and the default 33 to 33 (`confirm_base_vs_gl99`; the unlimited push won that comparison 50 to 16). So the strong push at the noisiest steps is the method: it lays the composition down while the image is still a blur, and the clipped pixels are its price. Saturation stays at 0.46 with the limit, against 0.37 for the default and 0.45 for Karlo's own pipeline, so the fuller colour is the picture's and not the clamp's.

### Recurrence on the class-conditional default, second-order reruns, and the command's time

`cc128` and `cc128_r2` are the class-conditional default (128, ViT-B/32, 16 cutouts) at `-recur 1` and `-recur 2`, 12 prompts × 3 seeds: +0.0119 ± 0.0145 on the CLIP judges (p 0.092, 9/12), +0.48 ± 0.34 on PickScore (p 0.016, 9/12), +0.0045 ± 0.0104 on HPS and +0.06 ± 0.20 on the rating, at 2.3 s an image against 1.5 s. A tie on agreement, so the class-conditional default stays at 1.

`r2c_e` is the prototype's recurrence 2 with `correct_reruns` (`search.py`): a rerun extrapolates from the x0 of the step before, as the first run of a step does, where cgd's `recurring_step()` makes a rerun first order. Against `r2_e4070`, the same without it, 12 prompts × 3 seeds: -0.0079 ± 0.0144 on the CLIP judges (5/12), -0.15 ± 0.56 on PickScore (20.31, 5/12), -0.0013 ± 0.0143 on HPS and -0.09 ± 0.22 on the rating, with 5.2% of pixels clipped against 2.5%, both at 29.2 s. A tie that leans the wrong way and clips twice as much, so reruns stay first order.

`time_cli.py` measured `uv run cgd -cutn 4 ... "A red fox in the snow"` on the RTX 4070 under Windows, as the README's Speed table does (`time_cli.jsonl`): 128 default 6.9 s and 2.36 GiB, 256 default 9.8 s and 5.31 GiB, 256 `-uncond -recur 1` 15.4 s and 5.31 GiB, 256 `-uncond` 21.3 s and 5.31 GiB, 512 `-uncond -recur 1` 34.1 s and 4.10 GiB, 512 `-uncond` 50.2 s and 4.11 GiB. The peaks equal the 2070's. Starting up takes longer on Windows, which is why 128 is not faster than the 2070's 6.5 s.


## Perturbed-attention guidance, four layers (RTX 4070, 2026-11-14)

The deep research of 2026-11-14 ranked this first among what needs no training: PAG in the configuration its paper used on this very checkpoint, four attention layers perturbed at scale 1 (`self_guidance.py`), which took unguided FID from 26.21 to 16.23. The repo's earlier test, which lost, perturbed one layer of the class-conditional 128 model. It costs a second U-Net call without a gradient.

`pag4_s1_e` against `cgd_uncond_e`, 12 held-out prompts × 3 seeds: -0.0286 ± 0.0181 on the CLIP judges (p 0.003, 2/12), -0.15 ± 0.68 on PickScore (20.56, 6/12), +0.0114 ± 0.0162 on HPS (0.2353, p 0.20, 8/12) and +0.29 ± 0.33 on the rating (5.43, 9/12), at 36.9 s against 30.2 s and 6.0 GiB against 5.9. A fresh agent that only read the sheets preferred it in 16 of 24 blind pairs at seeds 1 and 2 (`base_vs_pag4`, sign test p 0.15).

So it is a trade, not a gain. PAG's images are sharper and more often one coherent picture: a whole fox, a whole cat, a face in the pencil sketch. But it pulls towards what the unconditional model draws well, and where that is not the prompt the subject goes: the lighthouse shrinks to a silhouette on the horizon, the sailboat becomes a close-up of a hull. The CLIP judges count the second, the blind judge mostly the first, and 24 pairs do not decide between them.

**With thumbnail guidance it is a gain.** The noise-aware CLIP's thumbnail guidance brings the subject, which is what PAG loses, and PAG the coherence. `xres_nclip_w10_pag1_e` is `xres_nclip_w10_e` with PAG at scale 1 on the U-Net's call (`thumbnail.py`'s `pag_scale`), 12 prompts × 3 seeds, 39.0 s and 6.6 GiB:

| `xres_nclip_w10_pag1_e` against | CLIP judges | PickScore | HPS v2.1 | Rating | Blind pairs won, seeds 1 and 2 |
|---|---|---|---|---|---|
| `cgd_uncond_e`, the `-uncond` default | +0.0056 ± 0.0103, 8/12 | +1.01 ± 0.57, p 0.009, 10/12 | +0.0368 ± 0.0169, p 0.002, 11/12 | +0.58 ± 0.32, p 0.005, 10/12 | 20 of 24, p 0.0015 (`base_vs_nclip10_pag1`) |
| `xres_nclip_w10_e`, thumbnail guidance alone | -0.0058 ± 0.0066, 4/12 | +0.46 ± 0.53, p 0.064, 9/12 | +0.0143 ± 0.0115, p 0.016, 10/12 | +0.15 ± 0.21, 9/12 | 17 of 24, p 0.064 (`nclip10_vs_nclip10_pag1`) |
| `pag4_s1_e`, PAG alone | +0.0342 ± 0.0195, p 0.002, 11/12 | +1.16 ± 0.30, p < 0.001, 12/12 | +0.0253 ± 0.0102, p < 0.001, 12/12 | +0.28 ± 0.23, p 0.042, 8/12 | not judged |

Its means are 0.344, 0.347 and 0.161 on the CLIP judges, PickScore 21.72, HPS 0.2606 and rating 5.72: the highest PickScore, HPS and rating of any arm here that uses no text-to-image model, above GLIDE's thumbnail guidance (21.50, 0.2509, 5.55). It gives up half of thumbnail guidance's gain in agreement for that. Each blind judge was a fresh agent that only read the sheets. `check_seed.py` remade prompt 7 at seed 2 and it matched byte for byte.

**The 66 confirmation prompts keep the gain over the default, and do not decide whether PAG adds to thumbnail guidance.** `confirm_nclip_w10_pag1`, seed 0, 40.4 s and 6.6 GiB against the default's 28.1 s and 5.9 GiB and thumbnail guidance's 30.5 s:

| `confirm_nclip_w10_pag1` against | CLIP judges | PickScore | HPS v2.1 | Rating | Blind pairs won |
|---|---|---|---|---|---|
| `confirm_base`, the `-uncond` default | +0.0039 ± 0.0087, 32/66 | +0.53 ± 0.21, p < 0.001, 43/66 | +0.0252 ± 0.0075, p < 0.001, 51/66 | +0.38 ± 0.19, p < 0.001, 45/66 | 46 of 66, p 0.0019 (`confirm_base_vs_nclip10_pag1`) |
| `confirm_nclip_w10`, thumbnail guidance alone | -0.0120 ± 0.0064, p < 0.001, 17/66 | +0.18 ± 0.15, p 0.048, 40/66 | +0.0099 ± 0.0061, p 0.011, 37/66 | +0.21 ± 0.16, p 0.011, 41/66 | 37 of 66, p 0.39 (`confirm_nclip10_vs_nclip10_pag1`) |

Its means are 0.337, 0.358 and 0.161 on the CLIP judges, PickScore 20.99, HPS 0.2413 and rating 4.83, with 7.1% of pixels clipped and 29 images over 5%, the same as thumbnail guidance alone. So both sets agree on what PAG does on top of thumbnail guidance: PickScore, HPS and the rating go up, agreement with the prompt goes down by about 0.01, and an image takes a third longer. Over both sets a blind judge preferred the combination to thumbnail guidance alone in 54 of 90 pairs (sign test p 0.07), and against the default it won 66 of 90 where thumbnail guidance alone won 68 of 90. PAG is therefore an option on top of thumbnail guidance for a more finished picture, not a second gain of its size: thumbnail guidance alone gets the blind judge's preference at no extra time.

**At 4 cutouts it beats the default in the default's time, and adds little to thumbnail guidance.** `xres_nclip_w10_pag1_e_c4` is the combination at `-cutn 4`, 12 prompts × 3 seeds, 6.6 GiB. Its time is 31.5 s, the median of the 18 images made while nothing else ran: a CPU scoring job slowed the first 18, and the median of all 36, which `score.py` prints, is 32.9 s.

| `xres_nclip_w10_pag1_e_c4` against | CLIP judges | PickScore | HPS v2.1 | Rating | Seconds |
|---|---|---|---|---|---|
| `cgd_uncond_e`, the default at 16 cutouts | +0.0057 ± 0.0127, 8/12 | +0.74 ± 0.57, p 0.016, 10/12 | +0.0316 ± 0.0160, p 0.001, 11/12 | +0.50 ± 0.28, p 0.002, 10/12 | 31.5 against 30.2 |
| `uncond_e_c4`, the default at 4 cutouts | +0.0100 ± 0.0160, 5/12 | +0.76 ± 0.47, p 0.007, 10/12 | +0.0294 ± 0.0163, p 0.002, 10/12 | +0.33 ± 0.18, p 0.003, 10/12 | 31.5 against 20.5 |
| `xres_nclip_w10_e_c4`, thumbnail guidance alone at 4 cutouts | -0.0073 ± 0.0077, p 0.064, 4/12 | +0.01 ± 0.28, 7/12 | +0.0079 ± 0.0098, 8/12 | +0.09 ± 0.16, 9/12 | 31.5 against 22.4 |
| `xres_nclip_w10_pag1_e`, itself at 16 cutouts | +0.0000 ± 0.0076, 6/12 | -0.28 ± 0.29, p 0.064, 3/12 | -0.0052 ± 0.0093, 3/12 | -0.07 ± 0.22, 5/12 | 31.5 against 39.0 |

Its means are 0.342, 0.350 and 0.161 on the CLIP judges, PickScore 21.44, HPS 0.2555 and rating 5.64, with 2.9% of pixels clipped. A fresh agent that only read the sheets preferred it to the default in 19 of 24 blind pairs at seeds 1 and 2 (`base_vs_nclip10_pag1_c4`, sign test p 0.0066). But thumbnail guidance alone at 4 cutouts has the same PickScore in 22.4 s, so here the 9 s PAG costs buy +0.008 on HPS and +0.09 on the rating, neither of them clear of zero. The cheapest arm that beats the default on everything is `xres_nclip_w10_e_c4`, in three quarters of the default's time.

## Built into cgd (2026-11-15)

`cgd -uncond` now has thumbnail guidance (`-thumb clip`, `glide`, `laionide`, `karlo` or `none`, `cgd/thumbnails.py`) and PAG (`-pag`, `cgd/pag.py`), ported from `thumbnail.py` and `self_guidance.py` with the same arithmetic in the same order, and its defaults are `xres_nclip_w10_e_c4`: the ensemble, 4 cutouts, 60 steps, recurrence 2 and `-thumb clip`.

**The options make the experiments' images, byte for byte.** `check_port.py` makes prompt 3 at seed 1 of each of its ten arms again with `cgd.adm.generate()`, and all ten are the stored image. That took a fix. At first only the default, `xres_nclip_w10_e_c4`, differed, and only when it was the first image of a process: OpenAI's CLIP of noisy images is built straight on the GPU, and its weights' random initialisation drew from the GPU's random stream after the seed was set, which moved the noise the image starts from. `thumbnail.py` loads it before setting the seed, so the experiment never saw this. The thumbnail loaders now fork the GPU's stream as well as the CPU's, and `test_loading_a_thumbnail_model_draws_none_of_the_runs_random_numbers` fails without that. GLIDE, LAIONIDE and Karlo are built on the CPU and were never affected. So the measurements above are measurements of the options.

Still to run:

- The integration tests (`tests/integration`). A first run passed its first 32 tests and was stopped by the machine running low on memory.
- `time_cli.py` with the new configurations, GLIDE, LAIONIDE and Karlo at 4 cutouts, and thumbnail guidance at 512, where the thumbnail is an eighth of the image and nothing has been measured. They are queued in `todo.txt`, for `queue7.sh`.
- The README's examples and GIFs of each option.

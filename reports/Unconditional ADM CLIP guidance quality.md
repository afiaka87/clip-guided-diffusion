# Search narrows the gap, conditioning attacks it

Training-free methods alone can probably lift the unconditional 256 pipeline from **0.316 to roughly 0.33-0.35 ViT-B/16**, stacked at about 4-9x the current 20 s, but that number flatters it, and the real target is further away than 0.35 suggests. ViT-B/16 is close kin to the ViT-B/32 guide and is one of `-best`'s two pickers. On the class-conditional 256 model, the CLIP ensemble moved ViT-B/16 by only +0.002, while it moved ViT-H by +0.025 and SigLIP2 by +0.018. The gap to SD-Turbo there is **0.034 on ViT-B/16 but 0.064 on ViT-H and 0.062 (47%) on SigLIP2**. Most of SD-Turbo's lead is its text-conditioned prior: SD-Turbo with no CLIP at all already beats every ADM configuration measured, and CLIP noise optimisation adds only 0.01-0.02 on top. The training-free shortlist, in order, has three items, and all of them fit in 5.3-6.5 GiB on the 8 GB card:

| Change | Expected gain | Basis |
|---|---|---|
| The existing `ViT-B/32+PE-Core-B-16` ensemble on `-uncond` | +0.010 to +0.020 | Measured on class-conditional |
| Feynman-Kac steering with guided proposals, scored by a verifier that is not also a reporting judge | +0.015 to +0.035 | Estimated |
| Recurrence focused on the 31 calls where composition is set | +0.005 to +0.015 | Best-evidenced fix for a weak guide on unconditional ImageNet ADM |

Together they plausibly add **+0.03 to +0.05 on judges that did not guide, about half the distance to SD-Turbo**. The only lever on the remainder is giving the U-Net a CLIP input. That means a zero-initialised projection of a CLIP embedding into the 1024-d time embedding, the same additive slot the class-conditional checkpoint uses for its label. It would be trained without captions on the RTX 4070, at batch about 8. It should take 0.5-3 GPU-days to start working, plus one to two weeks for a classifier-free-guidance phase. Its gain is unmeasured, and it is both the highest-ceiling and the least certain item. Reward fine-tuning, RL, DPO and distillation either need that conditioning first or are out of budget, and there is no evidence that Disco-era settings or upsampler cascades help. The current 6-prompt baseline cannot detect any of these gains. Screening needs about 25 prompts per arm to detect a 0.02 change and 80-90 prompts for 0.01, scored by judges that neither guided nor selected.

## ViT-B/16 hides half the gap, and the rest is ImageNet's prior

The only measurement of `-uncond` quality is **0.316 ViT-B/16 on 6 held-out prompts**. It took 20.0 s and 5.3 GiB at 256px on the RTX 2070 SUPER. The removed 1000-step ancestral sampler scored 0.331 on the same prompts and was higher on 5 of 6, but the 95% interval of the difference spans zero (−0.040 to +0.014), and it took 18x the time ([README](../README.md)). That baseline has no ViT-H or SigLIP2 score. It also uses a different prompt set from the 12 prompts behind every SD-Turbo number, so the first thing to measure is the unconditional model on those 12 prompts with all three judges.

The class-conditional 256 model was scored on those 12 prompts, and it shows how much the guide's kin understates the distance. The table puts it next to SD-Turbo with the same judges.

| Pipeline, 12 held-out prompts | ViT-B/16 | ViT-H-14 | SigLIP2 | Source |
|---|---|---|---|---|
| ADM 256 class-conditional, ViT-B/32 guide, 2 seeds | 0.314 | 0.295 | 0.114 | ([scores](evidence/scores/sizes256.txt)) |
| ADM 256 class-conditional, ViT-B/32+PE-Core-B-16, 2 seeds | 0.316 | 0.320 | 0.132 | ([scores](evidence/scores/sizes256.txt)) |
| ADM 128 class-conditional, ViT-B/32, seed 0 | 0.289 | 0.277 | 0.087 | ([scores](evidence/scores/gap_seed0.txt)) |
| SD-Turbo, one step, no CLIP, seed 0 | 0.340 | 0.373 | 0.184 | ([scores](evidence/scores/gap_seed0.txt)) |
| SD-Turbo + noise optimisation, ensemble guide, seed 0 | 0.359 | 0.386 | 0.196 | ([scores](evidence/scores/gap_seed0.txt)) |
| SD-Turbo + noise optimisation, ViT-B/32 only, CLIP loss scaled 100x, seed 0 | 0.375 | 0.374 | 0.186 | ([scores](evidence/scores/gap_seed0.txt)) |
| `-bb sd-turbo` default, 3 seeds | 0.350 | 0.384 | 0.194 | ([README](../README.md)) |
| `-uncond` default, 6 different prompts | 0.316 | not measured | not measured | ([README](../README.md)) |

Three things follow. First, **the ensemble's gain at 256 is invisible to ViT-B/16 (+0.002) but clear on ViT-H (+0.025) and SigLIP2 (+0.018)**, so an arm that only reports ViT-B/16 will miss real improvements. Second, the reverse also happens. Optimising SD-Turbo hard against ViT-B/32 alone (its CLIP loss scaled 100x) pushed ViT-B/16 to 0.375, the highest number in the table, while ViT-H fell below the ensemble-guided run and 10.8% of pixels clipped. **ViT-B/16 rewards optimisation against its kin.** The README makes the same point: ViT-B/16 "is close kin to ViT-B/32, so it favours it" ([README](../README.md)). Third, **SD-Turbo with no CLIP at all scores 0.340/0.373/0.184**, above every ADM arm measured, and its CLIP noise optimisation adds 0.013-0.019. On judges that did not guide, roughly four-fifths of the target is the prior, not the guidance.

The unconditional model is likely worse off than the class-conditional one on the non-kin judges, because with no class to bring the subject, CLIP has to set the composition too ([README](../README.md)). That is inference, not measurement. It nonetheless sets the realistic goal: **0.35 ViT-B/16 measured without selection on ViT-B/16, with ViT-H and SigLIP2 both rising**, not a ViT-B/16 number reached by optimising against ViT-B/16.

## Extra compute pays where composition is set, in the first 31 of 60 calls

On the linear schedule the sampler covers logSNR −5.06 to 4.61 in 60 uniform steps. With `UNCONDITIONAL_GUIDANCE_WINDOW = (0.1, 1.0)` ([sampler.py](../cgd/sampler.py)), **39 calls are guided (t ≥ 100), 31 fall at t ≥ 300, and 27 at σ > 0.9**. That last region is 45% of the calls, where an unconditional model has nothing to work from but noise and the guide. Measured on the 2070 SUPER, one fp16 U-Net forward at 256 without gradients takes 146-149 ms. A forward and backward takes 290 ms at 4.94 GiB, or 445 ms at 2.17 GiB with the U-Net's own checkpointing ([local measurement](../research_notes/Unconditional%20ADM%20CLIP%20guidance%20quality/inference_time_search_and_optimisation.md)). With CLIP and cutouts, a guided call costs about 0.35-0.4 s. That accounts for the 20 s baseline and prices every option below.

**Recurrence is the best-evidenced way to spend that compute on this model.** Recurrence means re-noising back to the same level and redoing the guided step k times. On unconditional ImageNet ADM with label guidance, TFG's accuracy rose from 40.9% to 52.3% to 59.8% at 1, 2 and 4 recurrences, with FID 177, 163 and 165 ([TFG](https://arxiv.org/abs/2409.15761)). Universal Guidance used k = 5-10 with CLIP on OpenAI's unconditional ImageNet model ([UG](https://arxiv.org/abs/2302.07121)). FreeDoM used 10 repeats between t = 800 and 500 ([FreeDoM](https://arxiv.org/pdf/2303.09833)). On unconditional ImageNet with CLIP, Shen et al.'s one-step x̂0 with time travel and random augmentation scored 27.7, against 25.9 for FreeDoM, 25.7 for UG, 25.1 for MPGD and 24.3 for LGD-MC ([Shen](https://arxiv.org/abs/2403.12404)). The 1000-step sampler's +0.015 at 18x the time is probably the same effect, bought inefficiently, because it makes many small guided updates during composition. Recurrence at r = 2 over the 31 calls at t ≥ 300 adds about 11 s (31 s, 1.6x), and r = 4 adds about 33 s (53 s, 2.7x), with no change in memory. It must be compared at equal time against more steps, `-n 120` (about 37 s) and `-n 250` (about 75 s), because Restart showed that re-noising can beat simply adding steps (CIFAR-10 FID 1.90 at 43 calls against 1.97 at 63), but only with a matched baseline ([Restart](https://arxiv.org/html/2306.14878)).

Building it into `sampler.py` takes three decisions:

1. **Where the noise comes from.** Draw the re-noising noise from a dedicated `torch.Generator` seeded from the run seed, so the default path's `th.randn` order, and every README GIF, stays byte-identical.
2. **The solver's state on a redo.** After a redo, take a first-order step or recompute the solver's `Previous` x0, so DPM-Solver++(2M) does not extrapolate from a stale estimate.
3. **The guidance cap.** Leave `cap_guidance` unchanged, since it already keeps each repeated step from overshooting.

**Selection gains most when combined with gradient guidance, and the verifier must not be a reporting judge.** The repo has already measured best-of-4. It was picked by ViT-B/16 and SigLIP2 and scored by ViT-H, which did not pick. It gained **+0.015 ± 0.007 (11 of 12 prompts), or +0.025 ± 0.011 (12 of 12) with the ensemble**, on the class-conditional model ([README](../README.md); [scores](evidence/scores/best.txt)). At equal compute, plain random search matches or beats zero-order and path search on the verifier's own metric. However, searching on CLIP raised CLIPScore from 0.71 to 0.82 while aesthetic score fell from 5.79 to 5.68 ([Ma et al.](https://arxiv.org/abs/2501.09732)). Feynman-Kac steering does better than independent seeds when its proposals are gradient-guided. On SD 1.5, gradient guidance alone reached ImageReward 0.668 in 20 s. FK with 4 particles and plain proposals reached 0.898 in 8.1 s, and FK with 4 particles and gradient-guided proposals reached **1.290 in 55 s**. On SDXL, FK at k = 4 beat best-of-4, 1.298 against 1.236 ([FK steering](https://arxiv.org/html/2501.06848)). Without tempering, SMC over-optimises even at 32 particles; with tempering, 4-8 particles suffice ([DAS](https://arxiv.org/abs/2501.05803)). On an unconditional ImageNet model guided by a classifier, separate verifiers for the gradient and for selection gave FID 118.2 and 55.9% accuracy, against 155.7 and 35.8% for best-of-N with one verifier, at N = 8 and about half the compute ([Zhang et al.](https://arxiv.org/html/2505.23614v2)).

The concrete design runs k = 4-8 particles, each the existing ensemble-guided sampler on its own seed. Particles are resampled 4-6 times on a tempered potential, which `-best`'s pickers (ViT-B/16 and SigLIP2) compute on x̂0. The multinomial draws come from the dedicated generator, and each child inherits its parent's solver state. One detail matters specifically for the unconditional model: **x̂0 at σ > 0.9 is close to the dataset mean, so a potential computed there ranks blurs.** Resampling should start once x̂0 has structure, roughly between t ≈ 700 and 300, which is still inside the composition window. On the 2070 SUPER, k = 4 takes about 90 s and k = 8 about 180 s, at 5.6-6.5 GiB with an fp16 verifier resident. On the 4070, the extra 4 GB allows two particles per batch without inner checkpointing. A cheaper approximation of best-of-8 is progressive pruning, which keeps the best seed 80-90% of the time ([PSP](https://arxiv.org/abs/2607.21591)). Run 8 seeds to 25% of the schedule, keep 4 to 50%, then finish 2. That costs about 4 baseline runs (about 80 s). Recurrence and search compound: local refinement alone plateaus, while combined with global search it keeps gaining ([Zhang et al.](https://arxiv.org/html/2505.23614v2)). **Recurrence r = 2 inside FK k = 4 is therefore the natural 8x configuration.**

**Noise optimisation fits in memory but carries the largest hacking risk for the least evidence.** With step-level checkpointing, backpropagating through any number of sampler steps peaks at about 4.95 GiB (2.2 GiB with inner checkpointing as well), at 4.5, 11.1 and 23.7 s per iteration through 10, 25 and 50 steps. The naive graph runs out of memory at 2 steps ([local measurement](../research_notes/Unconditional%20ADM%20CLIP%20guidance%20quality/inference_time_search_and_optimisation.md)). DNO raised SD 1.5's aesthetic score from 5.367 to 7.496 in one minute through a 15-step sampler, but reached only 5.754 through 50 steps in the same time. Unregularised, it turned a "black duck" white ([DNO](https://arxiv.org/html/2405.18881)). DOODL gained only 3.1% and 2.6% CLIP score over one-step guidance ([DOODL](https://arxiv.org/abs/2303.13703)), and ReNO measured it at 24 s per iteration, 20 minutes and 40 GB for CLIPScore 0.261 to 0.289 ([ReNO](https://arxiv.org/abs/2406.04312)). For ADM, the design is DNO over x_T and the injected SDE noises, through a 15-25-step sampler, with a chi-norm penalty and a reward ensemble that is not also a reporting judge. Thirty iterations would take about 4.5-5 minutes at 5.5-6 GiB on the 2070 SUPER (about 15x). There is no published result for CLIP on a pixel ImageNet ADM. Tree and trajectory search are gradient-free, so they waste the CLIP gradient this pipeline already pays for. Their best reported CLIP gain, 0.246 to 0.297, cost 80x the model calls without a compute-matched baseline ([noise trajectory search](https://arxiv.org/html/2506.03164)). They do not belong on the shortlist.

| Option | 2070 SUPER time | 2070 SUPER peak | RTX 4070 time (est.) | Notes |
|---|---|---|---|---|
| `-uncond` today | 20.0 s | 5.3 GiB | 10-15 s | Measured on the 2070 SUPER |
| Ensemble `ViT-B/32+PE-Core-B-16` | about 23 s | about 5.3-5.5 GiB | 12-17 s | PE adds about 1.3 s per 30 calls at 256 ([README](../README.md)) |
| `-n 120` / `-n 250` | about 37 s / 75 s | 5.3 GiB | 19-56 s | Compute-matched controls |
| Recurrence r = 2 / r = 4, t ≥ 300 | about 31 s / 53 s | 5.3 GiB | 16-40 s | +31 / +93 guided calls |
| Hybrid x0 rollout, n = 3, σ > 0.9 | about 32 s | 5.3 GiB | 16-24 s | +81 forwards without gradients |
| Checkpointed x0 rollout, n = 3, σ > 0.9 | about 56 s | 5.3-5.5 GiB | 28-42 s | Exact gradient |
| Progressive pruning, 8 → 4 → 2 | about 80 s | 5.3-5.6 GiB | 40-60 s | Approximates best-of-8 |
| FK k = 4 / k = 8 | about 90 s / 180 s | 5.6-6.5 GiB | 45-135 s | Verifier resident |
| DNO, 30 iterations × 20 steps | 4.5-5 min | 5.5-6 GiB | 2.5-3.5 min | Opt-in only |

The 4070 column assumes 0.5-0.75x the 2070 SUPER's time per configuration. That is my estimate from the two cards' published specifications; nothing has run on the 4070. Peak VRAM per configuration should be about the same on both cards.

## At σ above 0.9 the guide reads a grey blur through a low-pass Jacobian

The unconditional pipeline's specific weak link is the gradient's path during composition. With no class, x̂0 at maximum noise is near the dataset mean, a grey blur, and CLIP's opinion of it says little about layout. The gradient then passes through the U-Net's Jacobian, which on this repo's class-conditional 128 pipeline is a strong low-pass filter. Of the raw CLIP gradient's power, 60-69% lies above 0.25 cycles per pixel. **After the Jacobian only 3% does at σ = 0.997, rising to 35% at σ = 0.33** ([gradient spectrum log](evidence/exp/e5.log)). Low frequencies are what composition needs, so the filtering itself is not the problem; the uninformative x̂0 is. That is why the methods worth testing here change what CLIP sees at high noise, and leave the magnitude of the gradient alone.

**A multi-step x0 is the cheapest such change.** Symplectic Adjoint Guidance estimates x0 with n ODE steps and backpropagates exactly through them. Gains stop being significant past n = 4. On a V100 with SD it used 15.66 GB, against 28.63 GB for direct backprop at n = 2 ([SAG](https://arxiv.org/html/2312.12030v1)). A consistency-model x0 raised a guiding classifier's accuracy from 0.79 to 0.90 and lowered FID from 15.73 to 13.45, for 150 s and 5.35 GB rising to 218 s and 6.32 GB. Without randomising x0 (τ = 0.2), guidance overfit the guiding model and held-out models scored worse ([Xu et al.](https://arxiv.org/html/2403.12063)). Neither study tested CLIP on ADM.

The proposed first test is an untested hybrid estimator that costs no extra memory. Run a 2-3-step rollout without gradients to get a sharp x0, take CLIP's gradient there, and pass it back through the one-step Jacobian only, so that ∇ ≈ (∂x̂0/∂x_t)ᵀ ∇L(x0_roll). Applied at σ > 0.9, that adds about 81 forwards, about 12 s. If it helps, the exact version with per-step checkpointing adds about 36 s at about the same peak. **Keeping one Jacobian matters.** Skipping it entirely, as MPGD does, collapses on hard semantic targets: 6.8% accuracy on ImageNet labels against DPS's 38.8% ([MPGD](https://arxiv.org/abs/2311.16424); [TFG](https://arxiv.org/abs/2409.15761)). Stale shortcuts in the same family cost FlowChef an FID of 19.28 against 5.39 ([FlowChef](https://arxiv.org/abs/2412.00100)).

**A noise-aware CLIP is the trained version of the same idea, and PPAP shows the recipe works on this exact checkpoint.** PPAP distilled an off-the-shelf classifier into noise-band LoRA experts, each covering 200 timesteps, using about 500K generated images, KL distillation, AdamW at 1e-4 and batch 256 for 300K iterations on 8 A100s ([PPAP](https://ar5iv.labs.arxiv.org/html/2212.05973)). It ran on OpenAI's unconditional ADM 256 with 25 DDIM steps:

| Guide on unconditional ADM 256, 25 DDIM steps | FID |
|---|---|
| Unguided | 40.24 |
| Off-the-shelf classifier | 38.74 |
| Off-the-shelf classifier on x̂0 | 38.14 |
| PPAP, 5 experts | 29.65 |
| PPAP, 10 experts | 27.86 |
| Labelled multi-experts, 5 | 19.98 |

PPAP also found that timestep conditioning barely matters and capacity per noise band does. One warning comes from the same paper: at 250 DDPM steps, a single noise-aware model scored 38.15, worse than unguided at 28.97.

The CLIP version would put LoRA band experts on the guiding tower and train them to map a noised image to the clean teacher's embedding. Such a student learns E[φ(x0) | x_t], an expected embedding whose norm shrinks with noise, so **guidance should use its unnormalised dot product with the text embedding, not cosine similarity**. My estimate for training on the 4070 is about 300 images per second for a ViT-B, so a ViT-B/32 student in under a day and a PPAP-scale run in about 3 days. At inference it needs no U-Net backward pass, which cuts the guided call's peak from 5.3 GiB to about 2.5-3 GiB and brings its time close to an unguided call.

The evidence against is real. A time-conditioned latent CLIP scored below plain CLIP on one-step-denoised images, 39.7 against 43.4 and 45.2 against 48.9 ([noise-aware guides notes](../research_notes/Unconditional%20ADM%20CLIP%20guidance%20quality/noise_aware_guides_and_gradient_paths.md)). On class-conditional ADM, calibrated off-the-shelf classifiers beat OpenAI's noise-trained one, FID 2.19 against 2.97 ([EluCD](https://arxiv.org/abs/2310.11311)). NoisyCLIP and StitchVM report noisy-input reward models, but for ranking and alignment rather than pixel guidance, and StitchVM's figures (3.2x faster, about half the memory, about 10 GPU-hours) are the authors' alone ([NoisyCLIP](https://arxiv.org/html/2512.08505v1); [StitchVM](https://gohyojun15.github.io/StitchVM)). The repo's existing secondary model, which predicts x0 from a cheap 13.8M-parameter network, tied at 256 and cut peak VRAM from 5.3 to 1.9 GiB ([MMagic source](https://mmagic.readthedocs.io/en/latest/_modules/mmagic/models/editors/disco_diffusion/secondary_model.html); [README](../README.md)). That is a reason to expect the student's main gain in memory and speed, with quality uncertain.

**GLIDE's public noised CLIP is the zero-training probe of the same hypothesis.** It is a 64px ViT-B, trained for 390K iterations at batch 32K on filtered data with no people, MIT licensed, and its `clip_image_enc.pt` (345.8 MB) is still downloadable from an archived repository ([GLIDE](https://arxiv.org/abs/2112.10741); [glide-text2im](https://github.com/openai/glide-text2im)). Average-pooling a 256px x_t 4x4 to 64px multiplies its SNR by 16, which maps to a GLIDE timestep by logSNR. Used as an extra term at σ > 0.9 only, it would judge composition and leave detail to the existing guide. It needs OpenAI's archived glide-text2im code as a new dependency and has not been run here. Its own record is a caution: in GLIDE's human evaluation, CLIP guidance lost to classifier-free guidance, Elo −73.2 against 82.7 for photorealism and 29.3 against 110.9 for caption match.

**Gradient magnitude is already solved, and robust towers are not the answer.** The repo's relative cap (`DEFAULT_GUIDANCE_RATE = 0.08`) does what DSG's spherical step does; on these settings DSG would change the step radius by about 0.3% ([DSG](https://arxiv.org/html/2402.03201)). Disco's `clamp_grad` is RMS clipping with no published evidence. Blurring the gradient and sine or rising schedules tied or lost here, because the cap dominates ([README](../README.md)). FARE and TeCoA towers lost as guides here, and robust plus x̂0 plus Adam scored worse in GradCheck ([GradCheck](https://arxiv.org/html/2406.17399v1)). The theory points to smoothing the guide over noise (Shen et al.), which is the noise-aware route above, rather than to adversarial training.

**ADM's own 256 classifier is the strongest published guide on this checkpoint, and the repo already has a better use for it.** It takes unconditional 256 from FID 26.21 to 12.00 at scale 10, and at scale 1 it hurts, 33.03. It cost 46 V100-days to train ([ADM](https://arxiv.org/abs/2105.05233)). It only applies to prompts that map to an ImageNet class, and for those the class-conditional checkpoint, which `closest_class()` already serves, is the better tool. The practical version of this finding is routing: send prompts with a close class to the class-conditional pipeline, and keep `-uncond` for the rest. PAG is similar: it took this exact unconditional checkpoint from FID 26.21 to 16.23 at 250 steps ([PAG](https://arxiv.org/abs/2403.17377)), but with CLIP at 30 calls on the class-conditional model it scored −0.013 ± 0.009 ([README](../README.md)). It is worth one retest on `-uncond` as a realism arm scored by preference models, not as a fix for alignment.

## Conditioning is the only lever on the prior, and it fits in 12 GB

An unconditional model cannot learn prompt following from a prompt-dependent reward, because nothing in its input tells it the prompt. Every reward-fine-tuning method below therefore either improves prompt-independent qualities, which for an aesthetic reward trades away CLIP similarity ([Diffusion-DPO](https://arxiv.org/abs/2311.12908)), or needs a conditioning input first ([reward notes](../research_notes/Unconditional%20ADM%20CLIP%20guidance%20quality/reward_finetuning_and_distillation.md)). The precedent for a pixel model of this size following CLIP is cc12m_1: 602M parameters, conditioned on CLIP text embeddings, trained from scratch for 3.1M steps and then 250K classifier-free steps. Crowson shipped it with both classifier-free guidance and CLIP guidance, so the two compose ([v-diffusion-pytorch](https://github.com/crowsonkb/v-diffusion-pytorch)).

**The architecture already has the slot.** The repo's unconditional U-Net has 552.81M parameters ([conditioning notes](../research_notes/Unconditional%20ADM%20CLIP%20guidance%20quality/clip_conditioning_finetune.md)):

| Part | Parameters | Share |
|---|---|---|
| Convolutions | 447.03M | 80.9% |
| `emb_layers` of the 42 ResBlocks (turn the time embedding into AdaGN scale and shift) | 53.00M | 9.6% |
| Attention `qkv` and `proj_out`, 16 blocks | 51.44M | 9.3% |
| Time MLP | 1.31M | 0.2% |

guided-diffusion conditions its class models with `emb = emb + self.label_emb(y)`, and its attention blocks have no encoder input.

The recommended design is a bias-free, zero-initialised MLP from a CLIP embedding into that 1024-d embedding. At step zero the model is exactly the current one, so a fine-tune starts with no regression. Through `emb_layers` the condition reaches every ResBlock, and unCLIP feeds its decoder the same way ([unCLIP](https://ar5iv.labs.arxiv.org/html/2204.06125)). The trainable set can grow in three stages:

1. The projection alone, a few million parameters.
2. The projection plus `emb_layers`, about 55M.
3. Both plus LoRA on attention and convolutions.

Decoupled cross-attention, about 38M new parameters, is more expressive but means writing new attention inputs. A DEFT-style side network is the other proven option on this exact model. DEFT trained a network of 4-17% of the model's size beside the frozen unconditional ADM 256, for inpainting in 3.9 hours, with KID 0.29 against 15.28 for DPS ([DEFT](https://arxiv.org/abs/2406.01781)). ControlNet-style copies cost about 23% more memory and 34% more time per step and are built for spatial conditions ([ControlNet](https://ar5iv.labs.arxiv.org/html/2302.05543)).

**Training can skip captions, but sampling has to bridge from text to image embeddings.** A decoder trained on CLIP image embeddings receives a text embedding at sampling time, and the two sit in different places. Each published method bridges the gap differently:

| Method | How it bridges | Result |
|---|---|---|
| LAFITE | Normalised Gaussian noise on the image embedding during training | Zero-shot COCO FID 26.94 ([LAFITE](https://arxiv.org/abs/2111.13792)) |
| KNN-Diffusion | Conditions on retrieved neighbours | FID 12.5 against 32.8 without them ([KNN-Diffusion](https://arxiv.org/abs/2204.02849)) |
| unCLIP | Learns a prior | FID 7.99, against 16.55 when the text embedding is fed in directly ([unCLIP](https://ar5iv.labs.arxiv.org/html/2204.06125)) |

The cheapest combination for this repo is LAFITE-style noise in training and an existing small prior at sampling. nousr's MIT-licensed text-to-image-embedding prior for ViT-B/32 is one ([nousr/conditioned-prior](https://huggingface.co/nousr/conditioned-prior)). Drop the condition for 10-20% of samples (unCLIP 10%, GLIDE 20%) so classifier-free guidance is available, at scales of 1.25-3.

**Data decides how far the prior moves.** Conditioning on ImageNet alone teaches the model to follow CLIP within ImageNet's content, which helps prompts like "a red fox in the snow" and not "a bowl of ramen" or "a neon street". Broader data is needed for the rest:

1. PD12M: 12.4M public-domain and CC0 images with synthetic captions, the cleanest licence ([PD12M](https://arxiv.org/abs/2410.23144)).
2. CC12M, Re-LAION and DataComp.
3. Synthetic pairs from the repo's own `-bb` pipeline.

Mix in 20-30% ImageNet replay against forgetting. A target of 100k-1M images is my estimate, between ControlNet's 200k-sample depth model and danbooru's 4.8M-image ADM fine-tune.

**Memory and time on the 4070 are workable.** A frozen-trunk bench with block checkpointing on the 2070 SUPER, run on a contended GPU, measured:

| Batch | Time per step | Peak |
|---|---|---|
| 1 | 1012 ms | 3.18 GiB |
| 2 | 1805 ms | 4.28 GiB |

That is about 1.1 GiB per extra sample, so **batch about 8 fits in 12 GB**. The bench ended with the GPU dropping off the bus (Xid 79), so multi-day runs should save checkpoints often and run under a power limit. A full fine-tune with fp32 AdamW and an EMA on the GPU needs about 11 GB before activations and does not fit. 8-bit AdamW with the EMA on the CPU fits at about batch 2.

My throughput estimate, from ADM's 6.4 images per V100-second, is 2-3 images per second on the 4070, or about 170k-260k samples a day:

| Stage | Cost |
|---|---|
| ControlNet's "sudden convergence" (under 10K steps) at batch 8 | about half a day |
| A classifier-free phase of 1-2M samples | about one to two weeks |

Recommended settings, from the precedents, are lr 1e-4 for new layers and 1e-5 for `emb_layers`, EMA 0.999-0.9995, bf16, and no `torch.compile`, which degraded glide-finetune ([glide-finetune](https://github.com/afiaka87/glide-finetune)). At inference on the 2070 SUPER, classifier-free guidance plus CLIP guidance on the 60-call schedule should take about 28-30 s at 5.3-5.7 GiB. If the model composes the subject itself, the class-conditional schedule (30 calls, window 0.1-0.8) may suffice, at about 12-15 s.

**The gain is unmeasured.** No public recipe adds CLIP or text conditioning to the ImageNet ADM checkpoints. The community fine-tunes were unconditional domain models, and Disco's best-known ones were small models trained from scratch ([HF KaliYuga](https://huggingface.co/KaliYuga/pixelartdiffusion4k)). My position is that it is the only item on this list with a mechanism for the non-kin gap of about 0.06-0.09. I expect large gains on ImageNet-like prompts and gains limited by data elsewhere, anywhere from 0 to +0.06 pooled, depending on how far the data reaches.

**Reward fine-tuning comes after conditioning.** It is cheap once there is an input to steer, but its held-out gains are small. Gradient methods are the efficient family:

1. **DRaFT.** It works best at K = 1, degrades past K = 10, and learns about 2x faster in its LV variant. A CLIP reward degraded image quality, and LoRA scaling beat KL and early stopping as the remedy, at rank 8 and lr 4e-4 ([DRaFT](https://arxiv.org/abs/2309.17400)).
2. **AlignProp.** It fits in about 15 GB with checkpointing and was 25x more data-efficient than DDPO ([AlignProp](https://arxiv.org/abs/2310.03739)).
3. **Adjoint Matching.** It gives the best trade between reward and diversity, with an explicit KL knob:

| Method | ClipScore | PickScore | HPSv2 | Diversity |
|---|---|---|---|---|
| Base | 28.32 | 18.15 | 17.89 | 56.53 |
| Adjoint Matching | 31.65 | 19.76 | 24.49 | 37.24 |
| DRaFT-1 | 30.95 | 19.37 | 24.37 | 27.39 |
| DPO | 27.77 | 17.92 | 17.30 | 54.11 |

Adjoint Matching's memoryless schedule for this model is plain DDPM ancestral sampling, which ADM supports ([Adjoint Matching](https://arxiv.org/abs/2409.08861)).

Before conditioning exists, the only reward option is DRaFT through the unguided tail of the guided sampler, which nobody has tested with gradient guidance. **RL and DPO are poor value here.** DDPO mode-collapsed on unconditional priors ([RTB](https://arxiv.org/abs/2405.20971)), policy gradients need one to two orders of magnitude more reward queries than backprop, and DiffusionNFT's efficiency is shown on flow models at scale ([DiffusionNFT](https://arxiv.org/abs/2509.16117)). **Distillation is out of budget.** One-step consistency distillation at LSUN 256 reached FID 7.80 on 64 A100s ([Consistency Models](https://arxiv.org/abs/2303.01469)), and the best ImageNet one-step results are at 64px on 4-8 large GPUs for days ([DMD2](https://arxiv.org/abs/2405.14867); [ECT](https://arxiv.org/abs/2406.14548)). The affordable piece is LD3, which learns a short solver's timesteps in 5-40 minutes and would shorten DNO's differentiable chain ([LD3](https://arxiv.org/abs/2405.15506)).

## Cascades and Disco habits stop below SD-Turbo

**Refining through SD-Turbo fixes texture, not composition.** Over a class-conditional ADM 128 image guided by ViT-B/32 (0.289/0.277/0.087), SDEdit through SD-Turbo raised the scores with strength, but never reached SD-Turbo alone (0.340/0.373/0.184) ([scores](evidence/scores/gap_seed0.txt)):

| SDEdit strength | ViT-B/16 | ViT-H | SigLIP2 |
|---|---|---|---|
| 0.3 | 0.303 | 0.297 | 0.120 |
| 0.5 | 0.303 | 0.306 | 0.126 |
| 0.7 | 0.322 | 0.332 | 0.147 |

It also keeps ADM's semantic mistakes; the pencil sketch stays a pen drawing. SDEdit's own paper puts the faithfulness sweet spot at 0.3-0.6 ([SDEdit](https://ar5iv.labs.arxiv.org/html/2108.01073)). Applied to `-uncond`'s output at 512, it would add roughly half a minute and produce an image that is partly SD-Turbo's. It is the right tool when the result matters more than the pipeline, and it does not raise ADM.

The other cascades are worse. OpenAI's upsamplers are class-conditional only, there is no unconditional 64 or 128 base, and 64 → 256 lost 0.043-0.068 to direct 256 here ([refinement notes](../research_notes/Unconditional%20ADM%20CLIP%20guidance%20quality/disco_practice_cascades_and_evaluation.md)). The reverse cascade, Centipede's text-conditioned layout followed by Disco detail ([Centipede](https://github.com/Zalring/Centipede_Diffusion/blob/main/README.md)), is already possible with no code: `-bb sd-turbo -size 256`, then `-uncond -init` with `-skip`. Its composition would be SD-Turbo's, so it is unlikely to beat SD-Turbo alone.

**Disco's settings were never ablated.** Its defaults are code facts:

1. 250 DDIM steps, eta 0.8.
2. Guidance 5000, range 150, TV 0.
3. Cutouts: `cutn_batches` 4, `clamp_max` 0.05.
4. The secondary model on, and ViT-B/32, ViT-B/16 and RN50 together.
5. The 256 model rendered at 512x448, 3.5x its training pixels.

Disco's own README calls them "heavily outdated" ([disco.py](https://raw.githubusercontent.com/alembics/disco-diffusion/main/disco.py); [disco-diffusion README](https://github.com/alembics/disco-diffusion)). Two are bugs or no-ops for this case: `fuzzy_prompt` clamps the embedding to [0, 1], and `randomize_class` does nothing on an unconditional model. Reanalysing this repo's class-conditional 128 pilot gives the only numbers:

| Arm (ViT-B/16, ViT-H, SigLIP2) | Change |
|---|---|
| Disco's three-model ensemble | +0.008, +0.011, +0.030 |
| ViT-B/32+PE-Core-B-16 | +0.012, +0.022, +0.037 |
| TV 0 against 150 | −0.013, −0.019, −0.008 |

The secondary model tied ([Disco notes](../research_notes/Unconditional%20ADM%20CLIP%20guidance%20quality/disco_practice_cascades_and_evaluation.md)). The community's real fixes for faces, composition and adversarial texture were layout inits, fine-tuned priors and best-of-N, and of those only selection and ensembles have numbers.

## Ranked shortlist

The table ranks by expected gain on judges that did not guide, weighted by the strength of the evidence, since quality comes before speed. Gains are the change in mean CLIP similarity against the current `-uncond` default. They are pooled over ViT-H and SigLIP2, plus ViT-B/16 for arms that do not select with it, the same scale as the repo's pooled column. All gains are estimates unless marked measured. "Strong" means measured here on the closely related class-conditional pipeline, "moderate" means published on this model family or closely analogous settings, and "weak" means inference with indirect support.

| Rank | Change | 2070 SUPER, 8 GB | RTX 4070, 12 GB | Expected gain | Evidence | Sources |
|---|---|---|---|---|---|---|
| 1 | Re-baseline `-uncond` on the 12 held-out prompts with all judges, then `-uncond -clip ViT-B/32+PE-Core-B-16` (no code) | about 23 s, 5.3-5.5 GiB | 12-17 s, same | +0.010 to +0.020 | Strong on class-conditional: +0.0188 ± 0.015 at 128 over 4 seeds, +0.015 at 256 | ([scores](evidence/scores/adm128_b.txt)), ([scores](evidence/scores/sizes256.txt)) |
| 2 | FK steering, k = 4-8 ensemble-guided particles, tempered potential from `-best`'s pickers on x̂0, 4-6 resamples between t ≈ 700 and 300; progressive pruning 8 → 4 → 2 as the cheap form | 80-180 s, 5.6-6.5 GiB | 40-135 s; 2 particles per batch | +0.015 to +0.035 over an unselected sample | Moderate: best-of-4 measured +0.015 / +0.025 here; FK beats best-of-4 on SDXL | ([README](../README.md)), ([FK](https://arxiv.org/html/2501.06848)), ([DAS](https://arxiv.org/abs/2501.05803)), ([Zhang](https://arxiv.org/html/2505.23614v2)), ([PSP](https://arxiv.org/abs/2607.21591)) |
| 3 | Recurrence r = 2-4 over the 31 calls at t ≥ 300, against `-n 120/250` at equal time | 31-53 s, 5.3 GiB | 16-40 s | +0.005 to +0.015 | Moderate: TFG, UG, FreeDoM and Shen on unconditional ImageNet ADM; 1000-step sampler +0.015 here at 18x | ([TFG](https://arxiv.org/abs/2409.15761)), ([UG](https://arxiv.org/abs/2302.07121)), ([Shen](https://arxiv.org/abs/2403.12404)) |
| 4 | CLIP-embedding conditioning: zero-init projection into `emb` plus `emb_layers`, LAFITE noise, 10-20% dropout, ImageNet replay plus PD12M or synthetic pairs, then CFG with CLIP guidance | Inference about 28-30 s, 5.3-5.7 GiB | Train at batch about 8, about 11 GiB; 0.5-3 GPU-days to take effect, 1-2 weeks for CFG | 0 to +0.06, unmeasured; the only item aimed at the 0.06-0.09 non-kin gap | Weak for this checkpoint, strong for the mechanism | ([unCLIP](https://ar5iv.labs.arxiv.org/html/2204.06125)), ([LAFITE](https://arxiv.org/abs/2111.13792)), ([DEFT](https://arxiv.org/abs/2406.01781)), ([cc12m_1](https://github.com/crowsonkb/v-diffusion-pytorch)) |
| 5 | High-noise x0 rollout at σ > 0.9: hybrid (n = 2-3 without gradients, one-step Jacobian) first, checkpointed exact rollout if it helps | 32 s / 56 s, 5.3-5.5 GiB | 16-42 s | −0.005 to +0.015 | Weak to moderate: SAG and Xu et al. on other models | ([SAG](https://arxiv.org/html/2312.12030v1)), ([Xu](https://arxiv.org/html/2403.12063)) |
| 6 | DNO over x_T and SDE noises, 20-step sampler, step checkpointing, chi-norm penalty, a reward ensemble that is not also a judge, about 30 iterations (opt-in) | 4.5-5 min, 5.5-6 GiB | 2.5-3.5 min | −0.01 to +0.03, high hacking risk | Weak: SD results only; ReNO adds 0.01-0.02 on SD-Turbo here | ([DNO](https://arxiv.org/html/2405.18881)), ([DOODL](https://arxiv.org/abs/2303.13703)), ([scores](evidence/scores/gap_seed0.txt)) |
| 7 | Noise-aware CLIP student: LoRA band experts on the guide, trained to E[φ(x0) \| x_t], dot-product guidance | Inference about 12-15 s, 2.5-3 GiB | Train 1-3 days | −0.01 to +0.015; certain savings in memory and time | Weak to moderate: PPAP on this checkpoint, negative latent result | ([PPAP](https://ar5iv.labs.arxiv.org/html/2212.05973)) |
| 8 | Zero-training borrowed priors: GLIDE's noised CLIP at σ > 0.9; nousr prior image targets; cc12m_1_cfg's text direction combined with ADM's prediction at matched logSNR; PAG on `-uncond` as a realism arm | 25-40 s, +0.5-1.5 GiB | 13-30 s | Unknown | Weak; each is a one-day experiment | ([GLIDE](https://arxiv.org/abs/2112.10741)), ([nousr](https://huggingface.co/nousr/conditioned-prior)), ([cc12m_1_cfg](https://huggingface.co/multimodalart/crowsonkb-v-diffusion-cc12m-1-cfg)), ([PAG](https://arxiv.org/abs/2403.17377)) |
| 9 | Reward fine-tuning after rank 4: DRaFT-LV LoRA K = 1 with a reward ensemble that is not also a judge, or Adjoint Matching with a λ sweep | Inference unchanged | 1-5 GPU-days | +0.005 to +0.02 held out, costs diversity | Moderate in general, none on pixel ADM | ([DRaFT](https://arxiv.org/abs/2309.17400)), ([Adjoint Matching](https://arxiv.org/abs/2409.08861)) |
| 10 | SDEdit through SD-Turbo at strength 0.3-0.5 (leaves ADM) | about +30 s, run in sequence, under 5.3 GiB | about +15 s | +0.015 to +0.04 over the ADM image, capped below plain SD-Turbo | Strong (measured at 128) | ([scores](evidence/scores/gap_seed0.txt)), ([SDEdit](https://ar5iv.labs.arxiv.org/html/2108.01073)) |
| 11 | Enablers: LD3 timesteps for rank 6, and a secondary model retrained on ADM's own ODE endpoints to cut rollout and particle VRAM | None | LD3 under an hour; secondary model 2-3 days | None directly | Moderate (LD3) to weak | ([LD3](https://arxiv.org/abs/2405.15506)) |

The build order differs slightly from the ranking. Ranks 1-3 are an afternoon's code each and share one evaluation harness. Rank 4 should start on the 4070 in parallel with them, since it is the only item whose ceiling reaches the non-kin target and it takes days.

Several things should not be built:

1. RL and DPO, for the reasons above.
2. One-step distillation.
3. Jacobian-free guidance (MPGD) and robust towers as guides.
4. More magnitude tricks: DSG, `clamp_grad`, Polyak steps, gradient blur and schedule shapes. The cap dominates them.
5. Tree search.
6. Golden Noise and InitNO, which need training or text cross-attention.
7. Noise pre-optimisation without a U-Net gradient, since x̂0 at maximum noise is the dataset mean.
8. Per-prompt fine-tuning.
9. ADM upsampler cascades from 64.
10. Disco folklore: oversize renders, `fuzzy_prompt`, eta, perlin inits and symmetry.

## An evaluation protocol that can see a 0.01 change

The repo's saved scores set the noise floor. They come from the class-conditional 128 pipeline. For a paired difference between two settings, the seed-level SD per image is about 0.029 on ViT-B/16. The prompt-level SD of the effect is 0.011 on ViT-B/16 and 0.018 pooled ([evaluation notes](../research_notes/Unconditional%20ADM%20CLIP%20guidance%20quality/disco_practice_cascades_and_evaluation.md)). At 80% power and α = 0.05 that gives the following sample sizes, with my arithmetic for the 0.02 row:

| Effect to detect | Judge | Prompts × 1 seed | Prompts × 4 seeds |
|---|---|---|---|
| 0.02 | ViT-B/16 or pooled | about 20-25 | about 8-12 |
| 0.01 | ViT-B/16 | 78 | 27 |
| 0.01 | Pooled | 88 | 41 |
| 0.005 | ViT-B/16 | 311 | 108 |
| 0.005 | Pooled | 350 | 162 |

The 6-prompt baseline resolves nothing smaller than about 0.04, and the existing 12 prompts × 4 seeds about 0.015-0.02. Published work explains why seeds matter. On 1,000 COCO prompts with 3 seeds, CLIPScore's model ranking flipped between seeds, while VQAScore's stayed stable. A model with a "significant" 1.15-point lead won only 50% of prompts against 49%, so per-prompt dominance has to be reported alongside significance ([Re-Thinking the Automatic Evaluation of Image-Text Alignment](https://arxiv.org/html/2506.08480)). Paired metric differences in text-to-image studies are non-normal, which argues for Wilcoxon signed-rank tests ([arXiv 2401.16092](https://arxiv.org/pdf/2401.16092)).

The protocol has four stages. **First, re-baseline.** Run `-uncond` on the 12 held-out prompts and 3 seeds behind the SD-Turbo numbers, with ViT-B/16, ViT-H and SigLIP2. Run the class-conditional 256 pipeline beside it as a reference arm, which takes about 12 minutes per arm. Use the run to re-estimate the per-image variance on the unconditional model, since every figure above comes from class-conditional 128. Translate the target into SD-Turbo's scores on every judge: 0.350, 0.384 and 0.194.

**Second, freeze at least 100 prompts.** Take 50 from PartiPrompts, 25 compositional prompts in GenEval's style and 25 of the repo's own kind. Keep a separate tuning set of about 24. Tag each prompt by whether it maps closely to an ImageNet class, and include people and faces, the model card's known weakness. **Third, screen and then confirm.** Screen each arm at 1 seed on about 25 prompts, enough to detect 0.02, or 80-90 prompts for 0.01. Confirm survivors at 40 prompts × 4 seeds. Expensive arms (8x and up) screen on 25 prompts only. At 20 s per image, a 100-prompt arm at 1x takes about 35 minutes on the 2070 SUPER, and 40 × 4 about 55 minutes. **Fourth, match compute and hold judges apart.** Every arm is compared at equal wall-clock time against more steps (`-n`) and against best-of-N.

| Role | Models | Rule |
|---|---|---|
| Guide | ViT-B/32, PE-Core-B-16 | Never reports |
| Selector or FK verifier | `-best`'s ViT-B/16 + SigLIP2, or a rank-averaged pair outside the reporting set | Never reports for arms that select with it |
| Reporting judges | ViT-H-14, SigLIP2 and ViT-B/16 for arms without selection; VQAScore for prompt fit; HPSv3 for preference; the aesthetic predictor | Scored offline on the 4070 or the CPU |
| Guardrails | Saturation and clip%, an OCR rate for rendered prompt text, diversity across seeds (LPIPS or DreamSim), CMMD against a few hundred reference images, ImageNet-class prompts for forgetting after fine-tunes, a native-resolution view | Reported for every arm |

VQAScore had the best agreement with humans and is stable across seeds ([VQAScore](https://arxiv.org/pdf/2404.01291)). HPSv3 reports 76.9% pairwise accuracy on HPDv3 ([HPSv3](https://arxiv.org/abs/2508.03789)). CMMD stays stable at small sample sizes, where FID does not ([Rethinking FID](https://openaccess.thecvf.com/content/CVPR2024/papers/Jayasumana_Rethinking_FID_Towards_a_Better_Evaluation_Metric_for_Image_Generation_CVPR_2024_paper.pdf)). None of these preference and alignment models has been validated on adversarially textured CLIP-guided images, which is one more reason to keep the CLIP judges and a human check. Ensembles of rewards resist hacking only in part, and the typical failure is an image with artifacts that still scores high ([reward hacking](https://arxiv.org/html/2601.03468)). That is the case for the guardrail columns and the human check.

Report each comparison as a paired mean difference with a bootstrap or Wilcoxon interval, the per-prompt win rate, and the number of arms tested, with a Holm correction. A blind human A/B test confirms the final pick. It needs about 85 pairs to detect a 65/35 preference and 194 pairs for 60/40.

## Conclusion

The question's target, 0.35 ViT-B/16, measures the wrong thing for this pipeline. ViT-B/16 is kin to the guide and is one of the selection judges. It barely registered the ensemble's +0.025 on ViT-H at 256, and it inflated a ViT-B/32-only optimisation that ViT-H scored lower. Stacked, compute-matched search in the composition window (ensemble, FK steering with a verifier that is not also a judge, and recurrence over the 31 calls at t ≥ 300) can plausibly bring ViT-B/16 to about 0.33-0.35 at 4-9x the time and within 8 GB. It would close only about half the distance on judges unrelated to the guide. The unconditional model's particular weakness is that 27 of its 60 calls guide a grey x̂0 through a low-pass Jacobian, so every cheap idea worth testing changes what CLIP sees at σ > 0.9, not how hard the gradient pushes.

The remaining gap is the prior, and SD-Turbo shows it: with no CLIP at all it outscores every ADM arm. The cheapest structural fix is architectural, not reward-based. A zero-initialised CLIP projection into the U-Net's existing additive embedding slot starts the fine-tune from exactly the current model, trains without captions on the 12 GB card, and is the precondition for reward fine-tuning to mean anything on an unconditional model. Whether it reaches SD-Turbo depends on the training data more than the method, and only the evaluation protocol above, with 100 frozen prompts, separate guide, selection and reporting judges, and compute-matched controls, can tell a real gain from a kin judge being flattered.

# Guidance formulations for CLIP guidance of a diffusion model without text conditioning

> **Status, 2026-11-02:** These are the notes as written on 2026-11-01, kept as the record of what was known then. Later measurements overturned some of it, including its leading recommendations. [The report](../../reports/CLIP%20guided%20diffusion%20quality%20improvements.md) has the current results, and wins wherever the two disagree.

## Q1. Which guidance formulations exist, and what are their update equation, hyperparameters, cost, evidence and fit to 30 to 100 model calls?

### Takeaway
I found only two studies comparing CLIP guidance head to head on an unconditional pixel U-Net, and the FreeDoM and Shen ImageNet code spends roughly 380 to 560 model calls, so no method has measured evidence of beating this repo's update at 30 calls. The repo already uses FreeDoM's normalisation rule plus a cap, which points to additive changes (schedule, solver, more views, class or perturbation guidance) over a replacement update.

### Cited Findings

#### Repo baseline and notation
- Notation used below: x_t = α x_0 + σ ε, x̂0 = (x_t − σ ε_θ)/α, L is the guidance loss, g = −∇_{x_t} L through the U-Net, h is the logSNR step — [cgd/sampling.py](/home/sam/GitHub/clip-guided-diffusion/cgd/sampling.py)
- Repo update: ratio = RMS(σ g)/RMS(ε); g ← g·min(1, rate/ratio); guided_x0 = clamp(x̂0 + (σ²/α) g, ±1.2); x_next = (σ_next/σ) e^{−ηh} x + w·guided_x0 + 0.5 w (h/h_prev)(x̂0 − x̂0_prev) + σ_next √(1 − e^{−2ηh}) z, with w = α_next(1 − e^{−h−ηh}). Defaults are 30 steps (60 unconditional), η 0.5, rate 0.08, window 0.1 to 0.8 of the schedule (0.1 to 1.0 unconditional) — [cgd/sampling.py](/home/sam/GitHub/clip-guided-diffusion/cgd/sampling.py)
- The multistep correction uses the unguided x̂0 only, because extrapolating guidance destabilised the solver — [cgd/sampling.py](/home/sam/GitHub/clip-guided-diffusion/cgd/sampling.py)
- `guidance_loss` feeds CLIP x_in = σ x̂0 + (1 − σ) x_t, with 4 cutouts of 70 to 100% of the frame, spherical distance, a range loss on x̂0 and TV on x_in, at cgs 1000 and tvs 150 — [cgd/cgd.py](/home/sam/GitHub/clip-guided-diffusion/cgd/cgd.py)
- Measured baseline on an RTX 2070 Super: 256px in 11.4 s and 5.3 GiB, 512px in 22.4 s and 4.1 GiB, 60 call unconditional 256px in 20 s. ViT-B/16 judged score ties the 1000 step sampler at 128px (0.292 vs 0.292, 12 prompts), and unconditional 256px is 0.316 vs 0.331 (CI −0.040 to +0.014, 6 prompts) — [README.md](/home/sam/GitHub/clip-guided-diffusion/README.md)

#### Tested with CLIP or text on a pixel or latent diffusion model
- **Shen et al., "Understanding Training-free Diffusion Guidance" (NeurIPS 2024).** Random augmentation: g = (1/|T|) Σ_{T∈𝒯} ∇_{x_t} ℓ(f(T(x̂0)), y), then x_{t−1} = DDIM(x_t) − η g. Polyak step: x_{t−1} = DDIM(x_t) − η (‖ε_θ‖₂/‖g‖₂²) g, where ‖ε_θ‖ estimates the gap to the optimum and balances the diffusion and guidance terms — [Source](https://arxiv.org/abs/2403.12404)
  - CLIP scores on unconditional ImageNet (CLIP-B/16 guides, CLIP-L/14 judges, DDIM 100 steps): LGD-MC 24.3, Universal Guidance 25.7, FreeDoM 25.9, MPGD-Z 25.1, Shen 27.7 — [Source](https://arxiv.org/abs/2403.12404)
  - CelebA-HQ text (distance/FID): UG 12.08/44.27, LGD-MC 11.84/41.74, FreeDoM 11.17/46.13, MPGD-Z 10.78/42.45, Shen 10.72/41.25 — [Source](https://arxiv.org/abs/2403.12404)
  - Augmentation cost on an A100 (ImageNet diffusion plus CLIP-B/16): 0.541 s/step without, 0.544 at |T|=1, 0.571 at 10, 0.592 at 20, 0.625 at 30. The table row is labelled "CLIP Score" but the caption says inference time, so treat the label as a typo — [Source](https://arxiv.org/abs/2403.12404)
  - Published ImageNet settings: 100 steps, `--rho_scale 2`, 20 DiffAugment replicates (policy color, translation, resize, cutout) of a 224px bicubic full frame, and 10 repeats for 800 ≥ t ≥ 300 — [denoising.py](https://github.com/BIGKnight/Understanding-Training-free-Diffusion-Guidance/blob/master/CelebA%20and%20ImageNet/functions/denoising.py), [base_clip.py](https://github.com/BIGKnight/Understanding-Training-free-Diffusion-Guidance/blob/master/CelebA%20and%20ImageNet/functions/clip/base_clip.py), [README](https://github.com/BIGKnight/Understanding-Training-free-Diffusion-Guidance/blob/master/CelebA%20and%20ImageNet/README.md)
  - Conflict: the paper presents Polyak as the method, but that code applies the Polyak factor only when `improved and domain == "face"`. The ImageNet branch uses FreeDoM's RMS rule with augmentation, so the 27.7 may not involve Polyak at all — [denoising.py](https://github.com/BIGKnight/Understanding-Training-free-Diffusion-Guidance/blob/master/CelebA%20and%20ImageNet/functions/denoising.py)
  - Stated limits: still more calls than CFG, and CLIP cannot do positional prompts — [Source](https://arxiv.org/abs/2403.12404)
- **ADMMDiff (CVPR 2025).** Alternates a reverse diffusion step and K_t gradient steps on z with coupling ρ; x̂_{t−1} = z_t − ν_t/ρ. CLIP score (L/14 judges, B/16 guides): DPS 24.5, LGD-MC 24.3, FreeDoM 25.9, MPGD 25.1, ADMMDiff 26.85 — [Source](https://arxiv.org/abs/2411.12773)
  - Conflict: that table is captioned unconditional CelebA-HQ, yet its LGD-MC, FreeDoM and MPGD values equal Shen's ImageNet row exactly, so the dataset label or the baselines need checking before use — [ADMMDiff](https://arxiv.org/abs/2411.12773), [Shen](https://arxiv.org/abs/2403.12404)
- **FreeDoM (ICCV 2023).** x_{t−1} = m_t − ρ_t ∇_{x_t} D(c, x̂0), with time travel j = 1 and r_t repeats; sampling stages are chaotic (1000 to 800), semantic (800 to 500, critical) and refinement (500 to 1) — [Source](https://arxiv.org/abs/2303.09833)
  - Exact step size in the official CLIP sampler: ρ = 0.02 · RMS(ε)·(σ/α)·c1 / RMS(g), where c1 = √ᾱ_{t−1}(1 − α_t)/(1 − ᾱ_t) is the DDPM x0 coefficient. ImageNet repeats 10 times for 800 ≥ t ≥ 500 and faces do not repeat. The loss is the L2 norm of the CLIP residual — [denoising.py](https://github.com/vvictoryuki/FreeDoM/blob/main/Face-GD/functions/denoising.py)
  - Conflict: Shen reports FreeDoM resampling from 800 to 300, while the official code uses 800 to 500. Shen's own fork uses 800 to 300 — [Shen](https://arxiv.org/abs/2403.12404), [denoising.py](https://github.com/vvictoryuki/FreeDoM/blob/main/Face-GD/functions/denoising.py)
- **Universal Guidance.** Forward: ε̂ = ε + s(t) ∇_{z_t} ℓ(c, f(ẑ0)). Backward: Δz0 = argmin_Δ ℓ(c, f(ẑ0 + Δ)) over m steps, then ε̃ = ε − √(α_t/(1−α_t)) Δz0. Recurrence: z'_t = √(α_t/α_{t−1}) z_{t−1} + √(1 − α_t/α_{t−1}) ε'. CLIP settings: s(t) = 10√(1 − α_t), k = 8, m = 0. The paper gives no quantitative text evaluation — [Source](https://arxiv.org/abs/2302.07121)
- **MPGD (ICLR 2024).** x̂0 ← x̂0 − c_t ∇_{x̂0} L(x̂0), with no U-Net backward, and the AE and Z variants add a manifold projection through an autoencoder — [Source](https://arxiv.org/abs/2311.16424)
  - CelebA FaceID, 50 steps (KID, FaceID, time): DDIM .0442/1.3914/3.41 s, FreeDoM .0452/.5690/10.65 s, LGD-MC .0448/.6783/14.64 s, MPGD .0473/.5163/5.82 s, MPGD-AE .0467/.5309/7.78 s, MPGD-Z .0445/.5791/6.93 s — [Source](https://arxiv.org/abs/2311.16424)
  - Stable Diffusion style guidance (Style, CLIP, time, VRAM): DDIM 761/31.61/13.89 s/10.8 GB, FreeDoM 498.8/30.14/26.5 s/17.3 GB, LGD-MC 404/21.16/37.43 s/31.65 GB, MPGD-LDM 441/26.61/19.83 s/15.53 GB — [Source](https://arxiv.org/abs/2311.16424)
- **DSG (ICML 2024).** d* = −√n σ_t g/‖g‖, d_sample = σ_t ε_t, d_m = d_sample + g_r(d* − d_sample), x_{t−1} = μ + r d_m/‖d_m‖ with r ≈ √n σ_t. g_r is 0.05 to 0.2, 0.1 for text style. FaceID at g_r 0.05 (FaceID/FID/KID): FreeDoM .545/39.61/.0165, LGD .533/39.13/.0170, FreeDoM+DSG .371/34.29/.0092. Inverse problem PSNR for DPS vs DPS+DSG: 12.08 to 27.53 dB at 20 steps, 22.74 to 30.67 at 50, 27.73 to 30.95 at 100 — [Source](https://arxiv.org/abs/2402.03201)
- **LGD (ICML 2023).** Replaces the delta posterior with Monte Carlo over N(x̂0, r_t²I): log (1/n) Σ exp(−ℓ(x_i)). Super-resolution FID at 100 steps: n=1 5.42, n=10 5.32, n=100 5.21. Text and label results are qualitative — [Source](https://proceedings.mlr.press/v202/song23k/song23k.pdf)
- **Unified training-free guidance, TFG.** Δ_t = ρ_t ∇_{x_t} log f̃(x̂0), Δ_0 ← Δ_0 + μ_t ∇_{x̂0} log f̃(x̂0 + Δ_0) for N_iter steps, x_{t−1} = Sample(x_t, x̂0, t) + Δ_t/√α_t + √ᾱ_{t−1} Δ_0, then renoise for N_recur. MPGD, LGD, UG, DPS and FreeDoM are all special cases. ImageNet label guidance at T=100, η=1, N_recur 1, N_iter 4 (accuracy %/FID): DPS 38.8/193, LGD 11.5/210, FreeDoM 19.7/200, MPGD 6.80/239, UGD 25.5/205, TFG 40.9/176 — [Source](https://arxiv.org/abs/2409.15761)
- **EluCD (ICLR 2024).** Off-the-shelf classifier guidance with softplus (β=3 improves FID 7.17 to 6.61), joint/marginal temperatures (τ2 1.0 to 0.5 improves 6.20 to 5.27) and a sine schedule γ_t = σ_t + γσ_T sin(πt/T) (γ 0 to 0.3 improves 5.57 to 5.24), on ADM ImageNet 128 with 250 DDPM steps. 50k FID: baseline 5.91, fine-tuned classifier 2.97, CFG 2.43, ResNet-50 2.36, ResNet-101 2.19. The authors report CLIP guidance of a conditional ADM with ViT-L at 336px running about 5 times faster than Universal Guidance, and "significantly higher" CLIP scores from the sine schedule, shown only as a plot over a handful of prompts — [Source](https://arxiv.org/abs/2310.11311)
- **DOODL (ICCV 2023).** Optimises the initial latent by backpropagating through EDICT invertible sampling. Gradient clipped to ±1e-3, momentum 0.9, noise N(0, 1e-4). CLIP score +3.1% and +2.6% over one-step classifier guidance on DrawBench DALL-E and Reddit prompts, and PASR +11.3% over Stable Diffusion and +6% over classifier guidance, across 9 seeds, in SD latent space — [Source](https://arxiv.org/abs/2303.13703)
- **DPS as a CLIP baseline.** x_{i−1} = x'_{i−1} − ζ_i ∇_{x_i}‖y − A(x̂0)‖², ζ_i = ζ'/‖y − A(x̂0)‖ — [Source](https://arxiv.org/abs/2209.14687). It scores 24.5 on the ADMMDiff text table and 38.8/193 on TFG's ImageNet label task — [ADMMDiff](https://arxiv.org/abs/2411.12773), [TFG](https://arxiv.org/abs/2409.15761)
- **ADM-G.** ε̂ = ε_θ − √(1 − ᾱ_t) s ∇_{x_t} log p_φ(y|x_t), with a noise-aware classifier. FID on ImageNet 256: conditional 10.94 to 4.59 at s=1, unconditional 26.21 to 12.00 at s=10 (and worse, 33.03, at s=1). With only 25 DDIM steps: 5.98 at 128px, 5.44 at 256px, 8.41 at 512px — [Source](https://arxiv.org/abs/2105.05233)
  - Classifiers for 64, 128, 256 and 512 are public, with README scales 1.0, 0.5, 1.0 and 4.0, and 10.0 for unconditional 256 — [guided-diffusion README](https://github.com/openai/guided-diffusion/blob/main/README.md)
- **GLIDE noised CLIP.** A CLIP trained on noised images guided the diffusion model and needed no augmentation or perceptual tricks, but human evaluators preferred classifier-free guidance — [Source](https://arxiv.org/abs/2112.10741)
- **CLIP-JEM (TMLR 2025).** An energy-trained, adversarially robust CLIP that the authors say turns unconditional ADM into a text-guided model in 25 diffusion steps. It requires training a new CLIP — [Source](https://arxiv.org/abs/2408.17046)
- **Blended Diffusion.** Extending augmentations reduce adversarial CLIP results — [Source](https://arxiv.org/abs/2111.14818). **SDG** injects text or image semantics into sampling as guidance — [Source](https://arxiv.org/abs/2112.05744). **PixelAsParam** projects away conflicts between the denoising, divergence and classifier gradients on ADM and IDDPM, with a claimed improvement for GLIDE noised CLIP (numbers not extracted) — [Source](https://proceedings.mlr.press/v202/dinh23a.html)

#### Arbitrary-loss guidance tested only outside CLIP
- **FlowChef.** On rectified flow: x_t ← Optimize(x_t, L(x_t + t·v)), then x_{t−1} ← x_t + dt·v, with N=1 sufficient and no Jacobian. ImageNet 64 at batch 32 (NFE, CG scale, FID, VRAM GB, time s): DDIM 50/-/5.39/3.67/14.22, MPGD 50/1/4.24/6.56/25.01, MPGD 50/10/5.46/6.56/25.01, FlowChef-style skip on IDDPM 50/1/19.28/6.56/24.95; RF++ 15/-/4.29/3.36/2.75, RF++ backprop 15/5/2.77/17.98/12.79, RF++ skip 15/50/3.13/6.64/5.85 — [Source](https://arxiv.org/abs/2412.00100)
- **DiffStateGrad (ICLR 2025).** Projects the guidance gradient onto the top singular vectors of the current state: U,S,V ← SVD(Z_t), r ← smallest rank retaining τ of the energy, τ = 0.99, every step for DPS and DAPS. ImageNet inpainting (LPIPS/SSIM/PSNR, box): DAPS .217/.762/20.90 to .191/.801/21.07. The choice of subspace matters, not the low rank alone — [Source](https://arxiv.org/abs/2410.03463)
- **PGDiff (NeurIPS 2023).** s_norm = (‖x' − μ‖₂/‖g‖₂)·s, with several gradient steps in chosen ranges of t. Restoration only — [Source](https://arxiv.org/abs/2309.10810)
- **Rethinking DPS.** Argues DPS behaves like MAP estimation and proposes multi-step gradients with a projection — [Source](https://arxiv.org/abs/2501.18913). Exact posterior sampling is intractable in general — [Source](https://arxiv.org/abs/2402.12727)

#### Noise or trajectory optimisation
- **D-Flow.** L-BFGS on the initial noise through the whole ODE: 5 to 15 minutes for ImageNet 128 and 30 to 40 minutes for text-to-image — [Source](https://arxiv.org/abs/2402.14017)
- **DMPlug** optimises the seed through a sampler of about 3 steps — [Source](https://arxiv.org/abs/2405.16749)

#### Editing methods
- **SDEdit**: noise an image to t0 and denoise — [Source](https://arxiv.org/abs/2108.01073). The repo's `-init -skip` already does this — [cgd/sampling.py](/home/sam/GitHub/clip-guided-diffusion/cgd/sampling.py)
- **GradOP**: optimise an image, then apply an SDEdit-like denoise — [Source](https://arxiv.org/abs/2211.17084)
- **DiffusionCLIP**: fine-tunes the model with a directional CLIP loss, at t0 300 to 600 with 40 inversion and 6 generation steps — [Source](https://arxiv.org/abs/2110.02711)
- **Asyrp**: x_{t−1} = √ᾱ_{t−1} P_t(ε(x_t|Δh_t)) + D_t(ε(x_t)) + σ z. It changes only the predicted-x0 term, because a shift of ε applied equally to P_t and D_t nearly cancels, and it learns Δh in the frozen U-Net's bottleneck — [Source](https://arxiv.org/abs/2210.10960)
- **Self-guidance**: energy on cross-attention maps, in Stable Diffusion — [Source](https://arxiv.org/abs/2306.00986)

#### Inverse-problem methods (inverse-only evidence)
- **MCG**: a DPS-like gradient plus a data-consistency projection — [Source](https://arxiv.org/abs/2206.00941)
- **ΠGDM**: p(y|x_t) ≈ N(y | A D_t(x_t), σ²I + r_t² AAᵀ), r_t² = σ_t²/(1 + σ_t²), which relies on a linear A. Taken from the OPC paper's restatement — [Source](https://arxiv.org/abs/2402.02149)
- **DDNM**: x̂0 ← A†y + (I − A†A) x̂0, plus time travel — [Source](https://arxiv.org/abs/2212.00490)
- **RED-diff**: Adam on μ with data fit plus λ_t (ε_θ(x_t) − ε)ᵀ μ, where the score is not differentiated, λ_t = λ/SNR, 1000 steps, lr 0.1, and descending t beats random t — [Source](https://arxiv.org/abs/2305.04391)
- **ReSample**: hard x0 optimisation about every 10 steps, then stochastic resampling, in latent space; only +26% memory against +175% for DPS — [Source](https://arxiv.org/abs/2307.08123)
- **DAPS**: alternates x0 sampling by Langevin and a jump to the next noise level; DAPS-1K is 4 ODE NFE by 250 steps with 100 Langevin steps — [Source](https://arxiv.org/abs/2407.01521)
- **PnP-DM and SGS**: alternate a likelihood step and a denoising prior step with coupling ρ — [PnP-DM](https://arxiv.org/abs/2405.18782), [PnP-SGS](https://arxiv.org/abs/2304.11134). InverseBench finds multi-query methods such as DAPS and PnP-DM win when the forward model is closed form and do poorly on PDE forward models — [Source](https://arxiv.org/abs/2503.11043)
- **DiffPIR** needs 100 NFE against 1000 for DPS — [Source](https://arxiv.org/abs/2305.08995). **DDS** runs conjugate gradient on the Tweedie estimate at 20 to 50 NFE — [Source](https://arxiv.org/abs/2303.05754). **CM4IR** restores with 4 NFE on consistency models — [Source](https://arxiv.org/abs/2412.20596)
- **Tweedie second order**: TMPD costs about 1.5 times ΠGDM — [Source](https://arxiv.org/abs/2310.06721); OPC optimises the posterior covariance — [Source](https://arxiv.org/abs/2402.02149); STSL estimates the Hessian trace in O(1) — [Source](https://arxiv.org/abs/2312.00852)
- **Guidance flow matching** — [Source](https://arxiv.org/abs/2502.02150) and **FlowDPS** — [Source](https://arxiv.org/abs/2503.08136) carry these ideas to flows.

### Inferences

#### Cost and fit against the repo
Units: F is a U-Net forward, B a U-Net backward, C one CLIP forward and backward over the 4 cutouts. On the linear schedule the repo guides 14 of 30 steps and 39 of 60 unconditional steps (computed from `NoiseLevels`), so the baseline is 30F + 14B + 14C.

| Method | Cost per guided step, vs F+B+C | Calls on the 30 step grid | Fits 30 to 100? | Main difference from the repo |
|---|---|---|---|---|
| FreeDoM rule, no repeats | same | 30 | Yes | Always normalised rather than capped |
| FreeDoM repeats, t 800 to 500 | + (r−1)(F+B+C) | 36, 42, 54, 84 for r=2,3,5,10 | r ≤ 5 | Needs multistep history handling (Q5) |
| Shen augmentation | C × views/4 | 30 | Yes | Full-frame DiffAugment rather than cutouts |
| Shen Polyak | same | 30 | Yes | Step is inverse in ‖g‖, so it depends on loss scale and needs its own ceiling |
| ADMMDiff | K_t inner steps | unknown | Unknown | K_t values not extracted |
| UG forward | same | 30 | Yes | Already what the repo does |
| UG backward, m steps | + mC | 30 | Yes | CLIP config used m=0, so untested |
| UG recurrence | × k | 128 at k=8, 100 at k=6 | k ≤ 6 | Published k=8 does not fit |
| MPGD, no projection | −B | 30 | Yes | Risk on hard semantic losses (Q2) |
| MPGD-AE, MPGD-Z | −B + autoencoder | 30 | By cost | Adds an autoencoder to a pixel pipeline |
| DSG | same | 30 | Yes | Near no-op at this rate (Q4) |
| LGD | C × n | 30 | Yes | Already covered by random cutouts in spirit |
| TFG | N_recur × (F+B+C) + N_iter C | 44 at N_recur 2 | Yes | Inner steps need Jacobian-free x̂0 gradient |
| EluCD schedule | same | 30 | Yes | Measured on classifiers, plotted on CLIP |
| ADM-G classifier | + classifier forward and backward, no B | 30 | Yes | New checkpoint; noise-aware, so it can run at all steps |
| DPS step size | same | 30 | Poor | 12.08 dB at 20 steps in DSG's inverse test |
| DiffStateGrad | + SVD per channel | 30 | Yes | Only inverse evidence |
| PGDiff norm | same | 30 | Yes | Same family as the repo cap |
| RED-diff | F + C per iteration, no B | 30 to 100 by cost | Unproven | Published at 1000 steps with Adam |
| FlowChef skip | −B | 30 | Risky | FID 19.28 on IDDPM |
| DOODL, D-Flow, DMPlug | chain backward per iteration | hundreds or more | No | Whole chain per optimiser step |
| ReSample, DAPS, PnP-DM | ≥ 250 steps or nested denoising | hundreds or more | No | Multi-query |
| MCG, DDNM, ΠGDM, DiffPIR, DDS, TMPD, OPC | n/a | n/a | No | Need a linear A or closed-form likelihood |
| SDEdit | same | fewer | Already present | |
| GradOP | C only | n/a | Only with an init image | Text-only runs have nothing to start from |
| DiffusionCLIP, Asyrp | training per prompt or attribute | n/a | No | Training step |
| Self-guidance | n/a | n/a | No | ADM has no cross-attention |

#### Determinism
- Closed-form changes (normalisers, schedules, projections, solver coefficients) add no randomness.
- Anything that draws noise or transforms (re-noising, augmentation, extra cutouts, Monte Carlo) must use the seeded generators in a fixed order, since changing the draw order changes every image a seed makes.
- New CLIP passes must stay on the plain attention path and avoid `tf.adaptive_avg_pool2d` and fused kernels, for the reason recorded in the project notes — [CLAUDE.md](/home/sam/GitHub/clip-guided-diffusion/CLAUDE.md)
- Dropping the U-Net backward removes one source of nondeterministic backward kernels and of fp16 overflow. A PAG or SAG branch runs without gradients, so it adds neither.
- SVD for DiffStateGrad should run on the CPU or be tested for run-to-run equality first, since I have no evidence on its CUDA behaviour.

#### Ranked candidates for this repo
None of these has been measured at 30 calls in this repo. They are ordered by evidence on the same or similar checkpoints, weighed against cost.
1. Sine or increasing guidance schedule inside the existing window. Free, and it has both FID and plotted CLIP evidence (Q6).
2. ER-SDE-Solver as the stochastic solver. Free in calls, with direct classifier-guided ADM 256 numbers (Q7).
3. ADM's noise-aware classifier on the chosen class. It has the only low-step evidence on these checkpoints (5.44 at 25 steps) and may help the unconditional path most (Q6).
4. More augmented views, following Shen's 20 replicates at about +10% step time.
5. PAG or SAG on part of the schedule, with a +14F style budget (Q6).
6. Jacobian-free x̂0 gradient plus 2 to 4 inner steps, on the class-conditional path first (Q2, Q3).
7. Time travel at r = 2 or 3 for t 800 to 500, for 36 to 42 calls (Q5).
8. A real cond and uncond checkpoint pair at 256 and 512 (Q6).

Judge each with the repo's held-out prompts and ViT-B/16 score over several seeds, since CLIP score alone can reward adversarial detail.

### Gaps
- No study found compares these methods with CLIP at 30 calls on a pixel U-Net. Everything measured sits at 50 steps or more, and the published ImageNet CLIP runs cost far more: FreeDoM's code is about 379 calls and Shen's about 559 on a uniform 100 step grid (my count from their repeat ranges).
- Whether the CLIP-specific claims of EluCD hold numerically is unknown, since they appear only as a plot over a few prompts.
- "CDM" in the brief is ambiguous (cascaded, composable or conditional diffusion). I covered PnP and splitting methods and could not identify a single intended method.
- ControlNet-free editing guides beyond Self-guidance were not read, and FlowChef's official repo is unverified.
- The ΠGDM paper itself could not be retrieved (OpenReview returned HTML), so its equation comes from the OPC restatement. Its budget and hyperparameters are not recorded here.
- I did not find whether CLIP-JEM's weights are public.
- The time split of a guided step into F, B and C on this repo's GPU is not measured. It can be read off with a short timing run.

## Q2. Should the gradient be taken at x̂0 or at x_t, and does skipping the U-Net Jacobian improve quality or only speed?

### Takeaway
Skipping the Jacobian reliably saves time and memory, but its effect on quality depends on the task: it holds on easy pixel-near losses and collapses on hard semantic ones, with MPGD at 6.80% accuracy against DPS at 38.8% on ImageNet labels. The form matters too: moving only x̂0 improved FID on IDDPM while moving x_t with a stale ε made it much worse.

### Cited Findings
- Theory: gradient through x_t equals mean guidance reweighted by the posterior covariance, up to a constant, so the Jacobian carries second-order information; positively correlated components reinforce and negatively correlated ones cancel — [Source](https://arxiv.org/abs/2409.15761)
- Hardest measured case: ImageNet label guidance has MPGD last at 6.80%/239 against DPS 38.8/193 and FreeDoM 19.7/200 — [Source](https://arxiv.org/abs/2409.15761)
- Quality kept or better on easier tasks: FaceID .5163 for MPGD vs .5690 for FreeDoM at 5.82 s vs 10.65 s; KID is slightly worse for plain MPGD at .0473 vs .0452 — [Source](https://arxiv.org/abs/2311.16424)
- Text results for MPGD-Z sit near FreeDoM (ImageNet 25.1 vs 25.9, CelebA distance 10.78 vs 11.17), but that variant needs an autoencoder projection — [Source](https://arxiv.org/abs/2403.12404)
- MPGD without projection "suffers from the shortcut" on the motion task, per Shen's Appendix B.2 — [Source](https://arxiv.org/abs/2403.12404)
- On IDDPM at 50 NFE, baseline FID 5.39: MPGD at scale 1 reaches 4.24, at scale 10 reaches 5.46, and x_t-moving skip gradient reaches 19.28, all at the same VRAM and time — [Source](https://arxiv.org/abs/2412.00100)
- On rectified flow at 15 NFE, baseline FID 4.29: full backprop 2.77 at 17.98 GB and 12.79 s, skip 3.13 at 6.64 GB and 5.85 s, i.e. backprop costs 2.7 times the VRAM and 2.2 times the time for 0.36 better FID — [Source](https://arxiv.org/abs/2412.00100)
- FlowChef's error term is ε(t) = t·J_uθᵀ∇L, small when t → 0 or the Jacobian varies slowly, which it argues holds for straight flows and not curved diffusion trajectories — [Source](https://arxiv.org/abs/2412.00100)
- SDS drops the U-Net Jacobian — [Source](https://arxiv.org/abs/2209.14988). RED-diff does not differentiate the score either, and calls this more memory efficient — [Source](https://arxiv.org/abs/2305.04391)
- Memory scale of the through-U-Net path: DPS adds 175% over the bare model — [Source](https://arxiv.org/abs/2307.08123)
- Different question, not to be conflated: EluCD finds a classifier better calibrated when fed x̂0 than x_t (ECE 0.28 vs 0.36). That concerns the classifier's input, not whether the U-Net Jacobian is kept — [Source](https://arxiv.org/abs/2310.11311)

### Inferences
- The swap in this repo: detach ε, so ∂x̂0/∂x_t = I/α, and with `fac` = σ the gradient becomes g = −(σ/α + 1 − σ)∇_{x_in} L. Because the cap renormalises, only the direction changes when the cap is active. It removes 14B.
- The repo's update is already MPGD's better form in one respect: guidance enters through x̂0 and x_next is written with x, so ε is never left stale. It is not MPGD's weighting: writing the update with x makes it equal to DDIM with a corrected ε̂, so Δ is scaled by w, which shrinks with h. Asyrp's cancellation result explains why that shrinks with step size. MPGD gives x̂0 the full √ᾱ_{t−1} weight.
- FlowChef's 19.28 should be read as a warning about moving x_t while keeping ε. It does not condemn Jacobian-free guidance on ADM, since MPGD on the same model improved FID.
- Class-conditional runs resemble the easier tasks, since the U-Net brings the subject. Unconditional runs resemble TFG's hard case, since CLIP must set the composition, so try Jacobian-free there last.
- The Jacobian likely acts as a learned low-pass on CLIP's gradient at high noise. Without it, raw high-frequency gradient reaches x̂0, so more views or a low-pass would be needed. This is my reasoning, not a measurement.
- If backward costs here resemble FlowChef's ratios, the time saved could fund about double the calls at the same wall-clock. The repo's docstring calls the U-Net backward the most expensive part of guidance — [cgd/sampling.py](/home/sam/GitHub/clip-guided-diffusion/cgd/sampling.py)

### Gaps
- No measurement exists of CLIP guidance on ADM with the Jacobian against without it on the same seeds, which is the deciding experiment.
- MPGD's Style and CLIP columns are read as lower-is-better from the direction in which guidance moves them; I did not confirm that against the paper's table caption.
- Partial schemes, such as a full backward only every few steps, turned up no evidence.
- FlowChef's "Ours w/ skip grad" row on IDDPM is read as x_t-moving DDIM from its surrounding text; the table does not name the backbone directly.

## Q3. Do multiple inner optimisation steps per model call help?

### Takeaway
Inner steps on x̂0 cost only CLIP and TFG's best ImageNet settings use four of them, but Universal Guidance used none for CLIP and MPGD saw artifacts at 7 to 15. Two to four is the range to test, and they should refine direction under the existing cap instead of adding magnitude.

### Cited Findings
- TFG's Table 3 setting is N_iter 4 with N_recur 1, and it reports inner iterations as cheaper than recurrence — [Source](https://arxiv.org/abs/2409.15761)
- Universal Guidance's backward step runs m optimiser steps on Δz0 with no model call, but its CLIP configuration sets m = 0 and relies on k = 8 — [Source](https://arxiv.org/abs/2302.07121)
- MPGD: harder tasks benefit from multi-step optimisation, 7 to 15 steps gave background artifacts, and conjugate gradient is possible — [Source](https://arxiv.org/abs/2311.16424)
- ADMMDiff runs K_t gradient steps per reverse step with ρ coupling, with a convergence analysis assuming L-smooth terms — [Source](https://arxiv.org/abs/2411.12773)
- PGDiff applies several gradient steps within selected t ranges for restoration — [Source](https://arxiv.org/abs/2309.10810)
- ReSample's hard data-consistency step and DAPS's 100 Langevin steps are the extreme end, published with hundreds of model calls — [ReSample](https://arxiv.org/abs/2307.08123), [DAPS](https://arxiv.org/abs/2407.01521)
- Rethinking DPS: multi-step gradient plus projection on inverse problems — [Source](https://arxiv.org/abs/2501.18913)
- LGD's n samples have small effect at equal step count: 5.42 to 5.21 FID going from 1 to 100 — [Source](https://proceedings.mlr.press/v202/song23k/song23k.pdf). TFG's analysis also says the number of Monte Carlo samples barely matters while the injected noise does — [Source](https://arxiv.org/abs/2409.15761)

### Inferences
- Through the U-Net an inner step costs F+B+C, the same as a whole extra call, so it can only compete if the x̂0 step is Jacobian-free at C each.
- For this repo: after each model call, run K steps of Δ ← Δ − μ∇_{x̂0+Δ} L with fresh seeded cutouts, then apply the existing cap to the total Δ. At K=4 that is 4×C at each of 14 steps, about 16 cutouts per step in total.
- The cap fixes the total shift, so extra steps change only direction. That is close to averaging over more views, so the comparison to beat is simply 16 cutouts at one step, which needs no new code path.
- Under TFG's reading, inner steps on x̂0 only reach the mean; only recurrence or variance guidance reaches second-order information.
- Keeping the U-Net gradient at the first inner step and using x̂0 gradients for the rest would match TFG's mix of ρ and μ terms, at no more B than today.

### Gaps
- No CLIP result varies m or N_iter, so the bracket of 2 to 4 is transferred from label and FaceID tasks.
- ADMMDiff's K_t values were not extracted.

## Q4. Which normalisation or rescaling should guidance use?

### Takeaway
The repo's cap is already FreeDoM's RMS rule with a ceiling, at 2 to 4 times its rate, and DSG's sphere changes the step radius by only about 0.3% here, so magnitude is not the open question. The untested levers are direction filters: APG's projection against x̂0, CFG rescale, and subspace or low-pass filtering of the gradient.

### Cited Findings
- FreeDoM's constant is 0.02 and Shen's ImageNet command gives 0.04 in the same formula — [FreeDoM](https://github.com/vvictoryuki/FreeDoM/blob/main/Face-GD/functions/denoising.py), [Shen](https://github.com/BIGKnight/Understanding-Training-free-Diffusion-Guidance/blob/master/CelebA%20and%20ImageNet/functions/denoising.py)
- Shen's Polyak form has scale 0.001 for faces in the published command, and a guard sets ρ to 0 when ‖g‖ < 1e-6 — [denoising.py](https://github.com/BIGKnight/Understanding-Training-free-Diffusion-Guidance/blob/master/CelebA%20and%20ImageNet/functions/denoising.py), [README](https://github.com/BIGKnight/Understanding-Training-free-Diffusion-Guidance/blob/master/CelebA%20and%20ImageNet/README.md)
- EluCD's EDM variant uses x_{i−1} = x_i + (t_i − t_{i−1}) d_i + γ_i g/‖g‖₂ — [Source](https://arxiv.org/abs/2310.11311)
- DPS divides by the residual norm; PGDiff scales by ‖x' − μ‖/‖g‖ — [DPS](https://arxiv.org/abs/2209.14687), [PGDiff](https://arxiv.org/abs/2309.10810)
- DSG reports large gains exactly where unnormalised steps are huge, at 20 to 50 steps — [Source](https://arxiv.org/abs/2402.03201)
- APG: decomposes ΔD into parallel and orthogonal parts relative to the denoised prediction, ΔD ← ΔD⊥ + ηΔD∥, rescales with ΔD ← ΔD·min(1, r/‖ΔD‖), and adds reverse momentum D̄_t = ΔD_t + βD̄_{t−1}, with β in [−0.75, −0.25] — [Source](https://arxiv.org/abs/2410.02416)
  - EDM2-S at w=4: CFG FID 10.42 and saturation .46, APG 6.49 and .33; without projection 6.63 and .37; without rescaling 7.93; without momentum 6.85. Projection mainly cuts saturation, while rescaling and momentum mainly improve FID and recall — [Source](https://arxiv.org/abs/2410.02416)
  - Projecting in x0 space matters: projecting onto ε gives results nearly identical to CFG — [Source](https://arxiv.org/abs/2410.02416)
- CFG rescale: x ← x·(φ std(x_pos)/std(x_cfg) + 1 − φ), with φ = 0.7 at w = 7.5 — [Source](https://arxiv.org/abs/2305.08891)
- Dynamic thresholding: s = percentile(|x̂0|, p), and if s > 1, clip to [−s, s] and divide by s, with p = 99.5 — [Source](https://arxiv.org/abs/2205.11487). DPM-Solver++ pairs data prediction with this — [Source](https://arxiv.org/abs/2211.01095)
- Clipping and momentum: DOODL clips to ±1e-3 with momentum 0.9 — [Source](https://arxiv.org/abs/2303.13703). RED-diff uses Adam at lr 0.1 — [Source](https://arxiv.org/abs/2305.04391)
- Conflict on momentum: APG finds negative β better and warns that excessive momentum degrades images — [Source](https://arxiv.org/abs/2410.02416); DOODL uses +0.9 on seed optimisation — [Source](https://arxiv.org/abs/2303.13703); and HB uses positive momentum on the solver, reducing artifacts at high guidance on ADM — [Source](https://arxiv.org/abs/2307.11118). They act on different quantities.
- Heavy ball: x_{n+1} = x_n + (1 − β)(x_n − x_{n−1}) + δβ f(x_n), order falls to 1, and GHVB keeps the order — [Source](https://arxiv.org/abs/2307.11118)
- CFG as predictor plus corrector: CFG equals a predictor plus a Langevin corrector on a powered distribution — [Source](https://arxiv.org/abs/2408.09000)
- Further 2025 to 2026 CFG reshaping: CFG-Zero* — [Source](https://arxiv.org/abs/2503.18886), TAG tangential amplification — [Source](https://arxiv.org/abs/2510.04533), CFG-Ctrl with PID and sliding mode — [Source](https://arxiv.org/abs/2603.03281), frequency-decoupled guidance — [Source](https://arxiv.org/abs/2506.19713), ADG — [Source](https://arxiv.org/abs/2506.11039). None was measured on loss guidance in what I read.
- DiffStateGrad's projection onto the state's top singular vectors improved robustness to step size — [Source](https://arxiv.org/abs/2410.03463)

### Inferences
- Mapping to the repo: the guidance shift in x_next has RMS w·(σ/α)·min(ratio, rate)·RMS(ε), and FreeDoM's has c1·(σ/α)·0.02·RMS(ε). Both weights scale with step length, so the rates compare directly: 0.08 here, 0.02 FreeDoM, 0.04 Shen ImageNet.
- So per unit of logSNR the repo pushes 2 to 4 times harder than those methods before their repeats. Raising the rate is unlikely to be the remaining lever, and this agrees with the rate having been tuned.
- The difference that remains is that they always normalise, while the repo only caps. That needs a log of how often the cap is active.
- Shift size at 30 steps: h ≈ 0.322 (logSNR from −5.06 to 4.61), giving about 0.042 of σ_next per step. The fresh noise is 0.525 σ_next, so guidance is about 8% of the noise. At 60 steps it is 0.020 and about 5%.
- That sits at the low end of DSG's 0.05 to 0.2. Adding a roughly orthogonal 8% vector changes the radius by about 0.3%, so DSG's sphere is nearly a no-op here, and the repo's cap already delivers what DSG's few-step gains come from. To test the idea, raise the rate rather than add the projection.
- Polyak is the odd one out: the step grows as ‖g‖ shrinks, and it scales inversely with the loss weight, so cgs would need to come out of it. It conflicts with the cap's aim of not overshooting, and the published ImageNet path does not appear to use it. It is low priority.
- APG applied to CLIP: Δ = guided_x0 − x̂0, project against x̂0, keep η ≈ 0 to 1, then apply the existing cap. It costs nothing and targets the saturation that the x0 clamp currently handles. CLIP's gradient has no reason to match CFG's geometry, so this is a test and not a prediction. Negative momentum across steps is riskier, since CLIP gradients are noisier and the 14 step window is short.
- CFG rescale and dynamic thresholding also act on x̂0. The repo's static clamp at 1.2 touches only guided_x0 while the multistep term uses the unclamped x̂0. Replacing it with a percentile rule should keep that split.
- Filtering (DiffStateGrad at τ = 0.99, or a low-pass) targets the docstring's complaint that the low-noise gradient is mostly adversarial high-frequency detail, and might allow a wider window.

### Gaps
- No source compares normalisers for CLIP guidance at a fixed budget.
- Sign-of-gradient guidance: I found no diffusion-guidance evidence.
- "Anderson view" in the brief: I found no source on Anderson acceleration for guidance. The only Anderson hit was the 1982 reverse-time SDE reference.
- APG's values of r and η per model were not extracted, and they would not transfer anyway since they are in EDM2 latent units.
- How often the repo's cap is active by step is unknown and cheap to log.

## Q5. Is time travel worth it, and how many repeats at which noise levels?

### Takeaway
Repeats are concentrated at t 800 to 500 and measured value ranges from indispensable in FreeDoM's ImageNet setup to nil in EluCD, so at this budget only r = 2 or 3 on the six steps in that range is defensible. The same 6 to 12 calls may do more as extra solver steps, which the repo's unconditional path already shows it needs.

### Cited Findings

| Source | Repeats | Noise range | Result |
|---|---|---|---|
| [FreeDoM code](https://github.com/vvictoryuki/FreeDoM/blob/main/Face-GD/functions/denoising.py) | 10 | t 800 to 500, ImageNet only | Authors tie it to the semantic stage; faces use none |
| [Shen code](https://github.com/BIGKnight/Understanding-Training-free-Diffusion-Guidance/blob/master/CelebA%20and%20ImageNet/functions/denoising.py) | 10 | t 800 to 300 | Used for the 27.7 run |
| [Universal Guidance](https://arxiv.org/abs/2302.07121) | k = 8 for CLIP | whole schedule | Qualitative |
| [EluCD](https://arxiv.org/abs/2310.11311) | 1, 2, 3 | whole schedule | FID 7.17, 7.06, 7.14 at 14.1, 16 and 18 h; calibration alone reached 5.19 at 14.1 h |
| [TFG](https://arxiv.org/abs/2409.15761) | 1, 2, 4 | whole schedule | Clear gain at N_recur× cost, then FID deteriorates when too large |
| [RePaint](https://arxiv.org/abs/2201.09865) | r = 10, j = 10 | inpainting | Saturates near 10 resamples |
| [DDNM](https://arxiv.org/abs/2212.00490) | length l | inverse | Time travel on top of the null-space step |
| [Restart](https://arxiv.org/abs/2306.14878) | K = 2 | σ in [0.1, 2] in SD with CFG 8 | 132 NFE, better FID at equal CLIP score, no principled hyperparameters |

- Shen gives a theoretical analysis of time travel in Appendix C.2 and says earlier uses had no justification — [Source](https://arxiv.org/abs/2403.12404)
- EluCD concludes recurrence does not fully exploit the potential and costs sampling time — [Source](https://arxiv.org/abs/2310.11311)
- The kernel: x_t = √b x_{t−1} + √(1 − b) z, b = ᾱ_t/ᾱ_{t−1} — [denoising.py](https://github.com/vvictoryuki/FreeDoM/blob/main/Face-GD/functions/denoising.py)

### Inferences
- On the 30 step grid t 800 to 500 holds 6 steps and t 800 to 300 holds 10. Calls at r = 2, 3, 5, 10 are 36, 42, 54, 84 and 40, 50, 70, 120.
- The ImageNet repeats make FreeDoM's and Shen's totals about 379 and 559 calls, against 100 without them. If the published code matches the paper's runs, those are the budgets behind their CLIP scores, so the scores should not be expected at 30 calls.
- EluCD's class-conditional setting is the closer analogue to the repo's conditional path. FreeDoM's unconditional setting is the closer analogue to `-uncond`. So time travel is more likely to pay on the unconditional path.
- Adaptation needs care in this solver: a repeat lands at the same t, so `previous_x0` would come from the same noise level. Keep the history from the previous distinct timestep or run repeats first order, and draw re-noise from the seeded generator in a fixed position. `h/previous_h` must not use a repeat's zero or stale step.
- Unconditional runs needed 60 steps here to avoid overshoot, so spending 12 calls on more steps is the fair baseline to test time travel against.

### Gaps
- No CLIP measurement of repeat count at under 100 calls exists.
- Interaction between time travel and second-order multistep solvers is not studied in anything I read.
- Whether repeats help at t below 300 for text guidance is unknown.

## Q6. Which schedule shapes and class or self-guidance methods apply to a class-conditional or unconditional pixel U-Net?

### Takeaway
A sine or increasing guidance schedule is free and has the most consistent evidence. ADM's own classifier has the best numbers on these exact checkpoints, including 5.44 FID at 25 steps, and PAG and SAG are the strongest options needing no new weights, though measured at 250 steps.

### Cited Findings

#### Schedule shape
- TFG: structure "increase" (∝ α_t) beats constant, which beats "decrease", for both ρ and μ, and the ordering is invariant to the other parameters — [Source](https://arxiv.org/abs/2409.15761)
- EluCD: the sine term at γ = 0.3 improves FID 5.57 to 5.24 and CLIP scores are plotted higher — [Source](https://arxiv.org/abs/2310.11311)
- CFG weight schedulers: monotonically increasing guidance performs best, and a linear ramp always improves on constant — [Source](https://arxiv.org/abs/2404.13040)
- Guidance interval: disabling guidance outside σ in (0.28, 2.9] improves EDM2-S 2.23 to 1.68, EDM2-XXL 1.81 to 1.40, DiT 3.04 to 2.40, and saves evaluations — [Source](https://arxiv.org/abs/2404.07724)
- Conflict in direction: TFG's "increase" and CFG schedulers ramp guidance toward low noise, while the repo's docstring finds low-noise CLIP gradient mostly adversarial and stops at 0.1 — [cgd/sampling.py](/home/sam/GitHub/clip-guided-diffusion/cgd/sampling.py)

#### Self and perturbation guidance on ADM
- SAG: ε̃ = ε̂ + (1 + s)(ε − ε̂), two forwards, s about 0.1 to 0.3. ADM unconditional 256 FID 26.21 to 20.08, conditional 256 10.94 to 9.41, ADM 128 with classifier guidance 2.97 to 2.58 — [Source](https://arxiv.org/abs/2210.00939)
- PAG: replaces self-attention with identity in the perturbed branch, s = 1.0, layer middle_block.1 for unconditional and input_blocks.14.1 for conditional, 250 DDPM steps. ADM unconditional 256 FID 26.21 to 16.23 and IS 39.7 to 88.53; conditional 256 10.94 to 6.32. The conditional precision and recall appear swapped in the table — [Source](https://arxiv.org/abs/2403.17377)
- SEG: SDXL only, FID 129.5 to 88.2, CLIP .271 to .292 — [Source](https://arxiv.org/abs/2408.00760)

#### Two-model and class contrast
- Autoguidance: D_w = D_0 + w(D_1 − D_0), with a smaller and earlier snapshot as D_0. EDM2-S 512 FID 2.56 to 1.34, XXL 1.25, 64px 1.01, and an unconditional EDM2-S 11.67 to 3.86 — [Source](https://arxiv.org/abs/2406.02507)
- In-situ autoguidance (dropout as the bad model, w = 2, p = 0.1): FID 2.57 and FD-DINOv2 90.05 against baseline 2.56 and 68.64, so no gain — [Source](https://arxiv.org/abs/2510.17136)
- GFCG: D̂ = ωD(x, t, c_des) − (ω − 1)D(x, t, c_ref), where c_ref is the classifier's confused class and ω = 1 + αe^{−β(p − τ)} when p(c_des|x̂0) < τ, with no gradients and no null model. FD-DINOv2 and precision on EDM2-S: ATG 38.50 at 90.6% vs GFCG_NG 36.93 at 93.3%. On EDM2-XXL: ATG 24.83 at 90.2% vs GFCG combined with ATG 23.09 at 94.3%. Both use 32 Heun steps — [Source](https://arxiv.org/abs/2411.15393)
- CFG++ renoises with ε_∅ and λ in [0, 1], so it needs an unconditional branch — [Source](https://arxiv.org/abs/2406.08070)
- SSG needs an adapter trained on a pixel transformer (JiT-H/16 1.86 to 1.67) — [Source](https://arxiv.org/abs/2607.29122). Feedback guidance scales by state — [Source](https://arxiv.org/abs/2506.06085). Representation guidance — [Source](https://arxiv.org/abs/2610.12978) and PixelDiT2 — [Source](https://arxiv.org/abs/2609.24919) are trained into new models.

#### What the repo's checkpoints allow
- guided-diffusion adds the class as `emb = emb + self.label_emb(y)` with `nn.Embedding`, so there is no null class and a continuous vector must bypass the embedding. Attention sits at 32, 16 and 8 — [guided-diffusion unet.py](https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/unet.py)
- 256px has conditional and unconditional diffusion checkpoints, and classifiers exist at all four sizes — [guided-diffusion README](https://github.com/openai/guided-diffusion/blob/main/README.md). The repo also uses a finetuned 512 unconditional checkpoint — [data/diffusion_model_flags.py](/home/sam/GitHub/clip-guided-diffusion/data/diffusion_model_flags.py)
- The repo found averaged class embeddings produce noise and uses one real label — [cgd/sampling.py](/home/sam/GitHub/clip-guided-diffusion/cgd/sampling.py)
- On DiT, class embeddings exceed 99% pairwise cosine, only 10 to 20 of 1152 dimensions carry large magnitude, and up to 66% can be pruned without harm. The paper links this to summation with the timestep embedding before adaptive normalisation — [Source](https://arxiv.org/abs/2602.21596)
- Optimised embeddings exist elsewhere: a class vector in a GAN — [Source](https://arxiv.org/abs/2005.01703), and text-embedding optimisation and blending, where IMPUS warns naive latent interpolation gives severe artifacts — [survey](https://arxiv.org/abs/2409.19365), [IMPUS](https://arxiv.org/abs/2311.06792)

### Inferences

| Method | Conditional ADM | Unconditional ADM | ADM evidence | Added cost at 30 steps |
|---|---|---|---|---|
| ADM-G classifier | Yes, all sizes | Yes, 256 at s=10 | Yes, including 25 steps | Classifier forward and backward per step |
| PAG | Yes | Yes | Yes, 250 steps | Up to +30F, less on an interval |
| SAG | Yes | Yes | Yes, 250 steps | Up to +30F plus a blur |
| SEG | Likely | Likely | None | Up to +30F |
| CFG from a checkpoint pair | 256, 512 | Is the pair | None | +30F and a second U-Net in VRAM |
| CFG++ | Same as above | Same as above | None | As above |
| Autoguidance | No degraded model | No degraded model | None | Needs training |
| GFCG-style class contrast | Yes | No | None on ADM | +F on guided steps |
| In-situ | Pointless | Pointless | Negative elsewhere | |

- Schedule change to try inside the window: replace the constant rate with rate·(1 + γ sin(π t/T)) or an α-weighted ramp, then renormalise so the summed guidance stays equal. Equal totals keep the comparison about shape.
- Mapped naively as σ/α on ADM's linear schedule, the guidance interval (0.28, 2.9] would be t of about 82 to 468, far from the repo's 100 to 800. EDM2 runs in latent space and that interval is for class guidance, so this is at most a hint that the upper part of the window could be tested for less CLIP or for class guidance only. The ramp conflict above suggests shaping within the window and leaving the low cut alone.
- ADM's classifier is the most surprising gap in the repo. It is noise-aware, so it needs no U-Net backward and no x̂0, and its gradient can run in the steps where CLIP is off. On the unconditional path, a CLIP-chosen class with s around 10 could supply the subject that the sampler's docstring says is missing, at FID 12.00 vs 26.21. It only helps prompts with a sensible ImageNet class, so it may fight CLIP on others, as class guidance raises precision and cuts recall.
- PAG numbers at 250 steps may shrink at 30, and they measure realism, not prompt fit. Applying it in about 10 mid-schedule steps keeps the call count near 40. Its perturbed branch takes no gradient, so the cost is F only.
- Cond and uncond pair: ε̃ = ε_u + w(ε_c − ε_u) is available at 256 and 512 with the same linear schedule. The pair was not trained jointly, unlike real CFG. It is closest to autoguidance with D_0 being unconditional, and at 5.3 GiB of 8 GiB, a second U-Net at 256px may not fit on this GPU.
- Class contrast: `imagenet_top_n()` already ranks classes by CLIP, so c_ref can be the runner-up. That skips GFCG's classifier. ε̃ = ωε(y_1) − (ω − 1)ε(y_2) is a valid evaluation at each label, which embedding averaging is not.
- Null-like embeddings: a zero or mean vector was not trained and the repo already saw the mean fail. The unconditional checkpoint is the only real null.
- Averaging in ε space across labels keeps each evaluation on the manifold, at K forwards. It may recover what embedding averaging was trying to do, though I have no evidence.
- Measure ADM's `label_emb` pairwise cosines first; it takes minutes and no GPU. If they are as aligned as DiT's, the noise from averaging needs another explanation, and interpolation between similar classes becomes more plausible. Optimising the embedding by CLIP would cost the same B as today, but has no precedent on ADM and risks leaving the valid set.

### Gaps
- I found no paper on interpolating or optimising class embeddings in ADM, and no paper applying SAG, PAG, SEG or CFG++ together with CLIP guidance.
- PAG, SAG and classifier numbers at about 30 calls with a modern solver are missing, apart from ADM-G at 25 DDIM steps.
- Dropout at 128, 256 and 512 is not stated for the ImageNet models in the repo's flags file, though 64px shows 0.1, and in-situ autoguidance gave nothing anyway.
- SSG and Feedback guidance have no ADM evidence. In-situ's claim of a "strong baseline" conflicts with its own table.

## Q7. How does guidance interact with the solver?

### Takeaway
Keep guidance first-order and out of the multistep history, as the repo already does, because the splitting evidence shows high-order treatment of the guidance term fails to beat Euler and can diverge. ER-SDE-Solver is the one direct upgrade with classifier-guided ADM 256 numbers, 5.47 to 4.62 FID at 30 NFE against the repo's solver family.

### Cited Findings
- Splitting numerical methods on ADM classifier guidance: high-order solvers on the conditional subproblem never beat Euler, some diverge, and condition gradients are "less compatible with classical high-order methods". The method: separate the ε ODE (PLMS4) and the guidance (Euler). LTSP is Lie-Trotter and STSP is Strang, which needs two gradients per step — [Source](https://arxiv.org/abs/2301.11558)
  - ImageNet 128 at 25 steps: DDIM 6.69, PLMS4 4.97, LTSP4 3.85, STSP4 at 20 steps 3.78, ADM-G 250 2.97. ImageNet 256: DDIM 5.47, PLMS4 4.71, STSP4 4.49 vs ADM-G 4.59, 9.4 times faster. ImageNet 512: DDIM 9.07, PLMS4 16.0, STSP4 8.24, ADM-G 7.72 — [Source](https://arxiv.org/abs/2301.11558)
- Classifier-guided ImageNet 256, s=2, FID at 20/30/50 NFE: SDE-DPM-Solver++(2M) 6.01/5.47/5.19; DPM-Solver++(2M) 5.54/5.32/5.16; DDIM 5.60/5.00/4.59; ER-SDE-Solver-3 4.76/4.62/4.57. Unguided conditional 128: SDE-DPM-Solver++(2M) 9.61/8.38/7.83, ER-SDE 8.33/7.84/7.72 — [Source](https://arxiv.org/abs/2309.06169)
- VP ER-SDE-Solver-2 step: x_i = r_α r x_{i−1} + α_i(1 − r) x_θ + δ_i + n_i, with r = φ(λ_i)/φ(λ_{i−1}), r_α = α_i/α_{i−1}, λ = σ/α, δ_i = α_i[λ_i − λ_{i−1} + S_i φ(λ_i)] D_i, D_i = (x_θ,i−1 − x_θ,i−2)/(λ_{i−1} − λ_{i−2}), n_i = α_i √(λ_i² − r²λ_{i−1}²) z. Default φ(x) = x(e^{x^{0.3}} + 10). S_i is a numerical integral — [Source](https://arxiv.org/abs/2309.06169)
- The ER-SDE authors suggest stochasticity may reduce inaccuracy in classifier gradients but leave it as speculation — [Source](https://arxiv.org/abs/2309.06169)
- Stochasticity against NFE: at small NFE smaller nonzero τ is better, and at 20 to 100 steps larger τ is better. Results: ADM ImageNet 64 FID 1.81 at 95 NFE and DiT 256 2.02 at 60 NFE vs DDPM 250 at 2.27. DPM-Solver++(2M) is a τ = 0 special case — [Source](https://arxiv.org/abs/2309.05019)
- Conflict: at low NFE unguided, stochastic solvers lose (EDM ImageNet 64 at 35 NFE: EDM stochastic 2.97, deterministic 2.46, SDE-DPM-Solver++(2M) 3.02), and only ER-SDE matches the ODE at 2.46 — [Source](https://arxiv.org/abs/2309.06169)
- DPM-Solver++ data prediction and multistep exist for stability at large guidance scale, and high order is unstable with large guidance — [Source](https://arxiv.org/abs/2211.01095)
- UniPC at guidance s=8, 256px, NFE 5 to 10: DDIM 38.11 to 10.34, DPM-Solver++ 55.64 to 8.49, UniPC-B2(h) 25.87 to 7.51, while UniPC-B1(h) is much worse at 16.02. B2 suits guided sampling — [Source](https://arxiv.org/abs/2302.04867)
- EDM Heun with an off-the-shelf ResNet: 18 steps 2.54 to 2.35, 36 steps 2.35 to 2.22, at 10 steps 3.64 to 3.38, unnoised classifier with normalised gradient — [Source](https://arxiv.org/abs/2310.11311)
- Heavy ball momentum on ADM improved DPM-Solver++(2M) and LTSP at 10 to 26 steps, and LTSP with GHVB 0.8 gave the lowest FID, shown as a plot — [Source](https://arxiv.org/abs/2307.11118)
- Restart: DDIM 30 steps plus a [0.1, 2] restart with K = 2 and 10 Heun steps, 132 NFE, with SD and CFG 8 — [Source](https://arxiv.org/abs/2306.14878)
- Others: PFDiff — [Source](https://arxiv.org/abs/2408.08822) and an adaptive momentum sampler — [Source](https://www.ijcai.org/proceedings/2024/157). No numbers extracted.

### Inferences
- The repo's design matches LTSP: a second-order ε-side and a first-order guidance term. That is independent support for the docstring's choice. It is not exact splitting, since the gradient is taken at the start of the step.
- DDIM beats DPM-Solver++(2M), ODE and SDE alike, at 30 and 50 NFE with classifier guidance. Those solvers extrapolate guided outputs. The repo does not, so it should sit nearer LTSP's numbers. The table does not directly rate the repo's solver.
- ER-SDE swap: at second order, replace the coefficients and keep x_θ unguided in D_i and guided in the x̃0 term. The full third-order result needs a two-step history. Noise stays one seeded draw per step and S_i is deterministic. The window, rate and step count would need re-tuning, since noise injection changes.
- At 30 steps the expected gain from a solver change is modest if it follows the FID gap, and CLIP judged score could differ. Since Q4 shows guidance is 8% of noise, the injected noise is the larger term and is worth tuning alongside η.
- Strang splitting costs 14B + 14C more, and its measured edge over LTSP is small (3.78 at 20 steps vs 3.85 at 25 steps at 128px). It is worth trying only after the cheaper items. Taking the half-steps with an x̂0 gradient would avoid B.
- UniC as a corrector needs a guidance choice at the corrected point. Using unguided ε in it follows the same rule.
- Heun and Restart spend calls at a low-order guidance rate. At 30 calls, Heun halves the steps, so it is a poor fit. At 100 calls it is an option.
- Keep guidance at Euler, and let any extra order go to the unguided part.

### Gaps
- No solver comparison uses CLIP guidance, so every number here is classifier-guided or CFG, and classifier gradients may behave differently from cutout CLIP gradients.
- The ER-SDE guided table uses s=2 while ADM's published optimum at 256 is 1.0, which may favour some solvers.
- How guidance should enter a corrector in SA-Solver or UniPC under stochastic steps is not addressed in what I read.
- Numbers from PFDiff, adaptive momentum and GHVB beyond the plot were not extracted.

# Inference-time search and optimisation for CLIP guidance of OpenAI's 256x256 unconditional ADM

Scope: methods that spend extra inference compute (more model calls, backprop through the sampler, particles, search) to raise prompt alignment and quality when sampling `256x256_diffusion_uncond.pt` with CLIP. The baseline here is the repo's DPM-Solver++(2M) SDE, 60 calls, capped CLIP guidance from 10% of the schedule. It takes 20.0 s and 5.3 GiB at 256px. It scores 0.316 ViT-B/16 similarity against 0.331 for the removed 1000-step sampler, which took 6 min. That gap is not significant: 95% CI of the difference -0.040 to +0.014 on 6 prompts ([README](/home/sam/GitHub/clip-guided-diffusion/README.md), Speed section). In the notes below, "1x" means one baseline sample, about 20 s on the RTX 2070 SUPER.

**Local measurement (2026-11-13, RTX 2070 SUPER, fp16 U-Net via `cgd.adm.load_model(256, unconditional=True)`, batch 1, loss = mean square of the predicted noise, no CLIP).** Every "measured" number below comes from this run:

| Configuration | Time | Peak VRAM |
|---|---|---|
| U-Net weights | – | 1.13 GiB |
| Forward, no grad | 146–149 ms/call | 1.52 GiB |
| Forward + backward to the input, no inner checkpointing (what the repo does at 256px) | 290 ms | 4.94 GiB |
| Same, with guided-diffusion `use_checkpoint=True` | 445 ms | 2.17 GiB |
| Backprop through K DDIM steps, `torch.utils.checkpoint` around each sampler step (only x_t kept per step), K = 10 / 25 / 50 | 4.5 / 11.1 / 23.7 s per iteration | 4.95 / 4.96 / 4.98 GiB |
| Same plus inner `use_checkpoint`, K = 10 / 25 / 50 | 6.2 / 21.6 / 53.4 s per iteration | 2.18 / 2.19 / 2.21 GiB |
| Naive full autograd graph through K steps, no inner checkpointing | K=2 out of memory | > 8 GiB |
| Naive full graph with inner checkpointing, K = 2 / 3 / 4 / 6 | 1.0 / 1.5 / 1.9 / 2.9 s | 2.79 / 3.41 / 4.02 / 5.25 GiB (about 0.6 GiB per extra step) |

The inner-checkpointed chain grew from 0.62 to 1.07 s per step between K=10 and K=50. That superlinear growth is probably GPU clock throttling and was not investigated. Gradients were finite at every K.

## 1. Initial-noise and trajectory optimisation through a multi-step sampler (DOODL, D-Flow, DNO, ReNO, InitNO, Golden Noise, adjoint/checkpointing)

### Takeaway
Memory is not the obstacle at 256px. With step-level gradient checkpointing, backprop through any number of sampler steps of the 550M U-Net peaks at about 5 GiB (2.2 GiB with inner checkpointing too). That leaves room for CLIP on 8 GB. The cost is time: about 0.45 s per sampler step per optimisation iteration (measured). So 30 iterations through a 20-step sampler is roughly 4.5–5 min, about 14x the baseline. The published gains are large on latent T2I models with learned rewards (DNO: aesthetic 5.37 to 7.50 in about 1 min on an A800) and modest with CLIP (DOODL: +3% CLIPScore at 20 min per image). There is no published result for a pixel ImageNet ADM with CLIP. Every method keeps the noise on the Gaussian shell with a norm projection or a chi-norm likelihood penalty. Without one, rewards get hacked out of distribution.

### Cited Findings
- **DOODL** optimises the initial noise x_T through the full 50-step DDIM chain of SD v1.4 using EDICT invertible sampling. EDICT backpropagates at constant memory in the number of steps. Naive activation caching "could approach a terabyte at S=50" for SOTA models — [DOODL, Wallace et al. 2023](https://arxiv.org/html/2303.13703)
- DOODL's stabilisers:
  - after each update, average x_T and y_T and renormalise to the norm of the initial draw ("Gaussian shell");
  - momentum 0.9;
  - element-wise gradient clipping at 1e-3;
  - Gaussian perturbation (variance 1e-4) per update;
  - 16 multicrop augmentations per image;
  - CLIP loss weight λ=0.1, against 30 for one-step guidance.
  — [DOODL](https://arxiv.org/html/2303.13703)
- DOODL improves DrawBench CLIP score over one-step guidance by 3.1% (DALLE prompts) and 2.6% (Reddit prompts), and human prompt alignment by 6% over one-step guidance. It is "more prone to warp or deform content" than baseline guidance, and its aesthetics gains are inconsistent (win-draw-loss 0.48-0.07-0.45 against baseline guidance) — [DOODL](https://arxiv.org/html/2303.13703)
- ReNO measured DOODL on SD2.1 with CLIP: 24 s/iter, 20 min total, 40 GB VRAM, CLIPScore 0.261 to 0.289. ReNO on one-step SD-Turbo with CLIP only: 0.2 s/iter, 10 s total, 10 GB, CLIPScore 0.362 to 0.483 — [ReNO, Eyring et al. 2024](https://arxiv.org/html/2406.04312)
- ReNO's regulariser is K(ε) = (d−1)·log‖ε‖ − ‖ε‖²/2, the log-likelihood of the norm of a Gaussian sample. It uses 50 steps of gradient ascent with Nesterov momentum 0.9 and gradient-norm clipping at 0.1, and returns the highest-reward image along the trajectory — [ReNO](https://arxiv.org/html/2406.04312)
- ReNO's reward ensemble weights are ImageReward 1.0, HPSv2 5.0, PickScore 0.05, CLIPScore 1.0. Optimising ImageReward or CLIPScore alone lowered the aesthetic score (CLIP only: 5.647 to 5.475), while all rewards together raised it to 5.704 — [ReNO](https://arxiv.org/html/2406.04312)
- ReNO's VRAM on an A100 at 512px: 15 GB for SD-Turbo (860M). fp16 plus a PyTorch memory technique save "another ~15%" — [ReNO](https://arxiv.org/html/2406.04312)
- **DNO** (Tang et al. 2024), noises optimised: in the SDE variant (DDIM η=1), both x_T and every injected noise z_t (about T·d dimensions). The paper argues it converges faster than optimising x_T alone through the ODE — [DNO](https://arxiv.org/html/2405.18881)
- DNO's probability regularisation: split the noise into m subvectors (k=2). Concentration bounds on their mean and covariance give a low-probability score, minimised over q=100 random permutations and added with weight γ. It keeps noise in-distribution and reduces OOD reward hacking — [DNO](https://arxiv.org/html/2405.18881)
- DNO's memory and time: gradient checkpointing through the sampler, under 15 GB, on a single A800. Budgets of 1, 3 and 5 min per image. With 15 steps and 1 min: aesthetic 7.496, HPS 0.341, PickScore 24.82, against SD v1.5 baseline aesthetic 5.367, HPS 0.278, PickScore 21.11. With 50 steps and 1 min: aesthetic only 5.754. The authors say 15–25 steps are often enough — [DNO](https://arxiv.org/html/2405.18881)
- DNO's reward hacking: without regularisation, brightness optimisation of "black duck" produced a white duck after 50 steps. At a fixed 10-min budget DNO beats best-of-N "by a large margin". It also offers zero-order and hybrid gradient estimators for non-differentiable rewards — [DNO](https://arxiv.org/html/2405.18881)
- **D-Flow** optimises only the source noise by differentiating through a 6-evaluation midpoint ODE solve, with L-BFGS and line search, and gradient checkpointing for memory. It takes 5–15 min per sample on ImageNet-128 on a V100, and 30–40 min for large T2I and text-to-audio models — [D-Flow, Ben-Hamu et al. 2024](https://arxiv.org/html/2402.14017)
- D-Flow's regularisation: backpropagating through the solver of a Gaussian-path flow projects the cost gradient onto high-variance data directions, an implicit manifold regulariser (Theorem 4.2, under zero-training-loss assumptions). An explicit chi-d norm prior helped only for noisy inverse problems and molecules — [D-Flow](https://arxiv.org/html/2402.14017)
- D-Flow's results: noiseless ImageNet-128 deblur PSNR 39.47 against 37.44 for OT-ODE. COCO free-form inpainting FID 16.92 against 23.31 for RED-Diff, CLIP 0.892 against 0.882 — [D-Flow](https://arxiv.org/html/2402.14017)
- **Adjoint methods**: AdjointDPM backpropagates through diffusion sampling with the adjoint ODE, optimising initial noise among other things. The search summary says 19.63 GB on a V100 for gradients with respect to one initial noise — [AdjointDPM](https://arxiv.org/pdf/2307.10711). That number comes from a search snippet and its exact setting was not verified.
- DITTO (music latents) reports that sampler-step checkpointing costs "only 2x the number of model calls" — [DITTO](https://arxiv.org/pdf/2401.12179)
- **Golden Noise (NPNet)** trains a small network to map random noise to "golden" noise. It was trained on 100k noise pairs and evaluated on SDXL, DreamShaper and Hunyuan-DiT with PickScore, HPSv2 and ImageReward. Reported overhead is about 3% (secondary summary) — [ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_Golden_Noise_for_Diffusion_Models_A_Learning_Framework_ICCV_2025_paper.pdf), [review](https://themoonlight.io/review/golden-noise-for-diffusion-models-a-learning-framework)
- **InitNO** is cited by the Golden Noise authors as an earlier initial-noise optimiser that is slow and generalises poorly — [papernotes summary](https://en.papernotes.org/ICCV2025/image_generation/golden_noise_for_diffusion_models_a_learning_framework/)
- Local measurement, U-Net only (table above):
  - step-checkpointed backprop through 10, 25 or 50 DDIM steps peaks at 4.95–4.98 GiB;
  - adding inner checkpointing brings it to 2.2 GiB;
  - one iteration costs about 0.45 s per step (no inner checkpointing).
  - A naive full graph fits at most 1 step without inner checkpointing, and about 9 steps with it (extrapolated from 0.6 GiB per step), before CLIP — measured on this machine with `cgd.adm.load_model`
- The repo's 256px unconditional sampler peaks at 5.3 GiB with CLIP — [README](/home/sam/GitHub/clip-guided-diffusion/README.md). So the default CLIP guide adds well under 1 GiB over the U-Net backward, an inference from the two numbers.

### Inferences
- **Feasibility on 8 GB:** DNO or DOODL-style optimisation through this U-Net fits easily at 256px with `torch.utils.checkpoint` per sampler step. EDICT or adjoint machinery is unnecessary because checkpointing already gives flat memory. The cost per iteration is about 1 forward without grad, 1 recomputed forward and 1 backward per step: roughly 3x a plain call, or 0.45 s per step on the 2070 SUPER.
- **Budget at 256px on the 2070 SUPER** (estimates, ignoring CLIP time):
  - 20-step sampler × 30 iterations ≈ 4.7 min (about 14x baseline);
  - 60-step sampler × 30 iterations ≈ 14 min (about 42x);
  - 10-step × 50 iterations ≈ 3.8 min.
  - DNO's "15–25 steps often enough" finding suggests a short sampler inside the optimisation loop.
- **Do not backprop through the guided sampler itself.** The guidance term is a gradient of CLIP through the U-Net, so differentiating it needs a second-order (double-backward) pass through the U-Net. That costs roughly twice the time and memory, and the guided-diffusion fp16 path has not been verified for double backward. Practical options:
  - (a) Optimise the noises of an unguided or lightly guided sampler, as DNO and DOODL do.
  - (b) Treat the capped guidance step as a constant (stop-gradient), a first-order approximation.
  - (c) Optimise only through the high-noise "composition" part of the schedule, then hand over to the existing guided sampler for detail.
  - This is an inference: none of these papers uses in-loop gradient guidance.
- **What to optimise:** the repo's sampler is an SDE with ETA=0.5, so DNO's SDE formulation fits directly: optimise x_T and the injected noises. Then reuse the same noises in the final sampling run, so the optimised trajectory is reproduced exactly and determinism is preserved.
- **Regulariser:** the repo already uses a noise-norm penalty for the one-step backbone (ReNO-style). The same chi-norm log-likelihood, applied per injected noise, is the simplest choice. DOODL's hard renormalisation to the original norm is stricter and parameter-free. DNO's permutation-based probability regulariser also constrains low-order statistics, catching structured but norm-preserving hacks, but it is more code.
- **Expected gain is uncertain for pixel ADM plus CLIP.** The only CLIP-reward number for multi-step noise optimisation is DOODL's +0.028 CLIPScore on SD2.1, at 40 GB and 20 min. The large DNO and ReNO jumps are for learned preference rewards or one-step models. CLIP gradients in pixel space are notoriously adversarial, but here they pass through the U-Net Jacobian. D-Flow's theorem suggests that Jacobian projects them toward the data manifold. That implicit regularisation is the main reason to expect it to work better than raw pixel guidance.
- **Golden Noise and InitNO do not transfer.** NPNet needs a trained noise-to-noise network per model plus a preference dataset. As far as these sources show, InitNO targets text-conditioned latent U-Nets, and the unconditional ADM has no text conditioning (InitNO's mechanism was not verified here).

### Gaps
- No paper found that runs DOODL, DNO or D-Flow on OpenAI's pixel ADM (any resolution) with a CLIP reward, so there is no direct gain estimate for this model.
- DOODL's optimisation-step count is not given in the fetched text. ReNO's learning rate and λ_reg values were cut off in the fetched text.
- AdjointDPM's 19.63 GB figure comes from a search snippet. Its exact setting (which model and resolution) was not verified.
- InitNO's own numbers and mechanism were not retrieved.

## 2. Search over the sampling process: random/zero-order/path search, FK steering, TDS/SMC, beam and tree search, and combination with gradient guidance

### Takeaway
At equal compute, plain random search (best-of-N) is a strong baseline: Ma et al. found it matched or beat zero-order and path search on the verifier's own metric. The particle methods' clearest advantage comes from **combining resampling with gradient-guided proposals**. FK steering with gradient proposals reached ImageReward 1.290, against 0.668 for gradient guidance alone and 0.898 for FK with base proposals. Tempering or a separate verifier for selection is what prevents over-optimisation. Tree search (DTS*) claims to match best-of-N at 2–5x less compute. ε-greedy noise-trajectory search raised SD CLIP similarity from 0.246 to 0.297, but at 80x the NFEs and not compute-matched.

### Cited Findings
- **Ma et al. 2025, setup:** ImageNet uses SiT-XL 256 (latent), Heun with 250 NFEs. T2I uses FLUX.1-dev, 30 steps. Algorithms:
  - random search = best-of-2^k (k up to 8);
  - zero-order search (ZO-N) = N neighbours of a pivot noise per iteration (N=4 on ImageNet);
  - search over paths = noise-and-redenoise of intermediate states (σ=0.11, Δb=0.81, Δf=0.78).
  — [Ma et al. 2025](https://arxiv.org/html/2501.09732)
- **Ma et al., equal compute** (FLUX, DrawBench, 2,880 search NFEs, Ensemble verifier), LLM grader: random 88.18, ZO-2 87.25, Paths-2 86.84, baseline 84.29. Random search "generally matches or beats" the local methods, which the paper attributes to their locality — [Ma et al.](https://arxiv.org/html/2501.09732)
- **Ma et al., ImageNet hacking:** searching against DINO or CLIP logits raises FID after some iterations while IS keeps improving. Precision rises and recall falls (lost diversity). ZO with a large step size λ raises FID beyond about 10³ NFEs. Gains per iteration diminish once NFEs per iteration ≥ 50 — [Ma et al.](https://arxiv.org/html/2501.09732)
- **FK steering (Singhal et al. 2025)**:
  - samples k particles and resamples them at intermediate steps, on potentials computed from rewards of the x0 prediction;
  - potentials: max, difference, sum;
  - proposals: the base kernel, or gradient-guided;
  - resampling can be restricted to an interval schedule to save compute.
  — [FK steering](https://arxiv.org/html/2501.06848)
- FK steering on SDXL with ImageReward, k=4 (GenEval / IR / HPS): FK 0.6489 / 1.298 / 0.302 against best-of-4 0.6347 / 1.236 / 0.296, and k=1 0.5571 / 0.871 / 0.289. SD v2.1 with FK k=4 (0.6289 GenEval) beats SDXL and SDXL-DPO — [FK steering](https://arxiv.org/html/2501.06848)
- **FK steering combined with gradient guidance (Table 3, SD v1.5)**:

  | Method | GenEval | IR | Time |
  |---|---|---|---|
  | Gradient guidance, k=1 | 0.45 | 0.668 | 20 s |
  | FK, k=4, base proposal | 0.54 | 0.898 | 8.1 s |
  | FK, k=4, gradient-guided proposal | 0.56 | 1.290 | 55 s |

  — [FK steering](https://arxiv.org/html/2501.06848)
- FK potentials, GenEval at k=4 / k=8: max 0.633 / 0.648 and difference 0.603 / 0.640 on SDXL. Max wins but reduces particle diversity. Lower λ (2 against 10) gives about twice the CLIP diversity at slightly lower GenEval — [FK steering](https://arxiv.org/html/2501.06848)
- FK on class-conditional ImageNet: mean class probability TDS 0.59 against FK (max, learned reward, k=4) 0.88. Timing: SD v2.1 k=1 4.6 s against FK k=4 17.4 s (9.1 s parallel) on an A100. No memory figures are given — [FK steering](https://arxiv.org/html/2501.06848)
- **TDS / twisted SMC** (Wu et al. 2023): SMC with twisting functions built from heuristic conditional samplers and corrected by weights, so it is asymptotically exact. It "already improves on heuristics with as few as two particles". Evaluated on MNIST class-conditional generation, inpainting and protein motif scaffolding — [TDS, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/63e8bc7bbf1cfea36d1d1b6538aecce5-Abstract.html), [arXiv](https://arxiv.org/html/2306.17775v2)
- **DAS** (Kim et al., ICLR 2025): SMC targeting p_pre·exp(r/α), with tempering λ_t=(1+γ)^t−1. Its Gaussian proposal is a first-order Taylor expansion of the reward (gradient-informed), with ESS-triggered resampling. "Without tempering, SMC suffers from over-optimization, even with 32 particles"; with tempering, 4–8 particles reach high aesthetic and ImageReward scores — [DAS](https://arxiv.org/html/2501.05803)
- DAS sits in the high-reward, high-diversity corner. Fine-tuning methods (DDPO, AlignProp) have high target reward but poor cross-reward generalisation. Gradient-guidance methods (DPS, FreeDoM, MPGD) have low target reward — [DAS](https://arxiv.org/html/2501.05803)
- **Diffusion Tree Sampling / Search** (Jain et al., NeurIPS 2025): MCTS-style rollouts refine value estimates of noisy states and reuse past rollouts. At 10⁶ NFEs, CIFAR-10 class-conditional FID: DTS 0.745, DAS 0.822, SMC/FK 1.409, DPS 2.609, TDS 2.675. It is gradient-free. DPS-RG and TDS-RG were unstable without gradient clipping — [DTS](https://arxiv.org/html/2506.20701)
- DTS efficiency claims conflict between versions. The v1 abstract says it matches the best baseline FID with up to 5x less compute and best-of-N with 2x less — [arXiv v1 / NeurIPS page](https://neurips.cc/virtual/2025/poster/120056). The current HTML abstract says up to 10x and 5x — [DTS HTML](https://arxiv.org/html/2506.20701)
- **ε-greedy noise trajectory search** (Ramesh & Mardani 2025) treats each denoising step's noise as a contextual bandit:
  - explores globally at the extreme timesteps and exploits locally at intermediate ones;
  - SD v1.5 CLIP reward: naive 0.2457, best-of-4 0.2673, beam 0.2682, zero-order 0.2689, MCTS 0.2885, ε-greedy 0.2973;
  - NFEs on EDM: 18 naive, 72 best-of-4, 144 beam, 1,440 ZO/ε-greedy, 3,888 MCTS, so baselines are **not compute-matched**;
  - stays within 0.002 of its best with an average K_t of 4.
  — [Noise trajectory search](https://arxiv.org/html/2506.03164)
- **Classical search** (Zhang et al., ICLR 2026) does global BFS and DFS particle search plus local annealed Langevin with verifier gradients.
  - ImageReward on SD v1.5 at N=8: BoN 0.896, FK 0.926, TreeG 1.023, DAS 1.052, BFS 1.087.
  - On ImageNet (an unconditional model guided by a classifier) at N=8: BoN-single FID 155.7 and 35.8% accuracy, against BFS-double FID 118.2 and 55.9%. "Double" means separate verifiers for local gradient steps and global selection, which also used about 2x less compute.
  - On mazes, TFG's local search alone plateaus.
  — [Zhang et al.](https://arxiv.org/html/2505.23614v2)
- DSearch (Li, Uehara et al. 2025) is a tree search with beam and width that adapt over timesteps, plus lookahead. Reported to beat best-of-N, DPS and SVDD on reward in bio, molecule and image tasks. Numbers not verified; a third-party summary notes little compute-cost analysis — [DSearch arXiv](https://arxiv.org/html/2503.02039v1), [summary](https://www.themoonlight.io/review/dynamic-search-for-inference-time-alignment-in-diffusion-models)

### Inferences
- **FK steering is the best-matched upgrade for this repo.** It keeps the existing gradient-guided sampler as the proposal and adds resampling of k particles on a potential. The evidence is the single largest compute-to-reward step reported for gradient guidance plus search (FK Table 3). Recommended configuration:
  - Evaluate the potential on the x0 prediction the sampler already computes.
  - Use a tempered or "difference" potential. Use a *different* CLIP or ensemble from the guide (Zhang's "double verifier"; DAS's tempering).
  - Resample only inside the guidance window, e.g. at 4–6 points.
- **Memory for particles:** each particle's state is 3×256×256 (0.75 MiB in fp32). Running particles one at a time keeps the baseline peak of about 5.3 GiB. Time scales linearly: k=4 ≈ 80 s, k=8 ≈ 160 s on the 2070 SUPER, plus a few CLIP forwards per resampling point. These are estimates.
- **The unconditional ADM may benefit more than the published T2I models.** It has no class or text to bring the subject, so early composition is the weak point. Resampling at high noise directly selects for compositions CLIP likes, and best-of-N only does that at full-sample cost.
- **Random search stays the baseline.** Ma et al. found local search does not beat it at equal compute. Any new method should be compared with `-best N` at equal wall-clock.
- **Tree search (DTS*, ε-greedy) is the expensive end.** It is gradient-free, so it ignores the CLIP gradient the repo already has. That makes it likely less efficient here than FK with gradient proposals. This is an inference: there is no head-to-head in the literature with gradient proposals.

### Gaps
- No search or SMC paper was found that uses OpenAI's 256px unconditional ADM with CLIP. FK's ImageNet experiment model was not identified in the fetched text.
- Ma et al.'s ImageNet scaling curves are figures only. No numeric reward-at-NFE values for ImageNet were extracted.
- DTS reward-vs-NFE numbers for T2I appear only in figures.
- DAS's best-of-N and compute comparisons were in the truncated part of the paper.

## 3. Recurrence and restarts: Universal Guidance self-recurrence, FreeDoM time-travel, Restart sampling, TFG recurrence, annealed Langevin/predictor-corrector

### Takeaway
Recurrence (re-noise to the same level and redo the guided step, k times) is the method with direct evidence on **unconditional ImageNet ADM with a weak, clean-image guide**. TFG's ImageNet label guidance went from 40.9% to 52.3% to 59.8% accuracy at N_recur 1, 2 and 4, with FID 177, 163 and 165. Universal Guidance used k=5–10 with CLIP on OpenAI's unconditional ImageNet model. FreeDoM used 10 repeats between t=800 and t=500 on ImageNet. Cost is linear in k, and only the guided window needs it. This is also the most plausible explanation for the 1000-step sampler's edge over the 60-call sampler: it makes many more small guided updates during composition.

### Cited Findings
- **TFG hyperparameters:**
  - N_recur, recurrence, about N_recur times the compute;
  - N_iter, gradient steps on x̂0 ("mean guidance");
  - ρ, gradient with respect to x_t ("variance guidance");
  - μ, mean-guidance strength;
  - γ̄, Gaussian smoothing of the predictor ("implicit dynamics").
  - An "increase over time" structure for ρ and μ worked best — [TFG, Ye et al. NeurIPS 2024](https://arxiv.org/html/2409.15761)
- **TFG recurrence ablation** (accuracy / FID):

  | N_recur | ImageNet | CIFAR10 | Fine-grained |
  |---|---|---|---|
  | 1 | 40.9 / 177 | 52.0 / 91.7 | 1.27 / 256 |
  | 2 | 52.3 / 163 | 66.8 / 88.7 | – |
  | 4 | 59.8 / 165 | 77.1 / 73.9 | 2.24 / 259 |

  N_iter plateaus (4 is enough). FID worsens at very large N_recur or N_iter (about 10) — [TFG](https://arxiv.org/html/2409.15761)
- TFG's ImageNet model is "a larger U-Net model trained on ImageNet-1k images" from Dhariwal & Nichol, guided by an off-the-shelf clean-image classifier (ResNet/ViT) and evaluated with DeiT-small. The authors conclude the main difficulty is "the lack of a smoothed classifier for noisy inputs". Added noise matters more than the number of Monte Carlo samples — [TFG](https://arxiv.org/html/2409.15761)
- **Universal Guidance** combines:
  - forward guidance, the loss on predicted x0;
  - backward guidance, m gradient steps on Δx0;
  - per-step self-recurrence, which re-injects noise and repeats each step k times.
  - For CLIP guidance on OpenAI's unconditional ImageNet model: k = 10, 5, 10 with strength w·√(1−ᾱ_t), w = 2, 5, 2. On Stable Diffusion with CLIP, k=8.
  - No quantitative ablation of k. Time is linear in k.
  — [Universal Guidance, Bansal et al. 2023](https://arxiv.org/html/2302.07121)
- **FreeDoM:**
  - energy guidance alone struggles on large domains like ImageNet, which the authors attribute to poor guidance;
  - its "time-travel" steps back j steps and resamples, repeated at each timestep;
  - it is applied only in the "semantic stage": on ImageNet, 10 repeats between t=800 and t=500;
  - for SD style guidance, 3 repeats; not needed for faces.
  — [FreeDoM, Yu et al. ICCV 2023](https://arxiv.org/pdf/2303.09833), [supplement](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Yu_FreeDoM_Training-Free_Energy-Guided_ICCV_2023_supplemental.pdf)
- **Restart sampling** alternates K cycles of large forward noise over [t_min, t_max] with a deterministic ODE back down. It contracts accumulated error more strongly than small per-step SDE noise.
  - CIFAR-10 FID 1.90 at 43 NFE against Heun ODE 1.97 at 63.
  - On SD v1.5 with classifier-free guidance, better FID at matched CLIP and aesthetic score, especially at high guidance weight. Example: w=8, t_min=0.1, t_max=2, K=2.
  - K should be larger for weaker models.
  - No reward-guided experiments.
  — [Restart, Xu et al. NeurIPS 2023](https://arxiv.org/html/2306.14878)
- **Annealed Langevin local search (Zhang et al., ICLR 2026)** interleaves verifier-gradient Langevin steps with DDIM/DDPM steps, and frames recurrent training-free guidance as Langevin MCMC.
  - TFG-style local search alone plateaus on mazes.
  - Local steps combined with best-of-N or BFS advance the Pareto front.
  - Gradient guidance "over-optimizes the verifier", fixed with a separate global verifier.
  — [Zhang et al.](https://arxiv.org/html/2505.23614v2)

### Inferences
- **Recurrence is the cheapest high-evidence change for this repo.** Repeat each guided step 2–4x inside the composition part of the window: re-noise from the solver's own next state back to the current level, then redo the step. Approximate cost at k=2 over the whole window is about 2x (~40 s); limiting it to the first half of the window costs less (estimate). TFG's ImageNet table shows about +11 accuracy points for 2x and +19 for 4x with a clean-image guide. A clean-image guide on noisy x̂0 is exactly the CLIP situation here.
- **Watch the multistep solver.** DPM-Solver++(2M)'s second-order correction uses the previous step's x0. After a recurrence the "previous" x0 is stale, so either fall back to first order on recurred steps or recompute. This is a design note, not from the literature.
- **Restart is a stronger, coarser version of the same idea.** Re-noise over a whole interval of the guided window K times with the guided sampler. That is close to FreeDoM's time-travel over t=800→500. The published evidence is for FID with classifier-free guidance, not for CLIP reward.
- **Recurrence compounds with search.** Zhang et al. find local refinement alone plateaus but compounds with global search. So recurrence 2x combined with FK k=4 or best-of-4 is a reasonable 8x configuration. This is an inference from the maze and ImageNet results.

### Gaps
- The exact TFG ImageNet checkpoint (resolution, the unconditional 256 checkpoint or not) and its step count were not stated in the fetched text. It is very likely an OpenAI guided-diffusion model, but that is unverified.
- Universal Guidance does not state resolution in the fetched text, and gives no quantitative k ablation.
- No quantitative CLIP-score results were found for FreeDoM time-travel or for Restart with a CLIP guide.

## 4. Guidance formulations 2024–2026: TFG search, DSG, MPGD, DSearch, DAS, annealed importance guidance

### Takeaway
On unconditional ImageNet, TFG's unified search is the strongest training-free guidance with published numbers (40.9% accuracy against DPS 38.8, UGD 25.5, FreeDoM 19.7, MPGD 6.8). MPGD's Jacobian-free shortcut is weak there. DSG's spherical step closely resembles the repo's existing noise-relative cap, so little is left to gain from it. DAS is the most principled reward-tilted SMC. Annealed importance guidance targets reward–diversity trade-offs, which is not the main problem here.

### Cited Findings
- **TFG against other methods** (accuracy % / FID): ImageNet label guidance DPS 38.8 / 193, LGD 11.5 / 210, FreeDoM 19.7 / 200, MPGD 6.80 / 239, UGD 25.5 / 205, TFG 40.9 / 176. Fine-grained birds TFG 1.27 against UGD 1.07. TFG is best on 13 of 14 settings — [TFG](https://arxiv.org/html/2409.15761)
- TFG's hyperparameter search is a beam search over ρ̄, μ̄ and γ̄. It starts small, doubles one at a time, and keeps the top K. It runs on about 1/8 of the sample size, for at most 6 steps, with accuracy as the metric. Chosen ImageNet values vary per target, e.g. (ρ̄, μ̄, γ̄) = (2, 0.5, 0.1), (1, 4, 1), (0.5, 2, 0.1) (read from Table 11; check column alignment) — [TFG](https://arxiv.org/html/2409.15761)
- **DSG** constrains the guided step to a sphere of radius √n·σ_t around the unconditional mean. The step is x_{t−1} = μ_θ(x_t) − √n·σ_t·∇L/‖∇L‖, blended with the sampled noise direction by g_r ∈ [0.05, 0.2]. It adds "almost no extra" compute — [DSG, Yang et al. ICML 2024](https://arxiv.org/html/2402.03201)
- DSG's results:
  - Inpainting with Dhariwal & Nichol ImageNet 256 and FFHQ models: ImageNet SSIM/LPIPS 0.879 / 0.116 against DPS 0.828 / 0.189.
  - At 20 steps on FFHQ: SSIM 0.812 against 0.408 for DPS.
  - FaceID on CelebA-HQ: FreeDoM FID 39.61, FreeDoM+DSG 34.29.
  - No CLIP text-guidance benchmark.
  - Authors acknowledge reduced diversity.
  — [DSG](https://arxiv.org/html/2402.03201)
- **DAS** (tempered SMC, see section 2) gets comparable or superior target rewards to fine-tuning without over-optimisation, and improves held-out rewards (HPSv2, ImageReward) "by up to 20%" in the online setting — [DAS](https://arxiv.org/html/2501.05803)
- **Annealed Importance Guidance (AIG)** (Jena et al. 2024) is an inference-time regulariser that mixes the base and reward-tuned scores with a time-varying weight, stronger regularisation early. It reports Pareto-optimal reward-diversity trade-offs on Stable Diffusion. Details come from search snippets only — [AIG, arXiv 2409.06493](https://arxiv.org/html/2409.06493v1)
- **DSearch** is described in section 2 — [DSearch](https://arxiv.org/html/2503.02039v1)

### Inferences
- **The repo's cap is already DSG-like.** It limits guidance RMS to a fraction (0.08) of the noise and is applied only in a window. DSG fixes the step at the sphere radius and blends direction with g_r = 0.05–0.2. The relevant DSG lesson is to project onto the sphere around the unconditional mean rather than add to the stochastic sample. That could allow larger effective steps at the same off-manifold risk. Worth an A/B test, but expect a small effect.
- **Do not adopt MPGD's Jacobian-free shortcut for this model.** The ImageNet result (6.8% against 38.8% for DPS) supports the repo's choice to backpropagate through the U-Net, at least while guidance must set the composition.
- **Adopt TFG's search procedure for tuning, not its full parameterisation.** Beam-search the guidance rate, window and recurrence count on a small held-out prompt subset, optimising a held-out judge rather than the guiding CLIP. This mirrors how the repo already validates. Adding TFG's γ̄ smoothing (evaluate CLIP on several noise-perturbed x̂0) is a cheap extra knob, because CLIP cutouts already average.

### Gaps
- No 2024–2026 paper found that reports CLIP **text** guidance numbers on OpenAI's unconditional ImageNet 256 ADM. The closest are TFG (classifier labels) and Universal Guidance (CLIP, qualitative only).
- AIG's numbers and exact formulation were not verified from full text.
- No ImageNet or CLIP results for DSearch were extracted.

## 5. Verifiers and judge hacking: ensembles and what fits on 8 GB

### Takeaway
Selecting or steering with the same CLIP that guides inflates its own score and can lower other metrics. Ma et al. report this "verifier hacking" (CLIP search: CLIPScore 0.71 to 0.82 while aesthetic fell 5.79 to 5.68). Zhang et al. fix it with a separate selection verifier. The best-performing verifiers in compute-matched studies are rank-averaged ensembles: in Ma et al., ImageReward was best per category and the ensemble best on the LLM grader. On 8 GB:
- **co-resident with the U-Net:** CLIP variants, the LAION aesthetic head, ImageReward and probably PickScore/HPSv2 (CLIP-H, ~1B parameters);
- **only sequentially:** HPSv3 (7B Qwen-VL);
- **too large:** VQAScore-XXL. The repo's held-out-judge `-best` is already the right structure.

### Cited Findings
- **Verifier hacking, Ma et al.** (FLUX, 2,880 NFEs):
  - Aesthetic search: aesthetic 6.38, CLIPScore 0.71 to 0.69.
  - CLIPScore search: CLIPScore 0.82, aesthetic 5.79 to 5.68.
  - ImageReward search: ImageReward 1.58, LLM grader 87.09.
  - Rank-averaged ensemble: LLM grader 88.18.
  - On T2I-CompBench, ImageReward beats CLIP and Ensemble as a search verifier in every category, and Aesthetic is flat or negative.
  — [Ma et al.](https://arxiv.org/html/2501.09732)
- **Pixel ImageNet hacking:** searching against CLIP or DINO logits raises FID after enough iterations while IS keeps rising, and recall drops — [Ma et al.](https://arxiv.org/html/2501.09732)
- **Double verifier:** separate verifiers for local gradient steps and global selection gave ImageNet FID 118.2 against 155.7 and accuracy 55.9% against 35.8% at N=8, with higher MSP (fewer OOD samples) — [Zhang et al.](https://arxiv.org/html/2505.23614v2)
- ReNO's single-reward optimisation (CLIP or ImageReward) lowers aesthetic score, and the 4-reward ensemble raises all metrics — [ReNO](https://arxiv.org/html/2406.04312)
- **HPSv3** is a VLM reward model from a Qwen2-VL-7B-Instruct training config; README metadata inconsistently says Qwen2.5-7B. Preference accuracy:

  | Benchmark | HPSv3 | HPSv2 | PickScore | ImageReward |
  |---|---|---|---|---|
  | HPDv3 | 76.9 | 65.3 | 65.6 | 58.6 |
  | HPDv2 | 85.4 | 83.3 | – | – |

  No VRAM figure given; flash-attn recommended — [HPSv3 model card](https://huggingface.co/MizzenAI/HPSv3/blob/main/README.md)
- The HPSv3 authors report less reward hacking when HPSv3 is the RL reward (self-reported) — [HPSv3 paper](https://arxiv.org/pdf/2508.03789). RewardDance reports that larger reward models make the generator less susceptible to reward hacking — [RewardDance](https://arxiv.org/pdf/2509.08826)
- **PickScore_v1** is fine-tuned from CLIP-ViT-H-14 (laion2B), 1.0B parameters, stored in F32 — [model card](https://huggingface.co/yuvalkirstain/PickScore_v1)
- **ImageReward** is trained on 137k expert comparisons and beats CLIP by 38.6%, Aesthetic by 39.6% and BLIP by 31.6% at preference prediction — [ImageReward repo](https://github.com/THUDM/ImageReward). Its BLIP backbone gives 65.14% preference accuracy against 62.98% for a CLIP backbone — [review summary](https://liner.com/review/imagereward-learning-and-evaluating-human-preferences-for-texttoimage-generation)
- **VQAScore's** recommended `clip-flant5-xxl` uses a CLIP ViT-L-336 vision encoder plus FlanT5-XXL. The project recommends a 40 GB GPU for the larger scorers — [t2v_metrics](https://github.com/linzhiqiu/t2v_metrics), [VQAScore paper](https://arxiv.org/pdf/2404.01291)
- Local: the U-Net's backward peak at 256px is 4.94 GiB, and the repo's whole unconditional pipeline peaks at 5.3 GiB — measured, and [README](/home/sam/GitHub/clip-guided-diffusion/README.md)

### Inferences
- **Recommended verifier roles for this repo:**
  - Guide with the default CLIP, or an ensemble.
  - Steer or resample (FK potential, tree search) with a *different* CLIP family, such as SigLIP2 or PE-Core, already supported.
  - Make the final pick with a rank-averaged ensemble disjoint from the guide: a held-out CLIP plus ImageReward or PickScore plus an aesthetic head.
  - Report with yet another held-out judge.
  - This extends the repo's existing held-out `-best` judges along the lines of Zhang's double verifier and Ma's ensemble.
- **8 GB fit** (estimates from parameter counts):
  - PickScore and HPSv2 (ViT-H/14, ~1B): about 2 GB in fp16. They fit alongside the 5.3 GiB pipeline only if run as forward-only evaluators at resampling or selection points.
  - ImageReward (BLIP-based, size not confirmed here) is smaller.
  - HPSv3 (7B) needs about 15 GB in bf16. It would need 4-bit quantisation (about 5 GB) and the U-Net unloaded, so final selection only.
  - VQAScore-XXL does not fit; a smaller VQAScore variant was not checked.
  - The 12 GB RTX 4070 could hold HPSv3 in 8-bit after the U-Net is unloaded (estimate).
- **Domain caveat:** ImageReward, PickScore and HPS were trained on 512px+ text-to-image outputs of SD-era models. Their calibration on 256px ImageNet-style images from an unconditional model is unknown. A CLIP-family judge ensemble may be more trustworthy here than a preference model, until checked against human picks.

### Gaps
- No measured VRAM figures found for ImageReward, HPSv2, HPSv3 or PickScore inference. All 8 GB fit statements are estimates from parameter counts.
- No study found on verifier calibration for 256px pixel-ADM outputs.
- Search snippets mentioned conflicting evidence on whether reward ensembles only partially mitigate hacking. The sources could not be pinned to specific papers, so this is excluded.
- No paper directly quantifies the inflation from "guide and select with the same CLIP" in CLIP-guided pixel diffusion.

## 6. Expected compute–quality curves (4x, 16x, 64x) and ranked assessment

### Takeaway
Reported scaling is roughly log-linear for best-of-N and particle methods. Ma et al.'s FLUX LLM grader gained +0.9 at 4 iterations, +2.1 at 16 and +3.8 at 64 (84.29 to 85.17, 86.42, 88.08). The steepest early gains come from **combining** gradient guidance with resampling (FK k=4 with gradient proposals: ImageReward +0.62 over gradient guidance at about 2.75x the time) and from **recurrence** on a weak guide (TFG ImageNet: +11.4 points at 2x, +18.9 at 4x). Noise optimisation buys the largest reward jumps on learned rewards, but only at 16–64x budgets, and it carries the highest hacking risk.

### Cited Findings
- **Best-of-N / random search scaling** (FLUX + PixArt-Σ, Verifier Ensemble, DrawBench LLM grader):

  | Model | Baseline | 4 iters | 16 iters | 64 iters |
  |---|---|---|---|---|
  | FLUX | 84.29 | 85.17 | 86.42 | 88.08 |
  | PixArt-Σ | 84.67 | 85.62 | 86.27 | 87.55 |

  PixArt-Σ with search at about 1/10 of FLUX's compute beats FLUX without search — [Ma et al.](https://arxiv.org/html/2501.09732)
- **FK steering:** SDXL k=4 against best-of-4 on ImageReward: 1.298 against 1.236, at about 4x a single sample (43.5 s against 11.5 s). On SD v1.5, FK with gradient proposals reached IR 1.290 in 55 s, against 0.668 for gradient guidance in 20 s. Scaling k from 4 to 8 raised GenEval from 0.633 to 0.648 on SDXL with the max potential — [FK steering](https://arxiv.org/html/2501.06848)
- **TFG recurrence:** ImageNet accuracy 40.9 → 52.3 → 59.8 at 1x → 2x → 4x — [TFG](https://arxiv.org/html/2409.15761)
- **DNO:** most of the gain arrives by about 1 min of A800 time with 15 steps (aesthetic 5.37 → 7.50), converging in 3–5 min. With a 10-min budget it beats best-of-N "by a large margin" — [DNO](https://arxiv.org/html/2405.18881)
- **DOODL:** CLIPScore +0.028 (0.261 → 0.289) at 20 min and 40 GB on SD2.1 — [ReNO's comparison](https://arxiv.org/html/2406.04312)
- **ε-greedy trajectory search:** SD CLIP 0.2457 → 0.2973 (+0.05) at 80x NFEs. Best-of-4 (4x) reached 0.2673 (+0.022) — [Ramesh & Mardani](https://arxiv.org/html/2506.03164)
- **DTS***: matches best-of-N reward with 2x (v1) or 5x (current version) less compute — [DTS](https://arxiv.org/html/2506.20701), [NeurIPS page](https://neurips.cc/virtual/2025/poster/120056)
- **Repo's own data:**
  - `-best N` gained +0.015 held-out CLIP on the class-conditional model (from the task brief).
  - The 1000-step sampler (18x the time) scored 0.331 against 0.316, a difference not significant on 6 prompts.
  — [README](/home/sam/GitHub/clip-guided-diffusion/README.md)

### Inferences

#### Ranked assessment for `256x256_diffusion_uncond.pt` + CLIP on an 8 GB RTX 2070 SUPER (estimates, 1x = 20 s)

| Rank | Method | Budget | Peak VRAM (est.) | Expected effect (estimate) | Evidence strength for this setting |
|---|---|---|---|---|---|
| 1 | **Self-recurrence / time-travel in the composition part of the guided window** (TFG N_recur, UGD k, FreeDoM) | 2–4x (40–80 s) | 5.3 GiB, unchanged | Probably recovers the 60-call vs 1000-step gap (0.316 → ~0.33). TFG saw +11–19 points of guide accuracy on uncond ImageNet ADM | High: same model family, same weak-guide problem; UGD and FreeDoM used it with CLIP on uncond ImageNet |
| 2 | **FK steering / tempered SMC with the existing guided sampler as proposal**, potential from a *different* CLIP or ensemble, resample 4–6 times in the window, k=4–8 | 4–8x (80–160 s) | 5.3 GiB plus a forward-only verifier (≤2 GB) | Larger than best-of-k at the same k. The FK gradient-proposal result suggests well above best-of-4 | Medium: strong T2I evidence; no pixel-ADM/CLIP study |
| 3 | **Best-of-N with held-out ensemble judges** (already `-best`), stacked on top of 1 and 2 | N× | unchanged | About log-linear; repo measured +0.015 on class-conditional | High as a baseline; Ma et al. say it is hard to beat at equal compute |
| 4 | **DNO-style noise optimisation** of x_T plus injected SDE noises through a 15–25-step unguided or stop-gradient-guided sampler, step checkpointing, chi-norm or DOODL renormalisation, cutout augmentation, final guided pass with the optimised noises | 16–64x (5–20 min) | measured 4.95 GiB for the U-Net chain + CLIP backward ≈ 5.5–6 GiB | Potentially the largest CLIP gain, but the highest adversarial and hacking risk. DOODL's CLIP result was only +3% | Low to medium: no pixel-ADM CLIP results; fits memory (measured) |
| 5 | **Tree / trajectory search** (DTS*, ε-greedy, DSearch) | 16–80x | unchanged | Gradient-free, so it wastes the CLIP gradient. Gains reported (SD CLIP +0.05 at 80x) are not compute-matched | Low |
| 6 | **Restart over an interval of the guided window** | 1.5–3x | unchanged | Mostly FID and realism. Overlaps with rank 1 | Low for reward |
| – | DSG projection (small effect, repo cap is similar). MPGD (weak on ImageNet). Golden Noise / InitNO (need training, or text cross-attention) | – | – | Not recommended | – |

#### Compute-matched expectations (estimates, extrapolated from the papers above, not measured on this model)
- **4x (~80 s):** recurrence 2x with best-of-2, or FK k=4. The literature suggests FK k=4 with gradient proposals is the best single 4x option for prompt alignment.
- **16x (~5 min):** FK k=8 with recurrence 2x, or DNO for about 30 iterations through 20 steps. Prefer FK plus recurrence, which is lower risk and more diverse. Try DNO as an opt-in.
- **64x (~20 min):** best-of-4 over (FK k=8 + recurrence 2x), or DNO for about 130 iterations with an ensemble reward. Expect diminishing returns and rising FID and diversity loss (Ma et al.'s ImageNet hacking), unless selection uses held-out judges.

#### Implementation notes
- **Determinism:** FK resampling uses multinomial draws. Draw them from a dedicated `torch.Generator` seeded from the run seed, so `th.randn` order elsewhere is unchanged (CLAUDE.md requires byte-identical seeds). Recurrence adds `th.randn` calls, which changes every image, so README GIFs would need regenerating.
- **The solver's second-order state:** after a resample, each new particle inherits its parent's `Previous` (x0, h). After a recurrence, recompute it or take a first-order step.
- **Backprop through the sampler:** wrap each sampler step in `torch.utils.checkpoint(..., use_reentrant=False)`. At 256px batch 1, inner `use_checkpoint` is not needed (4.95 GiB) unless a large verifier must co-reside, which costs about 1.5–2.3x time and brings memory to 2.2 GiB (measured).

### Gaps
- No compute-matched comparison exists in the literature among recurrence, FK steering, best-of-N and noise optimisation on a pixel-space unconditional ImageNet model with CLIP. Every number above for this model is an extrapolation and must be measured against the repo's held-out prompts.
- Reported curves use different verifiers, models and units (NFEs, wall time, iterations), so cross-paper multipliers are only roughly comparable.
- Measured timings are for the U-Net with a trivial loss. CLIP forward and backward per step (with 16 or 4 cutouts) adds time that was not measured here.

# Reward fine-tuning, preference alignment and distillation for OpenAI's 256x256 unconditional ADM, with CLIP and aesthetic rewards, on one 12 GB GPU

Scope: what each method optimises, how it copes with an unconditional model and a prompt-dependent reward, its memory and compute, reported gains, how it hacks the reward, and the recipe details needed to implement it on `256x256_diffusion_uncond.pt`. The papers were read from their arXiv PDFs. Numbers are quoted from them. Anything computed or guessed for this model is marked **(estimate)**.

## 1. Direct reward backpropagation (DRaFT, AlignProp, ReFL and successors): steps, memory tricks, VRAM, and fit on 12 GB

### Takeaway
Backpropagating a differentiable reward through only the last one or few sampler steps (DRaFT-1, DRaFT-LV, ReFL) is the most sample-efficient family, and the cheapest in memory. With LoRA and per-call gradient checkpointing, it fits a 553M-parameter pixel U-Net at 256 on 12 GB **(estimate)**. A full backward pass through this U-Net at batch 1 already runs in 5.3 GiB in the repo today. The catch is that a CLIP reward specifically has been reported to degrade image quality, and every method in this family collapses diversity unless regularised.

### Cited Findings
**This model, from the repo:**
- With the repo's flags (`num_channels` 256, `num_res_blocks` 2, attention at 32/16/8, learned sigma, linear schedule over 1000 steps, no `class_cond`), the U-Net has 552,814,086 parameters. Of these, 446.9M are in Conv2d layers, 54.3M in Linear layers and 51.4M in the attention Conv1d (qkv/proj) layers. Counted by instantiating it on the meta device. — [cgd/adm.py CHECKPOINT_FLAGS](/home/sam/GitHub/clip-guided-diffusion/cgd/adm.py)
- The repo already backpropagates CLIP through the U-Net on every guided step: `predict()` sets `requires_grad` on x and takes the gradient through `eps` to `x0`. The 256 unconditional run (60 model calls) takes 20.0 s and peaks at 5.3 GiB on an RTX 2070 SUPER. Gradient checkpointing turns on only above 256² pixels per batch. — [cgd/sampler.py](/home/sam/GitHub/clip-guided-diffusion/cgd/sampler.py), [README Speed table](/home/sam/GitHub/clip-guided-diffusion/README.md), [cgd/adm.py](/home/sam/GitHub/clip-guided-diffusion/cgd/adm.py)
- In the unconditional sampler, guidance runs only between fractions 0.1 and 1.0 of the schedule (`UNCONDITIONAL_GUIDANCE_WINDOW = (0.1, 1.0)`), so the final low-noise steps are unguided. — [cgd/sampler.py](/home/sam/GitHub/clip-guided-diffusion/cgd/sampler.py)

**DRaFT (Clark et al., ICLR 2024):**
- Covers DRaFT (full chain), DRaFT-K (stop-gradient so only the last K steps get gradients) and DRaFT-LV (the last-step gradient averaged over several noise draws). ReFL is the special case of stopping early at a random late step and scoring the one-step x̂0. ReFL with m=1 equals DRaFT-1. — [DRaFT](https://arxiv.org/abs/2309.17400)
- The setup:
  - Model and sampler: SD 1.4, 50 DDIM steps, CFG 7.5 at both train and test time.
  - Small-scale runs: LoRA rank 8, lr 4e-4, batch 4, 2k steps, about 1.5 h on 4 TPUv4.
  - Large-scale runs: LoRA rank 32, lr 2e-4, batch 16, 10k steps, about 8 h on 16 TPUv4.
  - Optimiser and precision: AdamW with weight decay 0.1, which with LoRA pulls toward the pretrained weights. Base weights in bf16, LoRA in fp32. DRaFT-LV uses n=2.
  - LoRA placement: on the feedforward and cross-attention layers of the U-Net only. — [DRaFT](https://arxiv.org/abs/2309.17400)
- Memory: LoRA plus gradient checkpointing. For small K (e.g. K=1), "the memory cost of unrolling is small, and we do not need gradient checkpointing." — [DRaFT](https://arxiv.org/abs/2309.17400)
- Truncation helps both compute and results:
  - Gradient norms explode as K grows, and aesthetic performance degrades for K>10. Gradient clipping narrows the gap but does not close it.
  - K∈{1,5,10,30,50} was ablated on HPSv2, and smaller K was better in both training time and final reward.
  - DRaFT-1 still changes the whole sampling chain, not only the last step.
  - DRaFT-LV learns about 2× faster than DRaFT-1, adds about 10% compute, and reached the best HPSv2 in their comparison. — [DRaFT](https://arxiv.org/abs/2309.17400)
- Sample efficiency on the LAION aesthetic predictor (45 animal prompts): DRaFT-LV/DRaFT-1 rise from 5.4 to well above 8 within a few thousand reward queries. DDPO (fixed implementation, per the authors' correspondence) reaches 7.4 after 50k reward queries. — [DRaFT](https://arxiv.org/abs/2309.17400)
- Collapse: "methods initially produced improved images, but eventually collapsed to producing very similar high-reward images" (aesthetic reward around 11). — [DRaFT](https://arxiv.org/abs/2309.17400)
- **CLIP reward specifically** (their Appendix B.10): with r = CLIP similarity(x0, c), "CLIP scores improved, but … image quality degraded after many steps of training" (samples shown from step 0 to 3000). — [DRaFT](https://arxiv.org/abs/2309.17400)
- DRaFT also learned sampler inputs by backprop, such as a per-step guidance weight w(t) at a 50× larger lr than the LoRA, and negative-prompt embeddings. This "did not improve results much at large scale." — [DRaFT](https://arxiv.org/abs/2309.17400)
- Adding CFG to ReFL during training gave similar results to ReFL without it. The authors suggest fine-tuning compensates for missing guidance, as guided models can be distilled into unguided ones. — [DRaFT](https://arxiv.org/abs/2309.17400)

**AlignProp (Prabhudesai et al., 2023/2024):**
- Full backprop through all 50 steps with a randomized truncation K ~ Uniform(0, 50), plus LoRA (about 800K trainable instead of 800M parameters) and gradient checkpointing. — [AlignProp](https://arxiv.org/abs/2310.03739)
- Memory: one SD training step at batch 1 needs about 20 GB. A naive end-to-end chain would need about 1 TB. LoRA alone brings that to about 500 GB, and checkpointing to about 15 GB. — [AlignProp](https://arxiv.org/abs/2310.03739)
- Untruncated full backprop made the model "consistently generate the same image," which is why truncation is randomized. — [AlignProp](https://arxiv.org/abs/2310.03739)
- Against DDPO on 4 A100s at the same batch size:
  - Speed: HPSv2 2.8 in 48 min against about 23 h.
  - Efficiency: "25 times more data-efficient."
  - Held-out-class rewards: aesthetic 8.71 against DDPO 6.82 and SD 5.64. HPS 3.32 against 2.93 and 2.86. — [AlignProp](https://arxiv.org/abs/2310.03739)
- The repo used 4×A100 40 GB and suggests K=1 when memory is a bottleneck. — [AlignProp GitHub](https://github.com/mihirp1998/AlignProp)

**ReFL (ImageReward; Xu et al., NeurIPS 2023):**
- Sampling: 40 PNDM steps. A random step among the last ten (t∈[30,40] in their notation, [T1,T2]=[1,10] in the hyperparameters) is denoised in one shot to x̂0, which is scored with ϕ=ReLU and λ=1e-3.
- Regularisation: the pretraining loss, with half of each batch of 128 being pretraining data.
- Compute: full fine-tuning in half precision at lr 1e-5 on 8×A100 40 GB. — [ImageReward/ReFL](https://arxiv.org/abs/2304.05977)

**Newer comparisons:**
- In Adjoint Matching's text-to-image benchmark, DRaFT-1 was the strongest baseline, with ImageReward as the reward. It still had the lowest diversity of all the methods compared. — [Adjoint Matching](https://arxiv.org/abs/2409.08861)
  - DRaFT-1 (ODE sampling): ClipScore 30.95, PickScore 19.37, held-out HPSv2 24.37, DreamSim diversity 27.39.
  - Base model: 28.32, 18.15, 17.89, diversity 56.53.
  - DRaFT-40 (backprop through all 40 steps): 30.07 / 19.45 / 24.06, diversity 36.53.
  - ReFL: 30.06 / 19.07 / 23.06, diversity 32.69.
- DiffusionOPSD (Aug 2026 preprint) states that "ReFL backpropagates a differentiable reward through a differentiable clean-output" prediction, and uses ReFL as a baseline. — [DiffusionOPSD](https://arxiv.org/abs/2608.24646)

### Inferences
- **Memory on 12 GB (estimate):**
  - Full fine-tuning of 553M parameters with fp32 AdamW needs about 2.2 GB of weights, 2.2 GB of gradients and 4.4 GB of Adam state, so about 8.8 GB before activations, CLIP, or a frozen reference copy for KL. That is too tight for 12 GB.
  - 8-bit Adam (about 5.5 GB of state plus weights) or LoRA makes it fit.
  - LoRA over this U-Net's attention and Linear layers touches only about 106M of the 553M weights. Most parameters are 3×3 convolutions, so a conv-LoRA (LoCon-style) or 8-bit-Adam full fine-tune may be needed for capacity. No paper tests which is better for ADM.
- **Activation memory (estimate):**
  - A DRaFT-1/ReFL step is the same graph as one of the repo's guided steps (U-Net forward and backward plus CLIP on cutouts) plus LoRA parameter gradients. The guided step peaks at 5.3 GiB on 8 GB in fp16, so batch 1–2 at 256 should fit 12 GB in bf16, and batch 4 with checkpointing.
  - DRaFT-K or AlignProp over many steps fits if each U-Net call is checkpointed, keeping only the 3×256×256 latent per step. The cost is about one extra forward per call.
- **Compute (estimate):** the repo samples one guided 256 image in about 20 s on the 2070 SUPER. A DRaFT small-scale run (2k steps × batch 4 = 8k samples) with the guided sampler in the loop would be about 1–2 GPU-days on the 4070. It would be faster without guidance in the loop, since unguided sampling skips every U-Net backward.
- **CLIP reward risk:** DRaFT's CLIP-reward failure (quality degrades) and the repo's own finding that the aesthetic term on ADM "costs score and adds saturation and clipping" ([README](/home/sam/GitHub/clip-guided-diffusion/README.md)) predict that naive DRaFT with CLIP ViT-B/32 as reward will amplify the adversarial high-frequency patterns guidance already produces. Use rewards different from the guide and judges different from both (see section 6).

### Gaps
- No published DRaFT, AlignProp or ReFL run on a pixel-space ADM, an ImageNet model, or an unconditional model. All reported runs are latent text-to-image (SD 1.4/1.5/SDXL) with 50-step DDIM.
- No measured VRAM for LoRA DRaFT on ADM-256. The shared RTX 2070 SUPER had crashed (Xid 79, `nvidia-smi` found no device), so nothing was run on a GPU for these notes. The parameter count was computed on the meta device, with no GPU. Every other memory, time or throughput figure is either quoted from a source or marked as an estimate.
- Whether attention-only LoRA has enough capacity to shift an ImageNet prior toward arbitrary CLIP prompts is untested.

## 2. RL methods (DDPO, DPOK, GRPO-family, DiffusionNFT): sample efficiency against direct backprop

### Takeaway
Policy-gradient RL needs one to two orders of magnitude more reward queries than gradient-based methods when the reward is differentiable, as CLIP and the simulacra head are. Its one real advantage, KL regularisation and clean on-policy exploration, can be had more cheaply (sections 4 and 6). On an unconditional prior with a classifier reward, DDPO mode-collapses. DiffusionNFT is the most efficient 2025–26 RL variant, but it targets flow models at scale, and nothing shows it beating reward backprop on a differentiable reward.

### Cited Findings
- **DDPO:**
  - Treats each DDIM step as an action. Collects 256 samples per iteration, uses PPO clip range 1e-4, and has no KL term.
  - 50k samples took about 4 h on a TPU v4-64 pod.
  - Exploits rewards: the LLaVA alignment reward was beaten by a typographic attack, writing the text in the image. Samples drift "cartoon-like." Overoptimisation "does begin to happen within 25k reward queries" for one prompt. — [DDPO](https://arxiv.org/abs/2305.13301)
- **ddpo-pytorch:** with LoRA it "requires less than 10GB of GPU memory to finetune Stable Diffusion". The author used 8×A100 for about 4 h per 100 epochs, and warns the defaults need "a much larger number of samples per epoch" to work. — [ddpo-pytorch](https://github.com/kvablack/ddpo-pytorch)
- **Sample efficiency against backprop:**
  - Aesthetic: DDPO reached 7.4 after 50k reward queries, while DRaFT-LV passes that in a few thousand. — [DRaFT](https://arxiv.org/abs/2309.17400)
  - HPSv2: AlignProp reached 2.8 in 48 min, against about 23 h for DDPO on the same 4 A100s. — [AlignProp](https://arxiv.org/abs/2310.03739)
- **DPOK:** adds a KL penalty to online policy gradient (reward weight α, KL weight β). RL without KL "can generate lower-quality images." — [DPOK](https://arxiv.org/abs/2305.16381)
- **Unconditional priors (RTB paper):** on unconditional MNIST and CIFAR-10 diffusion priors with a classifier reward, DDPO mode-collapses. RL with tuned KL "yield[s] inaccurate inference," and RL without KL mode-collapses. — [RTB](https://arxiv.org/abs/2405.20971)
- **Flow-GRPO (2025):**
  - Recipe: trains on a 10-step "Denoising Reduction" schedule with group size G=24. Smaller groups (12, 6) were unstable. Run on 24 A800 GPUs.
  - On KL: "the KL constraint effectively prevents reward hacking … KL regularization is not empirically equivalent to early stopping." Without KL, "visual diversity" collapsed. — [Flow-GRPO](https://arxiv.org/abs/2505.05470)
- **DiffusionNFT (ICLR 2026):**
  - Does RL on the forward process: generations are split into positive and negative by reward, which defines an implicit improvement direction.
  - Needs "only clean images rather than sampling trajectories," works with any black-box solver, and is CFG-free.
  - Reported as up to 25× more efficient than FlowGRPO, with GenEval going from 0.24 to 0.98 in 1k steps. — [DiffusionNFT](https://arxiv.org/abs/2509.16117)

### Inferences
- With differentiable CLIP and aesthetic rewards, RL is the wrong default for a one-GPU budget. Tens of thousands of guided 256 samples at about 20 s each on the 2070 SUPER is several GPU-days per run before any gain **(estimate)**.
- DiffusionNFT is the RL method most compatible with this repo, because it can collect samples with the existing DPM-Solver++(2M) SDE and capped guidance, which is any solver, and trains only on clean images with a standard denoising-style loss. That makes it a "reward-weighted self-distillation of the guided sampler" (section 5c). It is still sample-hungry relative to backprop **(inference)**.

### Gaps
- No RL fine-tuning result exists for ADM ImageNet-256 or any 256 pixel-space unconditional model.
- DiffusionNFT's absolute GPU-hours and VRAM were not extracted. Its experiments are on SD3.5-M at multi-GPU scale.

## 3. Preference optimisation (Diffusion-DPO, D3PO, SPO, Diffusion-KTO, online DPO) and using CLIP or aesthetic scores to make the pairs

### Takeaway
DPO-family methods are stable and KL-anchored, and they can be trained from AI-feedback pairs ranked by CLIP or an aesthetic predictor. Diffusion-DPO showed this works, including for CLIP-ranked pairs. Each step costs only forward and backward passes of the denoising loss on stored pairs, so it fits 12 GB. But they move rewards far less than reward backprop, they need many pairs, and a single-reward ranker trades one metric against another.

### Cited Findings
- **Diffusion-DPO:**
  - Data: Pick-a-Pic v2, 851,293 pairs over 58,960 prompts.
  - Compute: effective batch 2048 pairs, 16 A100s at local batch 1 with 128 steps of gradient accumulation. β∈[2000, 5000]; SD1.5 used β=2000, SDXL β=5000.
  - Human evaluation: DPO-SDXL was preferred 70.0% of the time over SDXL-base on PartiPrompts. — [Diffusion-DPO](https://arxiv.org/abs/2311.12908)
- **Diffusion-DPO from AI feedback:**
  - Pairs were ranked by HPSv2, CLIP (OpenCLIP ViT-H/14), the LAION Aesthetic Predictor or PickScore, training SD1.5 at β=5000 for 1000 steps.
  - "Training for Aesthetics and CLIP improves those capabilities more specifically, in the case of Aesthetics at the expense of CLIP." "Tuning on Aesthetics preferences … achieves high Aesthetics scores but has lower text-image alignment as measured by CLIP."
  - Pseudo-labelled PickScore pairs (2000 steps) raised the PartiPrompts win rate from 59.8% to 63.3%. — [Diffusion-DPO](https://arxiv.org/abs/2311.12908)
- Diffusion-DPO notes that DRaFT could not improve image-text alignment with a CLIP reward, while "Diffusion-DPO can improve image-text alignment using CLIP preference." It also says reward-maximisation methods "suffer from mode collapse with over-training." — [Diffusion-DPO](https://arxiv.org/abs/2311.12908)
- As a reward maximiser, DPO barely moves the needle in Adjoint Matching's benchmark: ClipScore 27.77, PickScore 17.92, HPSv2 17.30, diversity 54.11, against the base model's 28.32 / 18.15 / 17.89 / 56.53. — [Adjoint Matching](https://arxiv.org/abs/2409.08861)
- **D3PO** fine-tunes directly from human feedback "without any reward model." — [D3PO](https://arxiv.org/abs/2311.13231)
- **SPO:**
  - At each step it samples a pool of x_{t−1} candidates from a shared x_t and picks a win/lose pair with a *step-aware preference model* trained to judge noisy images, because standard rewards are "designed for assessing clean images."
  - SPO improves aesthetics on SD1.5/SDXL and "converges much faster than DPO methods." — [SPO](https://arxiv.org/abs/2406.04314)
- **Diffusion-KTO** needs only per-image binary like/dislike labels, not pairs, and beat SFT and Diffusion-DPO on PickScore and ImageReward. — [Diffusion-KTO](https://arxiv.org/abs/2404.04465)

### Inferences
- **For this repo (estimate):**
  - Pair generation: for each training prompt, sample 2–4 candidates with the guided sampler, which is effectively the `-best` machinery, and rank them by a reward ensemble that excludes the held-out judges.
  - Training: run Diffusion-DPO on the unconditional U-Net, with prompt conditioning absent, so the "policy" is the unconditional model and only the pair ranking depends on the prompt (see section 5).
  - Cost: generating, say, 10k pairs costs about 20k guided samples, several GPU-days on the 4070 at about 20 s each. The DPO training is cheap.
- Because the model is unconditional, prompt-ranked pairs only teach it which *images* are better in general, as in a CLIP-aesthetic prior. They cannot teach prompt following unless the model gets a conditioning input (section 5b).
- SPO's step-aware ranking would need a noise-aware CLIP or aesthetic scorer. Scoring the one-step x̂0, as the repo's guidance does, is a cheap substitute, but SPO argues it is less reliable at high noise **(inference)**.

### Gaps
- No preference-optimisation result for ImageNet ADM, pixel space, or unconditional models.
- No published number for DPO trained on AI feedback from small CLIPs like ViT-B/32 or PE-Core-B-16.
- Online and iterative DPO for diffusion was not covered by a fetched primary source.

## 4. Stochastic optimal control and KL-regularised fine-tuning (Adjoint Matching, relative trajectory balance, DEFT) that avoids reward hacking and mode collapse

### Takeaway
Adjoint Matching is the principled choice. It converges to the reward-tilted distribution p_base(x)·exp(λ r(x)), has an explicit KL knob in λ, and its memoryless schedule for DDIM-type models is plain DDPM ancestral sampling, which ADM supports. It also reached the best reward-versus-diversity Pareto front against DRaFT-1, ReFL and DPO. It needs no backprop through the chain, only one VJP per step, but it costs several U-Net passes per sample. Relative trajectory balance and DEFT were demonstrated on unconditional priors, and DEFT on this exact ADM-256 model.

### Cited Findings
- **Adjoint Matching: objective and schedule.**
  - Reward fine-tuning is cast as stochastic optimal control. The "memoryless noise schedule … completely removes the dependency between noise variables and the generated samples," which gives "provable convergence to the tilted distribution."
  - For DDIM-family models, the memoryless schedule "recovers what we derived as the continuous-time limit of the DDPM generative process." After fine-tuning, the model can be sampled with any schedule, including the ODE. — [Adjoint Matching](https://arxiv.org/abs/2409.08861)
- **Adjoint Matching: algorithm (Algorithm 2, for DDIM/DDPM denoisers).**
  - Sample trajectories with DDPM.
  - Integrate a "lean adjoint" ODE backwards from ã_K = ∇r(X_K), using vector-Jacobian products of the *base* ε model.
  - Regress the fine-tuned model's control onto −σ(t)·ã_t, with X_t and ã_t detached. No gradient flows through the sampling chain. — [Adjoint Matching](https://arxiv.org/abs/2409.08861)
- **Adjoint Matching: recipe (Appendix G).**
  - Schedule and optimiser: K=40 steps; Adam at lr 2e-5 with β1=0.95, weight decay 1e-2 and gradient-norm clip 1; bf16.
  - Batch and prompts: effective batch 40 on 2×A100 80 GB; 40k prompts.
  - Timestep subsampling: the loss is computed on only 10 random timesteps from the first 75% of the schedule plus always the last 10. "Fine-tuning the last ten steps … well is critical."
  - Clipping: the loss is clipped at LCT = 1.6·λ².
  - Endpoint: the adjoint starts at ∇r of a final noiseless step.
  - σ(t): offset slightly larger than the theoretical schedule near t≈1, because that gave "substantially faster fine-tuning." — [Adjoint Matching](https://arxiv.org/abs/2409.08861)
- **Adjoint Matching: results** (512² latent flow-matching text-to-image model, ImageReward reward, 1000 test prompts):
  - λ=12500, ODE sampling: ClipScore 31.65, PickScore 19.76, held-out HPSv2 24.49, DreamSim diversity 37.24.
  - λ=1000: 31.41 / 19.57 / 23.29, diversity 43.10.
  - DRaFT-1: 30.95 / 19.37 / 24.37, diversity 27.39.
  - Base: 28.32 / 18.15 / 17.89, diversity 56.53.
  - The authors report Adjoint Matching has "the best Pareto fronts." DRaFT-1 "must rely on early stopping" to trade reward against diversity. — [Adjoint Matching](https://arxiv.org/abs/2409.08861)
- **Relative trajectory balance (RTB):**
  - Trains a posterior p(x|c) ∝ p_θ(x)·r(x) from an unconditional diffusion prior (MNIST, CIFAR-10, with classifier rewards) and supports off-policy training such as replay buffers.
  - It matches the posterior with good diversity, while DDPO mode-collapses and classifier guidance with a classifier on noiseless data gives biased samples. — [RTB](https://arxiv.org/abs/2405.20971)
- **DEFT:**
  - Uses "a pre-trained unconditional diffusion model with ∼500M parameters" on 256×256 ImageNet, the guided-diffusion unconditional model, and leaves it frozen.
  - It trains a small "generalised h-transform" network of 4–17% of the base size (about 23M parameters in one configuration) that takes x_t, the base model's Tweedie x̂0 and the likelihood gradient.
  - The supervised variant needs a small paired dataset, with ablations at 10, 100 and 200 images. Inpainting training took 3.9 h.
  - A stochastic-control variant targets reward-tilted distributions without paired data, using VarGrad or trajectory balance so the trajectory can be detached. — [DEFT](https://arxiv.org/abs/2406.01781)
- **TDPO-R** traces reward overoptimisation partly to a mismatch with the temporal inductive bias of multi-step denoising, finds that dormant neurons in its critic act as a regulariser, and resets active critic neurons periodically. — [TDPO-R](https://arxiv.org/abs/2402.08552)

### Inferences
- **Fit for this model.** ADM is an ε-prediction DDPM with 1000 linear-β steps, and the memoryless schedule is ancestral DDPM sampling, so Algorithm 2 applies directly. Adjoint Matching optimises the *unguided* tilted distribution: the CLIP term enters only through ∇r at the endpoint.
- **Memory (estimate):** the lean adjoint needs one base-model VJP per step with respect to x, the same op the repo's guidance does. The regression loss is an ordinary denoising-sized backward on about 20 timesteps per trajectory. Peak memory is about one U-Net backward plus the optimiser state, so a LoRA or 8-bit-Adam variant fits 12 GB at batch 1–2 with gradient accumulation.
- **Compute per sample (estimate):** a 40–60-step DDPM rollout, 40–60 VJPs and about 20 training forward/backward passes, so about 3–4× one guided sample. A 1000-iteration run at batch 8–16 is therefore about 2–5 GPU-days on a 4070. That sits at the top of the stated budget.
- **What the tilt can and cannot do (inference from the target).** The tilted distribution can only reweight images the ImageNet prior already assigns probability to. With a prompt-dependent r, it needs a prompt input (section 5), otherwise the target is the prompt-averaged tilt. This is the main reason a better ADM prior is still bounded by ImageNet content.

### Gaps
- No Adjoint Matching result with LoRA, on pixel space, on ImageNet, or with CLIP as the reward.
- RTB's image experiments are on MNIST and CIFAR-10 scale, with no reported VRAM at 256.
- DEFT's reward-tilted (stochastic-control) variant has only "initial experiments" reported, not at ImageNet-256.

## 5. Making an unconditional model follow text prompts: (a) per-prompt adaptation, (b) adding CLIP conditioning, (c) fine-tuning the guided sampler; work that reward-tunes a model later used with gradient guidance

### Takeaway
(b) is the structurally right fix, and the cheapest strong version is not reward-based. Add a zero-initialised CLIP-embedding input to ADM's existing class-embedding slot, or a small DEFT-style side network, and train it by supervised denoising on images with their CLIP *image* embeddings (language-free, as in LAFITE or KNN-Diffusion). Reward fine-tuning can follow. (c), DRaFT-K through the last unguided steps of the existing guided sampler, is cheap and has precedent in DRaFT's training with CFG, but no paper tests it with gradient guidance. (a), per-prompt tuning, works mechanically but costs minutes to hours per prompt and collapses easily.

### Cited Findings
- **Where to inject conditioning.** guided-diffusion's U-Net implements class conditioning as `self.label_emb = nn.Embedding(num_classes, time_embed_dim)`, added to the timestep embedding. The unconditional checkpoint simply lacks it. — [guided-diffusion unet.py (local copy)](/home/sam/GitHub/clip-guided-diffusion/.venv/lib/python3.13/site-packages/guided_diffusion/unet.py)
- **Side network on this exact model (DEFT).** DEFT conditions the frozen ∼500M ImageNet-256 unconditional model through a separate network of 4–17% of its size, with "the unconditional Tweedie estimate x̂0" as an input. It was trained in hours and beat DPS and RED-diff on KID for inpainting (0.29 against 15.28 and 0.86). — [DEFT](https://arxiv.org/abs/2406.01781)
- **Language-free text conditioning:**
  - LAFITE trains text-to-image models from image-only data by using CLIP image embeddings, perturbed with noise, as pseudo text features. — [LAFITE](https://arxiv.org/abs/2111.13792)
  - KNN-Diffusion trains on images only, conditioned on the CLIP image embedding plus k retrieved neighbour embeddings, and swaps in the text embedding at inference. — [KNN-Diffusion](https://arxiv.org/abs/2204.02849)
- **Precedent for training with guidance in the loop (CFG).** DRaFT trains and tests with CFG weight 7.5, and its ReFL+guidance variant performed like ReFL. DRaFT also learned a per-step guidance-weight schedule by backprop, which "did not improve results much at large scale." — [DRaFT](https://arxiv.org/abs/2309.17400)
- **Theory of guidance plus fine-tuning.**
  - Guo et al. show gradient-guided diffusion samples a regularised optimisation problem, and that naively adding an external objective's gradient "would jeopardize the structure in generated samples."
  - They propose guidance via a forward-prediction loss, and an iterative variant in which "both pre-trained score network and guidance are iteratively updated using self-generated samples." — [Gradient Guidance (Guo et al., NeurIPS 2024)](https://arxiv.org/abs/2404.14743)
- **Distilling reward-gradient guidance into the model.**
  - DiffusionOPSD (Aug 2026 preprint) "converts image-level reward guidance into explicit targets for clean-output predictions." A frozen behaviour policy generates trajectories, reward gradients construct bounded positive and negative x0 targets, the student fits them, and an EMA refreshes the behaviour policy.
  - It reports the best held-out score in 19 of 20 reward-matched settings across ten evaluators, and 40–63% fewer GPU-hours than DiffusionNFT, on SD3.5-M and Z-Image-Turbo. — [DiffusionOPSD](https://arxiv.org/abs/2608.24646)
- **Per-prompt cost reference.** DreamBooth "takes 5 minutes on 1 A100", cited by ReNO as the cost of per-subject fine-tuning. Fine-tuning-based reward methods on small prompt sets "(e.g., <50 prompts …) … may not generalize well to unseen prompt[s]." — [ReNO](https://arxiv.org/abs/2406.04312)
- **Repo context.** Without a class, "CLIP has to set the composition too." The 256 unconditional fast sampler (0.316 ViT-B/16) trails the old 1000-step sampler (0.331; difference CI −0.040 to +0.014) on 6 held-out prompts. — [README](/home/sam/GitHub/clip-guided-diffusion/README.md)

### Inferences
- **(b), concrete recipe (estimate):**
  - Add `Linear(512 or 1024 → time_embed_dim=1024)` from a CLIP image embedding into the `emb` sum, zero-initialised so the model starts exactly as the unconditional checkpoint. Optionally add LoRA, or a DEFT-like side network taking x̂0.
  - Train it with the ordinary ε-loss on ImageNet, the model's own domain, plus any captioned or uncaptioned set, conditioning on the CLIP ViT-B/32 image embedding with LAFITE-style noise. At sampling time, feed the prompt's text embedding.
  - Keep the CLIP guidance as is: the model now composes the subject, and guidance only corrects. This is the "change the prior" lever the earlier report found to matter most, at about 1–3 GPU-days. It is supervised, so it carries no reward-hacking risk.
  - Afterwards, DRaFT-LV or Adjoint Matching can operate on a model that is actually prompt-conditional.
- **(c), concrete recipe (estimate):**
  - Run the existing guided sampler without gradient up to the unguided tail below 0.1 of the schedule. Then enable the gradient for the last K=1–3 U-Net calls and apply the reward to x̂0, using a reward ensemble that differs from the guide (e.g. PE-Core-B-16 plus simulacra aesthetic) and is evaluated with ViT-H-14 and SigLIP2.
  - This has no second-order terms through guidance, so its memory equals one guided step.
  - It is untested, and because DRaFT-1 changes the whole chain, the LoRA will also change the denoiser that guidance differentiates through. Whether that helps or fights guidance is unknown.
- **(a) per prompt (estimate):** a LoRA DRaFT-LV run of 100–300 steps at batch 2–4 per prompt is about 20–60 min on the 4070. It is likely to collapse to one image, which may be acceptable for single-prompt use, but CLIP's adversarial gradients go straight into the weights. It is inferior per GPU-hour to ReNO-style or test-time search, which do not modify the model.

### Gaps
- **No paper found that reward-fine-tunes a model which is then sampled with CLIP or classifier *gradient* guidance**, the exact (c) setting. The nearest are DRaFT with CFG, Guo et al.'s iterative guidance plus fine-tuning, and DiffusionOPSD, which distils reward-gradient guidance into the model.
- No quantitative result for language-free CLIP-image-embedding conditioning added to the guided-diffusion ImageNet U-Net, nor for how many images or GPU-hours it needs at 256.

## 6. Reward hacking with CLIP and aesthetic rewards, mitigations, and how much held-out-judged quality actually improves

### Takeaway
CLIP rewards are hacked by rendered text (typographic attacks), adversarial texture, and mode or style collapse. Aesthetic rewards are hacked by saturation, over-detail and stylisation. The two pull against each other. The best-evidenced mitigations are a KL anchor (Flow-GRPO, Adjoint Matching's λ), LoRA scaling, which beat KL and early stopping for DRaFT, reward ensembles, and held-out judges. Held-out gains are real but much smaller than train-reward gains, and some fine-tuning methods *lower* unseen rewards.

### Cited Findings
- **Typographic attack.** DDPO with a LLaVA-based prompt-alignment reward "exploits the VLM with a typographic attack," writing text instead of drawing the counted objects. — [DDPO](https://arxiv.org/abs/2305.13301)
- **CLIP-reward quality loss.** CLIP-similarity reward with DRaFT: "CLIP scores improved, but … image quality degraded after many steps." — [DRaFT](https://arxiv.org/abs/2309.17400)
- **Collapse and failed diversity fixes.** Aesthetic-reward collapse to near-identical images. Reward dropout of 0.95 still collapsed. An LPIPS in-batch diversity term "increased inner-batch diversity at the expense of the aesthetic score" and made images "overly stylized." — [DRaFT](https://arxiv.org/abs/2309.17400)
- **Cross-reward generalisation.** Trained on HPSv2 and evaluated on PickScore, "LoRA scaling [was] more effective than" early stopping or a KL term on predicted ε at the last step. — [DRaFT](https://arxiv.org/abs/2309.17400)
- **Verifier hacking in search (Ma et al., 2025).**
  - "Searching with Aesthetic and CLIP Verifier overfit to their inherent biases, negatively impacting each other": the aesthetic score favours stylised images off the prompt, and CLIP favours alignment "at the expense of visual quality." Because search does not change the model, the cross-metric loss was only about −2%.
  - On ImageNet, random search against a CLIP or DINO classifier verifier showed "mode collapse as the compute increases," which they call "Verifier Hacking." — [Inference-time scaling](https://arxiv.org/abs/2501.09732)
- **Single CLIP reward in ReNO.** On SD-Turbo, optimising CLIPScore alone raised attribute binding (Color 0.5513 → 0.6625) but lowered the LAION aesthetic score (5.647 → 5.475). HPSv2 raised it (5.752). ReNO therefore uses a four-reward ensemble (HPSv2, PickScore, ImageReward, CLIP), few steps, a noise-norm regulariser and gradient clipping. — [ReNO](https://arxiv.org/abs/2406.04312)
- **KL for Flow-GRPO.** "With a proper KL term, we can match the high reward of the KL-free" run, while omitting KL collapsed diversity. — [Flow-GRPO](https://arxiv.org/abs/2505.05470)
- **Held-out reward gains with diversity cost.** Adjoint Matching lifted held-out HPSv2 from 17.89 to 24.49 while keeping DreamSim diversity at 37.24 (base 56.53). DRaFT-1 reached 24.37 but with diversity 27.39. — [Adjoint Matching](https://arxiv.org/abs/2409.08861)
- **Test-time alternative without overoptimisation.** DAS (ICLR 2025) uses SMC with tempering at test time and reports "comparable or superior target rewards to fine-tuning methods while preserving diversity and cross-reward generalization." The same paper contrasts it with fine-tuning methods (DDPO, AlignProp, TDPO) that "drop the scores of" unseen rewards, and SEIKO ("AlignProp with KL") "distorts animal features." — [DAS](https://arxiv.org/abs/2501.05803)
- **The repo's own evidence on ADM:**
  - The aesthetic term "costs score and adds saturation and clipping" at every scale, so it stays off for ADM.
  - Best-of-4 judged by held-out CLIPs gains only +0.015 ± 0.007 (+0.025 with the ensemble) on a held-out ViT-H-14.
  - ViT-B/16, a judge, "is close kin to ViT-B/32" and favours it. — [README](/home/sam/GitHub/clip-guided-diffusion/README.md)

### Inferences
- **Protocol for this repo (inference):**
  - Rewards: never use the guide (ViT-B/32) as the training reward. Use a reward ensemble disjoint from both the guide and the judges, e.g. PE-Core-B-16 plus another open_clip model, with the simulacra head at low weight.
  - Judges: evaluate on held-out prompts with ViT-H-14-quickgelu and SigLIP2, the README's non-kin judges, plus the ViT-B/16 rating head.
  - Early stopping and scaling: early-stop on the held-out judge and sweep the LoRA scale at inference (DRaFT's best mitigation).
  - Diversity: track it, for example as LPIPS or DreamSim between seeds per prompt.
  - Sanity checks: check for rendered text and clipping fraction (the README already reports clipped pixels).
- **Expected gain size (inference).** In published work, held-out gains are typically a fraction of train gains. On this repo, best-of-N moved the held-out ViT-H-14 by only about 0.015–0.025, which sets the scale of what is easy. Gains much above that from fine-tuning should be verified against hacking by eye and across the non-kin judges.

### Gaps
- No study of reward hacking with *small* CLIPs (ViT-B/32, PE-Core-B) as fine-tuning rewards on a pixel-space model, where adversarial high-frequency patterns are directly expressible in the output (no VAE decoder to filter them).
- No published held-out-judge evaluation of reward fine-tuning on ImageNet diffusion models.

## 7. Few-step and one-step distillation of ADM ImageNet models (CD, PD, CTM, DMD/DMD2, ECT, sCM, SiD), and whether a distilled 256 unconditional model would enable ReNO-style noise optimisation

### Takeaway
Published one-step and few-step distillations of ADM-architecture models at 256 (consistency distillation on LSUN-256) needed 64 A100s, and the best one-step ImageNet results (CTM, DMD2, SiD, ECT) are at 64×64 on 7–8 A100s or 4–8 H100s for days. Distilling the 553M ADM-256 unconditional to a good one- or two-step generator on one 12 GB 4070 is **out of budget** at published quality (estimate: weeks to months). The cheap paths to ReNO-style optimisation are differentiating through a short, learned-timestep ODE of the existing model (LD3-style, minutes to train) with checkpointing, or a LoRA-DMD2 that accepts a mediocre FID, since ReNO only needs a differentiable, roughly on-manifold generator.

### Cited Findings
- **Consistency models (Song et al., 2023):**
  - Architectures "follow … Dhariwal & Nichol (2021)" (ADM).
  - Consistency distillation (CD) with LPIPS on LSUN Bedroom 256 gives FID 7.80 at 1 step and 5.22 at 2, from an EDM teacher at 3.57 with 79 NFE (ADM itself reaches 1.90 at 1000 NFE). LSUN Cat 256: 11.0 at 1 step, 8.84 at 2.
  - ImageNet 64: CD FID 6.20 at 1 step and 4.70 at 2. PD at 1 step: 15.39.
  - Compute: LSUN runs used batch 2048 and 600k iterations (Bedroom), on 64 A100s according to the hyperparameter table. — [Consistency Models](https://arxiv.org/abs/2303.01469)
- **Progressive distillation** gets to "as few as 4 steps without losing much perceptual quality." It notes that with ε-prediction at low SNR, small output changes have large effects on the implied x, so it reparameterises to x or v prediction. Its larger datasets (ImageNet 64, LSUN 128) start distillation from 1024 steps. — [Progressive Distillation](https://arxiv.org/abs/2202.00512)
- **CTM:** ImageNet 64 FID 1.92 at 1 step and 1.73 at 2, using a StyleGAN-XL discriminator GAN loss. 8×A100 40 GB, batch 2048, 30K iterations for ImageNet. — [CTM](https://arxiv.org/abs/2310.02279)
- **DMD:** ImageNet 64 FID 2.62. It needs a regression loss on a paired noise→image dataset generated by the teacher (256 steps for ImageNet). — [DMD](https://arxiv.org/abs/2311.18828)
- **DMD2:**
  - Removes the regression loss and stabilises with a two-time-scale update, with the fake-score critic updated more often than the generator.
  - Adds a GAN loss on real data.
  - ImageNet-64 1-step FID 1.51 with batch 280 on 7 A100s for 200K iterations, about 2 days. 1.28 with longer training plus GAN (about 5 days total). The EDM teacher (ODE) scores 2.32. — [DMD2](https://arxiv.org/abs/2405.14867)
- **ECT** (fine-tunes a pretrained diffusion model into a consistency model):
  - CIFAR-10: 2-step FID 2.73 "within 1 hour on a single A100."
  - ImageNet 64: ECM-S 2-step FID 3.18 on 4 H100s in 8.5 h. ECM-XL 1.96 on 8 H100s in 15 h. — [ECT](https://arxiv.org/abs/2406.14548)
- **sCM:** 1.5B parameters on ImageNet 512, 2-step FID 1.88. — [sCM](https://arxiv.org/abs/2410.11081)
- **SiD:** data-free one-step distillation, ImageNet 64 FID 1.524. — [SiD](https://arxiv.org/abs/2404.04057)
- **LD3** learns the time discretisation of an existing ODE sampler from 100 samples in "5 to 40 minutes on a single GPU." It reaches FID 2.38 (CIFAR-10) and 2.27 (AFHQv2) at 10 NFE, with pixel and latent results including 256² models. It uses rematerialisation to keep memory flat in NFE. — [LD3](https://arxiv.org/abs/2405.15506)
- **DDO** fine-tunes a diffusion model against a frozen copy, using the likelihood ratio as an implicit GAN discriminator. ImageNet-64 FID goes from 1.58 to 0.97 and ImageNet-512 from 1.96 to 1.26 (EDM2), with each round under 1% of pretraining epochs. It improves the prior's realism from data, not prompt following. — [DDO](https://arxiv.org/abs/2503.01103)
- **ReNO needs:**
  - Recipe: a one-step differentiable generator, 50 iterations of Nesterov gradient ascent on the initial noise with a χ-norm regulariser and gradient clipping.
  - Cost: SD-Turbo 512 with one CLIP reward takes 0.2 s/iter, 10 s total, 10 GB. With all four rewards, 0.4 s/iter, 20 s, 15 GB.
  - Comparison: DOODL, which backpropagates through a 50-step SD2.1 sampler, takes 24 s/iter, 20 min and 40 GB. — [ReNO](https://arxiv.org/abs/2406.04312)
- The repo's SD-Turbo with ReNO-style correction scores 0.350–0.357 ViT-B/16, against 0.316 for 256 unconditional ADM. — [README](/home/sam/GitHub/clip-guided-diffusion/README.md)

### Inferences
- **Cost of a DMD2-style one-step ADM-256 (estimate):**
  - It needs three 553M U-Nets: frozen teacher, trainable generator, trainable fake score. Full fine-tuning of two of them is about 17.7 GB of fp32-Adam state alone, so it does not fit 12 GB. LoRA on both, or 8-bit Adam plus bf16, might.
  - DMD2 needed about 14 A100-days for ImageNet-64 FID 1.51 on a smaller EDM network. At 256², with 16× the pixels per sample, on a single consumer GPU, a comparable-quality run is far beyond "a couple of weeks."
  - A short LoRA-DMD2 run that yields a blurry-ish but on-manifold one-step generator may be enough for ReNO's purpose, but no data supports that.
- **Consistency distillation and ECT at 256 (estimate):** the only 256 results used 64 A100s. ECT's ImageNet-64 cost of 34–120 H100-hours, scaled to 256, is weeks on one 4070.
- **Cheaper route to ReNO on ADM (estimate):**
  - Fit LD3 timesteps for a 6–10-step DDIM or DPM-Solver ODE of the unconditional model (minutes), then do DOODL-style noise optimisation by backprop through those 6–10 calls with per-call checkpointing.
  - Cost: about 3× forward per call, so roughly 20–30 U-Net-equivalents per iteration, and several minutes per image for 50 iterations. Memory is about one U-Net backward (fits in 8 GB, as the repo's guidance already shows).
  - This turns "noise optimisation" from a distillation project into an inference option. Quality at 6–10 ODE steps for this model is unmeasured.
- Distillation cannot raise the ceiling set by the prior. Few-step students match or trail their teacher in FID. The repo's gap to SD-Turbo is a prior gap, ImageNet content against LAION. So distillation buys speed and differentiability, not content **(inference)**.

### Gaps
- No published one-step or few-step distillation of the guided-diffusion 256 ImageNet *unconditional* checkpoint was found. The search turned up CoSIGN (consistency-model priors for inverse problems) and shortcut models (DiT at ImageNet-256, 1-step FID 10.6), neither of which is this model.
- No measurement of how ReNO behaves on a pixel-space ImageNet one-step generator, or of adversarial-pattern risk without a VAE decoder.

## 8. Best expected gain per GPU-day for this model and these rewards

### Takeaway
Ranked by expected held-out gain per GPU-day, all estimates:
1. Add CLIP-embedding conditioning to ADM and train it by supervised, language-free denoising (1–3 GPU-days, no hacking risk, attacks the prior gap).
2. DRaFT-LV or ReFL with LoRA, K=1, over a prompt distribution, with a disjoint reward ensemble and held-out early stopping. About 1–2 GPU-days per run, likely after step 1, and expect small held-out gains.
3. Adjoint Matching with λ sweeps if collapse appears (2–5 GPU-days).
4. A short-ODE noise optimisation (LD3 timesteps plus checkpointed backprop) as the training-free ReNO analogue.

RL (DDPO, GRPO), DPO from AI feedback, and full one-step distillation all have worse gain per GPU-day here.

### Cited Findings
- Gradient methods beat RL in sample efficiency by large factors (DRaFT against DDPO's 7.4 at 50k queries; AlignProp 48 min against about 23 h). — [DRaFT](https://arxiv.org/abs/2309.17400), [AlignProp](https://arxiv.org/abs/2310.03739)
- DRaFT small-scale runs (2k steps, batch 4, LoRA rank 8) took about 1.5 h on 4 TPUv4. — [DRaFT](https://arxiv.org/abs/2309.17400)
- Adjoint Matching gives the best held-out-reward against diversity Pareto front, with full fine-tuning at batch 40 on 2×A100 80 GB. — [Adjoint Matching](https://arxiv.org/abs/2409.08861)
- A CLIP reward alone degraded quality under DRaFT. DPO with CLIP preference improved alignment. Aesthetic and CLIP objectives trade off against each other. — [DRaFT](https://arxiv.org/abs/2309.17400), [Diffusion-DPO](https://arxiv.org/abs/2311.12908), [Inference-time scaling](https://arxiv.org/abs/2501.09732)
- DEFT trained a side network for the frozen ImageNet-256 unconditional model in hours. — [DEFT](https://arxiv.org/abs/2406.01781)
- Language-free training with CLIP image embeddings is an established way to get text conditioning without captions. — [LAFITE](https://arxiv.org/abs/2111.13792), [KNN-Diffusion](https://arxiv.org/abs/2204.02849)
- One-step distillation at 256 or ImageNet scale needed multi-GPU days or weeks. — [Consistency Models](https://arxiv.org/abs/2303.01469), [DMD2](https://arxiv.org/abs/2405.14867), [ECT](https://arxiv.org/abs/2406.14548)
- The repo's own held-out gains from cheap tricks are small (best-of-4 +0.015 to +0.025), and the prior matters most. — [README](/home/sam/GitHub/clip-guided-diffusion/README.md)

### Inferences
All costs and gains below are estimates.
- **Ranking by gain per GPU-day:**

  | Rank | Method | GPU-days on a 4070 | Expected held-out gain | Main risk |
  |---|---|---|---|---|
  | 1 | Supervised CLIP-embedding conditioning (zero-init projection into `emb`, plus LoRA or a DEFT-like side net, ε-loss on ImageNet or other images with noised CLIP image embeddings) | 1–3 | Largest structural gain: the model composes the subject instead of CLIP doing it | ImageNet content limits; gap between text and image embeddings (LAFITE noise or retrieval mitigates) |
  | 2 | DRaFT-LV or ReFL with LoRA (K=1, a disjoint reward ensemble, guided sampler in the loop with no gradient, gradient only on the unguided tail) | 1–2 per run | Small to moderate on held-out judges | CLIP adversarial texture, collapse. Mitigate with LoRA scaling and early stopping on ViT-H-14 and SigLIP2 |
  | 3 | Adjoint Matching (LoRA or 8-bit Adam, DDPM 40–60 steps, λ sweep) | 2–5 | Like DRaFT-1 on reward, much better diversity | Untested with LoRA or pixel space; compute heavy |
  | 4 | LD3 timesteps plus checkpointed short-ODE noise optimisation (ReNO analogue, no training) | under 0.1 to train, then minutes per image | Unknown; ReNO-like gains need an on-manifold differentiable path | Slow per image; adversarial noise |
  | 5 | Diffusion-DPO or KTO on CLIP-ranked guided samples | Several, mostly pair generation | Small, stable | Weak reward movement (Adjoint Matching table) |
  | 6 | DDPO, GRPO or DiffusionNFT | Many | Similar to the above for far more samples | Sample cost |
  | 7 | One-step distillation (DMD2, CD, ECT) of ADM-256 | Weeks or more | Speed and differentiability, not content | Exceeds the budget at published quality |

- Compare any of these against simply switching prior (SD-Turbo with ReNO at about 0.35). On current evidence, no ADM fine-tune within budget is likely to close a 0.035 ViT-B/16 gap that comes from the ImageNet training data. Steps 1–2 are worth it only if keeping the ADM look or licence matters.

### Gaps
- Every ADM-specific cost and gain above is an estimate. None of these methods has a published result on this checkpoint, and the GPU could not be used for a measurement during this research.
- The RTX 4070's training throughput on this U-Net (bf16, with or without checkpointing) is unmeasured. A single short benchmark (forward and backward at batch 1 and 4, with and without `use_checkpoint`) would turn most of these estimates into numbers.

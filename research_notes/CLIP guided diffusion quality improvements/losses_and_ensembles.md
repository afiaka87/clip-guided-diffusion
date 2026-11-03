# Better losses and model ensembles for text-to-image guidance via CLIP-like encoders

> **Status, 2026-11-02:** These are the notes as written on 2026-11-01, kept as the record of what was known then. Later measurements overturned some of it, including its leading recommendations. [The report](../../reports/CLIP%20guided%20diffusion%20quality%20improvements.md) has the current results, and wins wherever the two disagree.

Evidence labels: **[measured here]** was run on this machine (RTX 2070 SUPER, 7.62 GiB usable) on 2026-11-01. **[paper]** is a number a paper reports. **[folklore]** is community practice with no measurement behind it. The experiment scripts are in `../../reports/evidence/scripts/`. Each measurement also states its procedure here.

**The real-pipeline image comparison never ran.** It needed a free GPU, which siblings held until this work was stopped, so no entry here shows that any technique produces better diffusion images on this repo. Everything measured below is tower cost, determinism, gradient statistics, or a gradient transfer proxy. Each states its sample size. Where no interval is given, the per-pair values were not kept and none can be computed.

Terms: `T` is a vision tower's forward plus backward peak VRAM with 4 cutouts, `w` is its weights alone, and `g_m` is the gradient of model `m`'s loss with respect to the image. 14 of the fast sampler's 30 model calls are guided at the defaults (counted from `sampling.NoiseLevels` and `GUIDANCE_WINDOW`), so a tower's time per image is 14 times its seconds per step. The sampler caps the RMS of guidance at `DEFAULT_GUIDANCE_RATE`, so at the defaults mostly the gradient's direction matters, not its scale. That makes relative loss weights matter and absolute ones not.

## 1. Do multi-model CLIP ensembles improve guidance, how should they be weighted, and what do they cost?

### Takeaway
No paper shows a large quality gain from ensembling, and DAS calls it marginal, but clean towers' gradients are nearly orthogonal (cosine .01 to .16 measured here), so averaging them plausibly removes noise, and the best transfer proxy here was an ensemble. Backpropagate towers one at a time so VRAM follows the largest tower instead of the sum; whether any of this improves diffusion samples is still unmeasured.

### Cited Findings
- **[folklore]** Disco Diffusion enables ViT-B/32, ViT-B/16 and RN50 by default and weights them equally, accumulating `grad(losses.sum() * clip_guidance_scale) / cutn_batches` per model into `x_in_grad` with no per-model normalisation, then one backward through the U-Net. Its defaults are cgs 5000, TV 0, range 150, saturation 0, clamp_max 0.05, and the notebook has no quality comparison or VRAM guidance for any mix. [Source](https://raw.githubusercontent.com/alembics/disco-diffusion/main/Disco_Diffusion.ipynb)
- **[folklore]** Majesty Diffusion uses ViT-B-16 OpenAI, ViT-B-32 laion2b_e16 and ViT-B-16 laion400m_e32 at cgs 5000, and squashes the gradient with `softsign(grad * grad_scale / mag)`. [Source](https://huggingface.co/datasets/multimodalart/latent-majesty-diffusion-settings/raw/main/defaults_v1_3.cfg)
- **[paper]** StyleGAN-NADA found ViT-B/32 and ViT-B/16 used in tandem best for shape changes, qualitatively, with no numbers. [Source](https://ar5iv.labs.arxiv.org/html/2108.00946)
- **[paper]** Direct Ascent Synthesis averaged the gradients of 3 CLIP ViT-B/32 models (OpenAI plus 2 OpenCLIP from different datasets): it "(marginally) improves generation quality, however, a single model is sufficient", and "some CLIP models were particularly bad at being turned into generators without any obvious reason why". Ablation shows ensembling plus augmentation without multi-resolution still lacks coherence. [Source](https://arxiv.org/abs/2502.07753)
- **[paper]** Ensemble Everything Everywhere aggregates multi-resolution logits with CrossMax, a Vickrey-style robust vote, and claimed 72% and 48% adversarial accuracy on CIFAR. [Source](https://arxiv.org/abs/2408.05446)
- **[paper]** Zhang, Nikolić, Carlini and Tramèr broke that with an adaptive attack, to 4% and 1%, attribute it to CrossMax gradient masking, and note "gradients are highly perceptually aligned, but the model exhibits almost no robustness". So a perceptually aligned gradient is not the same thing as robustness, and it is the former that generation needs. [Source](https://arxiv.org/pdf/2411.14834v1)
- **[measured here]** Vision tower alone, random weights, forward plus backward with 4 and 16 cutouts, peak allocated GiB and seconds per step. Procedure: `arch_cost.py`, one seed, one random input per row, timing the median of 5 repeats, so no interval. Peaks are reliable, because allocation does not depend on the seed. Timings marked `*` may be inflated by a sibling job on the GPU. `ckpt` is gradient checkpointing, which the repo already enables for open_clip models.

| Tower | Input | Precision | `w` | `T` cut 4 | s/step cut 4 | `T` cut 16 | s/step cut 16 |
|---|---|---|---|---|---|---|---|
| ViT-B/32 | 224 | fp16 | 0.19 | 0.23 | .015 | 0.34 | .027 |
| ViT-B/32 | 224 | fp32 | 0.34 | 0.42 | .027 | 0.65 | .086 |
| ViT-B/16 | 224 | fp16 | 0.19 | 0.34 | .048 | 0.78 | .089 |
| ViT-B/16 | 224 | fp16 ckpt | 0.19 | 0.22 | .069* | 0.33 | .119* |
| ViT-B/16 | 224 | fp32 | 0.34 | 0.64 | .099 | 1.53 | .348 |
| RN50 | 224 | fp16 | 0.09 | 0.28 | .014 | 0.86 | .039 |
| RN101 | 224 | fp16 | 0.12 | 0.39 | .024 | 1.20 | .059 |
| RN50x4 | 288 | fp16 | 0.18 | 0.70 | .040 | 2.29 | .120 |
| RN50x16 | 384 | fp16 | 0.34 | 1.90 | .093 | OOM | |
| ConvNeXt-B-w | 256 | fp16 | 0.18 | 0.48 | .058 | 1.37 | .199 |
| ConvNeXt-B-w | 256 | fp16 ckpt | 0.18 | 0.26 | .041* | 0.48 | .266* |
| ConvNeXt-B-w | 256 | fp32 | 0.35 | 0.94 | .136 | 2.72 | .254 |
| ConvNeXt-B-w | 256 | fp32 ckpt | 0.35 | 0.50 | | 0.95 | |
| ConvNeXt-B-w-320 | 320 | fp16 | | 0.65 | .091 | 2.09 | .125 |
| ConvNeXt-B-w-320 | 320 | fp16 ckpt | | 0.30 | .132* | 0.66 | .402* |
| ViT-L/14 | 224 | fp16 | 0.59 | 1.08 | .209 | 2.56 | .393 |
| ViT-L/14 | 224 | fp16 ckpt | 0.59 | 0.67 | .310* | 0.93 | .541* |
| ViT-L/14 | 224 | fp32 | 1.15 | 2.14 | | 5.09 | |
| ViT-H/14 | 224 | fp16 | 1.20 | 2.01 | .465 | 4.45 | .844 |
| ViT-H/14 | 224 | fp16 ckpt | 1.20 | 1.32 | .676* | 1.70 | 1.173* |
| ViT-H/14 | 224 | fp32 ckpt | 2.44 | 2.69 | | 3.45 | |

- **[measured here]** Determinism, 5 repeated forward plus backward passes on one seeded input compared bit for bit, 4 cutouts, random weights. A pass means 5 of 5 identical, and a failure is proof, but a pass on one input is not. fp16 was deterministic for every tower. At fp32:

| Tower | Default attention | Plain attention |
|---|---|---|
| RN50, RN101, RN50x4, RN50x16 | Nondeterministic (their `AttentionPool2d` calls `F.multi_head_attention_forward`) | Not tested |
| ViT-B/32, ViT-B/16 | Deterministic | Deterministic |
| ViT-L/14, ViT-H/14 | Nondeterministic | Deterministic |
| ConvNeXt-B-w, ConvNeXt-B-w-320 | Deterministic, no attention at all | Not applicable |

- **[measured here]** Clean towers' gradients barely overlap, and robust ones agree even across architectures. Gradient cosine at fp32 on CPU, each gradient summed over 8 draws of 4 cutouts, on 1 image and 1 prompt (dog photo, "a red fox in the snow"), so no interval, and this is 1 pair:

| Pair kind | Cosine |
|---|---|
| Clean with clean, all 28 pairs | -.021 to .160 |
| Clean with FARE4, all 24 pairs | .012 to .099 |
| FARE4 ViT-B/32 with FARE4 ViT-B/16 | .685 |
| FARE4 ViT-B/16 with FARE4 ConvNeXt-B-w | .557 |
| FARE4 ViT-B/32 with FARE4 ConvNeXt-B-w | .521 |

- **[measured here]** Transfer proxy, with no diffusion involved. Start from a real image at 256px, take 20 steps of L2 size 0.5 along the guide's unit gradient with 4 cutouts per model, and report the rise in cosine with the prompt for judges. CPU, fp32, 4 pairs (2 images by 2 prompts), 1 seed sequence, so no interval, and differences of .01 between guides should not be read as real. Held-out mean leaves out any judge that is also a member of the guide. "(m)" marks such a judge, whose number is inflated.

| Guide | ViT-B/16 | PE-Core-B-16 | SigLIP2-L | laion ConvNeXt | Held-out mean |
|---|---|---|---|---|---|
| OpenAI RN50 | .009 | .007 | .006 | .036 | .015 |
| OpenAI ViT-B/32 (repo default) | .028 | .015 | .014 | .040 | .024 |
| FARE4 ViT-B/32 | .034 | .017 | .018 | .029 | .025 |
| Cone: FARE4 ConvNeXt within 30° of ViT-B/32 | .037 | .027 | .025 | .065 | .039 |
| laion ConvNeXt-B-w | .027 | .059 | .043 | .290 (m) | .043 |
| PE-Core-B-16 | .016 | .202 (m) | .059 | .057 | .044 |
| ViT-B/32 + FARE4 ConvNeXt, mean unit gradient | .039 | .034 | .039 | .080 | .048 |
| FARE4 ViT-B/32 + B/16 + ConvNeXt, mean | .045 | .048 | .041 | .063 | .049 |
| FARE4 ConvNeXt-B-w | .041 | .042 | .038 | .081 | .050 |
| ViT-B/32 + B/16 + RN50 (Disco), mean | .140 (m) | .058 | .028 | .065 | .050 |
| ViT-B/32 + B/16 + FARE4 ConvNeXt, median | .128 (m) | .056 | .021 | .085 | .054 |
| ViT-B/32 + B/16 + FARE4 ConvNeXt, mean | .143 (m) | .061 | .042 | .085 | .062 |

  FARE4 ConvNeXt's .081 on laion ConvNeXt may partly reflect that FARE was fine-tuned from that model. Without that judge it scores .040.
- **[measured here]** The repo's own peaks show that tower VRAM does not add to the U-Net's peak. At 128px ViT-B/32 peaks at 2.3 GiB and PE-Core-B-16 at 2.4 GiB, and the fp32 and fp16 pairs for SigLIP2-L, PE-Core-L and SO400M fit `peak = max(U-Net backward peak, G + T)` with `G` about 1.9 to 2.0 GiB at 128px and 4.9 to 5.0 GiB at 256px, if an fp16 tower takes half the VRAM of an fp32 one. `G` is what the U-Net holds while CLIP runs, its weights and forward graph. [Source](../../README.md)

### Inferences
- Aggregation options, cheapest first. Each `g_m` comes from a separate tower backward:
  - Equal loss sum, as in Disco: `g = Σ_m ∇L_m`. A spiky model dominates, because clean gradients differ in norm.
  - Unit gradient sum: `g = Σ_m a_m · g_m / ‖g_m‖`. It makes `a_m` mean what it says, and suits the capped sampler, which discards scale anyway.
  - Coordinate-wise median of unit gradients: `g_i = median_m(ĝ_{m,i})`. It needs at least 3 towers.
  - AND-mask: keep pixels where all `sign(g_m)` agree, zero the rest.
  - Cone projection onto a robust model's gradient, see section 2.
- Backpropagate towers one at a time, as Disco does. This keeps peak at `T_ens = Σ w_m + max_m(T_m − w_m)`, not `Σ T_m`, and the U-Net backward runs once:

```python
x_in_d = x_in.detach().requires_grad_()          # Cut the graph at the image.
total = 0
for model, target, weight in towers:
    embeds = model.encode_image(clip_util.preprocess(model, make_cutouts(x_in_d.add(1).div(2))))
    grad = th.autograd.grad(spherical(embeds, target).mean(), x_in_d)[0]
    total = total + weight * grad / grad.norm().clamp(min=1e-12)
gradient = th.autograd.grad(x_in, x, grad_outputs=total)[0]  # One U-Net backward.
```

- Every tower should see the same cutouts, drawn once at the largest input size and downsampled per tower, or the seed-to-image mapping depends on tower order.
- Reproducibility rules out OpenAI's RN models at fp32 unless their attention pool is rewritten. At fp16, which is how the repo always runs OpenAI models, they are fine. ConvNeXt is the simplest choice because there is no attention to patch.
- Budget at 4 cutouts, using `T_ens` and the `G` fit. 8 GB allows `T_ens` up to about 5.6 GiB at 128px and 2.6 GiB at 256px; 12 GB allows about 9.6 GiB and 6.6 GiB, taking 11.6 GiB as usable on the 4070.

| Ensemble | `T_ens` | s/image added | 128px peak | 256px peak | Plain or no attention | Deterministic |
|---|---|---|---|---|---|---|
| ViT-B/32 alone (fp16) | 0.23 | 0.2 | 2.3 | 5.3 | Yes | Yes |
| ViT-B/32 + ConvNeXt-B-w fp32 ckpt | 0.69 | 2.1 | 2.7 | 5.7 | Yes | Yes |
| ViT-B/32 + ViT-B/16 + RN50 (fp16, Disco) | 0.66 | 1.1 | 2.6 | 5.6 | Yes at fp16 | Yes |
| ViT-B/32 + ViT-B/16 + ConvNeXt-B-w fp16 ckpt | 0.71 | 1.5 | 2.7 | 5.7 | Yes | Yes |
| ViT-B/32 + ViT-L/14 fp16 ckpt | 0.86 | 4.5 | 2.9 | 5.9 | fp16 only on OpenAI | Yes |
| ViT-B/32 + ConvNeXt-B-w + ViT-H/14 (fp16 ckpt) | 1.70 | 10 | 3.7 | 6.7 | Yes | Yes |

  The peaks are predictions from the fit, and the time column uses the s/step of the rows above.
- Weights: start equal on unit gradients. Nothing in the literature supports a specific non-uniform weighting, and DAS's remark that some models are bad generators suggests picking models empirically rather than trusting architecture.
- On the transfer proxy the default ViT-B/32 is weak, and the first move is to replace or join it, not to add a third clean ViT. ViT-B/32 + ViT-B/16 + FARE4 ConvNeXt is the first ensemble to try in the pipeline, at `T_ens` 0.71 GiB and 1.5 s per image. A single FARE4 ConvNeXt or PE-Core-B-16 did about as well as most ensembles, so a single better tower is a fair baseline to beat.
- Plain mean beat the median and the cone on this proxy, which is no reason to pay for anything more complicated. The sample is too small to rank them.

### Gaps
- **Transfer proxy, not finished:** 12 of 16 guides completed. FARE4 ConvNeXt with 16 cutouts, cross-entropy against ImageNet texts, a −0.3 negative prompt, and template mean were stopped at the deadline with no result, so section 3 has no transfer numbers. The proxy scores gradient steps on clean images with no U-Net and no noise, so it can disagree with real sampling.
- **Pipeline design, ready to run:** 12 prompts, seed 0, 128px, about 22 variants, judged by ViT-B/16, ViT-H-14-quickgelu, SigLIP2-256, and SAC B/32 and B/16. It should take about 30 minutes on a free GPU, using `pilot.py`, `pilot2.py` and `pilot_judge.py` whose weights were in session storage. `pilot_judge.py` itself is untested.
- **Priority experiment, not run:** every ensemble through the real sampler, 128px, 12 prompts, seed 0, judged by held-out towers. `pilot.py` was written and passed CPU smoke tests of 4 steps on 1 prompt (FARE4 ViT-B/32, Disco mix, ViT-B/32 + FARE4 ConvNeXt), which shows they run, not what quality they give. VRAM peaks in the budget table are therefore predictions.
- No paper compares ensemble compositions for diffusion guidance with human or held-out scores. Disco's VRAM folklore is undocumented in its notebook.
- CrossMax was not tried on embeddings. It was built for class logits and the rebuttal found it masks gradients, which is the opposite of what guidance needs.
- The AND-mask idea has no source in the guidance literature found here, and it was not run.
- `convnext_large_d_320` was not measured: the run hit an out of memory error because a sibling job held 5.6 GiB. `T` for it is unknown.
- OpenAI RN attention pool determinism with a plain-attention rewrite was not tested.

## 2. Are adversarially robust CLIP models, or models with perceptually aligned gradients, better guides?

### Takeaway
On gradient quality yes, and this is the clearest measured effect in these notes: FARE4 gradients agree across cutouts at .74 to .86 against .09 to .16 for clean ViTs and hold only .07 to .16 of their energy at high frequency. Whether that yields better diffusion images has no evidence here or elsewhere beyond CLIPAG's CLIPDraw and VQGAN results, and robust models lose clean accuracy.

### Cited Findings
- **[paper]** FARE trains a vision tower to keep clean embeddings under attack: `L = max_{‖z−x‖∞≤ε} ‖φ(z) − φ_Org(x)‖²`, for 2 epochs on ImageNet. TeCoA instead uses a cross-entropy adversarial loss against text. [Source](https://arxiv.org/html/2402.12336v2)
- **[paper]** RobustVLM weights, MIT licensed, clean ImageNet accuracy: ViT-L/14 TeCoA4 54.2, FARE4 61.1, FARE2 67.0; FARE4 ConvNeXt-B 60.2; FARE4 ViT-B/16 56.6; FARE4 ViT-B/32 53.8; TeCoA4 ViT-B/32 46.8. [Source](https://github.com/chs20/RobustVLM)
- **[paper]** Hugging Face repos `chs20/FARE4-ViT-B-32-laion2B-s34B-b79K`, `chs20/FARE4-ViT-B-16-laion2B-s34B-b88K`, `chs20/FARE4-convnext_base_w-laion2B-s13B-b82K-augreg`, their `TeCoA4-*` twins, and `chs20/{fare2,fare4,tecoa2,tecoa4}-clip` for ViT-L/14 load through open_clip with `hf-hub:` names, as `open_clip_model.safetensors` of 598 to 718 MB. [Source](https://github.com/chs20/RobustVLM)
- **[measured here]** All three FARE4 repos load through open_clip 3.3.0 and run gradients with `fused_attn = False`. FARE4 ViT-B/32 and FARE4 ConvNeXt also ran through `clip_guided_diffusion()` in a 4 step CPU smoke test on 1 prompt, which proves the code path only. Their VRAM should be the laion parent's in section 1's table, because the architecture is identical, but that was not measured on the trained weights.
- **[measured here]** Gradient behaviour at fp32, CPU, mean over 2 images by 3 prompts (6 pairs), through `grad_quality.py`. Noise cos is the gradient's cosine with itself after adding noise of 1/2048 to the input, the size the repo uses to judge fp16. Cross-cutout cos compares two random draws of 4 cutouts. High frequency share is gradient energy above 0.25 cycles per pixel. Each pair used one noise draw and 2 cutout seeds. Intervals are 95% half-widths across the 6 pairs (t = 2.571), which cover variation between pairs only, and were computed for the five towers shown in a rerun. The other six have per-pair values no longer available, so they have no interval. The rerun drew different noise, so its noise cos can differ a little from the first run's, and the rerun's value is shown for those five. A one-pair smoke run is excluded.

| Tower | Noise cos | Cross-cutout cos | High frequency share |
|---|---|---|---|
| OpenAI ViT-B/32 | .971 ± .011 | .091 ± .019 | .611 ± .041 |
| OpenAI ViT-B/16 | .968 | .146 | .431 |
| OpenAI RN50 | .728 | .155 | .741 |
| laion ViT-B/32 | .966 | .140 | .532 |
| laion ViT-B/16 | .959 | .138 | .472 |
| laion ConvNeXt-B-w | .970 ± .005 | .551 ± .027 | .467 ± .128 |
| PE-Core-B-16 | .641 ± .140 | .094 ± .071 | .591 ± .015 |
| SigLIP2-L-16-256 | .729 | .096 | .426 |
| FARE4 ViT-B/32 | 1.000 ± .000 | .743 ± .042 | .071 ± .030 |
| FARE4 ViT-B/16 | 1.000 | .768 | .120 |
| FARE4 ConvNeXt-B-w | 1.000 ± .000 | .858 ± .027 | .160 ± .055 |

- **[paper]** CLIPAG trains ViT-B/32 on LAION with L2 ε=1.5 and 5-step PGD. In CLIPDraw, aesthetics scored 3.70 to 3.98 and preference 22 to 78% without augmentation, and 4.14 to 4.31 and 34 to 66% with it. VQGAN+CLIP's ViT-H similarity went 33.4 to 34.3. Generator-free synthesis over 1000 steps gave FID 42.3, IS 18.7, and needed no TV. [Source](https://ar5iv.labs.arxiv.org/html/2306.16805)
- **[paper]** CLIPAG's weights are one 1.31 GB `CLIPAG_ViTB32.pt` on Zenodo under CC-BY-4.0, documented as a `ViT-B-32` open_clip checkpoint. [Source](https://zenodo.org/records/10446026)
- **[measured here]** That file is a full training checkpoint, with epoch 100, optimizer and scaler state, and a run name recording L2 ε=1.5. open_clip 3.3.0 on torch 2.9.1 cannot load it as published, because `weights_only=True` rejects its numpy scalars. A static scan of the pickle showed only `OrderedDict`, numpy scalar and dtype, and torch tensor rebuilds, so allowlisting those three numpy types in torch's restricted unpickler is enough. Keeping `state_dict` alone gives a 605 MB file that open_clip loads.
- **[paper]** CLIP-JEM adds an energy-model term on top of adversarial training, with FID 23.4 for its XXL model against CLIPAG-L's 47.6, and calls cosine gradients of standard CLIP "meaningless". Its ADM guidance results are only qualitative, and its weights are not released. [Source](https://ar5iv.labs.arxiv.org/html/2408.17046)
- **[paper]** Croce et al.: clean CLIP inversion gives "only adversarial noise" while robust models give recognisable images. Robust ConvNeXt also beats clean on perceptual 2AFC, NIGHTS 87.2 to 92.3 and BAPPS 68.2 to 74.1. [Source](https://ar5iv.labs.arxiv.org/html/2502.11725)
- **[paper]** Srinivas et al. explain the effect as off-manifold robustness: gradients align with the data manifold at intermediate robustness, and excessive robustness makes them worse again. That argues against simply picking the largest ε. [Source](https://arxiv.org/html/2305.19101v2)
- **[paper]** LEAF robustifies the text encoder too, published as `LEAF-CLIP/*` including `CLIP-ViT-L-FARE2`, `OpenCLIP-ViT-H-FARE2` and `OpenCLIP-ViT-g-FARE2`, loading through `transformers.CLIPModel`. It targets attacks on text, which guidance does not face, since the prompt is fixed. [Source](https://arxiv.org/html/2506.03355v1)
- **[paper]** DVCE projects a non-robust gradient into a cone around a robust one, which is the standard way to combine the two: `P(w) = ⟨u, w⟩u` if `∠(w, v) > α`, where `u = sin α · P⊥(w)/‖P⊥(w)‖ + cos α · v/‖v‖`, with α=30° and a robust RN50. [Source](https://ar5iv.labs.arxiv.org/html/2210.11841)
- **[paper]** Perceptually aligned gradients are not tied to strong robustness: one study found adversarial training with a weak adversary gives little robustness and still produces them, and Kaur et al. get them from randomized smoothing. [Source](https://arxiv.org/pdf/2204.01099) [Source](https://arxiv.org/pdf/1910.08640)
- **[paper]** Kazemi et al. report NSFW content and biases when CLIP is inverted generatively, which applies to any of these used without a diffusion prior. [Source](https://arxiv.org/abs/2403.02580v1)

### Inferences
- A clean tower's 4 cutout gradient mostly changes from one draw to the next, so the sampler is averaging noise, and a robust tower agrees with itself far better. If a pipeline test confirms that, `num_cutouts` could fall for robust towers. This is the most useful thing to test first.
- Clean accuracy loss should matter less here than in a classifier, because the U-Net brings the subject and CLIP only steers. That is an inference from the sampler's design, not a measurement.
- ConvNeXt is the best-placed FARE tower: highest clean accuracy of the base models, no attention for determinism, and `T` of 0.50 GiB at fp32 with checkpointing. Noise cosine of 1.000 also suggests that the README's fp16 gradient disagreement will not apply to robust towers. The fp16 against fp32 comparison was written but not run, so test it before dropping fp32.
- `clip_util.OPEN_CLIP_MODELS` already takes these: add entries with `None` weights and `hf-hub:` names, and only `check_clip_model` and the tokenizer path need checking. The checkpoint's text tower stays on CPU as it does now.

### Gaps
- **Priority experiment, not run:** FARE4 ViT-B/32, B/16 and ConvNeXt and CLIPAG through the real sampler against their clean parents, and with TV at 0, using `pilot.py`.
- fp16 against fp32 gradient agreement on robust towers was written as part of `grad_quality.py` but never ran, because it needs the GPU.
- CLIPAG's gradient statistics were not measured here. Only its checkpoint conversion and a 4 step CPU smoke test ran.
- No paper found uses FARE or TeCoA for diffusion guidance.
- CLIP-JEM's weights being unavailable means its better FID cannot be reproduced.
- The ε sweep suggested by Srinivas et al. for guidance does not exist: FARE2 against FARE4 on ViT-L/14 is the only easy comparison and it is untested.

## 3. Which text targets and text-side losses help?

### Takeaway
Text-side changes are cheap but small. Evidence supports template averaging for recognition, not generation, and CLIPDraw found negatives often negligible. The strongest idea with real numbers is replacing the text embedding with a prior-sampled image embedding, as unCLIP does.

### Cited Findings
- **[paper]** CLIP's 80 templates added 3.5% to ImageNet zero-shot accuracy and prompt engineering about 5% in all. The templates ship with open_clip. [Source](https://arxiv.org/pdf/2103.00020)
- **[paper]** OpenAI's notebook ran forward selection over those 80 and stopped at 7: "itap of a {}", "a bad photo of the {}", "a origami {}", "a photo of the large {}", "a {} in a video game", "art of the {}", "a photo of the small {}". It also says the list was "pretty haphazard". [Source](https://raw.githubusercontent.com/openai/CLIP/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb)
- **[paper]** CLIPDraw used negatives at 0.3:1 and found them "often negligible", not a cure-all. Without augmentation, results "score high numerically" but are nonsense. [Source](https://ar5iv.labs.arxiv.org/html/2106.14843)
- **[paper]** DAS uses weighted prompts, for example −0.3 "Optical Character Recognition", −0.3 "multiple exposure", +0.3 "octane render, unreal engine, ray tracing, volumetric lighting", +0.6 "cohesive single subject", tuned by hand. [Source](https://arxiv.org/abs/2502.07753)
- **[paper]** StyleGAN-NADA's directional loss is `L = 1 − ΔI·ΔT / (‖ΔI‖‖ΔT‖)`, with `ΔI = E(x) − E(x_src)` and `ΔT = E(t) − E(t_src)`, and needs a source image and text. [Source](https://ar5iv.labs.arxiv.org/html/2108.00946)
- **[paper]** unCLIP conditioning: text-embedding decoder FID 9.16, full stack 7.99, and passing the text embedding zero-shot into the unCLIP decoder 16.55. Humans preferred the full stack 57.0% of the time. It samples 2 candidates from the prior and keeps the higher dot product with the text. [Source](https://ar5iv.labs.arxiv.org/html/2204.06125)
- **[paper]** LAION's `nousr/conditioned-prior` is an MIT-licensed text-to-image embedding prior with ViT-B/32 and ViT-L/14 checkpoints, so it matches the repo's default model. [Source](https://huggingface.co/nousr/conditioned-prior)
- **[paper]** Prompt Expansion studies learned rewriting, and Fooocus ships a small GPT-2 expander whose model repo lists conflicting licenses (CC-BY-NC against AGPL). [Source](https://arxiv.org/abs/2312.16720v1) [Source](https://huggingface.co/LykosAI/GPT-Prompt-Expansion-Fooocus-v2)
- **[paper]** CLIP Interrogator is MIT, uses 6.3 GB by default and 2.7 GB in low-VRAM mode, and finds words matching an image, so it suits image prompts rather than text ones. [Source](https://github.com/pharmapsychotic/clip-interrogator)
- **[paper]** Typographic attacks work because CLIP reads text in images. [Source](https://arxiv.org/pdf/2412.05538)
- **[measured here]** The repo's README records nonsense text written across a sushi image by the big SigLIP2 models. [Source](../../README.md)

### Inferences
- Multi-target forms the repo can run through today's `prompts` list, which sums per-target distances:
  - Mean of distances to templates, `L = (1/K) Σ_k d(E(x), t_k)`: already works by passing `K` prompts, at no VRAM cost after encoding.
  - Distance to mean template, `L = d(E(x), normalize(Σ_k t_k))`: needs one line in `clip_guided_diffusion`, and it is what the CLIP paper actually measured.
  - Negatives: `L = d(E(x), t+) − λ d(E(x), t−)`, supported by `text:-0.3` syntax, normalised through `weights /= weights.sum().abs()`, so a negative also rescales the positive.
- Softmax cross-entropy against distractors is `L = −log[exp(τ x̂·t+) / (exp(τ x̂·t+) + Σ_j exp(τ x̂·d_j))]`, τ=100, with `d_j` the 1000 ImageNet texts the repo already caches, minus the nearest ten to the prompt. Unlike cosine it only pushes away from distractors the image is drifting toward, so it is a form of automatic negative prompt. VRAM cost is nil.
- Mean embedding across cutouts versus mean of per-cutout distances: `d(mean_c E(x_c), t)` lets individual crops disagree as long as they average to the target, which is more like a whole-image view. `mean_c d(E(x_c), t)`, the repo's form, asks each crop to match on its own. Neither has been compared in the pipeline.
- The prior route: encode text, sample `z_img ~ p(z | t)` on CPU (the model is small), then use `z_img` or a mix with `t` as `target_embeds`. Existing code is unchanged after that, and it can be run before U-Net load, so the runtime VRAM is nil. The caution is that the prior was trained on ViT-B/32 from LAION, so its domain is not ImageNet.

### Gaps
- **Priority experiment, not run:** templates, a negative at −0.3, cross-entropy against ImageNet texts, and mean embedding across cutouts in the real sampler. All are in `pilot.py`, and passed the 4 step CPU smoke test on 1 prompt.
- No paper measuring softmax cross-entropy against distractors for CLIP guidance was found.
- No generation-side measure of template ensembling was found. The CLIP paper's numbers are recognition accuracy.
- VRAM and time for `nousr/conditioned-prior` were not measured. Its checkpoints were not tested on this pipeline.

## 4. Which losses beyond a global embedding are worth adding?

### Takeaway
Aesthetic predictors are the only one with a direct generation precedent and trivial cost, but they are folklore in CLIP guidance and vulnerable to adversarial optimisation. Patch tokens, Grad-CAM and attention losses have no guidance evidence, so they are experiments rather than recommendations.

### Cited Findings
- **[folklore]** Mainline Disco Diffusion has no aesthetic loss: "aesthetic" appears 0 times in the notebook. It is Majesty that adds one: `aes_loss = aesthetic_model_32(F.normalize(image_embeds)).mean()` at scale 500 against cgs 5000, with presets of 200 and a Replicate default of 400. [Source](https://raw.githubusercontent.com/alembics/disco-diffusion/main/Disco_Diffusion.ipynb) [Source](https://github.com/multimodalart/majesty-diffusion)
- **[paper]** Simulacra Aesthetic Models are MIT, Ridge-fit linear heads on CLIP embeddings for ViT-B/16, B/32 and L/14, with input `normalize(e) · √d`. [Source](https://github.com/crowsonkb/simulacra-aesthetic-models)
- **[paper]** LAION's predictor is MIT, linear on ViT-B/32 and L/14 embeddings. The improved predictor is Apache-2.0, an MLP (768→1024→128→64→16→1) on ViT-L/14. Predictor v2.5 uses SigLIP so400m-384, with 5.5+ called great. [Source](https://github.com/LAION-AI/aesthetic-predictor) [Source](https://github.com/christophschuhmann/improved-aesthetic-predictor) [Source](https://github.com/discus0434/aesthetic-predictor-v2-5)
- **[paper]** Sequential Aesthetic Guidance used a weighted mix of LAION aesthetic, PickScore and HPSv2 in Stable Diffusion, and scored 8.17 against 9.71 for SD v1.5, 9.18 for FreeDOM and 9.78 for DOODL on 10 prompts with one image each. The table's arrow marks lower as better, while the text calls it aesthetic scores, so the direction needs checking. [Source](https://ar5iv.labs.arxiv.org/html/2312.12030)
- **[paper]** Aesthetic gradients nudge the text embedding toward a user's image set, and find that appending an aesthetic keyword to the prompt has a negligible or dim effect. [Source](https://arxiv.org/abs/2209.12330v1)
- **[paper]** `pyiqa` exposes differentiable `topiq_nr`, `topiq_iaa`, `musiq-ava`, `nima-vgg16-ava`, `clipiqa+` and `laion_aes` with `as_loss=True`, but is under a noncommercial PolyForm and NTU S-Lab license, and NIQE, BRISQUE, PI and NRQM have no backward pass. [Source](https://github.com/chaofengc/IQA-PyTorch)
- **[paper]** TOPIQ reaches AVA SRCC .733 on ResNet50 and .791 on Swin. [Source](https://ar5iv.labs.arxiv.org/html/2308.03060)
- **[paper]** No-reference IQA models can be attacked by small perturbations, so maximising one by gradient may not raise perceived quality. [Source](https://arxiv.org/html/2402.17533v2) [Source](https://arxiv.org/abs/2404.13277v2)
- **[paper]** MaskCLIP removes the last layer's global pooling and takes per-patch value features, matching them to text with a 1×1 convolution. SCLIP replaces the last attention with Correlative Self-Attention, arguing that CLIP's dense weakness is spatial misalignment rather than poor features. CLIP-DINOiser and ClearCLIP refine the idea. [Source](https://ar5iv.labs.arxiv.org/html/2112.01071) [Source](https://arxiv.org/html/2312.01597v4) [Source](https://arxiv.org/html/2312.12359v2)
- **[paper]** CLIP Surgery reports that raw CLIP has "opposite visualization", blamed on query-key attention, and noisy activations from redundant features, and modifies the inference path to fix both. [Source](https://arxiv.org/abs/2304.05653v2)
- **[paper]** Zero-Shot Spatial Layout turns cross-attention maps into a guidance loss. It needs a cross-attention U-Net, which ImageNet ADM has not got. [Source](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Couairon_Zero-Shot_Spatial_Layout_ICCV_2023_supplemental.pdf)
- **[paper]** CLIPasso adds L2 on intermediate RN101 layers 3 and 4 for geometry, on top of a semantic CLIP loss, with later work setting that weight near 0.1. RN101 is reported as geometric and RN50x16 as conceptual. [Source](https://arxiv.org/abs/2202.05822v1) [Source](https://arxiv.org/pdf/2304.08204) [Source](https://arxiv.org/pdf/2505.04813)
- **[paper]** DreamSim concatenates CLIP, OpenCLIP and DINO features under an LPIPS-like head, MIT, with single-branch options. [Source](https://github.com/ssundaram21/dreamsim)
- **[paper]** In an inverse-problems study a direct DINOv2 consistency term on decoded images gave worse perceptual quality than aligning internal diffusion features. [Source](https://arxiv.org/pdf/2511.16870)
- **[paper]** DINOv2 feature space aligns with human preference better than Inception. That concerns metrics, not losses. DINOv2 code is Apache-2.0; DINOv3 weights are gated under their own license. [Source](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Karras_Analyzing_and_Improving_CVPR_2024_supplemental.pdf) [Source](https://github.com/facebookresearch/dinov2) [Source](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
- **[paper]** The self-similarity of DINO keys is an established structure representation. In DDPM-based translation, matching it alone gave weak regularisation, and the authors added a patchwise InfoNCE loss on the keys. [Source](https://arxiv.org/pdf/2311.12193) [Source](https://arxiv.org/pdf/2209.15264)
- **[paper]** A sliced Wasserstein loss matches full feature distributions at `O(M log M)`, where a Gram matrix matches only second order statistics. [Source](https://openaccess.thecvf.com/content/CVPR2021/papers/Heitz_A_Sliced_Wasserstein_Loss_for_Neural_Texture_Synthesis_CVPR_2021_paper.pdf)
- **[paper]** SDS: `∇_x L = E_{t,ε}[w(t)(ε̂(x_t, t, y) − ε)]`, with no U-Net Jacobian. SDS-CLIP reports diffusion scores beat CLIP on Winoground for much more time. [Source](https://ar5iv.labs.arxiv.org/html/2302.08510) [Source](https://arxiv.org/pdf/2307.09233.pdf)
- **[paper]** LPIPS is BSD-2-Clause, and the repo already depends on it for `init_scale`. [Source](https://github.com/richzhang/PerceptualSimilarity)

### Inferences
- Aesthetic term: `L_aes = −λ · (w·(normalize(E(x))·√d) + b)` per cutout, added to CLIP's distance. It reuses the embeddings already computed, so no extra VRAM and no extra time beyond a matmul. Majesty's ratio of 500 to 5000 gives λ of about 0.1 against cgs. Use the ViT-B/32 SAC head with the default tower. SAC weights are 3 KB files.
- Judging an aesthetic term with the head it optimises is circular. Judge with a head on a different backbone: optimise with SAC B/32 and score with SAC B/16, as `pilot_judge.py` was written to do.
- Patch-token text loss would be `L = (1/|P|) Σ_p (1 − cos(proj(z_p), t))`, or its top-k version so that only regions meant to show the subject pull. It reuses the forward pass on ViT towers, where `visual.forward_intermediates` or a hook gives `z_p`. It costs almost nothing, but the projection is only trained for the class token, which is why MaskCLIP drops the last attention.
- At 128px CLIP sees heavily upsampled images, so dense patch-text alignment may not carry much information anyway, and ImageNet-class conditioning already sets the layout. SDS adds nothing to this repo's own U-Net, which already runs every step. Using another diffusion model for SDS is a backbone change and belongs to that research thread.
- Style reference, when an image prompt is meant as a style: Gram loss `‖G_l(x) − G_l(y)‖²`, `G_l = F_l F_lᵀ / N`, or sliced Wasserstein `E_θ ‖sort(θ·F_l(x)) − sort(θ·F_l(y))‖²` on RN50 or ConvNeXt stages. VRAM is the tower's, because stage activations already exist. ConvNeXt-B-w fp16 is the natural host.

### Gaps
- **Priority experiment, not run:** the SAC aesthetic term at λ=0.1 in the real sampler. It is in `pilot.py`, and passed the 4 step CPU smoke test on 1 prompt.
- No use of Grad-CAM as a target or consistency term in diffusion sampling was found. Its form would be `M = ReLU(Σ_k α_k A_k)`, `α_k` the mean of `∂s/∂A_k`, and its second-order backward cost is unmeasured.
- MaskCLIP, SCLIP and CLIP Surgery have only been evaluated on segmentation.
- MUSIQ-specific attacks are unverified. NIMA, TOPIQ, CLIP-IQA and MUSIQ as guidance on 128px samples were not tested, and `pyiqa`'s license rules out shipping it in any non-noncommercial use.
- CAFE's aesthetic model is AGPL-3.0 and BEiT based, and is not evaluated here.

## 5. Which natural image priors beyond TV are worth having?

### Takeaway
Feature visualisation research says TV and blur discourage real edges, and that preconditioning the gradient toward low frequencies works better. DAS's multi-resolution parameterisation can be applied here as a preconditioner with no VRAM cost.

### Cited Findings
- **[paper]** Feature Visualization: TV and blur penalties discourage legitimate edges, while transformation robustness and Fourier or decorrelated preconditioning work best, and Laplacian pyramid gradient normalisation is an established alternative. [Source](https://distill.pub/2017/feature-visualization/)
- **[paper]** DAS parameterises an image as `I = ½ + ½ tanh(Σ_r resize(P_r))` for `r` from 1×1 to 224×224, and generates with SGD lr 0.2 over 100 steps, 32 augmentations in a batch, noise std 0.2 and ±56 px jitter. Neither shift nor noise is enough alone, and images have a spectrum slope near −2. [Source](https://arxiv.org/abs/2502.07753)
- **[paper]** The rebuttal to Ensemble Everything Everywhere keeps the multi-resolution prior as useful for visualisation and generation from a single CLIP. [Source](https://arxiv.org/pdf/2411.14834v1)
- **[paper]** Second order Total Generalized Variation removes TV's staircasing, which is the "cartoon-like" look TV gives. [Source](https://imsc.uni-graz.at/optcon/projects/bredies/tgv.html) [Source](https://imsc.uni-graz.at/mobis/imaging10/abstracts/knoll.pdf)
- **[folklore]** Majesty's multi-scale TV is `tv_scales=[1200]+[600]*3` on the image at 1, ½, ¼ and ⅛ size, and Disco's later defaults set TV to 0 where its older tutorial used 150. [Source](https://huggingface.co/datasets/multimodalart/latent-majesty-diffusion-settings/raw/main/defaults_v1_3.cfg) [Source](https://raw.githubusercontent.com/alembics/disco-diffusion/main/Disco_Diffusion.ipynb)
- **[paper]** Natural images have roughly `1/f²` power spectra, a fact frequency-domain guidance work such as FreSca builds on. [Source](https://arxiv.org/html/2504.02154v1)
- **[paper]** CLIPAG's generator-free synthesis worked with no TV at all, which suggests that in a robust model the need for TV falls away. [Source](https://ar5iv.labs.arxiv.org/html/2306.16805)

### Inferences
- The repo's terms: range `‖x̂₀ − clamp(x̂₀, −1, 1)‖²`, computed through the U-Net, and L2 TV `mean(Δ_x² + Δ_y²)` on `x_in`. L2 TV is a quadratic low-pass, so unlike TV proper it does not preserve edges at all. With the sampler's RMS cap, TV also spends budget that CLIP could use, since cap and sum share a single RMS.
- DAS as a preconditioner. Optimising `Σ_r R_r P_r` by gradient on `P_r` moves pixels along `Σ_r R_r R_rᵀ g`, so without changing anything else the sampler can use:

```python
def multires(g, power=1):
    out = g.clone()
    for s in (2, 4, 8, 16, 32):
        out = out + s ** power * tf.interpolate(tf.avg_pool2d(g, s), size=g.shape[-2:], mode="bilinear", align_corners=False)
    return out
```

  It runs after `th.autograd.grad` and before the RMS cap, so the cap still bounds it, and it costs no VRAM. This preconditioner is my derivation from DAS's parameterisation, not something DAS or anyone else has tested in a sampler, and DAS's own objects would weight scales differently because of how resizing normalises them.
- Laplacian normalisation: `g' = Σ_k L_k(g) / std(L_k(g))`, same place, same cost. It equalises energy per band, whereas `multires` boosts low bands.
- Spectrum loss: `L = Σ_r (log P_x(r) + β log r − c)²`, β of about 2, on `x̂₀`. It would be cheap through FFT but needs `x̂₀`'s graph, so U-Net backward. It targets an average over natural images rather than a given image. A saturation term in the repo's existing `sat_scale`, `|x_in − clamp(x_in)|`, is a one-sided version of the range loss, and `X0_CLAMP` already bounds the sampler's `x̂₀`.
- TGV: `min_{w} α₁‖∇x − w‖₁ + α₀‖ε(w)‖₁`. It adds an inner minimisation, so it is much less simple than the alternatives. It should only be tried if TV's look turns out to be the visible problem.

### Gaps
- **Priority experiment, not run:** TV at 0, 50 and 150, multi-scale TV, and the `multires` preconditioner at powers 1 and 2, in the real sampler. All are in `pilot.py`. Multi-scale TV and `multires` passed the 4 step CPU smoke test on 1 prompt. So TV's effect on this repo at 128px is still unknown.
- No paper was found that uses a spectrum loss in CLIP guidance.
- No paper measures TV's effect on CLIP-guided samples at low resolution.
- TGV inside a sampler was not found and not tested.

## 6. Can non-CLIP classifiers or noise-aware models guide better?

### Takeaway
Noise-aware guides skip the U-Net backward, which is the most expensive part of guidance here, and OpenAI's classifier and GLIDE's noised CLIP are the only public ones. GLIDE itself showed classifier-free guidance beating noised CLIP guidance on human Elo, so this is a cost win before it is a quality one.

### Cited Findings
- **[paper]** OpenAI publishes noise-aware classifiers at 64, 128, 256 and 512, MIT, with FID 2.07, 2.97, 4.59 and 7.72 at scales 1.0, 0.5, 1.0 and 4.0; unguided 128 is 5.91 per Kawar et al. The 128 file is 170 MB and resolves at the published URL. [Source](https://raw.githubusercontent.com/openai/guided-diffusion/main/README.md) [Source](https://ar5iv.labs.arxiv.org/html/2208.08664)
- **[paper]** GLIDE trains noised CLIP as `f(x_t, t)`, a 64×64 ViT with 4×4 patches, and guides with `μ̂ = μ + sΣ∇(f(x_t)·g(c))`. Human Elo was −88.6 and −106.2 unguided, −73.2 and 29.3 with CLIP guidance at 2.0, and 82.7 and 110.9 with CFG at 3.0, and it hypothesises CLIP guidance finds adversarial examples. [Source](https://ar5iv.labs.arxiv.org/html/2112.10741)
- **[paper]** Those noised CLIP weights are public: `clip_image_enc.pt` 345.8 MB and `clip_text_enc.pt` 286.8 MB. The repo is MIT and was archived on May 29 2026. Config is 512-dimensional embeddings, 768 width, 12 blocks, cosine schedule, 1000 steps, logit scale 100. Its model card reports 137M filtered pairs, with no people in the training data. [Source](https://github.com/openai/glide-text2im)
- **[paper]** Kawar et al. train a robust noise-aware classifier: 128px FID 2.97 to 2.85, precision .78 to .82, recall .59 to .56 at s from 0.5 to 1, with checkpoints on Google Drive. [Source](https://github.com/bahjat-kawar/enhancing-diffusion-robust) [Source](https://ar5iv.labs.arxiv.org/html/2208.08664)
- **[paper]** BIGRoC post-processes with a robust classifier for 128px FID 2.53 and 256px 3.63. Discriminator guidance reports 1.83 at 256. [Source](https://arxiv.org/pdf/2108.03702) [Source](https://arxiv.org/abs/2211.17091v3)
- **[paper]** Madry Lab's robust ImageNet checkpoints are MIT on Hugging Face, including `resnet50_l2_eps3.ckpt` (205 MB) and sibling ε values from 0 to 5. [Source](https://huggingface.co/madrylab/robust-imagenet-models)
- **[paper]** Singh, Croce and Hein find ConvNeXt with a convolutional stem the most robust ℓ∞ family on ImageNet, with ConvNeXt-T + ConvStem at 72.7% clean and 49.5% robust. These are the checkpoints behind RobustBench's ImageNet entries, and they are ordinary, noise-unaware classifiers. [Source](https://arxiv.org/pdf/2303.01870)
- **[measured here]** `create_classifier` is broken in the crowsonkb `guided-diffusion` fork the repo depends on: `EncoderUNetModel` passes `use_neighborhood_attention` to `ResBlock`, which does not take it, and its attention layers get `spatial=None`, which `QKVAttention` cannot iterate. With both patched at runtime, OpenAI's `128x128_classifier.pt` loaded strictly and ran. Any implementation needs those two fixes or its own copy of the classifier.
- **[paper]** SDG fine-tunes CLIP on noise only for FFHQ and LSUN, so it does not cover ImageNet. [Source](https://github.com/xh-liu/SDG_code)
- **[paper]** GradCheck reports near-orthogonal gradients from a non-robust classifier. Guiding through `x̂₀` plus Adam took FID 126.5 to 24.3 and accuracy 15.6 to 85.9% on a small dataset, while Adam alone did not help. Combining those with a robust model made things worse. [Source](https://arxiv.org/html/2406.17399v1)
- **[paper]** UPainting combines CFG with CLIP on `x̂₀`, skipping the first 10 steps of 50 DDIM, and reports better fidelity. [Source](https://arxiv.org/pdf/2210.16031)

### Inferences
- A noise-aware guide reads `x_t` directly, so the sampler can run the U-Net under `no_grad`. That removes the U-Net's backward peak and most of guided step time. Peak at 128px should fall to about `G` plus guide, without the U-Net's graph.
- Adding classifier guidance to CLIP is `x̂₀' = x̂₀ + s σ²/α ∇log p(y|x_t) + σ²/α · g_clip`, kept outside the CLIP RMS cap and applied on all steps, because the classifier is valid at high noise. ADM's 128px scale of 0.5 is the natural start. The cost is a classifier forward plus backward at each model call.
- Noised CLIP on this repo's U-Net needs two adaptations, because its noise schedule and resolution differ. A 2×2 area pool of 128px `x_t` halves white noise, so `SNR₆₄ = 4 α²/σ²`, then `ᾱ' = SNR₆₄ / (1 + SNR₆₄)`, the view is rescaled by `√ᾱ'/α`, and `t'` is the nearest step in GLIDE's cosine schedule. 256px would pool by 4 with `SNR × 16`.
- OpenAI's classifier would add ImageNet realism, not text fidelity. It also shares the sampler's weakness: it pushes toward the single class picked, which may not fit a prompt such as a pencil sketch. A robust ImageNet ConvNeXt or ResNet has the same limit, so it can only be a helper term.

### Gaps
- **Priority experiment, not run:** ViT-B/32 guidance plus OpenAI's 128px classifier at s of 0.5 and 1, and class-only baselines. `pilot2.py` holds it and passed the 4 step CPU smoke test on 1 prompt.
- GLIDE's noised CLIP was not measured here. An experiment to run it through this repo's fast sampler was written, but the session's permission classifier refused the run because it would execute code cloned from `openai/glide-text2im`. The cloned source and downloaded weights were never executed or loaded. Measuring it needs that repo allowed as a dependency.
- No public noise-conditional CLIP for ImageNet at 128px or above was found. GLIDE's is 64×64 and SDG's covers other datasets. Nobody seems to have fine-tuned a modern CLIP on noise and released it. Fine-tuning is out of scope here but is the most direct fix for high-noise guidance.
- GLIDE's model-card claims about its training data were read, but the absence of people in the data means `a portrait` prompts may be poor. Untested.
- Kawar's weights are on Google Drive with no stated license on the checkpoints. They were not downloaded.
- ImageNet logits from a robust ConvNeXt or `resnet50_l2_eps3.ckpt` as a helper term on `x̂₀` were not tested. Licenses for the RobustBench ConvNeXt checkpoints were not verified.

## 7. Can image prompts, retrieval or a fast generated init improve results?

### Takeaway
A generated or retrieved image is a stronger target than text, because CLIP then guides toward a real image instead of a sparse text embedding. Retrieval has licensing and availability problems, so LCM or SD-Turbo as a local init or image prompt is the practical route, and it needs a measured test.

### Cited Findings
- **[paper]** RDM conditions on CLIP nearest neighbours from a database and showed that retrieval improves generation. [Source](https://arxiv.org/abs/2204.11824v2)
- **[paper]** Re-LAION-5B was released 30 Aug 2024 with 2,236 links removed, in Research and Research-Safe versions. A ViT-H/14 kNN index was published earlier. [Source](https://laion.ai/blog/relaion-5b/) [Source](https://laion.ai/blog/h14_clip_retrieval/)
- **[measured here]** `https://knn.laion.ai/knn-service`, the backend named in clip-retrieval's README, did not answer within 12 s on 2026-11-01, so a hosted retrieval path cannot be relied on. [Source](https://github.com/rom1504/clip-retrieval)
- **[paper]** SD-Turbo is a distilled SD 2.1 at 512×512 for 1 to 4 steps, with `guidance_scale=0.0`, no negative prompt, and commercial terms through Stability's license page. SDXL-Turbo's card has a non-commercial community license. [Source](https://huggingface.co/stabilityai/sd-turbo) [Source](https://huggingface.co/stabilityai/sdxl-turbo)
- **[paper]** `SimianLuo/LCM_Dreamshaper_v7` is MIT, with 1 to 8 steps recommended. [Source](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7)
- **[paper]** Universal Guidance covers forward and backward steps with generic guides, which is how image-space guidance generally joins a sampler. [Source](https://arxiv.org/pdf/2302.07121)
- **[measured here]** The repo already takes `-init` and `-skip`, starts partway down the schedule, and `encode_image_prompt` encodes cutouts of an image into targets. [Source](../../README.md)

### Inferences
- Three ways to use a fast image `y`, each needing only `cgd.py`'s existing options:
  - Init only: `x_{t_s} = α y + σ ε`, then guide with text as now. It keeps layout and colours.
  - Image prompt only: encode cutouts of `y` as targets, so CLIP pulls toward a real image's embedding. It keeps semantics but not layout.
  - Both, with `-init` at a high `-skip` and text and image prompts mixed.
- SD models must leave the GPU before the U-Net loads, or be run before it, as both will not fit at 256px on 8 GB. Loading a second model per run will likely cost more than sampling from it, so this is only cheap per image in a long-lived process.
- Text embeddings of ViT-B/32 and a generated image's embeddings have different distributions, which is what unCLIP's prior exists to bridge. A generated image acts like a free one-sample prior, and section 3's prior is the cheaper way to get the same thing.
- If the U-Net brings a class and the init brings a different subject, the class conditioning fights the init. Choosing the ImageNet class from the image's embedding would help.

### Gaps
- No end-to-end measure of LCM or SD-Turbo as init for ADM ImageNet models exists. It was not run here, and it needs `diffusers` and weights outside the project.
- SD-Turbo and LCM VRAM on these GPUs are unverified. Neither card gives a number.
- The current state of LAION kNN services beyond this one check is unknown.
- License of images returned by retrieval is per image and unknown, so a retrieved image would not be safe to ship.

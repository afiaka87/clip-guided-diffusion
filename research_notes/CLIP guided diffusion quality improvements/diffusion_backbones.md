# Diffusion backbones for CLIP guidance on 8 GB and 12 GB GPUs

> **Status, 2026-11-02:** These are the notes as written on 2026-11-01, kept as the record of what was known then. Later measurements overturned some of it, including its leading recommendations. [The report](../../reports/CLIP%20guided%20diffusion%20quality%20improvements.md) has the current results, and wins wherever the two disagree.

## Which models should replace or complement the ADM U-Net, ranked for this hardware?

### Takeaway
SD-Turbo is the best first step on either card: ungated, cheap to run, and the base with the largest gain from gradient steering among ReNO's four GenEval figures. For quality beyond it, FLUX.2 klein 4B is the strongest open model that fits, on 12 GB at bf16 and on 8 GB only with fp8 storage. All guided VRAM below is an estimate unless the measurement section marks it as measured on the card.

### Cited Findings
Ranked on quality per unit of work, counting license and gating against a model. Guided figures are my estimates from the measurement section, as `tiny_ckpt` (through the denoiser, tiny decoder, checkpointed) and `full` (through the denoiser and full VAE). Download is denoiser plus decoder plus text encoders at the lightest published precision, read from the Hugging Face API. SD-Turbo's forward and `tiny_ckpt` figures are counted on SD 2.1's U-Net, which has the same 865.9M parameters. Its `full` figure is projected from SD 1.5 on the card. All are one run at seed 0, with no interval.

| # | Model, weights | px | Params | Download | Inference GiB | Guided GiB, tiny_ckpt / full | Steps | License, gating | Text | How guidance attaches |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo) | 512 | 866M | 2.6 GB | 2.81 | 2.79 / ~5.9 | 1 to 4 | [Stability AI Community](https://huggingface.co/stabilityai/sd-turbo/blob/main/LICENSE.md), ungated. Commercial use needs registration, and the grant ends above USD 1M annual revenue | Yes | eps. Noise optimisation through one step, or per step on its own timesteps |
| 2 | [Segmind-Vega](https://huggingface.co/segmind/Segmind-Vega) + [VegaRT](https://huggingface.co/segmind/Segmind-VegaRT) | 1024 | 745M | 3.3 GB | ~4.0 | ~3.6 / over 12 | 25, or 2 to 8 with VegaRT | Apache-2.0, ungated | Yes | eps, so `sampling.py` maps directly. taesdxl |
| 3 | [FLUX.2 klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | 1024 | 3.9B | 16 GB | ~9.8 at bf16 | ~10 at bf16, ~6.4 with fp8 storage / not estimated | 4 | Apache-2.0, ungated | Yes | Flow, `x0 = x - sigma * v`, dynamic shift. taef2 |
| 4 | [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) + [Lightning](https://huggingface.co/ByteDance/SDXL-Lightning) or [Hyper-SD](https://huggingface.co/ByteDance/Hyper-SD) | 1024 | 2.6B | 6.9 GB plus adapter | ~7.4 | ~7.1 / over 12 | 1 to 8 | OpenRAIL++ for SDXL and Lightning, Hyper-SD has no license tag, ungated | Yes | eps. Tied to the checkpoint's own timesteps. taesdxl |
| 5 | [SD 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) + [LCM-LoRA](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5) | 512 | 860M | 2.1 GB, already cached here | 2.9 (measured) | 2.85 / 5.92 (measured) | 20, or 2 to 8 | CreativeML OpenRAIL-M, ungated | Yes | eps, so `sampling.py` maps directly. TAESD |
| 6 | [PixArt-Sigma](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-512-MS) | 512, 1024 | 611M | 2.4 GB, plus VAE and T5 from the [1024 repo](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS) | 2.3 at 512 | 2.3 / ~5.5 at 512 | 20 or more | OpenRAIL++, ungated | Yes | eps with learned sigma, like ADM's `[:, :shape[1]]`. T5 stays on the CPU |
| 7 | [Sana Sprint 0.6B](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers), [1.6B](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers) | 1024 | 0.6B, 1.6B | 7.7, 9.7 GB | ~3.4, ~5.3 | no tiny decoder / ~7.5, ~9.4 with my DC-AE checkpointing | 1 to 4 | Apache-2.0 tag, ungated | Yes | `trigflow` under `SCMScheduler`, a third parameterisation. Full DC-AE only |
| 8 | [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | 1024 | 6.2B | 33 GB | over 12 at bf16 | ~8.6 with fp8 storage / not estimated | 8 NFE | Apache-2.0, ungated | Yes | Flow, static shift 3. Inputs are lists of tensors. taef1 |
| 9 | [Kandinsky 2.2](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder) + [prior](https://huggingface.co/kandinsky-community/kandinsky-2-2-prior) | 512 to 768 | 1.25B | 5.3 GB + 10.6 GB prior | 3.65 at 512 | no tiny decoder / run failed, not estimated | 50 | Apache-2.0 tag, ungated | Via CLIP image embeddings | U-Net takes an image embedding, so CLIP steers it by design. MoVQ decoder |
| 10 | [EDM2-S](https://raw.githubusercontent.com/NVlabs/edm2/main/README.md) | 512 | 280M | 0.56 GB plus VAE | not measured, weights alone about 1.2 | not measured | 63 NFE | CC BY-NC-SA, direct download | ImageNet classes | Same recipe as today: class condition plus CLIP. Latent, needs a decoder |
| 11 | [cc12m_1_cfg mirror](https://huggingface.co/multimodalart/crowsonkb-v-diffusion-cc12m-1-cfg) | 256 | 602M | 2.4 GB | about ADM 256 | ~5.5, no decoder | 50 | MIT tag, ungated | CLIP embedding | v prediction, pixel space, same CLIP family guides it |

Counted against, and left out of the ranking:
- SDXL-Turbo is non-commercial: `sai-nc-community` license on [the repo](https://huggingface.co/stabilityai/sdxl-turbo).
- [SD 3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) and [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) are gated at their official repos. Ungated copies exist at [Comfy-Org](https://huggingface.co/Comfy-Org/stable-diffusion-3.5) and [unsloth](https://huggingface.co/unsloth/FLUX.1-schnell). The Comfy-Org copy keeps the Stability AI Community License, and its old `-fp8` URL redirects there. FLUX.1's 11.9B parameters are about 22 GiB at bf16, so it needs fp8 even on 12 GB.
- [FLUX.3 klein](https://huggingface.co/black-forest-labs/flux-3-image-klein) is gated under the FLUX Kommunity License with a 9B prompt upsampler. The 5.6B model itself would be close to Z-Image's cost.
- [Stable Cascade](https://huggingface.co/stabilityai/stable-cascade) is non-commercial, and its decode path includes a diffusion stage.
- [Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo), [Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw) and [DeepFloyd IF](https://huggingface.co/DeepFloyd/IF-I-M-v1.0) are gated under their own licenses. Krea-2's transformer is also 26 GB.
- [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image), [HiDream-I1](https://huggingface.co/HiDream-ai/HiDream-I1-Fast) and [AuraFlow](https://huggingface.co/fal/AuraFlow-v0.3) are open, but their weights do not fit: AuraFlow is 12.8 GiB at fp16, and the other two repos are 47 to 58 GB.

`stabilityai/stable-diffusion-2-1` and its `-base` and `-unclip` repos now return 401. Community copies live at [sd2-community](https://huggingface.co/sd2-community/stable-diffusion-2-1).

### Inferences
- The ranking weighs evidence and effort over raw model quality. klein is the best image model here, but SD-Turbo is ahead because it fits both cards, needs the smallest adapter, and is the smallest of the models ReNO measured.
- Vega ranks above SDXL because it gives 1024 output on 8 GB with room to spare, under Apache-2.0. Expect lower quality than SDXL, since it is a smaller distillation of it. I have no comparison figures for that.
- Kandinsky 2.2 and cc12m_1_cfg are the conceptually neat fits, because they condition on CLIP embeddings directly. Neither is a route to modern quality on its own evidence, so they rank low despite that.

### Gaps
- No quality comparison across these models under CLIP guidance exists, so the order rests on published unguided reputation, ReNO's numbers and my memory estimates.
- Hyper-SD's repo carries no license tag, so its terms need reading before use.

## What does one guided step cost on a latent model, and what changes the answer?

### Takeaway
Once a tiny decoder and gradient checkpointing are in, weights set the limit and the backward pass barely matters. At 512, a guided step through the denoiser costs 0.4 to 0.6 GiB above resident weights, against about 3.6 GiB through the full VAE with no checkpointing. In a small test (2 prompts, 1 seed each, every 95% interval including zero), all four cheap and full paths raised the guiding CLIP score on both prompts, so TAESD and x0 gradients can steer, but how well is not established.

### Cited Findings
Published memory figures:
- ReNO's A100 peaks through a one-step generator and full VAE run from 15 to 50 GB, as cited in the section on CFG, and the paper's rewards include large CLIP models — [ReNO](https://ar5iv.labs.arxiv.org/html/2406.04312). They are not comparable with fp16 steps through a single small CLIP.
- FLUX.2 klein 4B's card says about 13 GB of VRAM for inference — [card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B). Z-Image-Turbo's says 16 GB — [card](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo). Community write-ups put Z-Image at 8 GB with fp8 and 14 to 16 GB at bf16, and klein 4B at 8 to 13 GB. These are secondary sources — [Thunder Compute](https://www.thundercompute.com/blog/z-image-turbo-comfyui), [Apatero](https://apatero.com/blog/flux-klein-vs-z-image-turbo)

Decoders:
- TAESD's decoder is 1.2M parameters against the SD VAE decoder's 49M. It scales linearly with pixels, costs "nearly zero", and loses fine detail. The README mentions image-space losses on latent models but says nothing about gradient quality — [README](https://raw.githubusercontent.com/madebyollin/taesd/main/README.md)
- Tiny decoders exist for SD 1/2, SDXL, SD3, FLUX.1, FLUX.2, Z-Image and others, under MIT — [README](https://raw.githubusercontent.com/madebyollin/taesd/main/README.md). There is none for DC-AE, so Sana has only its full decoder: `madebyollin/taesana` returned 401 when I checked.
- `AutoencoderDC` sets `_supports_gradient_checkpointing = False` in diffusers 0.42.1, and enabling it raises `ValueError` — [source](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_dc.py). Its decoder is 159M parameters, with 128 channels at full resolution, from the [Sana Sprint VAE config](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers/blob/main/vae/config.json)

Precision:
- The SDXL VAE produces NaNs at fp16 because its internal activations are too large. The fp16-fix repo rescales it, with slight output differences — [sdxl-vae-fp16-fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)
- On this machine `torch.cuda.is_bf16_supported(including_emulation=False)` returns False for the RTX 2070 SUPER and True with emulation, local test. The default counts emulation as support — [PyTorch docs](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_bf16_supported.md). Native bf16 starts at Ampere and older cards emulate it, generally slowly — [PyTorch forums](https://discuss.pytorch.org/t/bfloat16-native-support/117155). Turing lacks it, per a user on [NVIDIA's forum](https://forums.developer.nvidia.com/t/quadro-rtx-6000-does-not-handle-bf16-please-make-an-update/312545). Both are forum answers, not NVIDIA documentation, and the first thread adds that some non-high-end Ampere cards may have it disabled.
- A source conflict on Turing: one Quadro RTX 8000 user fixed FLUX.1-schnell output with `float16` — [discussion](https://huggingface.co/black-forest-labs/FLUX.1-schnell/discussions/43), while a hosting guide recommends bf16 for the same model — [GigaGPU](https://gigagpu.com/?p=12040). I saw that guide only through a search summary, because the page refuses direct fetches.
- FLUX text encoders overflow at fp16 — [diffusers PR 9097](https://github.com/huggingface/diffusers/pull/9097). A SANA-Video fp16 pipeline test produced NaNs and its example keeps the VAE at fp32 — [PR 12634](https://github.com/huggingface/diffusers/pull/12634). Sana's own examples use bf16 for the VAE — [docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/sana). I found no report on DC-AE gradients at fp16.

Memory levers in diffusers:
- Layerwise casting stores weights in fp8 and upcasts each layer to fp16 or bf16 as it runs, skipping normalisation and modulation layers because fp8 can degrade them. The docs warn it may not work with every model — [docs](https://huggingface.co/docs/diffusers/main/en/optimization/memory)
- Group offloading moves blocks to the CPU, uses less memory than model offloading and is faster than sequential offloading, which the same page calls "extremely slow" — [docs](https://huggingface.co/docs/diffusers/main/en/optimization/memory)

**My measurements.** Peak allocated GiB for one guided step, shaped like `cgd/sampling.py`: denoiser, x0, decoder, four cutouts through ViT-B/32 at fp32, then the gradient. Architectures are the real published configs with random weights at fp16, which does not change memory. Text encoders are excluded, on the assumption they run on the CPU first.

Modes: `full` is through denoiser and full VAE. `ckpt` is the same, checkpointed. `tiny` is through denoiser and tiny decoder. `tiny_ckpt` is that with a checkpointed denoiser. `x0_vae` and `x0_tiny` stop at x0, so the denoiser needs no backward pass.

The sources differ and the table marks each: **G** measured on the RTX 2070 SUPER. **C** counted from the CPU allocator timeline at doubled precision and halved. **P** projected from a fit in pixels, in depth, or from another row.

| Model | px | Weights | forward | full | ckpt | tiny | tiny_ckpt | x0_vae | x0_tiny | Source |
|---|---|---|---|---|---|---|---|---|---|---|
| SD 1.5 | 512 | 2.33 | 2.93 | 5.92 | 4.14 | 3.43 | 2.85 | 5.20 | 2.71 | G |
| SD 2.1, SD-Turbo's architecture | 512 | 2.34 | 2.81 | ~5.9 | | 3.36 | 2.79 | | 2.69 | C, P for full from SD 1.5 |
| PixArt-Sigma XL/2 | 512 | 1.86 | 2.34 | ~5.5 | ~3.5 | 2.99 | 2.28 | | 2.22 | C, P for full and ckpt |
| PixArt-Sigma XL/2 | 768 | 1.86 | | | | | 2.76 | | 2.62 | C |
| Segmind-Vega | 512 | 2.11 | 2.58 | ~5.4 | | 2.93 | 2.54 | | 2.46 | C, P for full |
| Segmind-Vega | 1024 | 2.11 | ~4.0 | over 12 | ~9.4 | ~5.1 | ~3.6 | | ~3.3 | P |
| SDXL | 512 | 5.51 | 5.98 | ~9.6 | | 7.11 | 5.97 | | 5.86 | C, P for full |
| SDXL | 1024 | 5.51 | ~7.4 | over 12 | over 12 | ~11.6 | ~7.1 | | ~6.8 | P |
| Sana Sprint 0.6B | 512 | 2.25 | 2.53 | ~5.5 | ~3.6 | none | none | ~5.0 | none | C for forward, P for the rest |
| Sana Sprint 0.6B | 1024 | 2.25 | ~3.4 | over 12 | ~7.5 | none | none | over 12 | none | P |
| Sana Sprint 1.6B | 512 | 4.14 | 4.43 | | | none | none | | none | C |
| Sana Sprint 1.6B | 1024 | 4.14 | ~5.3 | over 12 | ~9.4 | none | none | over 12 | none | P |
| Kandinsky 2.2 decoder | 512 | 3.02 | 3.65 | | | none | none | | none | C |
| FLUX.2 klein 4B, bf16 | 512 | 7.94 | 8.42 | | | | ~8.5 | | 8.30 | C at reduced depth, P to full depth |
| FLUX.2 klein 4B, bf16 | 1024 | 7.94 | ~9.8 | | | | ~10 | | ~9.2 | P |
| FLUX.2 klein 4B, fp8 storage | 512 | ~4.35 | | | | | ~5.2 | | ~4.7 | C at reduced depth, P to full depth |
| FLUX.2 klein 4B, fp8 storage | 1024 | ~4.35 | | | | | ~6.4 | | ~5.7 | P |
| Z-Image-Turbo, bf16 | any | 12.19 | over 12 | | | | over 12 | | over 12 | exact weight count |
| Z-Image-Turbo, fp8 storage | 512 | ~6.5 | | | | | ~7.5 | | ~6.8 | C at reduced depth, P |
| Z-Image-Turbo, fp8 storage | 1024 | ~6.5 | | | | | ~8.6 | | ~7.8 | P |

Each cell is one run at seed 0, so no seed count above 1 and no interval applies. Peak allocation depends on shapes, not weights, so repeats would not add much. The only error estimates are the checks below, and projected cells have none of their own.

Checks on the method:
- On the card, the harness read 2.57, 5.52 and 4.30 GiB for the real ADM 128, 256 and 512 models against the README's 2.3, 5.3 and 4.1, so it runs about 0.2 GiB high.
- CPU counting of SD 1.5 gave forward 2.80, full 5.95 and ckpt 4.06 against 2.93, 5.92 and 4.14 on the card, within 5%.
- The depth fit predicted PixArt's `tiny` extra as 1.13 GiB and `tiny_ckpt` as 0.417 from quarter and half depth. Running full depth gave 1.13 and 0.415.
- The fp8 rows emulate storage at half the compute size. The first call after enabling casting read 0.85 GiB high in the profiler and repeat calls read 0.357, so I used repeat calls.
- Weights include 0.56 GiB for both CLIP towers and 0.16 GiB for the full VAE. The text tower and the VAE encoder could stay off the GPU, saving about 0.3 GiB.

Time on the card, for SD 1.5 at 512: `full` 0.50 s, `ckpt` 0.62 s, `tiny` 0.25 s, `tiny_ckpt` 0.24 s, `x0_vae` 0.34 s, `x0_tiny` 0.11 s. Each is one timed run, so these have no interval.

**Do TAESD gradients point the right way?** No source says, so I tested it: real SD 1.5, real TAESD, real ViT-B/32, on the project's cutouts, on CPU at fp32, for 2 prompts at seed 0 at each of 3 noise levels, 6 gradient draws per point. A first pass found cosines near 0.03 between TAESD and full VAE gradients, which looks fatal until you measure the floor. Two draws of cutouts through the same decoder barely agree either. Cells are mean ± 95% half-width over the 2 prompts (df = 1), so a narrow interval can simply mean the two prompts happened to land close together:

| t | VAE vs VAE, other draw | VAE vs TAESD, same draw | VAE vs VAE, mean of 3 draws | VAE vs TAESD, mean of 3 draws |
|---|---|---|---|---|
| 751 | 0.16 ± 0.61 (low-pass 0.36 ± 0.91) | 0.16 ± 0.16 (0.41 ± 0.50) | 0.24 ± 0.62 (0.32 ± 0.08) | 0.12 ± 0.01 (0.24 ± 0.34) |
| 451 | 0.10 ± 0.13 (0.28 ± 0.31) | 0.10 ± 0.39 (0.47 ± 0.84) | 0.28 ± 0.49 (0.53 ± 0.02) | 0.10 ± 0.31 (0.29 ± 0.03) |
| 151 | 0.08 ± 0.43 (0.16 ± 1.98) | 0.15 ± 0.33 (0.60 ± 0.99) | 0.20 ± 0.05 (0.53 ± 1.28) | 0.10 ± 0.02 (0.42 ± 0.32) |

Low-pass means pooled to 8×8 latents, the part that steers composition. In the 4 points from the first pass (2 prompts, seed 0, one gradient each, range only, no interval), the gradient that stops at x0 agreed with the full gradient through the U-Net at cosine 0.30 to 0.64 raw, rising as noise fell. The first pass used one resized view, not cutouts, and I stopped it early.

The means of 3 draws are the fairer test of what a sampler accumulates. There TAESD's point estimate falls below the same-decoder figure at all 3 noise levels, which hints at a systematic difference from the full VAE on top of sampling noise. The intervals mostly overlap, so that is a hint and not a finding.

**Does guiding through the cheap paths change the result?** Real SD 1.5 at fp32 on the CPU, DDIM 20 steps, CFG 7.5, guided on steps 3 to 14 at one relative strength (rate 0.2 of the prediction's RMS), 4 cutouts through ViT-B/32. Images decoded with the full VAE, then scored by the guiding CLIP and by held-out PE-Core-B-16. 2 prompts, 1 seed (seed 0) each, 10 images. Deltas are guided minus unguided on the same prompt and noise. Intervals are 95% t-intervals on the 2 paired deltas (df = 1), mean ± half-width:

| Path | Prompt 0 guide | Prompt 1 guide | Guide delta, mean ± half-width | Prompt 0 held-out | Prompt 1 held-out | Held-out delta, mean ± half-width |
|---|---|---|---|---|---|---|
| none (absolute) | 0.3467 | 0.3598 | n/a | 0.3179 | 0.3247 | n/a |
| `xt_vae` | +0.0219 | +0.0457 | +0.034 ± 0.151 | -0.0148 | +0.0095 | -0.003 ± 0.154 |
| `xt_tiny` | +0.0070 | +0.0260 | +0.017 ± 0.121 | -0.0233 | +0.0083 | -0.008 ± 0.201 |
| `x0_vae` | +0.0059 | +0.0118 | +0.009 ± 0.038 | +0.0159 | -0.0033 | +0.006 ± 0.122 |
| `x0_tiny` | +0.0071 | +0.0140 | +0.011 ± 0.044 | +0.0270 | -0.0050 | +0.011 ± 0.203 |

Every interval includes zero. What the data do show is limited: all four paths raised the guiding score on both prompts, and held-out deltas changed sign between prompts on every path. In the 3 prompt 0 images I looked at (`none`, `xt_vae`, `x0_tiny`) neither guided result had adversarial artifacts. That is an anecdote.

### Inferences
- **The full VAE, not the U-Net, is the expensive backward pass.** On SD 1.5 at 512, adding U-Net backward to a tiny decoder adds 0.7 GiB, while swapping TAESD for the full VAE adds 2.5 GiB. So the tiny decoder is the first lever and checkpointing the second.
- **`x0_tiny` barely beats `tiny_ckpt` on memory.** The gap is 0.06 to 0.23 GiB at 512 at fp16 or bf16, and about 0.5 GiB with fp8 storage. Outside fp8, the real choice is time and gradient quality, not VRAM.
- **The peak is weights plus 1.5 to 2 GiB at 1024 on the tiny paths.** That makes 8 GB an arithmetic question: denoiser weights under about 5.2 GiB fit at 1024, and under about 6.4 GiB fit at 512. Those limits assume the 7.62 GiB torch reports on this card, plus 0.73 GiB for CLIP and the decoders.
- **fp8 storage is what brings the 4B and 6B transformers into range.** Quantised weights do nothing for activations, but here activations were never the problem. Layerwise casting avoids a custom loader, though its compatibility with each model is unverified.
- **Sana's headline efficiency does not carry over to backward.** Inference is cheap, 0.28 GiB above weights at 512. Backward through DC-AE is the heaviest decoder path here, about 13 GiB of activations at 1024 without checkpointing. My per-block checkpoint wrapper cuts that by more than half, but it has to be written and maintained here.
- **SDXL needs care on 8 GB.** About 7.1 GiB at 1024 leaves under 0.5 GiB of margin, and plain attention for reproducible seeds would use more. Treat 1024 as 12 GB territory and 768 as the safe 8 GB size.
- **Use taesdxl rather than the SDXL VAE at fp16.** The tiny decoder sidesteps the fp16 NaN entirely during guidance. The final decode can use the fp16-fix VAE.
- **On the 2070 SUPER use fp16 for U-Nets.** bf16 is emulated, so its speed there needs measuring before relying on it. The RTX 4070 is Ada, so it should have native bf16, but I could not test it.

### Gaps
- Only SD 1.5 and the ADM baselines were measured on the card. Other GPU jobs held it for the whole of this work, so I had to use CPU counting. The queued GPU sweep never got a slot and was stopped at wrap-up, along with the on-card gradient agreement run and the fp16 against bf16 timing. No model other than SD 1.5 has a guided-step time.
- The Part B effect test stopped at 2 of 4 prompts when time ran out, at 1 seed and 1 rate. Its intervals are too wide to rank the paths or to show any effect on the held-out score. The rate was not tuned per path, so paths with different gradient directions may not be compared at their best.
- Kandinsky 2.2's `full`, `ckpt` and `x0_vae` rows, and Sana Sprint 1.6B's `full`, `ckpt` and `x0_vae` rows, failed with allocation errors and were not rerun. The likely cause is my 18 GB address-space cap, though RAM was also tight at the time. The SD 3.5 depth rows and klein 384 rows were cancelled, so klein stays projected from 512.
- `full` and `ckpt` for PixArt, Vega, SDXL and klein are projections. They exceeded the RAM available for doubled-precision runs.
- Fused attention is not in these numbers. The repo's plain attention for seed reproducibility would raise them by an amount I did not measure.
- Nothing was run on the RTX 4070. Its capacity is inferred from the 2070's, at about 11.6 GiB.
- taef2 ships no diffusers config. The klein rows use an inline config with taef1's layout and 32 latent channels, which is my assumption. A real integration would use the taesd repo's own class.
- HunyuanDiT, Stable Cascade and SD 3.5 were not measured. SD 3.5 Medium's weights alone come to 4.9 GiB on the same accounting.

## What can CLIP guidance add over plain CFG on models that already have text conditioning?

### Takeaway
CLIP guidance does not beat CFG as a text conditioner, and on a text-conditional model its realistic job is correcting composition and attribute alignment, not raising image quality. The strongest evidence for the combination is ReNO, which optimises the initial noise through a one-step model and raised GenEval by 0.08 to 0.13 on the four models I have figures for, using CLIP among its rewards.

### Cited Findings
- GLIDE compared CLIP guidance against CFG head to head: human evaluators preferred CFG for both photorealism and caption similarity — [GLIDE abstract](https://arxiv.org/abs/2112.10741)
- UPainting combined the two and reports that "CLIP guidance can be combined with the classifier-free guidance to produce much higher quality samples", at CFG scale s = 8.0 and CLIP weight g = 10.0. FID-30K on MS-COCO 256: UPainting 8.34, GLIDE 12.24, Stable Diffusion 14.24. It uses plain CLIP (a ViT-B/32, ViT-B/16 and RN50 ensemble) on an x_in built from predicted x0 and x_t, no noise-aware CLIP, which is the same trick as `guidance_loss()` in this repo. No statement of released weights. The FID compares different models, so it is not an isolated measure of the CLIP term — [UPainting](https://arxiv.org/abs/2210.16031)
- Universal Guidance is frequently misread. Its Table 5 compares CLIP-guided unconditional Stable Diffusion against text-conditional Stable Diffusion, not CLIP plus CFG against CFG: CLIP score 0.2632 against 0.2818 over 20 DrawBench prompts with 10 images each, which the authors call "quantitatively similar". So pure CLIP guidance nearly matches conditioning on that small sample, and tells us nothing about stacking the two — [Universal Guidance, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/dfbb3d1807b21dadee735eb75069ada4-Paper-Conference.pdf)
- DOODL backpropagates to the latents through the whole sampler with EDICT, at constant memory in the number of steps. It improves CLIP score over one-step CLIP guidance by 3.1% (DALL-E prompts) and 2.6% (Reddit prompts). Human prompt alignment success rises 11.3% over vanilla Stable Diffusion and 6% over one-step guidance on the Reddit prompts. No runtime or memory figures — [DOODL](https://ar5iv.labs.arxiv.org/html/2303.13703)
- ReNO optimises initial noise for 50 steps through SD-Turbo, SDXL-Turbo, PixArt-alpha DMD, Hyper-SDXL and FLUX-schnell, against HPSv2, PickScore, ImageReward and CLIPScore (OpenCLIP ViT-H/14). GenEval: SD-Turbo 0.49 to 0.62, SDXL-Turbo 0.54 to 0.65, HyperSDXL 0.56 to 0.65, FLUX-schnell 0.64 to 0.72. T2I-CompBench on SD-Turbo: colour 0.55 to 0.78, shape 0.44 to 0.62, texture 0.57 to 0.75. In a pairwise user study on Parti-Prompts, SD-Turbo + ReNO wins up to 77% against plain SD-Turbo, and HyperSDXL + ReNO wins 54% against SD3 (8B) — [ReNO](https://ar5iv.labs.arxiv.org/html/2406.04312)
- ReNO's gradient runs from noise through the one-step generator and decoder to the reward. Peak memory on an A100: SD-Turbo 15 GB, PixArt-alpha DMD 25 GB, SDXL-Turbo 21 GB, HyperSDXL 39 GB, FLUX-schnell 50 GB, taking 20 to 50 s per image. The paper adds that fp16 and a PyTorch memory reduction technique cut VRAM about another 15% — [ReNO](https://ar5iv.labs.arxiv.org/html/2406.04312)
- Latent-CLIP runs CLIP directly on latents so nothing is decoded. Inside ReNO its rewards match CLIP's on GenEval and T2I-CompBench and cut total runtime 21% — [Latent-CLIP abstract](https://arxiv.org/abs/2503.08455). Each VAE needs its own expensive retraining — [Latent-CLIP paper](https://arxiv.org/pdf/2503.08455). Weights are public at [wendlerc/latent-clip-b-8-512-34b-80k](https://huggingface.co/wendlerc/latent-clip-b-8-512-34b-80k) (1.9 GB checkpoint, ungated, no license tag on the repo)
- MPGD takes the guidance gradient with respect to x0 rather than x_t, so there is no backward pass through the denoiser. Stable Diffusion style guidance on an A100, 100 DDIM steps: unguided DDIM 13.89 s and 10.80 GB, FreeDoM 26.50 s and 17.30 GB, LGD-MC 37.43 s and 31.65 GB, MPGD-LDM 19.83 s and 15.53 GB. The paper's Stable Diffusion loss was a style Gram matrix, not CLIP — [MPGD](https://arxiv.org/abs/2311.16424)
- FreeDoM applies its time-travel retries only in the middle "semantic" stage of sampling, and sets the energy step size relative to the text guidance step on classifier-free models. I found no quantitative Stable Diffusion text result in it — [FreeDoM, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Yu_FreeDoM_Training-Free_Energy-Guided_Conditional_Diffusion_Model_ICCV_2023_paper.pdf)
- Attend-and-Excite is the same machine with an attention loss: in diffusers it enables grad, marks the latents as requiring it, and steps them with `torch.autograd.grad(loss, [latents])` through the U-Net, on top of CFG 7.5 — [diffusers source](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/deprecated/stable_diffusion_attend_and_excite/pipeline_stable_diffusion_attend_and_excite.py); [docs](https://huggingface.co/docs/diffusers/v0.16.0/api/pipelines/stable_diffusion/attend_and_excite). It targets "catastrophic neglect", where a subject in the prompt is dropped — [paper](https://arxiv.org/pdf/2301.13826). It has since moved to diffusers' `deprecated` folder, as read from the installed 0.42.1 source
- Self-guidance "works similarly to classifier guidance, but uses signals present in the pretrained model itself" — [abstract](https://arxiv.org/abs/2306.00986)
- The `clip_guided_stable_diffusion` community pipeline runs a separate cond-only U-Net pass under grad, decodes the predicted x0 through the full VAE with grad, compares cutouts by spherical distance, and adds the gradient to the noise prediction with `sqrt(beta_prod_t)`. Defaults: `clip_guidance_scale` 100, 4 cutouts, and only PNDM, DDIM, DPMSolverMultistep and LMS — [source](https://raw.githubusercontent.com/huggingface/diffusers/main/examples/community/clip_guided_stable_diffusion.py)
- That pipeline went in without tests, samples or reference material, and issue #596 raised both its embeddings and its x0 blend. A maintainer called the latter "a great catch" and it closed through PR #667 — [issue 596](https://github.com/huggingface/diffusers/issues/596)

### Inferences
- CFG and CLIP guidance fix different failures. CFG supplies a subject the model trained on. CLIP guidance fixes what the text encoder dropped, such as colours, counts and attributes. This fits ReNO's biggest gains being in colour, shape and texture, and the existing observation in `cgd/sampling.py` that the model should bring the subject and CLIP should only steer.
- Ranking of the possible roles, best first:
  1. **Noise optimisation through a one-step model (ReNO style).** The only role with strong controlled evidence. It has no solver or schedule to retune, because it just needs a differentiable one-step generator.
  2. **Windowed guidance inside a few-step sampler.** This is what `sampling.py` already does. Plausible, but I found no evaluation of it on modern models.
  3. **Refinement: ADM with CGD first, then img2img through SD or a turbo model at partial strength, with no gradients at all.** The composition comes from CGD and the texture from a modern model, and it pays only inference VRAM. It is the cheapest option, and it is my inference, not a tested result.
  4. **Modern model as the source of an init image for ADM.** Worst of the four: it hands a better image to a worse model.
- Expect prompt alignment gains more than texture gains on these models. The user's stated goal is "results closer to modern models", and most of that comes from the base model itself, so CLIP guidance is what keeps this project's character on top of it rather than the source of the upgrade.

### Gaps
- No head-to-head of the diffusers CLIP-guided pipeline against plain CFG on the same prompts and metric exists that I could find. The community evidence is examples, not measurements.
- No published evaluation of per-step CLIP guidance, as opposed to noise optimisation, on FLUX.2, Z-Image or Sana. Searching found D-Flow, optimal-control and VGG-Flow work on flow models, but nothing with FLUX and CLIP together — [D-Flow](https://arxiv.org/pdf/2402.14017); [VGG-Flow](https://arxiv.org/pdf/2512.05116).
- The memory figures in MPGD and ReNO are from an A100 and are not comparable with fp16 runs on 8 GB. See the measurements in the section on latent models.

## Which pixel-space or small models have downloadable weights, and under what license?

### Takeaway
EDM2 is the one direct upgrade to ADM inside the same ImageNet class-conditional recipe: FID 2.56 against ADM's 23.24 unguided at 512, with half the parameters. Its catch is that it is CC BY-NC-SA and latent, so it needs a decoder. Most other research models here either stop at the same ImageNet domain or have no reachable weights.

### Cited Findings
- EDM2 on ImageNet-512, FID without and with CFG, then Mparams and Gflops. ADM 23.24 / 7.72, 559, 1983. DiT-XL/2 12.03 / 3.04, 675, 525. EDM2-XS 3.53 / 2.91, 125, 46. EDM2-S 2.56 / 2.23, 280, 102. EDM2-M 2.25 / 2.01, 498, 181. EDM2-L 2.06 / 1.88, 777, 282. EDM2-XL 1.96 / 1.85, 1119, 406. EDM2-XXL 1.91 / 1.81, 1523, 552. All EDM2 rows use 63 model evaluations — [EDM2 paper, Table 2](https://arxiv.org/pdf/2312.02696)
- Autoguidance takes EDM2-S from FID 2.56 to 1.34, ahead of CFG at 2.23 and guidance interval at 1.68, by guiding with a smaller XS model. An XXL model reaches 1.25 — [Autoguidance](https://arxiv.org/pdf/2406.02507). ImageNet-512 runs in latent space and ImageNet-64 in pixel space — [Autoguidance](https://arxiv.org/pdf/2406.02507)
- EDM2 weights are at `nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/` under CC BY-NC-SA 4.0, with `-fid`, `-dino`, `-guid` and `-autog` presets, and unconditional variants for guiding — [EDM2 README](https://raw.githubusercontent.com/NVlabs/edm2/main/README.md). HEAD requests returned 200: `edm2-img512-xs-2147483-0.135.pkl` 0.25 GB, `edm2-img512-s-2147483-0.025.pkl` 0.56 GB, `edm2-img64-s-1073741-0.075.pkl` 0.56 GB. A diffusers conversion is at [BiliSakura/EDM2-diffusers](https://huggingface.co/BiliSakura/EDM2-diffusers) (ungated, CC BY-NC-SA tag, third party)
- DiT-XL/2: FID 2.27 at 256 and 3.04 at 512, CC-BY-NC code and weights, repo archived — [README](https://raw.githubusercontent.com/facebookresearch/DiT/main/README.md); [facebook/DiT-XL-2-512](https://huggingface.co/facebook/DiT-XL-2-512) is ungated
- HDiT has no released checkpoints that I could find. Its project page links code and sample archives, with filenames implying 557M for ImageNet-256 and 85M for FFHQ-1024. Its claim is linear scaling with pixel count — [project page](https://crowsonkb.github.io/hourglass-diffusion-transformers/). The k-diffusion README has no weights either, and k-diffusion is MIT — [README](https://raw.githubusercontent.com/crowsonkb/k-diffusion/master/README.md)
- SiT: the 256 checkpoint is reachable, a ranged GET on the README's Dropbox link returned 206 for a 2.70 GB file, FID 2.06. The 512 row is commented out of the README and its URL returned 403. Code is MIT — [README](https://github.com/willisma/SiT)
- MDT weights are at [shgao/MDT-XL2](https://huggingface.co/shgao/MDT-XL2) (5.8 GB, ungated, no license tag) and its code is Apache-2.0 — [repo](https://github.com/sail-sg/MDT)
- U-ViT weights are Google Drive links, code MIT — [repo](https://github.com/baofff/U-ViT)
- REPA's README describes training and I found no checkpoint link in it — [README](https://github.com/sihyun-yu/REPA)
- iDDPM: MIT, and `imagenet64_uncond_100M_1500K.pt` returned 200 at 0.48 GB — [repo](https://github.com/openai/improved-diffusion)
- EDM: `edm-imagenet-64x64-cond-adm.pkl` returned 200 at 1.18 GB — [repo](https://github.com/NVlabs/edm)
- Consistency models: MIT, repo archived. `cd_imagenet64_l2.pt` and `ct_imagenet64.pt` returned 200 at 1.18 GB, and the other checkpoints are LSUN bedroom and cat — [repo](https://github.com/openai/consistency_models); diffusers versions at [openai/diffusers-cd_imagenet64_l2](https://huggingface.co/openai/diffusers-cd_imagenet64_l2)
- GLIDE: `base.pt` 1.54 GB, `upsample.pt` 1.59 GB and the noise-aware `clip_image_enc.pt` 0.35 GB all returned 200. MIT, repo archived. Trained on about 67M filtered pairs with images of people removed, and "can fail to produce realistic outputs for complex prompts" — [model card](https://raw.githubusercontent.com/openai/glide-text2im/main/model-card.md); [abstract](https://arxiv.org/abs/2112.10741)
- latent-diffusion: MIT, `txt2img-f8-large/model.ckpt` returned 200 at 6.15 GB. Also at [CompVis/ldm-text2im-large-256](https://huggingface.co/CompVis/ldm-text2im-large-256) with an Apache tag. `vq-f4.zip` and `celeba.zip` returned 200 — [repo](https://github.com/CompVis/latent-diffusion)
- DeepFloyd IF is pixel-space but gated under its own license on all stages — [IF-I-M](https://huggingface.co/DeepFloyd/IF-I-M-v1.0)

### Inferences
- EDM2-S would keep the current recipe almost unchanged: the U-Net "conditioned on a single CLIP-selected ImageNet class", with the existing fast sampler. Only the prediction form and a decoder change. It costs an NC-SA license, and ImageNet class conditioning is still the limit on subject matter, so this improves fidelity without closing the gap to modern models.
- Using an FID gap on ImageNet as a proxy for CGD output quality is a stretch, since CLIP gradients do not reward the same thing. Treat the size of the gap as direction, not prediction.

### Gaps
- CTM: `NVlabs/CTM` returned 404 and a GitHub search found no official replacement.
- Simple Diffusion: I located no public weights, and the one bucket path I tried is a guess that returned 404, so that is weak evidence.
- IMLE: I found no text or ImageNet-scale weights, on a thin search.

## How do v-diffusion cc12m_1, yfcc_2 and the k-diffusion latent upscalers hold up today?

### Takeaway
cc12m_1_cfg is architecturally the natural fit, because it conditions directly on CLIP embeddings, but it is an unevaluated 256px model from 2021 and I found no evidence it competes with anything from 2024 on. yfcc_2's only documented host is dead. The latent upscaler is alive and useful, but as a final stage rather than a base model.

### Cited Findings
- The README's only host is the-eye.eu. Both `cc12m_1_cfg.pth` and `yfcc_2.pth` returned 404 on it, and its TLS certificate expired on 2026-03-21. The `v-diffusion` S3 bucket returned 403 — [README](https://raw.githubusercontent.com/crowsonkb/v-diffusion-pytorch/master/README.md)
- A mirror exists at [multimodalart/crowsonkb-v-diffusion-cc12m-1-cfg](https://huggingface.co/multimodalart/crowsonkb-v-diffusion-cc12m-1-cfg): `cc12m_1.pth` and `cc12m_1_cfg.pth`, 2.41 GB each, MIT tag, HEAD returned 200. The README lists SHA-256 sums to verify them against. I did not download it, so the hash is unchecked. HF model searches for `v-diffusion` and `yfcc` found no yfcc mirror
- cc12m_1_cfg is 256x256, 602M parameters, trained on Conceptual 12M for 3.1M steps and fine-tuned 250K more for CFG, and is conditioned on CLIP text embeddings. yfcc_2 is unconditional 512x512 at 968M, and yfcc_1 481M. `cfg_sample.py` defaults to 50 steps and `clip_sample.py` to 1000 — [README](https://raw.githubusercontent.com/crowsonkb/v-diffusion-pytorch/master/README.md)
- `crowsonkb/v-diffusion-pytorch` is MIT and was last pushed in 2022-11 — [repo](https://github.com/crowsonkb/v-diffusion-pytorch)
- `sd-x2-latent-upscaler` was trained by Katherine Crowson with Stability AI on a high-resolution LAION-2B subset: 0.4B parameters, Stable Diffusion latents in, 2x, 20 steps in the example, OpenRAIL++, and described as intended for research — [model card](https://huggingface.co/stabilityai/sd-x2-latent-upscaler)
- k-diffusion ships `sample_clip_guided.py` for unconditional models, and wrappers for v-diffusion, guided-diffusion and CompVis models. It lists latent diffusion only as a to-do — [README](https://raw.githubusercontent.com/crowsonkb/k-diffusion/master/README.md)

### Inferences
- cc12m_1_cfg would cost about what ADM 256 costs (602M against 554M parameters, both pixel space), and would mainly replace "closest ImageNet class" with real CLIP conditioning. That is a cleaner design than class selection. It is not a route to modern quality: it saw about 12M images, far fewer than even the roughly 67M filtered pairs behind the small GLIDE model cited above.
- For this project it is a cheap, faithful experiment rather than a recommendation: the same CLIP family conditions and guides it, at 256px, with no VAE and no new dependency beyond the architecture file.

### Gaps
- I found no FID or human evaluation of cc12m_1_cfg against Stable Diffusion or anything else, only the README's claim that it is the recommended sampler. A side by side run is the only way to settle it.

## Which GANs or one-step generators can be optimised through CLIP, and are the weights available?

### Takeaway
StyleGAN-XL is the only large class-conditional GAN with reachable weights and it needs no text model, but the literature shows one-step diffusion generators getting strong results with the same optimisation loop, so those are the better bet. StyleGAN-T and GigaGAN have no official weights.

### Cited Findings
- StyleGAN-XL: ImageNet FID 2.26 at 256, 2.42 at 512, 2.51 at 1024 — [README](https://raw.githubusercontent.com/autonomousvision/stylegan-xl/main/README.md). The EDM2 table gives it 2.41 at 512, 168M parameters and 2067 Gflops, the last two approximated to within about 10% because they could not be determined from primary sources — [EDM2 paper](https://arxiv.org/pdf/2312.02696)
- StyleGAN-XL `imagenet256.pkl` and `imagenet512.pkl` returned 200 at 1.96 and 1.97 GB, and the repo is MIT. Its README lists third party StyleGAN-XL + CLIP projects. Smaller sizes are mirrored at [public-data/StyleGAN-XL](https://huggingface.co/public-data/StyleGAN-XL)
- StyleGAN-T: the official repo is training code, and states "We do not provide pretrained checkpoints" — [README](https://raw.githubusercontent.com/autonomousvision/stylegan-t/main/README.md)
- GigaGAN: only the MIT implementation `lucidrains/gigagan-pytorch` turned up, and an HF search for `gigagan` returned nothing — [repo](https://github.com/lucidrains/gigagan-pytorch)
- R3GAN stops at ImageNet 64 and FFHQ 256 on HF, without a license tag — [brownvc/R3GAN-ImgNet-64x64](https://huggingface.co/brownvc/R3GAN-ImgNet-64x64)
- BigGAN: `huggingface/pytorch-pretrained-BigGAN` is MIT and archived; `ajbrock/BigGAN-PyTorch` has Drive links — [repo](https://github.com/huggingface/pytorch-pretrained-BigGAN)
- SiD is a one-step ImageNet 64 generator from [UT-Austin-PML/SiD](https://huggingface.co/UT-Austin-PML/SiD) (Apache tag, HEAD 200 at 1.18 GB). One-step SD 1.5 and SDXL SiDA checkpoints are at [UT-Austin-PML/SiD-EDM2](https://huggingface.co/UT-Austin-PML/SiD-EDM2), as 3.44 and 5.14 GB pickles
- ReNO gets its gains through one-step diffusion generators, and SD-Turbo costs the least memory of its set — [ReNO](https://ar5iv.labs.arxiv.org/html/2406.04312)

### Inferences
- A one-step generator with the same noise optimisation loop as ReNO gets text conditioning and modern training data for free, where a class-conditional GAN must have CLIP supply the prompt entirely. Prefer SD-Turbo or Sana-Sprint over StyleGAN-XL for this role.
- StyleGAN-XL's 2067 Gflops is more than ADM at 512, so its backward pass is unlikely to be cheap despite 168M parameters. I did not measure it: the pickles execute code on load and the architecture is not in diffusers, so it was out of bounds for this harness.

### Gaps
- No measured VRAM for StyleGAN-XL guided steps, and no quality comparison of StyleGAN-XL + CLIP against ReNO.

## How do diffusers pipelines expose eps or v prediction, and what would an adapter involve here?

### Takeaway
Pipelines are not the integration point, because their `__call__` runs under `torch.no_grad()`. Load the bare denoiser and decoder, call them directly, and add a small per-family adapter that returns x0. `sampling.py` then needs that adapter plus a latent-aware loss, and most of the real work is retuning guidance rather than plumbing.

### Cited Findings
- `__call__` is decorated with `@torch.no_grad()` in the Stable Diffusion, FLUX.2 Klein and Z-Image pipelines, read from installed diffusers 0.42.1 — [SD pipeline source](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py). So `callback_on_step_end` cannot host guidance unless it re-enables grad and reruns the model itself. The community pipeline does exactly that, with an `@torch.enable_grad()` `cond_fn` — [source](https://raw.githubusercontent.com/huggingface/diffusers/main/examples/community/clip_guided_stable_diffusion.py)
- Scheduler configs differ across the candidates: eps for SD-Turbo, Vega and PixArt-Sigma, `flow_prediction` for Sana, `trigflow` under `SCMScheduler` for Sana Sprint, and `FlowMatchEulerDiscreteScheduler` for klein (dynamic shift) and Z-Image-Turbo (static shift 3) — [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo/blob/main/scheduler/scheduler_config.json), [Sana Sprint](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers/blob/main/scheduler/scheduler_config.json), [klein](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/main/scheduler/scheduler_config.json), [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/main/scheduler/scheduler_config.json)
- Prediction type lives on the scheduler config, not the model. `DPMSolverMultistepScheduler` accepts `epsilon`, `sample`, `v_prediction` and `flow_prediction`, and its `convert_model_output` holds the exact conversions: `x0 = (x - sigma_t * eps) / alpha_t`, `x0 = alpha_t * x - sigma_t * v`, and for flow `x0 = x - sigma_t * v` — [source](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py)
- The same scheduler has `use_flow_sigmas` and `flow_shift`, so DPM-Solver++ in its SDE form already runs on flow models in diffusers — [source](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py). `FlowMatchEulerDiscreteScheduler` adds `use_dynamic_shifting`, which demands a resolution-dependent `mu` at `set_timesteps` — [source](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py). The docs advise keeping it on flow checkpoints "unless you are intentionally experimenting with a compatible replacement" — [docs](https://huggingface.co/docs/diffusers/main/en/using-diffusers/schedulers)
- Forward signatures differ by family (inspected in 0.42.1). `UNet2DConditionModel` takes sample, timestep, `encoder_hidden_states` and `added_cond_kwargs`. SD3 adds `pooled_projections`. Sana takes `guidance` as an embedding on Sprint. FLUX.2 takes packed tokens with `img_ids`, `txt_ids` and `guidance`. Z-Image takes `x` and `cap_feats` as lists of per-image tensors. Local read of the [installed models](https://github.com/huggingface/diffusers/tree/main/src/diffusers/models/transformers)
- Distilled models are tied to their schedule. "Please always use the correct checkpoint for the corresponding inference steps" — [SDXL-Lightning](https://huggingface.co/ByteDance/SDXL-Lightning). SD-Turbo runs at `guidance_scale=0.0`, 1 to 4 steps — [card](https://huggingface.co/stabilityai/sd-turbo). Z-Image-Turbo: "Guidance should be 0 for the Turbo models", 8 NFE — [card](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo). FLUX.2 klein: 4 steps at `guidance_scale=1.0` — [card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
- `AutoencoderTiny` plugs in as `pipe.vae`, with `scaling_factor` defaulting to 1.0 because "no such scaling factor was used" for it — [docs](https://huggingface.co/docs/diffusers/main/en/api/models/autoencoder_tiny). So a full VAE needs `z / scaling_factor` and TAESD takes z directly, which `grad_agree.py` in the measurements follows
- diffusers 0.42.1, accelerate 1.15.0 and peft 0.21.2 installed against this repo's torch 2.9.1 and transformers without resolver changes, and every model in the measurement table built and ran on them. None are in `pyproject.toml` today. `transformers` and `sentencepiece` are already there — [pyproject.toml](https://github.com/afiaka87/clip-guided-diffusion/blob/main/pyproject.toml), local install

### Inferences
Changes to this repo, by file:
- **`cgd/sampling.py`**
  - Lines 142-143 hard-code eps and `[:, :shape[1]]`. Replace them with an adapter returning `(x0, eps)`, where `eps = (x - alpha * x0) / sigma` keeps the cap at line 150 meaningful for v and flow models.
  - `NoiseLevels` takes `alphas_cumprod`, which SD-family schedulers expose directly. Flow models need a second schedule with `alpha = 1 - sigma`. The uniform logSNR grid breaks at `sigma = 1`, where alpha is 0, so it needs a `sigma_max` below 1.
  - `X0_CLAMP = 1.2` (line 153) is in image units and is wrong for latents, so skip it in latent space.
  - Line 122's timestep scaling becomes per family: integer 0 to 999, or sigma, or `t / 1000`.
  - `DEFAULT_GUIDANCE_RATE`, `GUIDANCE_WINDOW` and the rest were tuned on ADM, so they need retuning. The module docstring says they were tuned on 128px and validated on held-out prompts, and that work has to be repeated per family.
  - Distilled models cannot use the DPM-Solver++ loop as is. Use their own timesteps, or the ReNO loop, which needs no solver.
- **`cgd/cgd.py`**
  - In `guidance_loss()` (172-219), blend `x_in` in latent space and then decode before `make_cutouts`. `range_loss` and `tv_loss` must move to decoded pixels.
  - `init_tensor` (128-134) needs a VAE encode.
  - `shape` (148) becomes latent shape.
  - The `imagenet_top_n` block (137-146) is replaced by prompt embeddings, computed before the denoiser loads so that T5 or Qwen never shares VRAM with it.
  - CFG needs an uncond pass under `no_grad` and a cond pass with grad, so it costs time rather than peak memory.
  - `GRADIENT_CHECKPOINT_PIXELS` becomes per backbone.
- **`cgd/script_util.py`, `data/diffusion_model_flags.py`:** `DIFFUSION_LOOKUP`, `check_parameters()` and `load_guided_diffusion()` assume ADM. A registry entry per model needs repo, family, prediction type, dtype, decoder, tiny decoder, step count and size stride. Saving frames in `log_image()` now means decoding, so previews should use the tiny decoder.
- **Dependencies:** `diffusers` and `accelerate` are new. `peft` is needed as well for LoRA checkpoints such as VegaRT, LCM-LoRA and Hyper-SD. `guided-diffusion` stays for the ADM path.
- **Gating:** there is no HF token on this machine, so gated repos fail without one. That also changes first-run behaviour from today's plain `requests` download.
- **Seed reproducibility is the hidden cost.** The repo's CLAUDE.md records that fused attention kernels have a nondeterministic backward pass, so a seed stops reproducing its image. diffusers defaults to those kernels: `AttnProcessor2_0` in U-Nets and `dispatch_attention_fn` in FLUX.2 and Z-Image. U-Nets can switch to plain `AttnProcessor`, which raises memory. Transformers would need a custom processor. Without that, `test.py`'s seed reproducibility test will fail for new backbones. I did not test this, it is inferred from the repo's finding.
- **Effort estimate, unmeasured:** SD-family U-Nets with TAESD are the smallest path, on the order of a new module of a few hundred lines. Flow transformers add packing, position ids, schedule shifts and a second `NoiseLevels`. The ReNO loop is simpler than either, because it has no solver.

### Gaps
- Whether plain attention on U-Nets fits the 8 GB budget was not measured, since the harness used diffusers' defaults.
- Whether `sampling.py`'s second-order correction stays stable on flow schedules is untested.


## Where is the biggest quality gain per unit of work, on 8 GB and on 12 GB?

### Takeaway
The base model is the upgrade, and CLIP is the steering. On 8 GB, start with SD-Turbo and a noise optimisation loop through a one-step model. On 12 GB, do the same and then add FLUX.2 klein 4B through the x0 path. Do not port `sampling.py` onto a distilled model first: it is more work, with no published evidence behind it.

### Cited Findings
- ReNO's GenEval gains and user study are from noise optimisation, not per-step guidance. SD-Turbo went from 0.49 to 0.62, and SD-Turbo + ReNO won up to 77% of pairwise comparisons against plain SD-Turbo — [ReNO](https://ar5iv.labs.arxiv.org/html/2406.04312)
- The same paper reports HyperSDXL + ReNO winning 54% against SD3 — [ReNO](https://ar5iv.labs.arxiv.org/html/2406.04312)
- Memory figures are from the measurement section above. Step times are from SD 1.5 on the RTX 2070 SUPER.

### Inferences
**8 GB (RTX 2070 SUPER), in order of work:**
1. **No-gradient refinement:** CGD output through SD-Turbo img2img at partial strength. It pays inference VRAM only, about 2.8 GiB, and it needs diffusers but no adapter. This is the cheapest visible jump, and `refinement_pipelines.md` covers the pipeline side.
2. **SD-Turbo with noise optimisation:** a new module, not a change to `sampling.py`. One-step generator, decoder, cutouts, CLIP, and a step on the noise. The full VAE fits at about 5.9 GiB, so TAESD is optional here. At about 0.5 s per step, 50 iterations take about 25 s on this card, using SD 1.5's step time as a proxy.
3. **Segmind-Vega with VegaRT at 768 to 1024:** the same loop or per-step guidance at about 3.6 GiB, if 512 proves limiting.
4. **klein 4B with fp8 storage:** possible at about 5 to 6.5 GiB, but only as an experiment. It stacks the flow adapter, layerwise casting and emulated bf16 on a card with no native bf16.

**12 GB (RTX 4070), in order of work:**
1. Steps 1 and 2 above, which are identical.
2. **SDXL with Hyper-SD or Lightning, one step, noise optimisation with taesdxl at 1024:** about 7.1 GiB, and the closest match to ReNO's published HyperSDXL result. Check Hyper-SD's license, or use Lightning.
3. **FLUX.2 klein 4B at bf16 at 768 to 1024:** about 9.2 GiB on the x0 path and about 10 GiB through the denoiser. This is the highest-quality option here, and the flow adapter is the main new work. Its 4-step schedule leaves little room for per-step guidance, so noise optimisation or a single corrective step is the natural fit.

**Keep the current ADM path.** It is the only one with this project's character, and it needs no new dependency. EDM2-S is the upgrade to consider for it if a non-commercial license is acceptable.

**Avoid for now:** Sana, because it has no tiny decoder and needs a custom checkpoint wrapper. Z-Image-Turbo, because bf16 is over 12 GiB. Anything gated.

### Gaps
- Noise optimisation with CLIP alone, on these cards, is unmeasured. ReNO used an ensemble of rewards.
- The tiny-decoder routes (Vega at 1024, SDXL at 1024, klein) rest on a 2 prompt, 1 seed effect test whose intervals all include zero, on SD 1.5 only. Run that test at more prompts and seeds before committing to any of them.
- The order of the 12 GB options depends on how klein responds to guidance, which nobody has published.
- These are recommendations from memory estimates and published results on other hardware. None was run end to end here.

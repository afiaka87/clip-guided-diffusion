p = "/home/sam/GitHub/clip-guided-diffusion/research_notes/CLIP guided diffusion quality improvements/diffusion_backbones.md"
s = open(p).read()

ranked = """## Which models should replace or complement the ADM U-Net, ranked for this hardware?

### Takeaway
SD-Turbo is the best first step on either card: ungated, cheap to run, and the base with the strongest published evidence for gradient steering. For quality beyond it, FLUX.2 klein 4B is the strongest open model that fits, on 12 GB at bf16 and on 8 GB only with fp8 storage. All guided VRAM below is an estimate unless the measurement section marks it as measured on the card.

### Cited Findings
Ranked on quality per unit of work, counting license and gating against a model. Guided figures are my estimates from the measurement section, as `tiny_ckpt` (through the denoiser, tiny decoder, checkpointed) and `full` (through the denoiser and full VAE). Download is denoiser plus decoder plus text encoders at the lightest published precision, read from the Hugging Face API.

| # | Model, weights | px | Params | Download | Inference GiB | Guided GiB, tiny_ckpt / full | Steps | License, gating | Text | How guidance attaches |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo) | 512 | 866M | 2.6 GB | ~2.9 | ~2.9 / ~5.9 | 1 to 4 | Stability Community, ungated | Yes | eps. Noise optimisation through one step, or per step on its own timesteps |
| 2 | [Segmind-Vega](https://huggingface.co/segmind/Segmind-Vega) + [VegaRT](https://huggingface.co/segmind/Segmind-VegaRT) | 1024 | 745M | 3.3 GB | ~4.0 | ~3.6 / over 12 | 25, or 2 to 8 with VegaRT | Apache-2.0, ungated | Yes | eps, so `sampling.py` maps directly. taesdxl |
| 3 | [FLUX.2 klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | 1024 | 3.9B | 16 GB | ~9.8 at bf16 | ~10 at bf16, ~6.4 with fp8 storage / not estimated | 4 | Apache-2.0, ungated | Yes | Flow, `x0 = x - sigma * v`, dynamic shift. taef2 |
| 4 | [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) + [Lightning](https://huggingface.co/ByteDance/SDXL-Lightning) or [Hyper-SD](https://huggingface.co/ByteDance/Hyper-SD) | 1024 | 2.6B | 6.9 GB plus adapter | ~7.4 | ~7.1 / over 12 | 1 to 8 | OpenRAIL++ for SDXL and Lightning, Hyper-SD has no license tag, ungated | Yes | eps. Tied to the checkpoint's own timesteps. taesdxl |
| 5 | [SD 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) + [LCM-LoRA](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5) | 512 | 860M | 2.1 GB, already cached here | 2.9 (measured) | 2.85 / 5.92 (measured) | 20, or 2 to 8 | CreativeML OpenRAIL-M, ungated | Yes | eps, so `sampling.py` maps directly. TAESD |
| 6 | [PixArt-Sigma](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-512-MS) | 512, 1024 | 611M | 2.4 GB, plus VAE and T5 from the [1024 repo](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS) | 2.3 at 512 | 2.3 / ~5.5 at 512 | 20 or more | OpenRAIL++, ungated | Yes | eps with learned sigma, like ADM's `[:, :shape[1]]`. T5 stays on the CPU |
| 7 | [Sana Sprint 0.6B](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers), [1.6B](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers) | 1024 | 0.6B, 1.6B | 7.7, 9.7 GB | ~3.4, ~5.3 | no tiny decoder / ~7.5, ~9.4 with my DC-AE checkpointing | 1 to 4 | Apache-2.0 tag, ungated | Yes | `trigflow` under `SCMScheduler`, a third parameterisation. Full DC-AE only |
| 8 | [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | 1024 | 6.2B | 33 GB | over 12 at bf16 | ~8.6 with fp8 storage / not estimated | 8 NFE | Apache-2.0, ungated | Yes | Flow, static shift 3. Inputs are lists of tensors. taef1 |
| 9 | [Kandinsky 2.2](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder) + [prior](https://huggingface.co/kandinsky-community/kandinsky-2-2-prior) | 512 to 768 | 1.25B | 5.3 GB + 10.6 GB prior | not measured, weights alone 3.0 | no tiny decoder / not measured | 50 | Apache-2.0 tag, ungated | Via CLIP image embeddings | U-Net takes an image embedding, so CLIP steers it by design. MoVQ decoder |
| 10 | [EDM2-S](https://raw.githubusercontent.com/NVlabs/edm2/main/README.md) | 512 | 280M | 0.56 GB plus VAE | not measured, proxy weights 1.2 | not measured | 63 NFE | CC BY-NC-SA, direct download | ImageNet classes | Same recipe as today: class condition plus CLIP. Latent, needs a decoder |
| 11 | [cc12m_1_cfg mirror](https://huggingface.co/multimodalart/crowsonkb-v-diffusion-cc12m-1-cfg) | 256 | 602M | 2.4 GB | about ADM 256 | ~5.5, no decoder | 50 | MIT tag, ungated | CLIP embedding | v prediction, pixel space, same CLIP family guides it |

Counted against, and left out of the ranking:
- SDXL-Turbo is non-commercial: `sai-nc-community` license on [the repo](https://huggingface.co/stabilityai/sdxl-turbo).
- [SD 3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) and [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) are gated at their official repos. Ungated mirrors exist at [Comfy-Org](https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8) and [unsloth](https://huggingface.co/unsloth/FLUX.1-schnell). FLUX.1's 11.9B parameters are about 22 GiB at bf16, so it needs fp8 even on 12 GB.
- [FLUX.3 klein](https://huggingface.co/black-forest-labs/flux-3-image-klein) is gated under the FLUX Kommunity License with a 9B prompt upsampler. The 5.6B model itself would be close to Z-Image's cost.
- [Stable Cascade](https://huggingface.co/stabilityai/stable-cascade) is non-commercial, and its decode path includes a diffusion stage.
- [krea/Krea-2](https://huggingface.co/krea) and [DeepFloyd IF](https://huggingface.co/DeepFloyd/IF-I-M-v1.0) are gated under their own licenses.
- [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image), [HiDream-I1](https://huggingface.co/HiDream-ai/HiDream-I1-Fast) and [AuraFlow](https://huggingface.co/fal/AuraFlow-v0.3) are open, but their weights do not fit: AuraFlow is 12.8 GiB at fp16, and the other two repos are 47 to 58 GB.

`stabilityai/stable-diffusion-2-1` and its `-base` and `-unclip` repos now return 401. Community copies live at [sd2-community](https://huggingface.co/sd2-community/stable-diffusion-2-1).

### Inferences
- The ranking weighs evidence and effort over raw model quality. klein is the best image model here, but SD-Turbo is ahead because it fits both cards, needs the smallest adapter, and is what ReNO's numbers were measured on.
- Vega ranks above SDXL because it gives 1024 output on 8 GB with room to spare, under Apache-2.0. Expect lower quality than SDXL, since it is a smaller distillation of it. I have no comparison figures for that.
- Kandinsky 2.2 and cc12m_1_cfg are the conceptually neat fits, because they condition on CLIP embeddings directly. Neither is a route to modern quality on its own evidence, so they rank low despite that.

### Gaps
- No quality comparison across these models under CLIP guidance exists, so the order rests on published unguided reputation, ReNO's numbers and my memory estimates.
- Hyper-SD's repo carries no license tag, so its terms need reading before use.
"""

recommend = """## Where is the biggest quality gain per unit of work, on 8 GB and on 12 GB?

### Takeaway
The base model is the upgrade, and CLIP is the steering. On 8 GB, start with SD-Turbo and a noise optimisation loop through a one-step model. On 12 GB, do the same and then add FLUX.2 klein 4B through the x0 path. Do not port `sampling.py` onto a distilled model first: it is more work, with no published evidence behind it.

### Cited Findings
- ReNO's GenEval gains and user study are from noise optimisation, not per-step guidance. SD-Turbo went from 0.49 to 0.62, and SD-Turbo + ReNO won up to 77% of pairwise comparisons against plain SD-Turbo — [ReNO](https://ar5iv.labs.arxiv.org/html/2406.04312)
- The same paper reports HyperSDXL + ReNO winning 54% against SD3 — [ReNO](https://ar5iv.labs.arxiv.org/html/2406.04312)
- Memory figures are from the measurement section above. Step times are from SD 1.5 on the RTX 2070 SUPER.

### Inferences
**8 GB (RTX 2070 SUPER), in order of work:**
1. **No-gradient refinement:** CGD output through SD-Turbo img2img at partial strength. It pays inference VRAM only, about 2.9 GiB, and it needs diffusers but no adapter. This is the cheapest visible jump, and `refinement_pipelines.md` covers the pipeline side.
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
- The order of the 12 GB options depends on how klein responds to guidance, which nobody has published.
- These are recommendations from memory estimates and published results on other hardware. None was run end to end here.
"""

for marker, text in (("<!-- SECTION-RANKED-TABLE -->", ranked), ("<!-- SECTION-RECOMMENDATION -->", recommend)):
    assert s.count(marker) == 1, marker
    s = s.replace(marker, text.rstrip())
s = s.replace("\n<!-- MORE-ADAPTER -->\n", "\n")
a = "- **Gating:** there is no HF token on this machine, so gated repos fail without one."
b = "- **Dependencies:** `diffusers` and `accelerate` are new. `peft` is needed as well for LoRA checkpoints such as VegaRT, LCM-LoRA and Hyper-SD. `guided-diffusion` stays for the ADM path.\n" + a
assert s.count(a) == 1
s = s.replace(a, b)
a = "- Prediction type lives on the scheduler config, not the model."
b = "- Scheduler configs differ across the candidates: eps for SD-Turbo, Vega and PixArt-Sigma, `flow_prediction` for Sana, `trigflow` under `SCMScheduler` for Sana Sprint, and `FlowMatchEulerDiscreteScheduler` for klein (dynamic shift) and Z-Image-Turbo (static shift 3) — [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo/blob/main/scheduler/scheduler_config.json), [Sana Sprint](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers/blob/main/scheduler/scheduler_config.json), [klein](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/main/scheduler/scheduler_config.json), [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/main/scheduler/scheduler_config.json)\n" + a
assert s.count(a) == 1
s = s.replace(a, b)
open(p, "w").write(s)

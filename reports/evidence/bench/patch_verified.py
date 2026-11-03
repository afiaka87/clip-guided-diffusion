p = "/home/sam/GitHub/clip-guided-diffusion/research_notes/CLIP guided diffusion quality improvements/diffusion_backbones.md"
s = open(p).read()
edits = [
    ("SD-Turbo is the best first step on either card: ungated, cheap to run, and the base with the strongest published evidence for gradient steering.",
     "SD-Turbo is the best first step on either card: ungated, cheap to run, and the base with the largest gain from gradient steering among ReNO's four GenEval figures."),
    ("| 1 to 4 | Stability Community, ungated | Yes | eps. Noise optimisation",
     "| 1 to 4 | [Stability AI Community](https://huggingface.co/stabilityai/sd-turbo/blob/main/LICENSE.md), ungated. Commercial use needs registration, and the grant ends above USD 1M annual revenue | Yes | eps. Noise optimisation"),
    ("| not measured, proxy weights 1.2 | not measured |", "| not measured, weights alone about 1.2 | not measured |"),
    ("read from the Hugging Face API.\n\n| # | Model",
     "read from the Hugging Face API. SD-Turbo's memory figures are SD 1.5's, measured on the card, because the two U-Nets are within 1% in parameters. SD-Turbo itself was not run.\n\n| # | Model"),
    ("Ungated mirrors exist at [Comfy-Org](https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8) and [unsloth](https://huggingface.co/unsloth/FLUX.1-schnell).",
     "Ungated copies exist at [Comfy-Org](https://huggingface.co/Comfy-Org/stable-diffusion-3.5) and [unsloth](https://huggingface.co/unsloth/FLUX.1-schnell). The Comfy-Org copy keeps the Stability AI Community License, and its old `-fp8` URL redirects there."),
    ("- [krea/Krea-2](https://huggingface.co/krea) and [DeepFloyd IF](https://huggingface.co/DeepFloyd/IF-I-M-v1.0) are gated under their own licenses.",
     "- [Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo), [Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw) and [DeepFloyd IF](https://huggingface.co/DeepFloyd/IF-I-M-v1.0) are gated under their own licenses. Krea-2's transformer is also 26 GB."),
    ("and is what ReNO's numbers were measured on.", "and is the smallest of the models ReNO measured."),
    ("while a hosting guide recommends bf16 for the same model — [GigaGPU](https://gigagpu.com/?p=12040).",
     "while a hosting guide recommends bf16 for the same model — [GigaGPU](https://gigagpu.com/?p=12040). I saw that guide only through a search summary, because the page refuses direct fetches."),
    ("[UPainting](https://ar5iv.labs.arxiv.org/html/2210.16031)", "[UPainting](https://arxiv.org/abs/2210.16031)"),
    ("[MPGD](https://ar5iv.labs.arxiv.org/html/2311.16424)", "[MPGD](https://arxiv.org/abs/2311.16424)"),
    ("- StyleGAN-T: the official repo is training code, and its authors provide no checkpoints — [listing](https://awesome.ecosyste.ms/projects/github.com%2Fautonomousvision%2Fstylegan-t)",
     "- StyleGAN-T: the official repo is training code, and states \"We do not provide pretrained checkpoints\" — [README](https://raw.githubusercontent.com/autonomousvision/stylegan-t/main/README.md)"),
]
for old, new in edits:
    assert s.count(old) >= 1, old[:60]
    s = s.replace(old, new)
open(p, "w").write(s)

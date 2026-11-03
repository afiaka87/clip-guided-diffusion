"""Text embeddings of the prompts for a backbone. argv: <SP> <REPO> <backbone>."""
import sys
S, REPO, NAME = sys.argv[1:4]
sys.path.insert(0, REPO)
import torch as th
from diffusers import StableDiffusionXLPipeline
from cgd import modern
exec(open(f"{S}/scripts/prompts.py").read())
config = modern.BACKBONES[NAME]
pipe = StableDiffusionXLPipeline.from_pretrained(config.repo, variant="fp16", torch_dtype=th.float16, vae=None)
pipe.text_encoder.to("cuda"); pipe.text_encoder_2.to("cuda")
out = {}
with th.no_grad():
    for p in PROMPTS:
        embeds, _, pooled, _ = pipe.encode_prompt(p, device="cuda", do_classifier_free_guidance=False)
        out[p] = (embeds.detach(), pooled.detach())
th.save(out, f"{S}/turbo/emb_{NAME}.pt"); print("done", flush=True)

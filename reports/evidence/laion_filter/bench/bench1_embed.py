"""Embedding throughput (end to end and GPU-only) for SigLIP2-B/16-256, PE-Core-B/16 and OpenAI ViT-L/14,
plus LAION aesthetic V1 and V2 recomputed from the downloaded files, and SigLIP2 text embeddings of the alt-text."""
import os, sys, numpy as np, torch, open_clip, clip
from common import sample, tar_loader, aligned, sid, Timer, gpu_only, log, OUT, split
def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    df = sample(); paths = df.path.tolist(); N = len(paths)
    log(f"== bench1 on {N} images, torch {torch.__version__}, {torch.cuda.get_device_name()}")

    def run(name, model, tf, bs, n=None, save=True):
        sub = df if n is None else df.sample(n, random_state=0)
        ids_req = None if n is None else [sid(a, b) for a, b in zip(sub.shard, sub.key)]
        feats, ids, tm = [], [], Timer(skip=max(3, len(sub) // bs // 10)); cpu_tf, norm = split(tf)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for i, x in tar_loader(df, cpu_tf, bs, ids_req):
                f = model.encode_image(norm(x)).float()
                feats.append(torch.nn.functional.normalize(f, dim=-1).cpu()); ids += i.tolist(); tm.step(len(x))
        r = tm.rate(); f = aligned(df, ids, torch.cat(feats).numpy())
        log(f"{name}: end-to-end {r:.0f} img/s over {len(ids)} images (batch {bs}, streamed from tars), peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
        if save: np.save(os.path.join(OUT, f"emb_{name}.npy"), f.astype(np.float16))
        return f

    # SigLIP2 B/16 at 256: native size, so the transform is a no-op resize.
    m, _, tf = open_clip.create_model_and_transforms("hf-hub:timm/ViT-B-16-SigLIP2-256", device="cuda"); m.eval()
    log("siglip2 transform:", tf)
    f_sig = run("siglip2_b16_256", m, tf, 256)
    xx = torch.randn(256, 3, 256, 256, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16): log(f"siglip2_b16_256 GPU-only: {gpu_only(m.encode_image, xx):.0f} img/s")
    tok = open_clip.get_tokenizer("hf-hub:timm/ViT-B-16-SigLIP2-256")
    caps = df.caption.fillna("").tolist(); tfe = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        import time; t = time.time()
        for i in range(0, N, 1024):
            tfe.append(torch.nn.functional.normalize(m.encode_text(tok(caps[i:i+1024]).cuda()).float(), dim=-1).half().cpu())
        torch.cuda.synchronize(); log(f"siglip2 text tower: {N/(time.time()-t):.0f} captions/s")
    np.save(os.path.join(OUT, "txt_siglip2_b16_256.npy"), torch.cat(tfe).numpy())
    del m; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    # PE-Core B/16: timing only.
    m, _, tf = open_clip.create_model_and_transforms("hf-hub:timm/PE-Core-B-16", device="cuda"); m.eval()
    run("pe_core_b16", m, tf, 256, n=30000, save=False)
    xx = torch.randn(256, 3, 224, 224, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16): log(f"pe_core_b16 GPU-only: {gpu_only(m.encode_image, xx):.0f} img/s")
    del m; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    # OpenAI ViT-L/14 with LAION's aesthetic heads.
    m, tf = clip.load("ViT-L/14", device="cuda", jit=False, download_root=os.path.expanduser("~/.cache/clip-guided-diffusion/clip")); m.eval()
    f_l14 = run("openai_l14", m, tf, 128)
    xx = torch.randn(128, 3, 224, 224, device="cuda").half()
    log(f"openai_l14 GPU-only (fp16 weights): {gpu_only(m.encode_image, xx):.0f} img/s")
    v1 = torch.nn.Linear(768, 1); v1.load_state_dict(torch.load("sa_0_4_vit_l_14_linear.pth", map_location="cpu"))
    sd = torch.load(r"C:\Users\Clay\.cache\huggingface\hub\models--camenduru--improved-aesthetic-predictor\snapshots\7b2449be1264fcd9a1cf92e3d30dd29af989c836\sac+logos+ava1-l14-linearMSE.pth", map_location="cpu")
    v2 = torch.nn.Sequential(torch.nn.Linear(768, 1024), torch.nn.Dropout(.2), torch.nn.Linear(1024, 128), torch.nn.Dropout(.2),
                             torch.nn.Linear(128, 64), torch.nn.Dropout(.1), torch.nn.Linear(64, 16), torch.nn.Linear(16, 1)).eval()
    v2.load_state_dict({k.replace("layers.", ""): v for k, v in sd.items()})
    e = torch.from_numpy(f_l14).float()
    with torch.no_grad():
        np.save(os.path.join(OUT, "aes_v1.npy"), v1(e).squeeze(1).numpy()); np.save(os.path.join(OUT, "aes_v2.npy"), v2(e).squeeze(1).numpy())
    log("bench1 done")

if __name__ == "__main__":
    main()

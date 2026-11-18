"""Aesthetic Predictor v2.5 (SigLIP so400m-384) and the BMD YOLO11n watermark detector on a 20k subset: speed and labels."""
import os, time, glob, numpy as np, torch
from PIL import Image
from common import sample, tar_loader, aligned, sid, Timer, gpu_only, log, OUT, Images
from torch.utils.data import DataLoader
import torchvision.transforms as T
ProcTf = T.Compose([T.Resize((384, 384), interpolation=T.InterpolationMode.BICUBIC), T.PILToTensor()])  # SiglipImageProcessor: bicubic 384, /255, mean .5 std .5

def main():
    df = sample(); n = int(os.environ.get("N", 20000)); sub = df.sample(n, random_state=0); paths = sub.path.tolist()
    ids_req = [sid(a, b) for a, b in zip(sub.shard, sub.key)]
    log(f"== bench2 on {n} images")

    from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
    model, proc = convert_v2_5_from_siglip("aesthetic_predictor_v2_5.pth", low_cpu_mem_usage=True)
    model = model.to(torch.float16).cuda().eval()
    tf = ProcTf
    scores, ids, tm = [], [], Timer(skip=30)
    with torch.no_grad():
        for i, x in tar_loader(df, tf, 64, ids_req):
            s = model(((x.cuda(non_blocking=True).float() / 255 - .5) / .5).half()).logits.float().squeeze(1).cpu(); scores.append(s); ids += i.tolist(); tm.step(len(x))
    log(f"aesthetic v2.5: {tm.rate():.0f} img/s end-to-end (batch 64, fp16), peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    np.save(os.path.join(OUT, "aes_v25.npy"), aligned(df, ids, torch.cat(scores).numpy()))
    del model; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    from ultralytics import YOLO
    w = glob.glob(os.path.expanduser(r"~\.cache\huggingface\hub\models--BitcrushedHeart--bmd_watermark_n\snapshots\*\*.pt"))[0]
    yolo = YOLO(w)
    log("yolo names:", yolo.names, "trained imgsz:", yolo.overrides.get("imgsz"))
    ny = 10000; allims = [Image.open(p).convert("RGB") for p in paths[:ny]]  # decoded up front: this times the detector alone
    for imgsz in (320, 640):
        conf = []; t = None; k = 0
        for i in range(0, ny, 128):
            ims = allims[i:i+128]
            if i == 256: torch.cuda.synchronize(); t = time.time()
            res = yolo.predict(ims, imgsz=imgsz, conf=0.05, half=True, verbose=False, batch=128)
            conf += [float(r.boxes.conf.max()) if len(r.boxes) else 0.0 for r in res]
            if t: k += len(ims)
        torch.cuda.synchronize()
        log(f"watermark yolo11n @{imgsz}: {k/(time.time()-t):.0f} img/s (images already decoded); share with a box >= .5: {np.mean(np.array(conf) >= .5):.3f}")
        np.save(os.path.join(OUT, f"wm_yolo_{imgsz}.npy"), aligned(df, ids_req[:ny], np.array(conf, dtype=np.float32)))
    log("bench2 done")

if __name__ == "__main__":
    main()

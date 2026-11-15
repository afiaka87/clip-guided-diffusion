"""Score arms with judges that guided none of them, and compare each with a baseline, paired on (prompt, seed).

uv run python reports/evidence/uncond/score.py BASELINE ARM [ARM ...]

The judges are those of every README number: OpenAI's ViT-B/16, DFN5B's ViT-H-14 and SigLIP2
ViT-L-16-256, each the cosine similarity of the whole image, resized bicubically, to the prompt.
Also Simulacra's aesthetic predictors on ViT-B/16 and ViT-B/32, saturation and the share of
clipped pixels. Scores are cached in scores.json by path and modification time.

CLIP similarity cannot tell a coherent picture of the prompt from a collage of its textures,
which CLIP guidance is good at making. So two models trained on people's choices between
images of the same prompt also score each image: PickScore (pick) and HPS v2.1 (hps), both
fine-tunes of LAION's CLIP ViT-H-14, which guides nothing here. A second table compares
them and the aesthetic rating with the baseline's, paired the same way.

An arm that selects with a judge must not be reported on it: set its excluded judges with
ARM=judge,judge, e.g. fk4=b16,sig2.

SCORE_DEVICE=cpu scores on the CPU in float32, into scores_cpu.json, so it can run while the GPU
is busy. Its scores differ from the GPU's half precision ones in the third decimal, so compare
arms only within one device.
"""

import gc
import json
import os
import sys
from pathlib import Path

import clip
import numpy as np
import open_clip
import torch as th
import torch.nn.functional as tf
import torchvision.transforms.functional as tvf
from PIL import Image
from scipy import stats

from cgd.download import download
from common import HERE, IMAGES, located, prompt_of

CACHE = Path.home() / ".cache" / "clip-guided-diffusion"
OPENAI_MEAN, OPENAI_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
SAC_URL = "https://github.com/crowsonkb/simulacra-aesthetic-models/raw/master/models/"
JUDGES = ("b16", "vith", "sig2")
device = os.environ.get("SCORE_DEVICE", "cuda")
DTYPE = th.float16 if device == "cuda" else th.float32
DB = HERE / ("scores.json" if device == "cuda" else f"scores_{device}.json")


def key(path: str) -> str:
    """A recorded path and its file's modification time, so a regenerated image is scored again."""
    return f"{path}|{int(located(path).stat().st_mtime)}"


def resize(x: th.Tensor, size: int) -> th.Tensor:
    return tf.interpolate(x, (size, size), mode="bicubic", antialias=True, align_corners=False).clamp(0, 1)


def load_runs(arm: str) -> dict[tuple[int, int], dict]:
    log = IMAGES / arm / "runs.jsonl"
    return {(r["prompt"], r["seed"]): r for r in map(json.loads, log.read_text().splitlines())}


@th.no_grad()
def score_missing(db: dict, runs: list[dict]) -> None:
    todo = [r for r in runs if key(r["path"]) not in db]
    print(len(todo), "images to score", flush=True)
    if not todo:
        return
    images = {r["path"]: tvf.to_tensor(Image.open(located(r["path"])).convert("RGB")).unsqueeze(0) for r in todo}
    for r in todo:
        db[key(r["path"])] = {}

    def sweep(name, encode_image, size, mean, std, encode_text):
        texts = {p: tf.normalize(encode_text([p]).float(), dim=-1) for p in sorted({prompt_of(r) for r in todo})}
        for r in todo:
            x = tvf.normalize(resize(images[r["path"]].to(device), size), mean, std)
            db[key(r["path"])][name] = (tf.normalize(encode_image(x).float(), dim=-1) @ texts[prompt_of(r)].T).item()

    def openai(file: str, name: str) -> None:
        model = clip.load(str(CACHE / "clip" / file), device=device, jit=False)[0].float().eval()
        sweep(name, model.encode_image, 224, OPENAI_MEAN, OPENAI_STD,
              lambda t: model.encode_text(clip.tokenize(t, truncate=True).to(device)))
        head_file = f"sac_public_2022_06_29_vit_{name[0]}_{name[1:]}_linear.pth"
        head = th.load(download(SAC_URL + head_file, head_file), map_location=device, weights_only=True)
        for r in todo:
            x = tvf.normalize(resize(images[r["path"]].to(device), 224), OPENAI_MEAN, OPENAI_STD)
            e = tf.normalize(model.encode_image(x).float(), dim=-1) * 512**0.5
            db[key(r["path"])]["aes" + name] = (head["linear.weight"] @ e.T).item() + head["linear.bias"].item()

    def open_clip_judge(arch: str, weights: str, name: str) -> None:
        model = open_clip.create_model(arch, pretrained=weights, cache_dir=str(CACHE / "clip")).eval().to(device, DTYPE)
        tokenizer = open_clip.get_tokenizer(arch, cache_dir=str(CACHE / "clip"))
        sweep(name, lambda x: model.encode_image(x.to(DTYPE)), model.visual.image_size[0],
              model.visual.image_mean or OPENAI_MEAN, model.visual.image_std or OPENAI_STD,
              lambda t: model.encode_text(tokenizer(t).to(device)))

    for file, name in (("ViT-B-16.pt", "b16"), ("ViT-B-32.pt", "b32")):
        openai(file, name)
        gc.collect()
        th.cuda.empty_cache()
    for arch, weights, name in (("ViT-H-14-quickgelu", "dfn5b", "vith"), ("ViT-L-16-SigLIP2-256", "webli", "sig2")):
        open_clip_judge(arch, weights, name)
        gc.collect()
        th.cuda.empty_cache()
    for r in todo:
        x = images[r["path"]]
        high, low = x.max(1).values, x.min(1).values
        db[key(r["path"])]["sat"] = ((high - low) / high.clamp(min=1e-3)).mean().item()
        db[key(r["path"])]["clip_px"] = ((x <= 0) | (x >= 1)).float().mean().item()
    DB.write_text(json.dumps(db))


@th.no_grad()
def score_preference(db: dict, runs: list[dict]) -> None:
    """PickScore and HPS v2.1 of the images that lack them: how people would rate each as a picture of its prompt."""
    todo = [r for r in runs if "pick" not in db[key(r["path"])] or "hps" not in db[key(r["path"])]]
    print(len(todo), "images for the preference judges", flush=True)
    if not todo:
        return
    from huggingface_hub import hf_hub_download
    from transformers import AutoModel, AutoProcessor

    images = {r["path"]: Image.open(located(r["path"])).convert("RGB") for r in todo}
    prompts = sorted({prompt_of(r) for r in todo})

    def features(output) -> th.Tensor:
        return tf.normalize((output if isinstance(output, th.Tensor) else output.pooler_output).float(), dim=-1)

    processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", cache_dir=str(CACHE / "hf"))
    model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1", cache_dir=str(CACHE / "hf")).eval().to(device, DTYPE)
    scale = model.logit_scale.exp().item()
    texts = {}
    for prompt in prompts:
        inputs = processor(text=prompt, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
        texts[prompt] = features(model.get_text_features(**inputs))
    for r in todo:
        pixels = processor(images=images[r["path"]], return_tensors="pt")["pixel_values"].to(device, DTYPE)
        db[key(r["path"])]["pick"] = scale * (features(model.get_image_features(pixel_values=pixels)) @ texts[prompt_of(r)].T).item()
    del model
    gc.collect()
    th.cuda.empty_cache()

    model = open_clip.create_model("ViT-H-14", pretrained=None).eval()
    state = th.load(hf_hub_download("xswu/HPSv2", "HPS_v2.1_compressed.pt", cache_dir=str(CACHE / "hf")),
                    map_location="cpu", weights_only=True)
    model.load_state_dict(state["state_dict"])
    model.to(device, DTYPE)
    tokenizer = open_clip.get_tokenizer("ViT-H-14")
    texts = {p: tf.normalize(model.encode_text(tokenizer([p]).to(device)).float(), dim=-1) for p in prompts}
    for r in todo:
        x = tvf.normalize(resize(tvf.to_tensor(images[r["path"]]).unsqueeze(0).to(device), 224), OPENAI_MEAN, OPENAI_STD)
        db[key(r["path"])]["hps"] = (tf.normalize(model.encode_image(x.to(DTYPE)).float(), dim=-1) @ texts[prompt_of(r)].T).item()
    del model
    gc.collect()
    th.cuda.empty_cache()
    DB.write_text(json.dumps(db))


def paired(db: dict, runs: dict, base: dict, metrics: tuple[str, ...]) -> tuple[float, float, float, int, int]:
    """The change in the mean of `metrics` from `base` to `runs`, per prompt over its seeds: the mean,
    the half width of its 95% t interval, the Wilcoxon p, the prompts improved and how many there were."""
    pairs = sorted(set(runs) & set(base))
    prompts = sorted({p for p, _ in pairs})
    per_prompt = np.array([
        np.mean([np.mean([db[key(runs[k]["path"])][m] - db[key(base[k]["path"])][m] for m in metrics])
                 for k in pairs if k[0] == p])
        for p in prompts
    ])
    n = len(prompts)
    same = runs is base or not np.any(per_prompt)
    half = 0.0 if same or n < 2 else stats.t.ppf(0.975, n - 1) * per_prompt.std(ddof=1) / n**0.5
    p_value = 1.0 if same else stats.wilcoxon(per_prompt).pvalue
    return per_prompt.mean(), half, p_value, int((per_prompt > 0).sum()), n


def main() -> None:
    names = sys.argv[1:]
    excluded = {}
    for i, name in enumerate(names):
        arm, _, judges = name.partition("=")
        names[i], excluded[arm] = arm, set(judges.split(",")) - {""}
    arms = {name: load_runs(name) for name in names}
    db = json.loads(DB.read_text()) if DB.exists() else {}
    score_missing(db, [r for runs in arms.values() for r in runs.values()])
    score_preference(db, [r for runs in arms.values() for r in runs.values()])
    base_name = names[0]
    base = arms[base_name]
    print(f"Baseline {base_name}. Change in the mean over the judges each arm may be reported on, paired on")
    print("(prompt, seed); 95% t interval and Wilcoxon p over prompts; prompts improved; time is the median.")
    print(f"{'arm':34} {'change':>7} {'±':>6} {'p':>6} {'wins':>5} {'n':>3} | {'b16':>5} {'vith':>5} {'sig2':>5} | "
          f"{'aes16':>5} {'aes32':>5} {'sat':>4} {'clip%':>5} | {'s':>5} {'GiB':>4}")
    for name, runs in arms.items():
        judges = [j for j in JUDGES if j not in excluded[name] | excluded[base_name]]
        pairs = sorted(set(runs) & set(base))
        prompts = sorted({p for p, _ in pairs})
        per_prompt = np.array([
            np.mean([np.mean([db[key(runs[k]["path"])][j] - db[key(base[k]["path"])][j] for j in judges])
                     for k in pairs if k[0] == p])
            for p in prompts
        ])
        n = len(prompts)
        mean = per_prompt.mean()
        half = stats.t.ppf(0.975, n - 1) * per_prompt.std(ddof=1) / n**0.5 if name != base_name and n > 1 else 0.0
        p_value = stats.wilcoxon(per_prompt).pvalue if name != base_name and np.any(per_prompt) else 1.0
        raw = {m: np.mean([db[key(r["path"])][m] for r in runs.values()])
               for m in (*JUDGES, "aesb16", "aesb32", "sat", "clip_px")}
        seconds = np.median([r["seconds"] for r in runs.values()])
        peak = max(r["peak"] for r in runs.values())
        shown = {j: f"{raw[j]:.3f}" if j in judges or name == base_name else f"({raw[j]:.3f})"[:7] for j in JUDGES}
        print(f"{name:34} {mean:+.4f} {half:.4f} {p_value:6.3f} {int((per_prompt > 0).sum()):2d}/{n:<2d} {len(pairs):3d} | "
              f"{shown['b16']:>5} {shown['vith']:>5} {shown['sig2']:>5} | {raw['aesb16']:5.2f} {raw['aesb32']:5.2f} "
              f"{raw['sat']:.2f} {100 * raw['clip_px']:5.1f} | {seconds:5.1f} {peak:4.1f}")
    print()
    print(f"What people would prefer, against {base_name}: PickScore, HPS v2.1 and the ViT-B/16 aesthetic rating. Each is the")
    print("mean, then the paired change with its 95% t interval, Wilcoxon p and prompts improved.")
    for name, runs in arms.items():
        cells = []
        for metric, digits in (("pick", 2), ("hps", 4), ("aesb16", 2)):
            value = np.mean([db[key(r["path"])][metric] for r in runs.values()])
            mean, half, p_value, wins, n = paired(db, runs, base, (metric,))
            cells.append(f"{metric} {value:.{digits}f} {mean:+.{digits}f} ±{half:.{digits}f} p {p_value:.3f} {wins:2d}/{n}")
        print(f"{name:26} " + " | ".join(cells))


if __name__ == "__main__":
    main()

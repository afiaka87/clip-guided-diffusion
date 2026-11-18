"""Rough content composition: CLIP ViT-B/32 zero-shot over a random sample, on the CPU."""
import io, sys, tarfile, random, glob, os, collections
import numpy as np, torch, clip
from PIL import Image
torch.set_num_threads(int(os.environ.get("THREADS", 16)))
CATS = {
 "portrait of a person": ["a portrait photo of a person", "a close-up photo of a face"],
 "group of people / event": ["a photo of a group of people", "a photo of people at an event"],
 "wedding": ["a wedding photo", "a bride in a wedding dress"],
 "fashion / clothing": ["a photo of a dress", "a fashion photo of clothing", "a model wearing an outfit"],
 "jewelry / accessories": ["a photo of jewelry", "a photo of a handbag", "a photo of shoes"],
 "food": ["a photo of food", "a photo of a meal on a plate"],
 "cake / dessert / baking": ["a photo of a cake", "a photo of cupcakes or cookies"],
 "drink": ["a photo of a drink", "a cocktail in a glass"],
 "interior / room / furniture": ["a photo of a room interior", "a photo of a living room", "a photo of furniture"],
 "building / architecture": ["a photo of a house", "a photo of a building", "architecture"],
 "garden / flowers / plants": ["a photo of flowers", "a photo of a garden", "a photo of a plant"],
 "landscape / nature": ["a landscape photo", "a photo of mountains", "a photo of a forest", "a photo of a lake"],
 "beach / sea": ["a photo of a beach", "a photo of the ocean"],
 "city / street": ["a photo of a city street", "a city skyline"],
 "animal": ["a photo of an animal", "a photo of a dog", "a photo of a cat", "a photo of a bird", "a photo of a wild animal"],
 "car / vehicle": ["a photo of a car", "a photo of a motorcycle", "a photo of a vintage car"],
 "boat / plane / train": ["a photo of a boat", "a photo of an airplane", "a photo of a train"],
 "painting / fine art": ["a painting", "an oil painting", "a watercolor painting"],
 "drawing / sketch / line art": ["a pencil drawing", "a line art drawing", "a coloring page"],
 "cartoon / clipart / vector": ["a cartoon illustration", "clipart", "a vector illustration"],
 "anime / comic": ["an anime drawing", "a comic book illustration"],
 "3d render": ["a 3d render", "a computer generated image"],
 "text / poster / quote": ["a poster with text", "an inspirational quote", "a flyer with text", "a book cover"],
 "logo / icon": ["a logo", "an icon"],
 "pattern / texture": ["a seamless pattern", "a texture background"],
 "crafts / handmade": ["a crochet project", "a knitted item", "a handmade craft", "a sewing pattern"],
 "toys / dolls": ["a photo of a toy", "a photo of a doll"],
 "holiday decoration": ["christmas decorations", "a christmas tree", "halloween decorations"],
 "product on white": ["a product photo on a white background"],
 "collage": ["a collage of several photos"],
 "screenshot / map / diagram": ["a screenshot", "a map", "a diagram"],
 "tattoo / nails / hair / makeup": ["a tattoo", "nail art", "a hairstyle", "makeup"],
}
if __name__ == "__main__":
    n_shards, per = int(sys.argv[1]), int(sys.argv[2])
    tars = sorted(glob.glob(r"D:\laion2b-en-aesthetic\*.tar"))[17:-1]
    random.seed(1); tars = random.sample(tars, n_shards)
    model, pre = clip.load("ViT-B/32", device="cpu", download_root=os.path.expanduser("~/.cache/clip-guided-diffusion/clip")); model.eval()
    names = list(CATS); prompts = [p for k in names for p in CATS[k]]; owner = [i for i, k in enumerate(names) for _ in CATS[k]]
    with torch.no_grad():
        t = model.encode_text(clip.tokenize(prompts)).float(); t /= t.norm(dim=-1, keepdim=True)
    keys, feats = [], []
    for tp in tars:
        ims = []
        with tarfile.open(tp) as tf:
            for m in tf:
                if m.name.endswith(".jpg"):
                    ims.append((int(os.path.basename(tp)[:5]), m.name[:-4], tf.extractfile(m).read()))
        random.shuffle(ims)
        for i in range(0, per, 100):
            batch = ims[i:i + 100]
            x = torch.stack([pre(Image.open(io.BytesIO(b)).convert("RGB")) for _, _, b in batch])
            with torch.no_grad():
                f = model.encode_image(x).float(); f /= f.norm(dim=-1, keepdim=True)
            feats.append(f); keys += [(s, k) for s, k, _ in batch]
        print(tp, len(keys), flush=True)
    f = torch.cat(feats)
    sim = (100 * f @ t.T)
    # each category = max over its prompts
    cat = torch.full((len(f), len(names)), -1e9)
    for j, o in enumerate(owner):
        cat[:, o] = torch.maximum(cat[:, o], sim[:, j])
    top = cat.argmax(1).numpy()
    c = collections.Counter(top)
    for i, k in sorted(c.items(), key=lambda x: -x[1]):
        print(f"{names[i]:32s} {k/len(top)*100:5.1f}%")
    np.save(sys.argv[3], f.numpy()); 
    import pandas as pd
    pd.DataFrame({"shard": [s for s, _ in keys], "key": [k for _, k in keys], "cat": [names[i] for i in top]}).to_parquet(sys.argv[3].replace(".npy", ".parquet"))

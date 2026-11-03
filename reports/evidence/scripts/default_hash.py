"""Hash the default model's output through whichever tree is on PYTHONPATH."""
import hashlib, shutil, tempfile, warnings
warnings.filterwarnings("ignore")
import cgd.cgd as module
for size in (128, 256):
    for seed in (0, 1):
        with tempfile.TemporaryDirectory() as prefix:
            paths = [p for _, p in module.clip_guided_diffusion(prompts=["a red fox in the snow"], image_size=size, seed=seed,
                                                                prefix_path=prefix, save_frequency=1000, progress=False)]
            print(size, seed, hashlib.sha256(open(paths[-1], "rb").read()).hexdigest()[:16], flush=True)
print(module.__file__, flush=True)

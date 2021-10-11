import os
import cog
from cgd.cgd import clip_guided_diffusion
from cgd import clip_util, script_util
from pathlib import Path


class ClipGuidedDiffusionPredictor(cog.Predictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        prefix_path = Path(os.path.realpath("./cog_output"))
        prefix_path.mkdir(exist_ok=True)     
        self.prefix_path = Path(prefix_path)
        # theres no need to set the following to class members, the predict function uses their cached values.
        if not os.path.exists(os.path.expanduser("~/.cache/clip-guided-diffusion/128x128_diffusion.pt")):
            _ = script_util.download_guided_diffusion(image_size=128, checkpoints_dir=script_util.CACHE_PATH, class_cond=True)
        if not os.path.exists(os.path.expanduser("~/.cache/clip/ViT-B-32.pt")):
            _, _ = clip_util.load_clip("ViT-B/32", "cuda")

    @cog.input("prompt", type=str, help="a caption to visualize")
    @cog.input("init_image", type=cog.Path, help="an image to blend with diffusion before clip guidance begins. Uses half as many timesteps.", default=None)
    @cog.input("respace", type=str, help="Number of timesteps. Fewer is faster, but less accurate.", default="250", options=["25", "50", "100", "200", "250", "ddim25", "ddim50", "ddim100", "ddim200", "ddim250"])
    @cog.input("clip_guidance_scale", type=int, default=1000, min=0, max=2500, help="Scale for CLIP spherical distance loss. Values will need tinkering for different settings.",)
    @cog.input("tv_scale", type=float, default=150., min=0., max=250., help="Scale for a denoising loss that effects the last half of the diffusion process. 0, 100, 150 and 200")
    @cog.input("range_scale", type=float, default=50., min=0., max=250., help="Controls how far out of RGB range values may get.")
    @cog.input("sat_scale", type=float, default=0., min=0.0, max=128.0, help="Controls how much saturation is allowed. Use for ddim. From @nshepperd.",)
    @cog.input("use_augmentations", type=bool, default=False, help="Whether to use augmentation during prediction. May help with ddim and respacing <= 100.")
    @cog.input("use_magnitude", type=bool, default=False, help="Use the magnitude of the loss. May help (only) with ddim and respacing <= 100")
    @cog.input("seed", type=int, default=0, help="Random seed for reproducibility.")
    def predict(self, prompt: str,  respace: str, init_image: cog.Path = None, size: int=256, clip_guidance_scale: int = 1000, tv_scale: float = 150., range_scale: float = 50., sat_scale: float = 0., use_augmentations: bool = False, use_magnitude: bool = False, seed: int = 0):
        # this could feasibly be a parameter, but it's a highly confusing one. Using half works well enough.
        timesteps_to_skip = int(respace.replace("ddim", "")) // 2 if init_image else 0
        init_scale = 1000 if init_image else 0
        cgd_generator = clip_guided_diffusion(
            clip_guidance_scale=clip_guidance_scale,
            tv_scale=tv_scale,
            range_scale=range_scale,
            sat_scale=sat_scale,
            prompts=[prompt],
            init_image=init_image,
            skip_timesteps=timesteps_to_skip,
            timestep_respacing=respace,
            save_frequency=5,
            batch_size=1, # not sure how replicate handles multiple outputs, i have a batch index to deal with it
            image_size=256, # image size is fixed to the checkpoint, so we can't change it without breaking the cache.
            class_cond=False,
            randomize_class=False,
            clip_model_name="ViT-B/32", # changing works, but will break the cache
            cutout_power=1.0,
            num_cutouts=16,
            device="cuda",
            prefix_path=self.prefix_path,
            progress=True,
            use_augs=use_augmentations,
            use_magnitude=use_magnitude,
            init_scale=init_scale,
            seed=seed,
        )
        for _, batch in enumerate(cgd_generator): yield cog.Path(batch[1]) # second element is the image path, first is the batch index if batch_size > 1
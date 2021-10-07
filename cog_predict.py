import os
from posixpath import realpath
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
    @cog.input("size", type=int, options=[128, 256], help="image size", default=256)
    @cog.input("clip_guidance_scale", type=int, default=1000, min=0, max=2500, help="Scale for CLIP spherical distance loss. Values will need tinkering for different settings.",)
    @cog.input("tv_scale", type=float, default=150., min=0., max=250., help="Scale for TV loss. 0, 100, 150 and 200")
    @cog.input("range_scale", type=float, default=50., min=0., max=250., help="Controls how far out of RGB range values may get.")
    @cog.input("sat_scale", type=float, default=0., min=0.0, max=128.0, help="Controls how much saturation is allowed. Use for ddim. From @nshepperd.",)
    @cog.input("respace", type=str, help="Number of timesteps", default="250", options=["25", "50", "100", "200", "250"])
    @cog.input("init_image", type=cog.Path, help="an image to blend with diffusion before clip guidance begins. Uses half as many timesteps.", default=None)
    def predict(self, prompt: str,  respace: str, init_image: cog.Path = None, size: int=256, clip_guidance_scale: int = 1000, tv_scale: float = 150., range_scale: float = 50., sat_scale: float = 0.):
        # this could feasibly be a parameter, but it's a highly confusing one. Using half works well enough.
        timesteps_to_skip = int(respace.replace("ddim", "")) // 2 if init_image else 0
        class_conditional = (size == 128)
        cgd_generator = clip_guided_diffusion(
            clip_guidance_scale=clip_guidance_scale,
            tv_scale=tv_scale,
            range_scale=range_scale,
            sat_scale=sat_scale,
            prompts=[prompt],
            init_image=init_image,
            skip_timesteps=timesteps_to_skip,
            timestep_respacing=respace,
            save_frequency=1,
            batch_size=1, # not sure how replicate handles multiple outputs, i have a batch index to deal with it
            image_size=size, # image size is fixed to the checkpoint, so we can't change it without breaking the cache.
            class_cond=class_conditional,
            clip_model_name="ViT-B/32", # changing works, but will break the cache
            randomize_class=class_conditional,
            cutout_power=0.5,
            num_cutouts=48,
            device="cuda",
            prefix_path=self.prefix_path,
            progress=True,
            use_augs=False,
            use_magnitude=False,
            init_scale=1000 if init_image else 0,
        )
        for _, batch in enumerate(cgd_generator):
            yield cog.Path(batch[1]) # second element is the image path, first is the batch index if batch_size > 1

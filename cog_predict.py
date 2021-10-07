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
    @cog.input("respace", type=str, help="Number of timesteps", default="ddim100", options=["25", "50", "100", "200", "250", "500", "1000", "ddim25", "ddim50", "ddim100", "ddim200", "ddim250", "ddim500", "ddim1000"])
    @cog.input("init_image", type=cog.Path, help="an image to blend with diffusion before clip guidance begins. Uses half as many timesteps.", default=None)
    def predict(self, prompt: str, respace: str, init_image: cog.Path = None):
        # this could feasibly be a parameter, but it's a highly confusing one. Using half works well enough.
        timesteps_to_skip = int(respace.replace("ddim", "")) // 2 if init_image else 0
        use_magnitude = False if int(respace.replace("ddim", "")) <= 50 else False
        use_augmentations = False if int(respace.replace("ddim", "")) <= 50 else False
        cgd_generator = clip_guided_diffusion(
            prompts=[prompt],
            init_image=init_image,
            skip_timesteps=timesteps_to_skip,
            timestep_respacing=respace,
            save_frequency=1,
            batch_size=1, # not sure how replicate handles multiple outputs, i have a batch index to deal with it
            image_size=128, # image size is fixed to the checkpoint, so we can't change it without breaking the cache.
            class_cond=True, # fixed to checkpoint
            clip_model_name="ViT-B/32", # changing works, but will break the cache
            randomize_class=True, # only works with class conditioned checkpoints
            cutout_power=0.5,
            num_cutouts=50,
            device="cuda",
            prefix_path=self.prefix_path,
            progress=True,
            use_augs=use_augmentations,
            use_magnitude=use_magnitude,
            sat_scale=0,
            init_scale=1000 if init_image else 0,
        )
        for _, batch in enumerate(cgd_generator):
            yield cog.Path(batch[1]) # second element is the image path, first is the batch index if batch_size > 1
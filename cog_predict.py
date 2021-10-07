import os
from PIL.Image import init
import cog
from pathlib import Path
from cgd.cgd import clip_guided_diffusion
from cgd import clip_util, script_util


class ClipGuidedDiffusionPredictor(cog.Predictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        prefix_path = Path("outputs")
        prefix_path.mkdir(exist_ok=True)     
        if not os.path.exists(os.path.expanduser("~/.cache/clip-guided-diffusion/128x128_diffusion.pt")):
            _ = script_util.download_guided_diffusion(image_size=128, checkpoints_dir=script_util.CACHE_PATH, class_cond=True)
        if not os.path.exists(os.path.expanduser("~/.cache/clip/ViT-B-32.pt")):
            _, _ = clip_util.load_clip("ViT-B/32", "cuda")
        # theres no need to set these to class members, the predict function uses their cached values

    # The arguments and types the model takes as input
    @cog.input("prompt", type=str, help="a caption to visualize")
    @cog.input("init_image", type=str, help="a path to an image to use as the initial image", default="")
    @cog.input("respace", type=str, help="Number of timesteps", default="ddim100")
    def predict(self, prompt="", init_image="", respace="ddim100"):
        try:
            """Run a single prediction on the model"""
            # this could feasibly be a parameter, but it's a highly confusing one. Using half works well enough.
            timesteps_to_skip = int(respace.replace("ddim", "")) // 2 if len(init_image) > 0 else 0
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
                num_cutouts=32,
                device="cuda",
                prefix_path="cog_output",
                progress=True,
                use_augs=True,
                sat_scale=32,
                init_scale=0 if len(init_image) > 0 else 1000,
                use_magnitude=True,
            )
            for batch_idx, output in enumerate(cgd_generator): yield output
        except KeyboardInterrupt as e:
            print("KeyboardInterrupt")
            print("Exiting")
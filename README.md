# CLIP Guided Diffusion

From [RiversHaveWings](https://twitter.com/RiversHaveWings).

<a href="https://colab.research.google.com/drive/12a_Wrfi2_gwwAuN3VvMTwVMz9TfqctNj?usp=sharing">
  <img alt="Original notebook" src="https://colab.research.google.com/assets/colab-badge.svg">
  Original Notebook
</a>
</br>
<a href="https://colab.research.google.com/drive/1V66mUeJbXrTuQITvJunvnWVn96FEbSI3">
  <img alt="Original 512x512 notebook" src="https://colab.research.google.com/assets/colab-badge.svg">
  Original 512x512 HQ Notebook (Recommended)
</a>
</br>
<a href="https://github.com/afiaka87/clip-guided-diffusion/blob/main/cgd_clip_selected_class.ipynb">
  <img alt="Modified 512 px Notebook" src="https://colab.research.google.com/assets/colab-badge.svg">
  Modified 512 px Notebook
</a>

> This code is currently under active development and is subject to frequent changes. Please file an issue if you have any constructive feedback, questions, or issues with the code or colab notebook.

[Gallery](/images/README.md)

---

## Installation
```sh
git clone https://github.com/afiaka87/clip-guided-diffusion.git
cd clip-guided-diffusion
python3 -m venv cgd_venv
source cgd_venv/bin/activate
(cgd_venv) $ pip install -r requirements.txt
```

## Quick start:

```sh
(cgd_venv) $ python cgd.py "32K HUHD Mushroom"
Step 999, output 0:
00%|███████████████| 1000/1000 [00:00<12:30,  1.02it/s]
```
![](/images/32K_HUHD_Mushroom.png?raw=true)

Generations will be saved in the folder from `--prefix` (default:'./outputs')

with the format `{caption}/batch_idx_{j}_iteration_{i}.png`


### Blending an existing image

This method will blend an image with the diffusion for a number of steps. 
You may need to tinker with `--skip_timesteps` to get the best results.
```sh
(cgd_venv) $ python cgd.py --init_image=images/32K_HUHD_Mushroom.png --skip_timesteps=500 \
    "A mushroom in the style of Vincent Van Gogh"
```
![](images/a_mushroom_in_the_style_of_vangogh.png?raw=true)

### Image size
- Default is 256px
- Available image sizes are `64, 128, 256, 512 pixels (square)`
- The 512x512 pixel checkpoint **requires a GPU with at least 12GB of VRAM.**
- clip guidance scale and tv scale will require experimentation for image sizes less than 256px.
- the 64x64 diffusion checkpoint is challenging to work with and often results in an all-white or all-black image.
  - This is much less of an issue when using an existing image of some sort.
```sh
(cgd_venv) $ python cgd.py \
    --init_image=images/32K_HUHD_Mushroom.png \
    --skip_timesteps=500 \
    --image_size 64 \
    "8K HUHD Mushroom"
```
![](images/32K_HUHD_Mushroom_64.png?raw=true)

```sh
(cgd_venv) $ python cgd.py --image_size 512 "8K HUHD Mushroom"
```
![](images/32K_HUHD_Mushroom_512.png?raw=true)

## Full Usage:
```sh
cgd.py [-h] [--image_size IMAGE_SIZE] [--init_image INIT_IMAGE] [--skip_timesteps SKIP_TIMESTEPS] [--num_cutouts NUM_CUTOUTS] [--prefix PREFIX] [--batch_size BATCH_SIZE] [--clip_guidance_scale CLIP_GUIDANCE_SCALE] [--tv_scale TV_SCALE] [--seed SEED] [--save_frequency SAVE_FREQUENCY] [--device DEVICE] [--diffusion_steps DIFFUSION_STEPS]
              [--timestep_respacing TIMESTEP_RESPACING] [--cutout_power CUTOUT_POWER] [--clip_model CLIP_MODEL] [--class_cond CLASS_COND] [--clip_class_search]
              prompt

positional arguments:
  prompt                the prompt

optional arguments:
  -h, --help            show this help message and exit
  --image_size IMAGE_SIZE
                        Diffusion image size. Must be one of [64, 128, 256, 512]. (default: 256)
  --init_image INIT_IMAGE
                        Blend an image with diffusion for n steps (default: None)
  --skip_timesteps SKIP_TIMESTEPS
                        Number of timesteps to blend image for. CLIP guidance occurs after this. (default: 0)
  --num_cutouts NUM_CUTOUTS, -cutn NUM_CUTOUTS
                        Number of randomly cut patches to distort from diffusion. (default: 8)
  --prefix PREFIX, --output_dir PREFIX
                        output directory (default: outputs)
  --batch_size BATCH_SIZE, -bs BATCH_SIZE
                        the batch size (default: 1)
  --clip_guidance_scale CLIP_GUIDANCE_SCALE, -cgs CLIP_GUIDANCE_SCALE
                        Scale for CLIP spherical distance loss. Default value varies depending on image size. (default: 1000)
  --tv_scale TV_SCALE, -tvs TV_SCALE
                        Scale for denoising loss. Disabled by default for 64 and 128 (default: 100)
  --seed SEED           Random number seed (default: 0)
  --save_frequency SAVE_FREQUENCY, -sf SAVE_FREQUENCY
                        Save frequency (default: 100)
  --device DEVICE       device (default: None)
  --diffusion_steps DIFFUSION_STEPS
                        Diffusion steps (default: 1000)
  --timestep_respacing TIMESTEP_RESPACING
                        Timestep respacing (default: 1000)
  --cutout_power CUTOUT_POWER, -cutpow CUTOUT_POWER
                        Cutout size power (default: 1.0)
  --clip_model CLIP_MODEL
                        clip model name. Should be one of: [ViT-B/16, ViT-B/32, RN50, RN101, RN50x4, RN50x16] (default: ViT-B/16)
  --class_cond CLASS_COND
                        Use class conditional. Required for image sizes other than 256 (default: True)
  --clip_class_search   Lookup imagenet class with CLIP rather than changing them throughout run. Use `--clip_class_search` on its own to enable. (default: False)
```
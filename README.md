# CLIP Guided Diffusion
From [RiversHaveWings](https://twitter.com/RiversHaveWings).

Generate vibrant and detailed images using only text.

<a href="https://colab.research.google.com/github/afiaka87/clip-guided-diffusion/blob/main/cgd_clip_selected_class.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg"> </a>

<img src="images/THX_sound_Spaceship.png" width="128px"></img><img src="images/Windows_XP_background_Mushroom.png" width="128px"></img><img src="images/a_red_ball_resting_on_top_of_a_mirror.png" width="128px"></img>
<img src="images/dog_looking_at_the_camera.png" width="128px"></img><img src="images/goose_on_the_run.png" width="128px"></img><img src="/images/photon.png" width="128px"></img>

See captions and more generations in the [Gallery](/images/README.md)

See also - <a href="https://github.com/nerdyrodent/VQGAN-CLIP">VQGAN-CLIP</a>

---

### Installation
```sh
git clone https://github.com/afiaka87/clip-guided-diffusion.git
cd clip-guided-diffusion
python3 -m venv cgd_venv
source cgd_venv/bin/activate
❯ (cgd_venv) pip install -r requirements.txt
❯ (cgd_venv) git clone https://github.com/afiaka87/guided-diffusion.git
❯ (cgd_venv) python guided-diffusion/setup.py install
```

### Download checkpoints

You only need to download the checkpoint for the size you want to generate.
Checkpoints belong in the `./checkpoints` directory.

- 64: https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_diffusion.pt
- 128: https://openaipublic.blob.core.windows.net/diffusion/jul-2021/128x128_diffusion.pt
- 256: https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt
- 512: https://openaipublic.blob.core.windows.net/diffusion/jul-2021/512x512_diffusion.pt

There is only one unconditional checkpoint. This one doesn't require a randomized class like the others do. Use `--class_cond False` to use.
- 256 (unconditional):  https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt

### Generate an image

- Outputs are saved in `--prefix` (default:'./outputs')
- Filename format `f"{caption}/batch_idx_{j}_iteration_{i}.png"`
- The most recent generation will also be stored in the file `current.png`

```sh
❯ (cgd_venv) python cgd.py \
  --image_size 256 \
  --prompt "32K HUHD Mushroom"
Step 999, output 0:
00%|███████████████| 1000/1000 [00:00<12:30,  1.02it/s]
```
![](/images/32K_HUHD_Mushroom.png?raw=true)


### Class scoring
- Scores are used to weight class selection.
```sh
❯ (cgd_venv) python cgd.py \
  --class_score \
  --top_n 50 \
  --prompt "An image of a cat"`
```


### Penalize a prompt
- `--prompt_min` loss is weighted half.
- `--prompt_min` is also used to weight class selection with `--class_score`.
```sh
❯ (cgd_venv) python cgd.py \
    --prompt "32K HUHD Mushroom" \
    --prompt_min "green grass"
```

<img src="images/32K_HUHD_Mushroom_MIN_green_grass.png" width="256"></img>
### Blending an existing image

This method will blend an image with the diffusion for a number of steps. 
You may need to tinker with `--skip_timesteps` to get the best results.
```sh
❯ (cgd_venv) python cgd.py \
    --init_image=images/32K_HUHD_Mushroom.png \
    --skip_timesteps=500 \
    --prompt "A mushroom in the style of Vincent Van Gogh"
```
![](images/a_mushroom_in_the_style_of_vangogh.png?raw=true)

### Image size
- Default is 128px
- Available image sizes are `64, 128, 256, 512 pixels (square)`
- The 512x512 pixel checkpoint **requires a GPU with at least 12GB of VRAM.**
- `--clip_guidance_scale` and `--tv_scale` will require experimentation.
- the 64x64 diffusion checkpoint is challenging to work with and often results in an all-white or all-black image.
  - This is much less of an issue when using an existing image of some sort.
```sh
❯ (cgd_venv) python cgd.py \
    --init_image=images/32K_HUHD_Mushroom.png \
    --skip_timesteps=500 \
    --image_size 64 \
    --prompt "8K HUHD Mushroom"
```
<img src="images/32K_HUHD_Mushroom_64.png?raw=true" width="256"></img>

```sh
❯ (cgd_venv) $ python cgd.py --image_size 512 --prompt "8K HUHD Mushroom"
  ```
<img src="images/32K_HUHD_Mushroom_512.png?raw=true" width="360"></img>


> This code is currently under active development and is subject to frequent changes. Please file an issue if you have any constructive feedback, questions, or issues with the code or colab notebook.

## Full Usage:

```sh
❯ (cgd_venv) python cgd.py --help
usage: cgd.py [-h] [--prompt PROMPT] [--prompt_min PROMPT_MIN] [--image_size IMAGE_SIZE] [--init_image INIT_IMAGE]
              [--skip_timesteps SKIP_TIMESTEPS] [--prefix PREFIX] [--batch_size BATCH_SIZE]
              [--clip_guidance_scale CLIP_GUIDANCE_SCALE] [--tv_scale TV_SCALE] [--class_score CLASS_SCORE] [--top_n TOP_N]
              [--seed SEED] [--save_frequency SAVE_FREQUENCY] [--device DEVICE] [--diffusion_steps DIFFUSION_STEPS]
              [--timestep_respacing TIMESTEP_RESPACING] [--num_cutouts NUM_CUTOUTS] [--cutout_power CUTOUT_POWER]
              [--clip_model CLIP_MODEL] [--class_cond CLASS_COND]

optional arguments:
  -h, --help            show this help message and exit
  --prompt PROMPT       the prompt to reward (default: None)
  --prompt_min PROMPT_MIN
                        the prompt to penalize (default: )
  --image_size IMAGE_SIZE, -size IMAGE_SIZE
                        Diffusion image size. Must be one of [64, 128, 256, 512]. (default: 128)
  --init_image INIT_IMAGE
                        Blend an image with diffusion for n steps (default: None)
  --skip_timesteps SKIP_TIMESTEPS, -skipt SKIP_TIMESTEPS
                        Number of timesteps to blend image for. CLIP guidance occurs after this. (default: 0)
  --prefix PREFIX, -dir PREFIX
                        output directory (default: outputs)
  --batch_size BATCH_SIZE, -bs BATCH_SIZE
                        the batch size (default: 1)
  --clip_guidance_scale CLIP_GUIDANCE_SCALE, -cgs CLIP_GUIDANCE_SCALE
                        Scale for CLIP spherical distance loss. Default value varies depending on image size. (default: 900.0)
  --tv_scale TV_SCALE, -tvs TV_SCALE
                        Scale for denoising loss (default: 0.0)
  --class_score CLASS_SCORE, -score CLASS_SCORE
                        Enables CLIP guided class randomization. Use `-score False` to disable CLIP guided class generation.
                        (default: True)
  --top_n TOP_N, -tn TOP_N
                        Top n imagenet classes compared to phrase by CLIP (default: 1000)
  --seed SEED           Random number seed (default: 0)
  --save_frequency SAVE_FREQUENCY, -sf SAVE_FREQUENCY
                        Save frequency (default: 5)
  --device DEVICE       device to run on .e.g. cuda:0 or cpu (default: None)
  --diffusion_steps DIFFUSION_STEPS, -steps DIFFUSION_STEPS
                        Diffusion steps (default: 1000)
  --timestep_respacing TIMESTEP_RESPACING, -respace TIMESTEP_RESPACING
                        Timestep respacing (default: 1000)
  --num_cutouts NUM_CUTOUTS, -cutn NUM_CUTOUTS
                        Number of randomly cut patches to distort from diffusion. (default: 16)
  --cutout_power CUTOUT_POWER, -cutpow CUTOUT_POWER
                        Cutout size power (default: 1.0)
  --clip_model CLIP_MODEL, -clip CLIP_MODEL
                        clip model name. Should be one of: ['ViT-B/16', 'ViT-B/32', 'RN50', 'RN101', 'RN50x4', 'RN50x16']
                        (default: ViT-B/16)
  --class_cond CLASS_COND, -cond CLASS_COND
                        Use class conditional. Required for image sizes other than 256 (default: True)
```
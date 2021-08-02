# CLIP Guided Diffusion
From [RiversHaveWings](https://twitter.com/RiversHaveWings).
![Windows XP Background](/images/Windows_XP_background_Mushroom.png?raw=true)

> Windows XP background mushrooms

<p>
<a href="https://github.com/afiaka87/clip-guided-diffusion/blob/main/colab_clip_guided_diff_hq.ipynb">
  <img alt="Generate with Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
  Colab Forms Notebook - WIP
</a>
</br>
<a href="/notebooks/original-hq-notebook.ipynb">
  <img alt="Original notebook" src="https://colab.research.google.com/assets/colab-badge.svg">
  Original Notebook (Recommended)
</a>


[More images](/images/README.md)

---

## Installation
```sh
git clone https://github.com/afiaka87/clip-guided-diffusion.git
cd clip-guided-diffusion
python -m venv cgd_venv
source cgd_venv/bin/activate
(cgd_venv) $ # Should be inside virtual environment now.
(cgd_venv) $ which python # double check your python binary is from your virtual env
/path/to/cgd_venv/bin/python
(cgd_venv) $ pip install -r requirements.txt
```

## Usage:
```sh
(cgd_venv) $ python cgd.py \
    --num_cutouts=8 \
    --prefix='output_dir' \
    --batch_size=1 \
    --timestep_respacing=250 \
    --save_frequency=100 \
    --device=[none|cpu|cuda] \
    --txt_clip_guidance_scale=1000 \
    --img_clip_guidance_scale=50 \
    --tv_scale=100 \
    --clip_model='ViT-B/32' \
    'THX sound spaceship'
```


### Image Prompts
```sh
(cgd_venv) $ python cgd.py \
    --img_prompts='images/THX_sound_Spaceship.png' \
    --img_clip_guidance_scale=50 \
# ...


```
### (WIP) - 64x64 class conditioned checkpoint
You can try using the 64x64 px checkpoint with a random imagenet class.
```sh
# (WIP) Sample class conditional checkpoints using a random imagenet class idx.
(cgd_venv) $ python cgd.py 
  --class_cond True \
  --image_size 64 \
# ...
```


## CLI Output:
```
Step 1, output 0:
00%|██              █▋| 1/250 [??:??<00:00,  ?.??it/s]
```
Results `{caption}/batch_idx_{j}_iteration_{i}.png` in the directory specified with `--prefix`. Defaults to `./outputs/`.
![THX sound spaceship](/images/THX_sound_Spaceship.png?raw=true)
> THX sound spaceship

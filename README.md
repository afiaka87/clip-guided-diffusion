# CLIP Guided Diffusion
From [RiversHaveWings](https://twitter.com/RiversHaveWings).

<a href="https://github.com/afiaka87/clip-guided-diffusion/blob/main/colab_clip_guided_diff_hq.ipynb">
  <img alt="Generate with Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
</a>

![Windows XP Background](/images/Windows_XP_background_Mushroom.png?raw=true)

> Windows XP background mushrooms

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
    --prefix='outputs' \
    --batch_size=1 \
    --timestep_respacing=250 \
    --save_frequency=100 \
    --device=[none|cpu|cuda] \
    --text_prompt_weight=1000 \
    --img_prompt_weight=50 \
    --tv_weight=100 \
    --clip_model='ViT-B/32' \
    'THX sound spaceship'
```

## CLI Output:
```
Step 1, output 0:
00%|██              █▋| 1/250 [??:??<00:00,  ?.??it/s]
```
Results `batch_idx_{j}_iteration_{i}.png` in the directory specified with `--prefix`. Defaults to `./outputs/`.
 - ![THX sound spaceship](/images/THX_sound_Spaceship.png?raw=true)
THX sound spaceship

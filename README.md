# CLIP Guided Diffusion

From <a href='https://twitter.com/RiversHaveWings'>RiversHaveWings</a>

<p align='center'>
  <a href="https://github.com/afiaka87/clip-guided-diffusion/blob/main/colab_clip_guided_diff_hq.ipynb">
         <img alt="Generate with Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
  </a>
<p>
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
    --cutout_power=1. \
    --clip_model='ViT-B/32' \
		'An avocado with the shape and form of an armchair. An armchair imitating an avocado. Avocado armchair.'
```

## CLI Output:
```
Step 1, output 0:
00%|██              █▋| 1/250 [??:??<00:00,  ?.??it/s]
```
Results `batch_idx_{j}_iteration_{i}.png` in the directory specified with `--prefix`. Defaults to `./outputs/`.
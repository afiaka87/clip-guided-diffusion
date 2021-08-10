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


---

## Installation
```sh
git clone https://github.com/afiaka87/clip-guided-diffusion.git
cd clip-guided-diffusion
python3 -m venv cgd_venv
source cgd_venv/bin/activate
(cgd_venv) $ pip install -r requirements.txt
```

## Usage:
```sh
(cgd_venv) $ python cgd.py \
  --init_image=images/32K_HUHD_Mushroom.png \
  --image_size 256 \
  --num_cutouts 8 \
  --prefix 'outputs' \
  --batch_size 1 \
  --clip_guidance_scale 1000 \
  --tv_weight 150 \
  --seed 0 \
  --save_frequency 10 \
  --device 'cuda' \
  --diffusion_steps 1000 \
  --skip_timesteps 500 \
  --timestep_respacing '1000' \
  --cutout_power 1.0 \
  --clip_model 'ViT-B/32' \
  "32K HUHD Mushroom"
```

## CLI Output:
```
Step 1, output 0:
00%|██              █▋| 1/250 [??:??<00:00,  ?.??it/s]
```

## Outputs:
Generations will be saved in the folder from `--prefix (default:'./outputs')`

with the format `{caption}/batch_idx_{j}_iteration_{i}.png`

![](/images/32K_HUHD_Mushroom.png?raw=true)
> 32K HUHD Mushroom

[More images](/images/README.md)

> This code is currently under active development and is subject to frequent changes. Please file an issue if you have any constructive feedback, questions, or issues with the code or colab notebook.

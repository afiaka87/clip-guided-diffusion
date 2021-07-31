Installation
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

Usage:
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
    $caption
```

CLI Output:
```
Step 1, output 0:
00%|██              █▋| 1/500 [04:44<00:00,  1.77it/s]
```
Results `batch_idx_{j}_iteration_{i}.png` in the current directory.
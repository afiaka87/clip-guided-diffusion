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
    --clip_guidance_scale=500 \
    --tv_scale=100 \
    --seed=0 \
    --save_frequency=100 \
    --use_fp16 \
    --device=cuda \
    --diffusion_steps=1000 \
		--timestep_respacing=250 \
    --cutout_power=1.0 \
    --clip_model=ViT-B/16 \
    $caption
```

CLI Output:
```
Step 1, output 0:
00%|██              █▋| 1/500 [04:44<00:00,  1.77it/s]
```


Results are saved as `progress_{batch_id}.png` in the current directory.



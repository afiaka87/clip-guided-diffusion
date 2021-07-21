Installation
```sh
python -m venv cgd_venv
source cgd_venv/bin/activate
(cgd_venv) $ # Should be inside virtual environment now.
(cgd_venv) $ git clone https://github.com/afiaka87/clip-guided-diffusion.git
(cgd_venv) $ pip install -r requirements.txt

```

Usage:
```sh
(cgd_venv) $ which python # double check your python binary is from your virtual env
/path/to/cgd_venv/bin/python
(cgd_venv) $ python clip_guided_diffusion.py \
	--batch_size 1 \
	--clip_guidance_scale 3000 \
	--seed 0 \
	"a red apple on a wooden table"
```

CLI Output:
```
Step 1, output 0:
00%|██              █▋| 1/500 [04:44<00:00,  1.77it/s]
```


Results are saved as `progress_{batch_id}.png` in the current directory.



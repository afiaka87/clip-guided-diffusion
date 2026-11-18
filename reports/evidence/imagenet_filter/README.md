# ImageNet for the fine-tune

These scripts fetch ILSVRC 2012, the ImageNet that `guided-diffusion` was trained on, so it can be mixed with the LAION tiers under the same rules. They are a record, like `../laion_filter/`, and are not tested.

## `download.py`

It downloads [timm/imagenet-1k-wds](https://huggingface.co/datasets/timm/imagenet-1k-wds) to `F:\imagenet-1k-wds`: 1,024 train shards (1,281,167 images, 154.4 GB) and 64 validation shards (50,000, 7.0 GB). It runs in its own environment, from the inline dependencies at its top:

```sh
uv run reports/evidence/imagenet_filter/download.py --dry-run   # What it would fetch, and the free space
uv run reports/evidence/imagenet_filter/download.py             # Train and validation
uv run reports/evidence/imagenet_filter/download.py --sha256    # The same, then hash every file against the repository
```

Before running it, accept ImageNet's terms on the dataset's page and log in with `hf auth login`. The terms allow only non-commercial research and education.

Run it again to resume:

- It skips files already complete, and continues files cut off part way.
- It retries a dropped connection 20 times by itself.
- It stops with an error if a split's shard count differs from the expected one, or if the drive is short of space.
- `--sha256` renames any file that does not match to `.badNNNN`, and the next run fetches it again.

**Measured here** on 2026-11-15:

- **Xet stalled.** Hugging Face's Xet protocol, the default, stalled on its first shard: 68 MB of 108 in ten minutes, retrying one range request after another.
- **Plain HTTPS from the CDN worked.** One shard alone came at about 1 MB/s. Eight at once gave 31 MB/s, and 16 at once, the script's default, 52 MB/s. So the script turns Xet off and fetches 16 files at a time; the whole download should take about an hour.
- **The card gives one count wrong.** It says 63 validation shards; the repository holds 64, `imagenet1k-validation-00.tar` to `-63.tar`.

## The format

Each shard holds about 1,250 samples, each in three members:

- `KEY.jpg`: the original JPEG, at its original size.
- `KEY.cls`: the class index, 0 to 999, in sorted synset order as `cgd/imagenet_classes.py` has it.
- `KEY.json`: for example `{"label": 196, "width": 500, "height": 375, "filename": "n02097047/n02097047_2079.JPEG"}`.

The shards were converted from TensorFlow Datasets' ImageNet, and the first samples checked were RGB JPEGs. The width and height are what the no-upscaling rule needs, so the first filtering pass can drop small images without decoding them.

## Filtering and captioning

These apply the LAION tiers' rules to ImageNet, reusing `../laion_filter/pipeline/` through its `CGD_` and `STAGEC_` environment overrides. Working files go to `D:\imagenet-filter\{train,validation}\`, logs to `D:\imagenet-filter\logs\`, sheets to `D:\imagenet-filter\sheets\`. They run in the GPU environment of `../laion_filter/bench/pyproject.toml`.

| Script | What it does |
|---|---|
| `prepare.py` | Crops every image to 256 as img2dataset cropped LAION (area filter, centre crop, JPEG q95), into `crops256\`, and measures it into `scan\`. Images under 256 on the short side or wider than 1.8:1 are not cropped |
| `select_a.py` | The classical filters, writing `selection\stage_a.parquet` |
| `calibration.json` | The stage C thresholds for ImageNet, and why three of them differ from LAION's |
| `run_rest.sh` | Stage C (junk, near-duplicates, scores) and stage D (Florence-2 captions, webdataset shards) for one split |
| `chain.sh` | The run of 2026-11-15: waits for stage B, then `run_rest.sh` for both splits |
| `sheets.py`, `dup_pairs.py` | Contact sheets of what each test removed, and of near-duplicate pairs |
| `verify.py` | Checks a finished split against its selection: members, labels, a sample of decoded images, caption statistics |

## The run of 2026-11-15

| | Train | Validation |
|---|---|---|
| Source | 1,281,167 | 50,000 |
| After the classical filters | 1,068,059 | 45,470 |
| Kept | 964,246 | 40,568 |

- **Output:** `F:\imagenet-1k-clean-256\train` (193 tars, 32 GB) and `\validation` (9 tars, 1.4 GB). The README there has the format, the JSON fields and what each test removed.
- **Time:** cropping and measuring took minutes on 28 processes, the embedding pass for train 31.5 minutes, and captioning train 3.79 hours at 71 images a second on an RTX 4070.
- **Checked** with `verify.py`: every selected image is there exactly once with its three members, no label differs from the selection, and all 9,666 sampled train images decoded as 256 x 256 RGB. Its output is in `D:\imagenet-filter\logs\verify_train.txt` and `verify_validation.txt`.
- **Classes:** all 1,000 remain in both splits. In train the median is 1,036 images, 47 classes have under 500, and the thinnest is website with 36.
- **A gap the captions show:** 1.6% of train captions mention a watermark. Of 32 of those looked at, about 13 had a visible mark the detector missed.

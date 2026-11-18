# Scripts behind "LAION aesthetic data for the unconditional model"

These were run on this machine, against `D:\laion2b-en-aesthetic`. Each writes to `D:\laion2b-filter\`. They are a record of how the report's numbers were made, not part of `cgd`. None of them is tested or linted.

## Analysis on the CPU (the repository's environment)

| Script | What it does |
|---|---|
| `stats.py` | Sums img2dataset's `_stats.json` files: URLs attempted, success rate, failure reasons |
| `scan.py` | Makes one pass over every shard, recording each sample's metadata and 30 image statistics, into `analysis\scan\NNNNN.parquet` |
| `load.py` | Reads the scan back as one table |
| `meta_report.py`, `img_report.py`, `extras.py` | The report's counts over all images: metadata, image statistics, captions and domains |
| `dups.py`, `exif.py`, `pad_check.py`, `pad_report.py` | Placeholders and duplicates, EXIF rotation, strict padding on a 30-shard sample |
| `funnel.py` | The classical filters as a funnel |
| `zeroshot.py` | CLIP ViT-B/32 zero-shot composition on the CPU |
| `sheet_keys.py` | Contact sheets of chosen samples |

## Models on the GPU (`bench/`, its own CUDA environment)

`bench/pyproject.toml` is that environment: torch 2.11 with CUDA 12.8, transformers, open_clip, ultralytics and aesthetic-predictor-v2-5. Run the scripts from the directory holding it with `uv run python <script>`. They load weight files that are not copied here: LAION V1 (`sa_0_4_vit_l_14_linear.pth` from LAION-AI/aesthetic-predictor), V2 (`sac+logos+ava1-l14-linearMSE.pth`, from the Hugging Face cache) and v2.5 (`aesthetic_predictor_v2_5.pth`).

| Script | What it does |
|---|---|
| `make_sample.py` | Copies 14 random shards (82,285 images) to `bench_sample\` |
| `common.py` | Streams images from tars with uint8 batches normalised on the GPU, plus timers |
| `bench1_embed.py` | Measures SigLIP2-B/16, PE-Core-B/16 and ViT-L/14 throughput, and recomputes LAION V1 and V2 |
| `bench2_judges.py` | Runs Aesthetic Predictor v2.5 and the YOLO11n watermark detector |
| `bench3_caption.py`, `flo384.py` | Compares captioners: Florence-2-base at 768, 512 and 384, SmolVLM-256M and Qwen3-VL-2B |
| `probes.py` | Measures score drift, agreement between judges, and how well probes on the SigLIP2 embedding reproduce them |
| `zs_siglip.py`, `zs_b32.py` | Zero-shot composition |
| `calib.py` | Makes the calibration sheets for padding, watermarks and the zero-shot junk scores |

## The pipeline (`pipeline/`)

| Script | What it does |
|---|---|
| `stage_a_select.py` | Applies the classical hard filters, writing `selection\stage_a.parquet` |
| `stage_b_embed.py` | Computes a SigLIP2 embedding, a watermark confidence and edge-band features for each survivor, written to `stageB\part_*.npz`. It resumes |
| `stage_c_select.py` | Drops junk and near-duplicates, scores quality, clusters, and writes `selection\clean.parquet` and `best1m.parquet`. Thresholds come from `selection\calibration.json` |
| `stage_d_write.py` | `best` or `clean`: writes webdataset shards to `F:\`, captioning the best tier with Florence-2-base. It resumes |
| `run_overnight.sh` | Runs B, C, then D for both tiers. Logs go to `D:\laion2b-filter\logs\` |
| `composition.py` | Compares the zero-shot content of the stage-A survivors and both tiers, and how evenly each spreads over the clusters. Runs on the CPU |
| `dup_pairs.py` | Makes a sheet of dropped near-duplicates beside the copy kept |
| `verify_best.py` | Checks the written best tier against `best1m.parquet`: every sample once, every tar's members, caption statistics |

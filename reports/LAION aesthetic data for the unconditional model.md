# What is in D:\laion2b-en-aesthetic, and how it was cut down to a clean set and a best million

`D:\laion2b-en-aesthetic` holds 5,586,730 images. All of them are 256 × 256 centre crops made by img2dataset, and none is white-padded by the download. Each scored above 7 on LAION's oldest aesthetic predictor when LAION crawled it in 2021 or 2022. The files, though, were fetched again in September 2025, and the scores no longer describe all of them:

- 4.6% are now dead-link placeholders; VectorStock's "removed" picture alone is 1.8%.
- 6.9% are byte-identical copies of another file.
- 2.5% are a different picture from the one LAION scored.
- 28.8% are upscales of sources smaller than 256px.

Even where the picture is unchanged, the stored score is useless for ranking inside this set: it correlates at .02 with Aesthetic Predictor v2.5. Cheap probes on one SigLIP2-B/16 embedding reproduce the judges that do rank (LAION v2 at r = .86, v2.5 at r = .88, a YOLO watermark detector at AUC .88), so one embedding pass at 767 images a second can do the work of models that run at 59.

The pipeline built on this ran overnight (see "The run"):

- Classical tests keep 2,862,639 images.
- One GPU pass then embeds them and drops padding, watermarks, text, collages, screenshots, logos and nudity (206,917), plus near-duplicates (303,645). That leaves a **clean tier** of 2,352,077, written to `F:\laion-aesthetic-clean-256` (471 shards, 80 GB).
- A **best tier** of 1,000,000 is chosen by quality inside 1,000 embedding clusters and captioned with Florence-2-base, into `F:\laion-aesthetic-best1m-256` (200 shards, 36 GB). The captions ran at 77 images a second, and the best tier finished at 08:49 on November 15.

Both tiers are webdataset shards with the metadata inline, and each folder has a README describing its fields. These limits stand:

- No human rated anything; every threshold was set by one reading of contact sheets.
- The quality score's weights are a judgement, not a fit.
- Food and cakes make up a fifth of the best tier.

## How to read this report

Each claim is labelled **Measured here** (run for this report on this copy of the data, with the script linked), **Published** (from the source cited), or **Estimate** (worked out, not run). The labels mean this:

- **Counts over all images** come from a scan of all 989 shards ([scan.py](evidence/laion_filter/scan.py), 57 minutes on 30 CPU processes). It wrote every sample's metadata plus 30 image statistics to `D:\laion2b-filter\analysis\scan\`.
- **Model numbers** come from a fixed sample of 14 random shards, 82,285 images ([bench scripts](evidence/laion_filter/bench/)). Timings are from an RTX 4070 with torch 2.11 and CUDA 12.8, in fp16, with nothing else on the GPU.
- **"On the sheets"** means one reading of the contact sheets in `D:\laion2b-filter\analysis\sheets\`, which is an anecdote. The sheets are kept out of the repository because they are scraped web images.

## Where the data came from

**Published.** `laion2B-en-aesthetic` is LAION's 52M-row subset of laion2B-en. LAION first kept rows with `pwatermark < 0.8` and `punsafe < 0.5`, then rows whose predicted aesthetic score is above 7 ([LAION's notes][laesthetic]). The aesthetic score comes from the LAION-Aesthetics Predictor V1, a single linear layer on OpenAI CLIP ViT-L/14 image embeddings ([repo][lap]). Its weight file is named `sa_0_4_vit_l_14_linear.pth`, which points to Simulacra Aesthetic Captions ratings. It is not the better-known v2 predictor (an MLP trained on SAC + LAION-Logos + AVA). The other columns come from LAION-5B ([paper][l5b], [blog][l5bblog]):

| Field | Meaning | Made by | Range here (measured) |
|---|---|---|---|
| `aesthetic` | Predicted rating, 0 to 10 | LAION-Aesthetics V1 | 7.00 to 9.75, median 7.30, half between 7.13 and 7.56. 76 rows hold corrupt values up to 10.38 |
| `similarity` | Cosine similarity of image and alt-text | OpenAI CLIP ViT-B/32. LAION-2B-en kept ≥ 0.28 | Median 0.32. About 3,400 rows hold NaN, 0 or values over 1 |
| `punsafe` | Probability the image is unsafe | LAION's CLIP-based NSFW detector | < 0.45, median 0.00014 |
| `pwatermark` | Probability of a watermark | LAION's watermark CNN | < 0.80, median 0.13 |
| `WIDTH`, `HEIGHT` | Image size when LAION crawled it | LAION | |
| `hash` | LAION's 64-bit row hash, for joining with other LAION metadata | LAION | |
| `caption` (= `.txt`) | The HTML alt-text | Common Crawl | |
| `original_width/height` | Size of the file fetched in September 2025 | img2dataset | 1 × 1 to 30720 × 20562 |
| `width/height` | Stored size | img2dataset | Always 256 × 256 |
| `sha256` | Hash of the downloaded file | img2dataset | 5,202,688 distinct |
| `exif` | The source file's EXIF, as JSON | img2dataset | 27.8% non-empty |

**Measured here.** This copy is not all 52M. The 988 `_stats.json` files account for 9,765,322 attempted URLs, about 19% of the subset, and which 19% cannot be told from the data. img2dataset ran on September 19 and 20, 2025, and 57.2% of downloads succeeded. The failures were 1.13M DNS errors, 1.06M HTTP 404, 0.69M HTTP 403 and 0.22M undecodable files, plus timeouts. Shards 00000 to 00016 lost about 91% of their URLs to "Temporary failure in name resolution", a resolver outage while those first shards ran, not dead links; about 150,000 of those URLs could be fetched again. Shard 00988 has no stats file, and its tar is cut off after 354 samples (`tar: Unexpected EOF in archive`). A webdataset reader raises on it, so the pipeline skips it.

## How the images were processed

**Measured here.** Every image is a 256 × 256 RGB JPEG at quality 95 with 4:2:0 chroma subsampling, a median of 29.6 KB, 192 GB in all. The luma quantisation table is libjpeg's at q95 in every file checked, none are progressive, and no ICC profile was kept. Non-square sources fill the square with no bars, and a 1600 × 605 source shows its sides cut away. That is img2dataset's `center_crop` mode, not its `border` mode (which pads with white, `value=[255,255,255]`). From img2dataset's resizer ([source][i2dresize]):

- The short side is resized to 256, then the centre square is cut. A 4:3 source loses a quarter of its long side, and a 16:9 source loses 44%.
- Downscaling uses `cv2.INTER_AREA` (box averaging, no aliasing). Upscaling uses `cv2.INTER_LANCZOS4`.
- Sources under 256 were upscaled, because the run did not set `resize_only_if_bigger`. 1 × 1 placeholders became flat squares.
- Images with an alpha channel were composited onto **white**. Every transparent PNG (logos, cut-out products, clipart) now sits on pure white.
- The result was re-encoded once at q95. Colour profiles were dropped, so wide-gamut sources look slightly desaturated.
- EXIF rotation affects under 0.01% of images (196 of 2.5M carry a 90° tag). A few of those are sideways.

So img2dataset added no white padding. White bars and white backgrounds come from the source pictures, or from transparent PNGs.

## What is wrong with the files

**Measured here** over all 5,586,730 images unless a row says otherwise ([meta_report.py](evidence/laion_filter/meta_report.py), [img_report.py](evidence/laion_filter/img_report.py), [extras.py](evidence/laion_filter/extras.py), [probes.py](evidence/laion_filter/bench/probes.py)).

| Problem | Share | How it was found | What it does to a fine-tune |
|---|---|---|---|
| Upscaled: source short side under 256 | 28.8% (1.61M). Under 224: 15.5%. Under 128: 1.2% | `original_width/height` | Teaches blur. Spectral energy above 0.4 cycles/pixel has a median of .0002 for sources under 128 and .0008 for 128 to 192. It rises to .012 at 256 to 384, .022 at 384 to 768 and .029 above 768, so larger sources are crisper even at 256 |
| Dead-link placeholders | 4.6% have a sha256 shared by 3 or more files. VectorStock alone is 1.8% (102,436 copies of one "removed" picture, reported as 30720 × 20562). Others include eBay's grey photo frame, "Image not available", Magento's logo, Fox News's logo, and blank white or black | sha256 and dHash groups (sheet `dups_top.png`) | Recomputed on the file, V1 gives placeholders a median of 3.85, but every one carries the dead image's stored score of 7 or more, so an aesthetic filter on the stored score keeps them |
| Exact duplicates | 6.9% are copies of an earlier file (384,396) | sha256 | Memorisation, and an over-weighted handful of images |
| A different picture from the one LAION scored | 2.5% (138,514): the downloaded aspect is more than 5% off LAION's `WIDTH/HEIGHT`. Their recomputed V1 has a median of 3.85 | `WIDTH/HEIGHT` against `original_width/height` | All four LAION scores and the caption belong to another image |
| Stored aesthetic no longer matches the file | Stored V1 against V1 recomputed on the file: Pearson .44 (Spearman .62) overall, .69 on unchanged images | ViT-L/14 + V1 on the 82k sample | Even unchanged files moved, presumably because they were recompressed or resized since 2022 |
| Heavy centre-crop loss | 51% of sources are wider than 4:3, 5.0% wider than 16:9, 1.7% wider than 2:1 | Aspect ratio | Cut-off subjects. The caption describes the whole picture |
| Cut-outs and clipart on pure white | 10.0% have an outer 4px ring at least half near-white (every channel above 240), and 4.7% at least 90% | Ring statistic (sheet `s_ringwhite.png`) | Reads as white padding. Pulls the prior towards catalogue shots |
| Straight-edged bands (letterbox, pillarbox, frames) | 9.8% of a 168k sample have a flat band ending in a sharp step, most of them cut-outs. Requiring a straight edge as well leaves 0.4% of the ring-filtered survivors, which on the sheets are mostly real letterboxing: video frames with black bars, white bars | Strict band test with straightness (sheets `strict_pad_white.png`, `c_straight_pad.png`) | Teaches borders |
| Watermarks and overlay text | A YOLO11n watermark detector at 320px finds a box with confidence ≥ .5 on 1.5% of a 10k random sample, and ≥ .2 on 3.9%. On the sheets about half the .2 to .35 hits carry visible stock banners, logos or overlay text. LAION's `pwatermark` predicts the detector's hits at AUC .55 to .69 | `bmd_watermark_n` (sheets `c_wm_*.png`) | Teaches overlays. The LAION score is close to useless for this |
| Stock-photo and vector sites | 9.6% (Dreamstime, Adobe Stock thumbnails, Shutterstock, Depositphotos, Freepik, VectorStock and others) | URL domain | Most carry visible watermarks |
| Search-engine thumbnails | 4.6% (`t0.gstatic.com`, `tse*.mm.bing.net`) | URL domain | Small, recompressed sources |
| Greyscale | 1.6% have mean chroma under 4 | Chroma | ImageNet has little |
| Bright high-key images | 7.8% have mean luma over 200. Mean luma is 134, against about 117 for ImageNet | Luma | Pulls the prior towards white backgrounds |
| Truncated shard | 00988 | `tar` EOF | Crashes webdataset readers |

Some tests that looked promising did not hold up, and the pipeline does not use them:

- **Edge bands at a loose tolerance** (line std < 6) flag 28%. On the sheets those are mostly clear skies, studio backdrops and white products (sheet `s_bands.png`).
- **Colour count** picks greyscale and pale wedding dresses, not graphics (sheet `s_fewcolors.png`).
- **Low spectral energy** at a source of 256 or more is mostly shallow depth of field and smooth illustration (sheet `s_soft_highres.png`).
- **Flat interior lines as a collage detector** mostly found patterns and blank images (sheet `gutters.png`).
- **Zero-shot SigLIP2 prompts at the 97th percentile** for text, collage, screenshot and logo caught mostly good images. At the 99th they catch game screenshots, posters, sewing-pattern covers, grids and multi-panel pictures (sheets `c_zs_*`), so the pipeline cuts there.

## What the dataset contains

**Measured here** with SigLIP2-B/16-256 zero-shot over 32 categories on the 82k sample ([zs_siglip.py](evidence/laion_filter/bench/zs_siglip.py)). Each image goes to its best-matching category, so treat the shares as rough. CLIP ViT-B/32 on 10,000 other images gave the same picture, with crafts 11%, cars 9%, food 9%, cakes 7% and animals 1%.

| Category | All | After the classical filters |
|---|---|---|
| Cars and vehicles | 9.7% | 9.6% |
| Cakes, desserts, baking | 8.8% | 9.2% |
| Crafts and handmade (crochet, knitting, sewing) | 8.5% | 8.8% |
| Food | 7.1% | 8.6% |
| Painting and fine art | 5.4% | 6.3% |
| Fashion and clothing | 5.7% | 4.7% |
| Interiors and furniture | 4.8% | 5.3% |
| Groups of people and events | 4.1% | 4.6% |
| Buildings and architecture | 3.8% | 4.5% |
| Landscape and nature | 3.7% | 4.5% |
| Gardens, flowers, plants | 3.6% | 4.3% |
| Toys and dolls | 4.5% | 3.5% |
| Holiday decoration | 3.5% | 3.4% |
| Wedding | 2.7% | 3.0% |
| Portraits | 2.7% | 2.6% |
| Jewellery and accessories | 2.1% | 2.2% |
| Boats, planes, trains | 1.6% | 2.1% |
| Drinks | 1.8% | 2.0% |
| Beach and sea | 1.7% | 1.8% |
| Animals | 1.0% | 1.2% |
| Cartoon, clipart, vector | 3.9% | 1.0% |
| Logo and icon | 1.8% | 0.9% |
| Product on white | 1.6% | 0.7% |
| City and street | 0.6% | 0.7% |
| Everything else (3D render, pattern, tattoo/nails/hair, anime, screenshot, drawing, text, collage) | 6.2% | 5.0% |

The set is a catalogue of things people photograph to sell, cook or make: Pinterest serves 16% of it and shops (Shopify, Etsy, eBay, Amazon, AliExpress, Wix, Squarespace) another 16%. It is nothing like ImageNet, where about 40% of classes are animals: here animals are 1%, and city scenes and landscapes together are about 5%. Sampling evenly across embedding clusters flattens the big categories (classic cars, cupcakes, crochet), but it cannot create animals that are not there. **Fine-tuning on any subset of this will move the unconditional model's prior a long way from ImageNet.**

## Captions

**Measured here.** The `.txt` file is the raw HTML alt-text. Captions have a median of 9 words; 8.3% have 3 or fewer, and 5% have 51 or more, up to 4,378. The rest of the profile:

- 4.4% contain non-ASCII text.
- 1.8% contain a URL, 0.9% a file name, and 3.6% name a stock site.
- 4.9% say "royalty-free", "vector", "clipart" or "illustration", and 4.1% carry prices or sales talk.
- 0.35% contain HTML tags, for example `<i>The Goldfinch</i>`.
- The most repeated captions are listing titles ("1965 Ford Mustang" 421 times), news headlines reused across many photos, and forum chrome ("Click image for larger version" 266 times).

The unconditional model never reads a caption, so this matters only if the set later trains a conditioned model. The best tier gets generated captions anyway (below), and the alt-text stays in its JSON.

## Which models are worth running

### Speed

**Measured here** ([bench1_embed.py](evidence/laion_filter/bench/bench1_embed.py), [bench2_judges.py](evidence/laion_filter/bench/bench2_judges.py), [bench3_caption.py](evidence/laion_filter/bench/bench3_caption.py), [flo384.py](evidence/laion_filter/bench/flo384.py)).

| Model | Job | Images per second | Notes |
|---|---|---|---|
| SigLIP2 ViT-B/16 @256 | Embedding | 767 end to end (787 GPU-only) | Native 256px, 3.5 GB. Text tower: 2,743 captions/s |
| PE-Core ViT-B/16 @224 | Embedding | 760 (785) | |
| OpenAI ViT-L/14 @224 | Embedding, LAION V1/V2 heads | 212 (212) | 2.3 GB |
| Aesthetic Predictor v2.5 (SigLIP so400m-384) | Aesthetics | 59 | |
| BMD YOLO11n watermark detector | Watermarks | 1,484 at 320px, 508 at 640px | Images already decoded |
| Florence-2-base `<DETAILED_CAPTION>` | Captions | 31 at 768px, 67 at 512px, 98 at 384px | 41 words at 768px. 384px and 512px need the processor's image-token count set to match (145, 257) |
| Florence-2-base `<MORE_DETAILED_CAPTION>`, 768px | Captions | 31 | 77 words. The vision encoder dominates the cost, so longer captions cost nothing extra |
| SmolVLM-256M-Instruct | Captions | 18 | 16 words, vague ("In this image we can see a car") |
| Qwen3-VL-2B-Instruct (HF `generate`) | Captions | 8.5 | 55 words, the best captions (it names a placeholder graphic as one), but 1M images would take 33 hours |

Two lessons from getting these numbers on Windows:

- **Read from the tars, not from loose files.** Reading 82k extracted JPEGs runs at 240 files a second on this drive, while streaming the tars gives about 15,000 images a second.
- **Hand uint8 tensors between processes, not float32.** Moving float32 batches between DataLoader workers held SigLIP2 to 146 images a second. With uint8 batches normalised on the GPU, it ran at 767.

### Do the judges agree, and can a probe stand in for them?

**Measured here** on the 82k sample, after removing placeholders and changed images ([probes.py](evidence/laion_filter/bench/probes.py)).

| Pair | Pearson | Spearman |
|---|---|---|
| Stored V1 against v2.5 | .02 | .03 |
| V1 recomputed on the file against v2.5 | .13 | .14 |
| V1 recomputed against V2 | .25 | .24 |
| V2 against v2.5 | .55 | .55 |

Inside a set already filtered on V1 > 7, V1 carries no information the newer predictors share, and even V2 and v2.5 agree only moderately. **No single judge is ground truth**, so the quality score below averages two, with v2.5 weighted higher because it was trained on a wider range of images, illustrations included.

A ridge probe from the SigLIP2 embedding, trained on 80% and tested on the other 20%:

| Probe target | Labels | Pearson | Spearman | Overlap of predicted and true top 40% |
|---|---|---|---|---|
| V2 (L/14 MLP) | 66k | .855 | .843 | .79 |
| v2.5 (so400m) | 16k | .880 | .866 | .79 |
| v2.5 | 1.6k | .848 | .838 | .78 |
| V1 recomputed (L/14 linear) | 66k | .830 | .550 | .62 |
| YOLO watermark hit (logistic) | 8k | AUC .884 | | |

So the two aesthetic judges cost almost nothing once their outputs on a sample exist: 2,000 labels already give .85. Running ViT-L/14 on all 2.86M survivors would have taken 3.75 hours, and v2.5 13.5 hours.

## The pipeline

The design is a cascade: cheap tests on everything, one embedding pass on what survives, and the expensive judges only on the 82k sample, to train the probes. Captions come last, on the million kept. Every score is kept per sample, so a threshold can change and the selection re-run in minutes. The scripts are in [evidence/laion_filter/pipeline](evidence/laion_filter/pipeline/), and [run_overnight.sh](evidence/laion_filter/pipeline/run_overnight.sh) chains them.

**Stage A. Classical hard filters, CPU** ([stage_a_select.py](evidence/laion_filter/pipeline/stage_a_select.py), 3 minutes on the scan). Each image is dropped for the first test it fails, in this order: undecodable; blank; placeholder (sha256 seen 3+ times, or a dHash shared by 5+ images with 3+ different captions); exact duplicate; picture changed since LAION scored it; source short side under 256; aspect over 1.8; outer ring half or more near-white; greyscale; fewer than 256 colours; LAION `pwatermark` ≥ .5; LAION `punsafe` ≥ .1; corrupt metadata. **Measured here:** 2,862,639 of 5,586,730 survive. The drops were:

| Reason | Images |
|---|---|
| Source under 256 | 1,450,707 |
| Ring half white | 314,766 |
| Placeholder (sha256) | 248,769 |
| Exact duplicate | 147,193 |
| Aspect over 1.8 | 145,943 |
| `pwatermark` | 141,726 |
| Placeholder (dHash) | 137,970 |
| `punsafe` | 71,056 |
| Greyscale | 27,081 |
| Changed since scored | 15,385 |
| Under 256 colours | 13,676 |
| Blank | 6,400 |
| Corrupt metadata | 3,419 |

**Stage B. One GPU pass over the survivors** ([stage_b_embed.py](evidence/laion_filter/pipeline/stage_b_embed.py)). Streaming from the tars, each image gets three things:

- a SigLIP2-B/16-256 embedding;
- the YOLO detector's top watermark confidence at 320px, from the same decoded batch;
- strict edge-band features computed in the loader workers: band length, grey level, step into the content, and how straight that step is.

It writes one part file per 20 completed shards and resumes from them. **Measured here:** 599 images a second, all 2,862,639 in 79.7 minutes.

**Stage C. Selection** ([stage_c_select.py](evidence/laion_filter/pipeline/stage_c_select.py), 2.5 minutes on the GPU; thresholds in `D:\laion2b-filter\selection\calibration.json`):

1. **Junk:** it drops straight-edged bands of 3px or more, watermark confidence ≥ .25, and the top 1% of survivors on zero-shot scores for text, collage, screenshot/chart/map, logo/icon and nudity.
2. **Quality:** `0.45 z(v2.5 probe) + 0.30 z(V2 probe) + 0.10 z(log spectral energy above 0.4 c/px) + 0.15 z(log min(source short side, 768)) − 1 if watermark confidence ≥ .2`. The probes are ridge fits on the sample's real judge outputs.
3. **Clusters and near-duplicates:** spherical k-means on the GPU gives 20,000 leaves, then 1,000 parents over the leaf centroids, a two-level form of Vo et al.'s balanced sampling ([arXiv 2405.15613][vo]). Inside each leaf, any image with cosine > .95 to a better-scored one is a near-duplicate (SemDeDup, [arXiv 2303.09540][semdedup]).
4. **Clean tier** = everything left.
5. **Best tier** = 1M from the clean images above the 35th quality percentile. The parents are water-filled (every parent gets the same cap, which rises until the total is 1M), then the leaves inside each parent, taking the highest quality first. Illustration-like images (zero-shot painting/drawing/cartoon/3D over photo) are capped at 20% by swapping the weakest for the best photographs left.

The settings were chosen on a test run over the first 951k stage-B results. On the sheets a random draw from that test best tier was varied and good-looking (sheet `t_best_random.png`). Its weakest picks above a 25th-percentile floor were mostly catalogue product shots (sheet `t_best_lowest.png`), so the floor went up to the 35th.

**Measured here** on the full run over all 2,862,639 survivors:

| Step | Removed | Share of survivors |
|---|---|---|
| YOLO watermark ≥ .25 | 85,655 | 3.0% |
| Zero-shot nudity | 25,557 | 0.9% |
| Zero-shot text | 22,750 | 0.8% |
| Zero-shot logo/icon | 21,127 | 0.7% |
| Zero-shot collage | 20,655 | 0.7% |
| Zero-shot screenshot/chart/map | 17,407 | 0.6% |
| Straight-edged padding or frame | 13,766 | 0.5% |
| Near-duplicate | 303,645 | 10.6% |
| **Clean tier** | **2,352,077 left** | 82.2% |

- **Near-duplicates.** These were 10.6% of the full set but 4.9% of the test run, because in a set three times larger each image has three times as many others to match. On a sheet of 16 random dropped images beside the copy kept (`near_dup_pairs.png`, [dup_pairs.py](evidence/laion_filter/pipeline/dup_pairs.py)), 14 are the same picture re-encoded or slightly re-cropped. The other two are near-identical scenes: two floral fabrics, and two elephant-and-Kilimanjaro composites.
- **The best tier.** It is 1,000,000 images from 995 of the 1,000 parents. Its quality floor is −0.29 (z units), and 13.9% of it is illustration-like, so the 20% cap did not bind. Its largest parent holds 0.14% of it, against 0.36% before selection, and its 10 largest hold 1.4% against 3.2%.

**Measured here:** composition by the same SigLIP2 zero-shot categories as above, from the stage-B embeddings ([composition.py](evidence/laion_filter/pipeline/composition.py)):

| Category | Stage A survivors | Clean tier | Best tier |
|---|---|---|---|
| Food | 8.5% | 8.8% | 10.4% |
| Cakes, desserts | 9.1% | 9.3% | 9.8% |
| Cars | 9.8% | 9.9% | 7.1% |
| Painting, fine art | 6.3% | 6.4% | 6.9% |
| Crafts | 8.9% | 8.9% | 6.4% |
| Groups of people | 4.7% | 4.9% | 5.5% |
| Landscape | 4.6% | 4.7% | 5.2% |
| Interiors | 5.2% | 5.2% | 5.0% |
| Gardens, flowers | 4.1% | 4.2% | 4.9% |
| Architecture | 4.3% | 4.3% | 4.7% |
| Fashion | 4.7% | 4.7% | 4.4% |
| Portraits | 2.7% | 2.7% | 3.5% |
| Weddings | 2.9% | 2.9% | 3.4% |
| Toys, dolls | 3.6% | 3.7% | 1.9% |
| Animals | 1.3% | 1.4% | 1.8% |
| City, street | 0.7% | 0.7% | 1.1% |
| Logo, screenshot, text, collage together | 1.8% | 1.0% | 1.0% |

Balancing over clusters flattened cars, crafts and toys, and raised landscapes, portraits, animals and city scenes. Food and cakes rose instead of falling. Food photographs vary enough to fill many clusters, and both aesthetic judges score them well, so together they are 20% of the best tier. A per-category cap would be the fix if that matters; it is not applied.

**Measured here** over all 1,000,000 captions, and by reading the first and last shards' sheets (`out_best_00000.png`, `out_best_00199.png`):

- The captions are specific and mostly right: "a Japanese garden with trees in the foreground and a pagoda in the background", "three white rabbits dressed in formal attire" (wedding plush toys).
- Their errors are small, but names and fine categories are guessed. The last shard calls a view of Glasgow "Edinburgh", strawberry eclairs "strawberry shortcakes", and a girl "a woman". The alt-text, kept in the JSON, had all three right.
- The median is 38 words; 98% fall between 19 and 72, and the longest is 88.
- 0.27% are cut off at the 96-token limit.
- All but 126 open "The image shows", which a trainer may want to strip. The 126 open "In this image we can see".
- 2.8% contain a word such as text, sign, logo or lettering, and most of those describe text in the picture that the junk filters let through, such as a title strip under a painting. On the first shard, 4.7% did.
- 1.9% of images share their caption word for word with another. These are generic: "a red car parked on the side of a road, surrounded by trees and a clear blue sky" is 45 images, and its blue and black versions about as many.
- A few end in guessed qualities ("adding a sweet and nutty flavor").

**Stage D. Writing** ([stage_d_write.py](evidence/laion_filter/pipeline/stage_d_write.py)). It streams the selected samples from the source tars and writes webdataset shards of 5,000 to `F:\` (a separate USB SSD, so reads and writes do not share a drive):

- Each sample keeps its **original JPEG bytes**, with no re-encode, and its **original JSON**.
- The JSON gains `source_shard`, `source_key` and a `cgd_filter` object with every score: both aesthetic probes, quality, watermark confidence, the zero-shot scores, the illustration score, leaf and parent cluster, spectral energy, Laplacian variance, ring whiteness, colourfulness and mean luma.
- In the **best tier**, `.txt` is a Florence-2-base `<DETAILED_CAPTION>` caption made at 512px, and the JSON adds `generated_caption` and `caption_model`. The alt-text stays in the JSON's `caption`.
- In the **clean tier**, `.txt` is the alt-text, as in the source.
- Each shard has an index parquet beside it (key, source id, text), and a restart skips everything already written.

Florence at 512px was chosen over 768px because it is twice as fast, and on the sheets its captions were as good as 768px's and better than 384px's. Qwen3-VL-2B's are better than either, at a quarter of the speed, so it is the upgrade if the captions ever matter.

### The run

**Measured here,** on 2026-11-15. Logs are in `D:\laion2b-filter\logs\` and `D:\laion2b-filter\stageB\log.txt`.

| Step | Time | Output |
|---|---|---|
| Full scan | 57 min (01:02 to 01:59) | `D:\laion2b-filter\analysis\scan\` |
| Stage A | 3 min | `D:\laion2b-filter\selection\stage_a.parquet` (2,862,639) |
| Stage B | 79.7 min (03:49 to 05:09) | `D:\laion2b-filter\stageB\part_*.npz` (50 parts) |
| Stage C | 2.5 min (05:09 to 05:12) | `D:\laion2b-filter\selection\{scored,clean,best1m}.parquet` |
| Stage D, clean tier | 8.4 min (05:12 to 05:20), 4,700 img/s | `F:\laion-aesthetic-clean-256\`: 471 shards, 2,352,077 samples, 80 GB. Checked: every selected sample once, every member present, every image 256 × 256 RGB |
| Stage D, best tier with captions | 3.6 h (05:12 to 08:49), 77 img/s | `F:\laion-aesthetic-best1m-256\`: 200 shards, 1,000,000 samples, 36 GB. Checked: every selected sample exactly once, keys consecutive and in their shard, jpg, txt and json present for every key in every tar, no caption empty. Shards 0 and 199 decoded in full, every image 256 × 256 RGB, and shard 0 read back with webdataset 1.0.2 |

Two fixes were made during the run. Stage C's k-means as first written made a float32 copy of all 2.86M embeddings on the GPU, 8.8 GB on top of the 4.4 GB fp16 copy, which would not fit in 12 GB. It now accumulates in batches, and its results are identical on a CPU test. Its near-duplicate test now works in blocks of rows, so a large cluster cannot exhaust memory either.

## Answers to your questions

**How many images?** 5,586,730 in shards 00000 to 00987, plus 354 in the truncated 00988. That is 5,202,688 distinct files, and about 4.3M distinct real pictures once placeholders and copies are gone.

**What size?** 256 × 256, every one, RGB JPEG q95 with 4:2:0 chroma. The sources behind them have a median short side of 360; 28.8% are under 256 and 30.8% are 512 or more.

**How were they processed?** img2dataset `center_crop` at 256:

- short side resized to 256 (area filter down, Lanczos-4 up, upscaling allowed), then the centre square cut;
- alpha matted onto white;
- re-encoded at q95 with ICC profiles dropped.

There was no white padding from the download. The white bars and backgrounds come from the sources and from transparent PNGs.

**Which aesthetic classifier?** LAION-Aesthetics Predictor V1: a linear layer on OpenAI CLIP ViT-L/14 image embeddings, cut at above 7. A 2026 benchmark puts its rank correlation with human ratings at .29 ([ERNIE-Image report][ernie]). Here its stored values correlate with v2.5 at .02, so use it for nothing.

**Other questions that affect the fine-tune:**

| Question | Answer | Consequence |
|---|---|---|
| Do the stored scores describe the files? | Not reliably: placeholders and changed pictures keep the old score, and unchanged files correlate at only .69 with their recomputed V1 | Re-score from the files, as stage B does |
| How much true resolution is there? | 71% of sources are 256 or more; sharpness at 256 keeps rising with source size up to 768 and beyond | Both tiers require 256 or more. Quality prefers larger sources |
| How much of each picture survives the crop? | Half lose a quarter or more of the long side | Aspect capped at 1.8. Captions written from the crop (Florence) describe what is actually there |
| Is there NSFW content? | `punsafe` < .45 everywhere, but .5 is loose and the score belongs to the 2022 file | `punsafe` < .1, plus the top 1% of a zero-shot nudity score dropped |
| Is this the cleaned Re-LAION? | Unknown. LAION pulled LAION-5B in December 2023 after links to CSAM were found, and released Re-LAION-5B in August 2024 with them removed ([LAION][relaion]). Whether this copy predates that cannot be told from the files | **Before training,** join on `hash` or `url` against Re-LAION's metadata and drop anything absent from it. That needs access to the gated Hugging Face set, which only you can grant |
| How diverse is it? | 290k domains, but Pinterest is 16% and shops another 16%. By content it is cars, cakes, crafts and food | Balanced across embedding clusters, not domains |
| How close is it to ImageNet? | Far: animals 1% against about 40% of ImageNet's classes | Consider mixing ImageNet back in during the fine-tune |
| Will horizontal flips hurt? | Only where there is text, and the text-heavy images are cut | Flips are safe |
| Are there faces? | Many: portraits, weddings, fashion, groups | ImageNet's prior is poor at faces. Not filtered; a face detector could down-weight small faces in crowds |
| Colour and tone? | Brighter than ImageNet (mean luma 134 against 117) | Dropping the white cut-outs removes most of it |
| Orientation? | EXIF rotation affects under 0.01% | Ignored |
| Duplicates? | 6.9% exact, plus near-copies | sha256 in stage A, SemDeDup in stage C |
| Licensing? | Web-scraped with no licence; LAION distributes only links | Research use. Every sample keeps its URL |
| Can the set grow back? | About 150k URLs in shards 00000 to 00016 failed on DNS, not on dead links | Optional re-fetch |

## Decisions made while you slept, and how to undo them

You gave full autonomy, so these were made without you. Each is one setting.

| Decision | Why | To change |
|---|---|---|
| Products and clipart on white are dropped from both tiers (ring half or more white) | You ruled out white padding, and these read the same to the model | `white_ring_over_half` in stage A |
| Greyscale is dropped | ImageNet has little, and it is 1.6% | Stage A |
| The resolution floor is 256 for both tiers | You asked for the highest resolution possible | Stage A; 224 would keep 84% instead of 71% |
| Illustration-like images are capped at 20% of the best tier | ImageNet is photographs, but art is a fifth of what LAION calls beautiful | `illus_cap` in `calibration.json` |
| The best tier is captioned with Florence-2-base at 512px; the clean tier keeps alt-text | 3.6 hours, against about 33 for Qwen3-VL-2B | `stage_d_write.py`; the clean tier can be captioned later the same way |
| The quality floor for the best tier is the 35th percentile | Below it the picks were mostly catalogue shots | `floor_quantile` |
| No ImageNet mixing and no Re-LAION check | Both are yours: one is a training choice, the other needs your access | |

## What is left

- The Re-LAION join, before any training.
- Have a person read a few hundred best-tier images and captions. Every threshold here was set by one reading of contact sheets, and the zero-shot junk filters are weak below their 99th-percentile cuts.

[laesthetic]: https://github.com/LAION-AI/laion-datasets/blob/main/laion-aesthetic.md
[lap]: https://github.com/LAION-AI/aesthetic-predictor
[l5b]: https://arxiv.org/abs/2210.08402
[l5bblog]: https://laion.ai/blog/laion-5b/
[i2dresize]: https://github.com/rom1504/img2dataset/blob/main/img2dataset/resizer.py
[ernie]: https://arxiv.org/pdf/2605.25347
[relaion]: https://laion.ai/blog/relaion-5b/
[vo]: https://arxiv.org/abs/2405.15613
[semdedup]: https://arxiv.org/abs/2303.09540

# Fine-tuning the unconditional 256 model on CLIP embeddings

These scripts teach OpenAI's unconditional ImageNet 256 model to take a CLIP embedding as its condition, so that CLIP guidance refines an image the model was already told to draw instead of steering an unconditional one alone. They are scripts beside the `cgd` package, not part of it, and nothing in `cgd` uses the result yet.

All three steps have run on an RTX 4070, and two fine-tunes were started and stopped on 2026-11-16, see [What has run](#what-has-run). A pilot of the second approach ran the same day, see [Baking CLIP guidance in](#baking-clip-guidance-in), and then a pilot of the third, which starts from the class-conditional model instead, see [Class-anchored CLIP conditioning](#class-anchored-clip-conditioning).

## The steps

```sh
uv run python -m finetune.index              # 1. A table of every image and its metadata. CPU, minutes.
uv run python -m finetune.embed              # 2. CLIP embeddings of every image and text. GPU. Rerun to resume.
uv run python -m finetune.train --hours 12   # 3. The fine-tune. GPU. Add --wandb to chart it.
```

Each takes `-h`. The defaults are the LAION best tier (`F:\laion-aesthetic-best1m-256`), the filtered ImageNet train split (`F:\imagenet-1k-clean-256\train`), and the CLIP models `ViT-B/32` and `PE-Core-B-16`. Steps 1 and 2 write into a `finetune` folder inside each dataset, and step 3 into `D:\cgd-finetune\uncond256`.

| File | What it is |
|---|---|
| `index.py` | Step 1. One row per image: where its JPEG is in its tar, and every field of its JSON |
| `embed.py` | Step 2. Per CLIP model, the embedding of each image and of three texts for it |
| `data.py` | The loader. Mixes datasets in exact proportion, filters on metadata, returns images, texts and embeddings |
| `cgd/conditioned.py` | What is added to the frozen U-Net: the condition's path in, tokens, LoRA. In `cgd`, which loads it for `-ft` |
| `noise.py` | Which noise levels to train at, the loss weights, and the noise put on an embedding |
| `train.py` | Step 3 |
| `distill.py` | The second approach: train the adapters to do what CLIP guidance does. See [Baking CLIP guidance in](#baking-clip-guidance-in) |
| `cond.py` | The third: keep the class-conditional model's class and train its decoder to take CLIP embeddings too. See [Class-anchored CLIP conditioning](#class-anchored-clip-conditioning) |
| `prior.py` | A linear map from a text's embedding to its image's, to sample `cond.py`'s model from a prompt |
| `export.py` | Writes a `cond.py` run's weights and the prior as the file `cgd -ft` draws with. See [In cgd](#in-cgd) |
| `parallel.py` | Runs `cond.py` on several GPUs, a process each. See [Several GPUs](#several-gpus) |
| `vast/` | Shell scripts for a rented instance: upload, setup, the run, and copying it back. See [On vast.ai](#on-vastai) |

`tests/unit/test_finetune.py` covers the model, the noise levels and the loader, on a small U-Net of the real blocks and on shards it writes.

## The loader

```python
from finetune import data

models = ["ViT-B/32", "PE-Core-B-16"]
sources = [
    data.open_source("laion", data.LAION_ROOT, models, where="aesthetic >= 6.5 and pwatermark < 0.3"),
    data.open_source("imagenet", data.IMAGENET_ROOT, models, where="label in [281, 282, 283]"),
]
for batch in data.batches(sources, weights=[1, 1], batch_size=8, seed=0):
    ...
```

A batch holds:

- `image`: uint8, `[N, 3, 256, 256]`.
- `texts`: three lists of strings, see below.
- `embeds[model][kind]`: float32 `[N, dim]`, unit length. `kind` is `image` or one of the three texts.
- `label`: the ImageNet class, or -1.
- `source`: which dataset each image came from, as its place in `sources`.

**The mix is exact.** With equal weights every batch is half LAION and half ImageNet. Each dataset is drawn in its own shuffled order and reshuffled when it runs out, so the smaller one repeats before the larger one finishes. A seed gives the same batches, and `start` resumes part way.

**`where` is a pandas query** over the index's columns. LAION has `aesthetic`, `punsafe`, `pwatermark`, `similarity`, `alt_text`, `url`, `original_width` and `original_height`. ImageNet has `label`, `class_name`, `wnid` and `filename`. Both have `generated_caption` and the filter's own scores, such as `quality`, `wm_yolo`, `colorfulness`, `illus` and `zs_text`. For example `class_name.str.contains('terrier')` or `quality > 0 and illus < 0`.

**Images captioned as watermarked are left out.** The captioner mentions a watermark in 1.9% of the LAION tier and 1.6% of ImageNet, and many of those are marks the watermark detector missed: about 13 of 32 looked at in ImageNet and 5 of 32 in LAION. Leaving them all out takes the training split from 995,000 to 975,656 LAION images and from 960,000 to 944,720 ImageNet images, with all 1,000 classes still present. `keep_watermarked=True`, or `--keep-watermarked` when training, keeps them. The images stay in the shards.

**The last shard of each dataset is held out.** `split="holdout"` reads it, and training never does.

### The three texts

| Kind | LAION | ImageNet |
|---|---|---|
| `caption` | The generated caption, without its "The image shows" | The same |
| `first` | That caption's first sentence | The same |
| `short` | The alt text | `a photo of a <class>` |

`PE-Core-B-16` reads 32 tokens and the generated captions run to about 50, so the first sentence is embedded on its own. The short text is the closest to what a person types into `cgd`.

## What is trained

The U-Net's 554M weights stay frozen. About 32M are added, each starting so that the model is exactly the pretrained one:

- **The condition's path in.** Each CLIP model's embedding is projected and added to the timestep embedding, where a class label goes in the class-conditional checkpoints.
- **Tokens.** Four tokens made from the same projection, which a new cross-attention after each attention block reads (`--tokens`, 0 for none).
- **LoRA**, rank 16, on the attention projections and on each residual block's embedding layer (`--rank`).

Only the middle and the decoder half are adapted by default, so the encoder half runs without gradients, unless the noisy image itself needs one, as it does under CLIP guidance. `--train-encoder` adapts it too.

`cgd/conditioned.py` has four more additions that `train.py` leaves off and `distill.py` turns on, about 72M weights with them. Each also starts at zero, and each answers something [the second run's checkpoint](#what-the-second-runs-checkpoint-does) measured:

- **Standardised embeddings.** Each entry of an embedding is centred and scaled by statistics kept with the weights, and the hidden layer is normalised, so the condition no longer arrives small.
- **A scale and shift per block.** Each residual block gets its own projection of the condition, beside the shared one.
- **A layout.** The condition is projected to an 8 by 8 feature map, added after the middle block and the first six decoder blocks, because nothing else that is trained knows position.
- **Convolution updates.** A low-rank update, rank 8, on each adapted residual block's last convolution.

Each training step:

- **Picks a condition per image.** The image's own embedding with noise on it, or with `--p-text` (0.4) the embedding of one of its three texts. Each CLIP model's embedding is dropped 15% of the time on its own, and all of them 10% of the time, which is what classifier-free guidance needs.
- **Picks a noise level** uniform in logSNR over the schedule, as `cgd`'s sampler steps, leaning to high noise (`--noise-bias`).
- **Weights the loss** with Min-SNR, gamma 5, and counts the noisiest levels for more (`--snr-floor`, 0.1). At those levels the noise prediction's error is tiny however wrong the image behind it, so without the floor they hardly move the weights, and they are where the condition matters most. Only the noise prediction is trained. The learned variance is left alone, because the sampler does not use it.

It keeps a running average of the added weights (`--ema`, 0.999), saves every 20 minutes, and stops after `--hours`.

**`--resume` continues as if the run had not stopped.** The images come in the same order, and each step draws its noise, noise level and condition from the seed and its own number, so a resumed run trains on exactly what an uninterrupted one would have. A run that was killed goes back to its last checkpoint, at most 20 minutes earlier, and the records it logged past that are taken out of `log.jsonl` and logged again. On the CPU, six steps run in one go and six run as two, two more, then back to the checkpoint at step 2 and four more gave identical weights and logs. On a GPU the numbers differ in their last bits from run to run, as the fused attention's do.

### Weights & Biases

`--wandb` also sends each record of `log.jsonl` to Weights & Biases: the loss, the images per second, and the check's `conditioned`, `unconditioned` and `gap`, charted against `step`. The project is `cgd-finetune`, or the one named, as in `--wandb other-project`, and `--wandb-name` names the run. The options are saved as the run's configuration.

Log in once first, with the key from https://wandb.ai/authorize:

```sh
uv run --no-sync wandb login
```

The checkpoint keeps the run's id, so `--resume --wandb` continues the same run and its charts. Records are logged at their own step, and a step that is already there is ignored, so a run that went back to its checkpoint adds no repeats: the test above left steps 0 to 6 there once each. With `WANDB_MODE=offline` set, the run is written to `wandb` inside `--out` and `wandb sync` uploads it later. Without `--wandb` nothing is sent.

The defaults for the condition's probabilities, the embedding noise and the noise bias are first guesses. None has been tuned.

## The check to watch

Every 2,000 steps the run measures the loss at high noise on held-out images twice, with their embeddings and without, on the same noise. `gap` is how much lower the loss is with them, and `gap_image` is the same with each image's error counted in terms of the image and not the noise, which weighs the noisiest ones as much as the rest. Both are exactly 0 before training, and should grow.

If `gap` is still under 0.5% after an hour, the model is not using the condition, and the run prints a warning. Stop it then and try `--train-encoder`, more `--tokens`, or a higher `--lr`, before the remaining hours are spent.

## The checkpoint

`adapter.pt` holds the configuration, the added weights (`weights`), their running average (`average`), and the optimizer. To load it for sampling:

```python
import torch as th
from cgd.adm import load_model
from cgd.conditioned import load_adapter

checkpoint = th.load("D:/cgd-finetune/uncond256/adapter.pt", map_location="cpu", weights_only=True)
unet, log_alpha_bar = load_model(256, "cuda", False, False, True)
model = load_adapter(unet, checkpoint | {"weights": checkpoint["average"]}, fused=False).to("cuda")
noise_prediction = model(x, timesteps, {"ViT-B/32": text_embedding, "PE-Core-B-16": other_embedding})[:, :3]
```

`model(x, timesteps)` without embeddings is the unconditional prediction, for classifier-free guidance. `fused=False` keeps the U-Net's own attention, whose backward pass is deterministic, as guidance needs for a seed to reproduce.

Wiring this into `cgd.adm` as an option is not written.

## What has run

On the CPU:

- **The unit tests**, 61 of them. For `parallel.py`: each process draws its own batches and together they draw what one would, two processes end with the weights one gets from both their images, one failing stops the rest (those two on Linux only), and a run started by itself is unchanged. For `cond.py` and `prior.py`: a new model is the pretrained one told the class, the middle and decoder are trained whole and the encoder gets no gradient, classifier-free guidance scales the embeddings' move and keeps one gradient, the anchor is the closest class, a condition keeps the image's embedding, the text's, both or neither in the shares asked for, patch tokens become a grid, the pretrained weights get their own share of the learning rate, `--decay` brings it to zero by `--max-steps`, and the map recovers a linear one. For `distill.py`: the teacher's move is `cgd`'s guided step, the loss is 1 for the base model and 0 for the teacher, training on the guidance loss lowers it within the cap, wide layers get their share of the learning rate, samples use the averaged weights, and a gradient reaches the noisy image through the whole adapted model. For the rest: a new model's output equals the pretrained one's, the fused attention equals the U-Net's own, a dropped embedding is ignored, only the added weights get gradients, the noise levels and timesteps match the sampler's, and the loader's mix, filters, resume and pairing of each image with its own texts and embeddings.
- **All three steps on 96 real images**, 48 from each dataset. Each image's embedding retrieved its own caption 96 to 100% of the time with both CLIP models. Four training steps of the real 256 model ran, the check read exactly 0 before the first and rose after, and a resumed run continued from the same numbers.
- **The index of the whole LAION best tier**: 1,000,000 rows in 2 minutes.
- **Resuming the embedding step.** On the 48 LAION images, with the record of progress set back to 16 image rows and one text array not started, and everything past those points overwritten, a second run redid only what was missing and every array came out identical to the first run's.

On the GPU, an RTX 4070 with 12 GB:

- **A CUDA build of PyTorch** is needed for steps 2 and 3. The lock file installs PyTorch 2.9.1 from PyPI, which on Windows is the CPU build, so `uv run` and `uv sync` put that back. On 2026-11-15 the CUDA build of the same version was installed over it with `uv pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128 --reinstall-package torch --reinstall-package torchvision`. Run the GPU steps with `uv run --no-sync` so it stays.
- **Step 2 on the GPU** was then tried on the 48 LAION images. At half precision every embedding's cosine to the CPU's was 0.9997 or more. The full run measured about 580 images a second through both models on an RTX 4070.
- **Step 2 in full** took 3 h 35 min for both datasets, 1,964,246 images: about half an hour per dataset for the images and an hour for the texts, most of it PE-Core's text tower. Every row of every array is unit length. Of 2,000 images of each dataset, 90% retrieved their own caption among the 2,000 with PE-Core and 74 to 76% with ViT-B/32.
- **Memory.** Training takes about 3.4 GB plus 3 GB an image at half precision: 6.4 GB at `--batch 1`, 9.4 GB at 2 and 15.4 GB at 4. So the default is `--batch 2 --accumulate 4`, an optimizer step every 8 images.
- **Do not exceed the card's memory.** On Windows PyTorch does not fail then, it pages to main memory. `--batch 8` ran at 0.06 images a second, 127 seconds a step, and a 200 step trial took 7.5 hours. The log's `gpu_gb` is the peak so far, and should stay under the card's size.
- **Speed.** 4.5 images a second at `--batch 2`, and the same per image at `--batch 1`, so 12 hours is about 195,000 images or 24,000 optimizer steps.
- **The first run**, at `--lr 1e-4` and without `--snr-floor`, was stopped after an hour, 2,150 optimizer steps: `gap` stayed within 0.02% of zero, although every added weight had moved. It is in `D:/cgd-finetune/uncond256`.
- **The second run**, with `--lr 3e-4` and the floor, ran from 11:55 to 13:29 on 2026-11-16 and was stopped at step 12,800, 3,200 optimizer steps. `gap_image` rose to between 2 and 4%. It is in `D:/cgd-finetune/uncond256-v2`, last saved at step 11,800. At 13:27 it stopped with a CUDA out of memory error and resumed itself from that checkpoint.

## What the second run's checkpoint does

Measured on 2026-11-16 at step 11,800, with the scripts and grids in `reports/evidence/finetune/`.

- **It barely steers a sample.** 24 held-out images were drawn again from their own embeddings with no CLIP guidance, 40 steps, and judged by `ViT-L-16-SigLIP2-256`, which the model never saw. Similarity to the real image, LAION then ImageNet: 0.41 and 0.47 from the unconditional model, 0.43 and 0.48 from the adapter at guidance scale 3, and 0.50 and 0.63 from the class-conditional checkpoint given the ImageNet class closest to the caption. The grids show the adapter's images are the unconditional ones slightly changed (`diag_sample.py`).
- **The check understates conditioning.** The class-conditional checkpoint given the true class is only 3.5% better than the unconditional one on `gap_image`, over 128 ImageNet images. So a few percent there is what a fully trained condition looks like, and the check cannot tell a weak one from a strong one. Judge samples instead (`diag_levels.py`).
- **The embeddings know a thumbnail.** A ridge regression from an image's two CLIP embeddings to a 4 by 4 thumbnail predicts 26% of the whole image's pixel variance, and nothing more is predictable at 8, 16 or 32. From the caption's embeddings it is 13%, and from the short text's 7%. At the noisiest quarter of the probes the adapter had 4% on LAION and none on ImageNet (`diag_ceiling.py`).
- **Text embeddings are not image embeddings.** An image's `ViT-B/32` embedding is closer to another image's (cosine 0.48) than to its own caption's (0.32). Conditioned on the caption's embeddings the adapter was 3 to 5% worse than with nothing at the noisiest quarter. A linear map from the text embedding, fitted by ridge regression, reaches a cosine of 0.84 to the image's (0.77 for `PE-Core-B-16`).
- **The condition arrives small.** Unit embeddings give the first layer activations of about 0.014. At step 11,800 what is added to the timestep embedding had a norm of 1.1 against the embedding's 18, 5%. In the class-conditional checkpoint a class embedding is 15% of it (`diag_scale.py`).
- **A mixture of class embeddings breaks the class-conditional checkpoint.** Weighting the 1,000 class embeddings by a softmax of CLIP similarity at temperature 100 gave errors several times the unconditional model's. Only that one temperature was tried.

## Baking CLIP guidance in

`distill.py` was written on 2026-11-16, after the runs above. It does not ask the model to learn from images what an embedding means. It trains the adapters to do at each step what `cgd` does with CLIP.

```sh
uv run --no-sync python -m finetune.distill --hours 0.5 --wandb             # Copy the teacher's move
uv run --no-sync python -m finetune.distill --hours 0.2 --overfit --wandb   # Only the texts it samples
uv run --no-sync python -m finetune.distill --max-steps 0                   # Only the teacher's own samples
uv run --no-sync python -m finetune.distill --hours 0.2 --direct --wandb    # The CLIP loss itself. It collapses, see below
```

- **The teacher** is `cgd` as it samples, by its own functions. The unconditional model predicts the noise, `ViT-B/32` and `PE-Core-B-16` each score 16 cutouts of the denoised image against a text, and the gradient of that, capped at 0.08 of the noise, moves the prediction. A unit test holds `guided_shift()` to `cgd.sampler.predict`.
- **The student** is the same model with the adapters, told the text's CLIP embeddings, the ones step 2 saved. It learns to predict the teacher's moved noise.
- **Each example** is one training image at a noise level drawn as the sampler steps, within the part of the schedule `cgd` guides, and one text: one of the image's three, or half the time another image's (`--p-swap`), because guidance mostly has to move an image towards a text it does not match yet.
- **The loss** is the squared error from the teacher's noise, over the size of the teacher's move. `missed` is 1 for a student that ignores the text and 0 for one that reproduces the move, and `cosine` compares the two moves' directions alone. Two draws of the teacher's cutouts agree at a cosine of 0.88 to 0.90 (`teacher/agreement`), so about a tenth of the move cannot be learnt.
- **The samples.** Every `--eval-minutes` the run draws 8 held-out texts, 4 from each dataset, in 30 steps and each from its own fixed noise, with the averaged weights and no CLIP: the student's move as it is (`student x1`) and three times it (`student x3`). The grid has them under the real images, the unconditional model's and the live teacher's, and goes to Weights & Biases as `samples`. `text/...` is a row's similarity to the texts by `ViT-L-16-SigLIP2-256`, which neither the teacher nor the student has seen, and `image/...` its similarity to the real images.

One example at a time is what fits: 8.8 GB and 1.8 examples a second. `--accumulate` averages several before an optimizer step. It resumes as `train.py` does, and a resumed run keeps its sample texts and its Weights & Biases run.

**Wide layers get less of the learning rate** (`--widest`, 64). Adam moves every weight by about the learning rate, so a layer's output moves by that times its number of inputs, and standardising makes the entries the condition's first layers read about thirty times larger. The first attempt, at `--lr 3e-4` on every layer, overshot the teacher within 100 examples: `missed` went to 13. With each layer's rate divided by its inputs over 64, `missed` stayed under 1.2 at 1e-4 and under 1.4 at 3e-4. `train.py` has the option too, off.

`--direct` trains on the guidance loss itself instead of the teacher's move. The student's own denoised image is scored on the same kind of cutouts, and the loss is backpropagated into the adapters, with the student's move held to the same cap. `--tv` is how much of `cgd`'s smoothness term that loss keeps, 0 by default.

### The pilot

All on 2026-11-16, each run in the Weights & Biases group `distill-pilot` under the name given, at the defaults. The judge's scores of the 8 samples, text then image: the real images 0.156 and 1, the unconditional model's -0.054 and 0.476, the teacher's 0.127 and 0.682. At 60 steps the teacher scores 0.126 and 0.711, so the samples are drawn at 30.

| Run | Trained on | Minutes | Examples | Student x1 | Student x3 |
|---|---|---|---|---|---|
| `distill-overfit8-lr1e-4` | the 8 sample texts | 12 | 929 | 0.014, 0.557 | 0.002, 0.551 |
| `distill-general-regress` | every text | 15 | 1,242 | -0.002, 0.562 | 0.003, 0.549 |
| the same run, resumed | every text | 45 | 3,928 | -0.005, 0.546 | -0.008, 0.555 |
| `distill-overfit8-direct` | the 8 sample texts, `--direct --tv 1` | 12 | 1,108 | 0.024, 0.560 | 0.016, 0.556 |
| `distill-overfit8-direct-notv` | the 8 sample texts, `--direct` | 9 | 932 | 0.029, 0.573 | 0.011, 0.560 |

With 8 samples a score moves by about 0.02 from one evaluation to the next. `distill-overfit8-diverged` is the first attempt, stopped by hand.

- **The teacher is worth copying.** On the unconditional model, with no class to help, its samples show what the text names: a muffin on a plate, a yellow coat, a waterfall, a tiger in leaves, a yellow eel, a tank with a fish. They have guidance's saturated look.
- **Copying it steers colour and texture, and no more.** Within 15 minutes the student alone gives texts it never trained on green for forests and fields and blue for water and beaches, on images that still look like the unconditional model's. By the judge that is under a third of the way from the unconditional model to the teacher on the text, and about two fifths on the image. From minute 15 to minute 45 the scores did not rise and the colours only grew more saturated. No object the text names appears.
- **It learns a small, generic part of the move and then stops.** `cosine` reached 0.16 within 250 examples on the 8 texts, 0.11 on every text, and stayed there, with `missed` at 1. `reports/evidence/finetune/diag_distill.py` on the 8-text student: the cosine is 0.14 to 0.19 at every noise level and at every pooling from none to 64 times, and its move is 9 to 29% of the teacher's size. The teacher's move is not fine detail. At 0.9 of the schedule 52% of its energy survives pooling by 16, and 28% pooling by 64. What the student misses is how the move depends on the image in front of it, which is what CLIP sees. In the first 250 examples the cosine rose equally fast at 1e-4 and at 3e-4.
- **Training on the CLIP loss itself responds faster and collapses.** `--direct` reached a higher text score in 4 minutes than copying did in 12, at 2.2 examples a second. But its samples are fields of texture in the text's colours, crumbs for the muffin and foam for the beach, with nothing of an image left. Without the smoothness term they are the same, only grainier. The teacher's gradient passes back through the U-Net, which keeps a move among images the model finds likely. A loss backpropagated into the adapters has no such filter.
- **CLIP guidance on the student beats guidance on the unconditional model.** `diag_distill_guided_more.py` draws each text twice from the same noise, with `cgd`'s guidance on the unconditional model and with the same guidance on the 45 minute student, on 16 held-out texts the run never drew. At 30 steps the student's images score 0.132 against the texts and the unconditional model's 0.100: 0.033 better, with a standard error of 0.008, and the better of the pair for 13 texts. Against the real image it is 0.612 and 0.567, 0.044 better with an error of 0.020. At 15 steps it is 0.109 and 0.084 on the text, 0.025 with an error of 0.012, so the student at 15 steps scores what the unconditional model does at 30. On the run's own 8 texts (`diag_distill_guided.py`) the student led at 15 steps, 0.111 and 0.092, and at 30 the two tied on the text, 0.122 and 0.127, with the student behind on the image, 0.643 and 0.682. The grids are `guided_*.jpg` in the run's folder. One checkpoint, one judge, one noise per text.

That last test needs a gradient through the whole student. The model ran the frozen encoder half without one, so guidance on any adapter would have skipped the U-Net and raised nothing. It now lets the gradient through when the noisy image asks for one, and a unit test holds it to the pretrained model's.

### What it says

In 45 minutes the adapters did not absorb CLIP's judgement of an image. They absorbed the part of guidance that does not depend on looking: the palette and texture a text calls for. Alone that is not text-to-image. Under CLIP guidance it is a better start than the unconditional model, which is what these scripts set out to make. So what the numbers support is a distilled student as the model `cgd` guides, not as a replacement for guidance.

Whether hours of copying would get past colour is not known. The measure to watch in a longer run is `cosine`, which has to leave 0.11.

Not run: any distillation longer than 45 minutes, guidance on the student against `cgd`'s default of the class-conditional model with the closest class, a second judge, `--plain`, `--accumulate` above 1, a single CLIP model or fewer cutouts as the teacher, and wiring a student into `cgd.adm`. `--compile` (which on Windows needs a C++ compiler) and `--train-encoder` have not run either.

## Class-anchored CLIP conditioning

`cond.py` and `prior.py` were written on 2026-11-16, after the two approaches above. CLIP guidance approximates, one step at a time, what a model conditioned on a CLIP embedding would predict by itself. `cond.py` trains that model directly, with the denoising loss on real images, and it starts from `cgd`'s default pipeline and not from the unconditional model.

- **The class stays.** The U-Net is the class-conditional 256 checkpoint, told the ImageNet class closest to the text by `ViT-B/32`, as `cgd` picks it. What the embeddings add starts at zero, so at step 0 the model is the one `cgd` draws with.
- **The encoder half is frozen and the rest is trained whole.** The middle and decoder's own weights, 390M of 554M, with no LoRA beside them, and 31M of new weights: the condition's path and the token attention of `cgd/conditioned.py`.
- **Four conditions.** Per CLIP model, the image's own embedding with noise on it, and the embedding of one of its three texts. An image keeps both, either or neither, so the model can be drawn from a text alone, from an image embedding, and with classifier-free guidance against the class alone.
- **CLIP guidance still applies.** `cfg_denoiser()` is a denoiser for `cgd.sampler`. Under guidance only the pass with the embeddings carries the gradient, so a guided step costs one backward pass through the U-Net, as now, and one more forward pass.
- **`prior.py`** fits a linear map from a text's embedding to its image's, to fill the image condition from a prompt.

```sh
uv run --no-sync python -m finetune.prior                              # The map. CPU, a minute.
uv run --no-sync python -m finetune.cond --hours 12 --wandb            # The fine-tune
uv run --no-sync python -m finetune.cond --hours 12 --resume --wandb   # Continue it
```

It writes to `D:\cgd-finetune\cond256`, and resumes and logs as `train.py` does. `-h` has the options, and the defaults are what the pilot below left standing.

### The pilot on the RTX 4070

Three hours on 2026-11-16, to find what breaks before a longer run on rented GPUs. One image per optimizer step is what fits 12 GB, at 3 images a second, so each run saw 2,000 to 5,000 images. That tests every part, and shows what damages the pretrained model. It is far too few to learn the condition: the published fine-tune nearest to this one, Lambda's image-conditioned Stable Diffusion, saw 2.1M.

The samples are the distillation pilot's 8 held-out texts, from the same noise, in 30 steps, judged the same way. Text then image: the real images 0.156 and 1, the pretrained model told the class (`class only`) 0.041 and 0.640, and the same under CLIP guidance (`class guided`) 0.138 and 0.707. That last row is `cgd` as it is with `ViT-B/32` and `PE-Core-B-16` guiding, and the row to beat. Every model row is classifier-free guidance at 3. The runs are in the Weights & Biases group `cond-pilot`, all with `--ema 0 --embed-noise 0.5` and the U-Net's own weights at 1e-5.

| Run | Loss weights | Rate of the new layers | Images | Class alone, noisy and cleaner levels | Gap, text | `class now` | `text x3` |
|---|---|---|---|---|---|---|---|
| `cond-align-stopped-at-3000` | capped at 5, floor 0.1, lean 2, patch loss 0.01 | 1e-4, `--widest 64` | 3,000 | +2.1%, not measured | -0.3% | not drawn | 0.037, 0.610 at 2,100 |
| `cond-plain-objective` | every level alike, no lean | 3e-4, `--widest 256` | 2,098 | +0.4%, -3.4% | -0.1% | 0.048, 0.643 | 0.056, 0.640 |
| `cond-floor-no-cap` | floor 0.1, lean 1.5 | 3e-4, `--widest 256` | 2,097 | +1.2%, -3.2% | -4.1% | 0.046, 0.628 | 0.048, 0.628 |
| the same run, resumed | | | 4,741 | +1.2%, -3.7% | -1.8% | 0.049, 0.604 | 0.056, 0.596 |
| `cond-floor-align` | floor 0.1, lean 1.5, patch loss 0.01 | 3e-4, `--widest 256` | 1,654 | +1.5%, -3.0% | -3.3% | 0.050, 0.606 | 0.032, 0.599 |

`Class alone` is how much the loss told the class alone changed from the pretrained model's, on held-out images at the noisiest 40% of levels and at the rest. `Gap, text` is how much lower the loss at high noise is with the text's embeddings than without, in terms of the image. `class now` is the fine-tuned model told the class alone. With 8 samples a score moves by about 0.02 between evaluations.

- **What the fine-tune costs the pretrained model depends on the loss weights.** The first run used `train.py`'s: the cleaner levels capped at a signal to noise ratio of 5, which leaves the cleanest a thousandth of their weight, and noise levels leaning hard to the noisy end. By 2,100 images its samples had lost their detail: in `reports/evidence/finetune/cond_align_2100.jpg` the lower three rows are a blurred version of the second. With every level at its whole weight, as the model was pretrained, the model told the class alone draws as sharply as before (`cond_plain_2098.jpg` and `cond_floor_2100.jpg`, fourth row) and its loss at the cleaner levels fell 3%, with or without more weight on the noisiest levels. The patch feature loss does damage of its own. `cond-floor-align` is the third run's settings with that loss at 0.01, and by 1,654 images `class now` and every conditioned row had washed out to pale images of low contrast (`cond_floor_align_1654.jpg`), with the loss told the class alone at the noisiest levels 25% higher in terms of the image, against 8% without it. The first run had that loss and the cap together, and the cap without the loss was not run, so whether the cap alone also costs detail is not known.
- **The new layers need more of the learning rate than `distill.py` gave them.** At 1e-4 with `--widest 64` the layer that adds the condition to the timestep embedding trains at 6e-6, and in 3,000 images the conditions changed nothing: the three conditioned rows of that grid are the same images. At 3e-4 with `--widest 256` they changed the samples within 2,100.
- **The condition hurts before it helps.** In `cond-floor-no-cap` the text's embeddings made the loss at high noise 6.9% worse at 1,000 images, 3.0% worse at 2,000 and 0.2% better at 3,000. At 4,000 it was 0.1% worse and at 4,741 1.8% worse, so from 3,000 on it is zero within the noise of 48 probes. At 2,100 images the conditioned rows differ from `class now` in lighting, contrast and colour, a sunset for the beach and a silhouette for the coat, and not yet in what is drawn. By 4,741 several columns have taken on the real image's palette, with grass under the tiger, green for both forest scenes and a low sun over water for the beach (`cond_floor_4700.jpg`), all exaggerated by the scale of 3. `class now` is still sharp there, with its image score down from 0.640 to 0.604, more than the 0.02 a score moves by. The judge does not yet prefer any conditioned row.
- **Memory.** 9.2 GB at one image: the weights 2.2, their gradients 1.7, AdamW's state 3.4, and about 2.3 an image. Accumulating gradients over several images does not fit 12 GB, because the gradients then stay through the next forward pass, so `--accumulate` is 1. `train.Run` now uses the fused AdamW, which makes no temporary copies. The running average of the weights is another 1.7 GB and was off. On a 32 GB card that leaves room for 6 to 8 images a step with the average on.
- **Half precision needs the gradient scaler.** Without it the denoising loss's gradient on the middle block is exactly zero, lost to underflow. The trainer scales it. A script that takes gradients by hand has to as well.
- **The patch feature loss was 275 times too heavy.** At a weight of 1, with every image counted alike, its gradient on the middle block's weights is 30 to 700 times the denoising loss's at high noise and up to 700,000 times at low noise, a median of 275. With each image counted by its share of noise the ratio is 45 to 140 at every level, so a weight of 0.01 puts the two on a par. REPA's 0.5 would have swamped the denoising loss there. In the first run the head reached a cosine of 0.47 to PE-Core's patch features within 2,000 images, the same with and without the condition.
- **The map from a text's embedding to its image's.** `prior.py`'s ridge regression, on 60,000 images of each dataset, predicts held-out image embeddings to a cosine of 0.80 to 0.83 for `ViT-B/32` and 0.72 to 0.76 for `PE-Core-B-16`. A text's own embedding is at 0.26 to 0.33. Noise of relative length up to 0.5 on the image embedding only reaches a cosine of 0.89, so `--embed-noise` is now 1.0, which reaches 0.71.
- **The anchor.** `reports/evidence/finetune/diag_anchor.py` on 50,000 images of each dataset. For ImageNet the class closest to `a photo of a <class>` is the label 99.8% of the time, and the class closest to a generated caption 53%, in its five closest 74%. LAION's texts land on 840 to 940 of the 1000 classes, but `dining table` takes 10% of its captions and `website` 12% of its alt texts, so the model has to learn to draw past a wrong anchor from the embeddings.
- **CLIP guidance on the fine-tuned model.** `reports/evidence/finetune/diag_cond_guided.py` on the 4,741 image checkpoint, with 8 held-out texts the run never drew, each from the same noise in 30 steps. `cgd` as it is scores 0.128 against the texts and 0.594 against the real images, at 7.0 seconds an image. The fine-tuned model under the same guidance, with classifier-free guidance at 3 from the text's embeddings, scores 0.142 and 0.635: 0.014 better on the text with a standard error of 0.013, and 0.041 better on the image with an error of 0.018, the better of the pair for 5 and 7 texts. It takes 10.6 seconds an image, half as long again, for the second forward pass. With the prior's image embedding as well it scores 0.144 and 0.612. Without CLIP the same model scores -0.001 and 0.492: at this point its conditions alone, tripled, spoil the image. So guidance works on the result, costs half as much again, and on a model that has learnt next to nothing is not worse. The guided images are harsher in contrast than `cgd`'s, with black areas (`cond_guided_8texts_30steps.jpg`, rows 4 and 5 against row 2). 8 texts, one judge, one noise each.

### What the pilot does not show

That the condition is learnt. No run has a gap or a sample score that is distinguishable from the class alone, and none could have at 5,000 images. What stands is that the model keeps its detail through the fine-tune with the weights it now has, though `class now` has to be watched in a longer run, that the conditions reach the samples at the new layers' higher rate, and that every part of a run works: the samples, the checks, the checkpoint, resuming, and guidance on the result.

Not run: the cap on the cleaner levels without the patch feature loss, more than one image a step, the running average, `--embed-noise 1.0`, the prior's embedding as a condition during training, two GPUs or more, and any run long enough to learn the condition.

### For a rented run

What the pilot settled, and what it leaves to the first hour of a longer run:

- **Settled, and now the defaults:** every noise level at its whole weight with a floor of 0.1 and a lean of 1.5, the new layers at 3e-4 with `--widest 256`, the U-Net's own weights at 1e-5, `--embed-noise 1.0`, and no patch feature loss, which washed the samples out.
- **Set by the hardware:** 6 to 8 images a step on a 32 GB card with the running average on. `parallel.py` runs `cond.py` on all four, see [Several GPUs](#several-gpus), so a step is 24 to 32 images.
- **To watch from the start:** `probe/class_clean` and the `class now` row should hold, and the gaps should turn positive and stay there. If the conditioned rows are not ahead of `class now` by the judge when about 250,000 images are in, the run is not on course.
- **Not known:** how many images the condition takes to learn, whether the anchor helps or holds it back, and whether the prior's embedding in training gains anything. Each is one option, so a first hour with one variant per GPU would answer them on 50,000 images each.

### Several GPUs

`parallel.py` runs `cond.py` on more than one GPU, on Linux:

```sh
uv run --no-sync python -m finetune.parallel --devices 0,1,2,3 finetune.cond --hours 12 --wandb
```

It starts one process per GPU, each seeing only its own, and stops them all when one fails, so a crash never leaves the rest waiting. The processes hold the same weights at every step: each draws `--batch` images of its own, and every backward pass leaves each of them the mean gradient (PyTorch's `DistributedDataParallel`, over NCCL). So a step is `--batch` times the number of GPUs, and the first process alone logs, checks the probes, draws the samples, saves, and says when the time is up. The others wait while it does. Each process takes its own batch of every round from the one shuffled stream, so together they draw what one process would. A run resumes only with the number of GPUs it was started with.

What was checked, on 2026-11-17, in WSL on the RTX 4070 with the locked Linux environment (PyTorch 2.9.1+cu128, NCCL 2.27.5):

- **Two processes end with the weights one gets from both their images.** `tests/unit/test_finetune.py`, on the small model on the CPU: after three steps the two hold the same weights and running averages bit for bit, and they match one process that took each step on all four images. The worker makes its calls in `cond.py`'s order. The same on the GPU, two processes over Gloo: identical weights.
- **One process failing stops the rest**, with its exit code.
- **NCCL with one process**, on the small model: connecting, sharing the model, the decision and the mean record all work on CUDA tensors.
- **The real model through the launcher**, one process on the CPU: 6 steps with probes, samples and the checkpoint, then 4 more resumed from it. Resuming it with two processes is refused with a message. On the GPU it runs out of memory at the first step, for the reason below.
- **Not run: two or more processes over NCCL.** NCCL refuses two processes on one GPU, and this machine has one. Not run either: two processes of the real model, which need more memory than this machine has (28 GB for one on the CPU, and sharing the gradients adds a copy of them, 1.7 GB, which no longer fits 12 GB). `finetune/vast/train.sh` with `SMOKE=1` is that check, on the rented GPUs, before the run: 40 steps with samples, then 20 more resumed.

On the rented machine, four RTX 5090s on 2026-11-17 (vast.ai, PyTorch 2.9.1+cu128, NCCL 2.27.5, Python 3.12):

- **Four processes over NCCL end with identical weights** and running averages, with the small model of the tests, and within 0.019 of four processes on the CPU, which is as close as one GPU comes to the CPU.
- **Memory:** 9.3 GB for the weights, gradients, AdamW state and running average, and 3.0 GB an image, so 6 images fit a 32 GB card for training alone (27.4 GB) and `train.sh` takes 5 to leave room for the samples. The 4070's 2.4 GB an image was too low: 7 images ran out of memory.
- **Speed**, on random images, per second: one GPU 7.0 at 1 image a batch, 12.9 at 6. Four GPUs at 6 each: 25.9, twice one GPU and not four times. The cards have no direct link (`nvidia-smi topo -p2p r` says `CNS`), so NCCL sends the 1.7 GB of gradients through main memory every step, which took as long as computing them. Sending them as bfloat16 made it 35.5, and averaging only every second batch (`--accumulate 2`, `parallel.quiet()`) as well, 36.9 at 4 images a batch, 32 a step, and 40.1 at 5, 40 a step. The run takes 4 and 2: 1.6M images in 12 hours.

PyTorch's builds for Windows cannot connect processes at all (`unsupported gloo device`), so the two tests that start processes are skipped there.

One bug this found before it cost anything: the first process built and shared the model while the others were still waiting for it to finish downloading, which leaves them answering different calls. Nothing shared between processes is now inside that wait.

### On vast.ai

`finetune/vast/` has what a rented instance needs. From this machine, in Git Bash, with the address and port `vastai ssh-url <instance>` prints:

```sh
bash finetune/vast/upload.sh <host> <port>   # The code, the pretrained models and both datasets, by rsync. Run again to resume.
bash finetune/vast/pull.sh <host> <port>     # While it trains: copy the run back every 20 minutes
```

and on the instance:

```sh
bash /workspace/code/finetune/vast/setup.sh            # uv, the locked environment, and a check of the GPUs and the data
SMOKE=1 bash /workspace/code/finetune/vast/train.sh    # 60 steps on every GPU, with samples and a resume
HOURS=12 bash /workspace/code/finetune/vast/train.sh   # The run, in tmux. It resumes itself after an error
```

The learning rate stays constant after its warmup, so the run's measures level off with a jitter. To let the weights settle, resume with a batch to stop at and how many batches before it the rate falls to zero over, and enough `HOURS` to get there:

```sh
HOURS=16 EXTRA="--max-steps 120000 --decay 24000" bash finetune/vast/status.sh <host> <port> start
```

On four RTX 5090s a batch is 16 images at 36 a second, about 8,100 batches an hour, so 24,000 is about three hours.

`train.sh` uses every GPU the instance has, or those in `GPUS`, and a step is 32 images however many there are (`TOTAL`), as many on each as their memory allows: 4 each on eight GPUs, 4 each twice on four 32 GB cards, 32 on one large one. `BATCH` sets the images on each directly. The other processes wait up to an hour for the first one to draw its samples and save, where NCCL would give up after ten minutes. `bash finetune/vast/status.sh <host> <port>` from this machine prints the GPUs' load, the disk, and the end of the run's log, and with `start` or `smoke` as a third word starts the run or the smoke run in a tmux session named `run`.

A checkpoint with the running average is 6.7 GB. An instance that is destroyed takes its disk with it, so `pull.sh` is the only copy.

### The rented run

On 2026-11-17, on the four RTX 5090s above, with `cond.py`'s defaults: 4 images on each GPU, twice per update, so 32 an update, at 36 images a second, with no restarts. 12 hours at the constant learning rate reached batch 97,200 (1.56M images, 0.8 of a pass over the 1.92M), then a resume with `--max-steps 117200 --decay 20000` brought the rate to zero over another 2.5 hours. 1.88M images in all, about $72 of rented time. The log is `reports/evidence/finetune/vast_log.jsonl`, and the checkpoints `D:/cgd-finetune/cond256-vast/adapter.pt` (the end) and `adapter_12h.pt` (before the decay), with hourly ones in `snapshots/`.

- **The condition is learnt.** The share of the high-noise loss the image's embedding removes, on the held-out probes, rose from 7% at batch 6,000 to about 10.5% by 70,000 and then held, between 9.6% and 11.9%. The text's was 3 to 4%, both together 9 to 10%. The decay raised the image's and both by about a point, to 11.4% and 10.2% at the end, and the readings jitter less.
- **The training samples.** The judge's score against the real images, with classifier-free guidance at 3 from three conditions and no CLIP: both embeddings 0.72 to 0.74 from batch 48,000 on, the image's 0.70 to 0.73, the text's 0.65 to 0.68, the best of the run at the end. `cgd` as it is, CLIP-guided, scores 0.709 on the same 8 texts. `vast_samples_117200.jpg` is the last grid.
- **What it costs the class alone.** `class now` went from 0.600 to about 0.57 against the real images by batch 32,000 and held there. `probe/class_clean` fell from 0.1423 to 0.1340 and `probe/class` stayed at 0.0009, so the detail of the cleaner levels was not lost.

CLIP guidance on the end checkpoint, `reports/evidence/finetune/diag_cond_guided.py D:/cgd-finetune/cond256-vast 1 32 30 weights` and the same with `average`: 32 held-out texts the run never drew, each from the same noise in 30 steps, judged by SigLIP2 (`vast_guided.txt`):

| Row | Against the texts | Against the real images | Better than `cgd` for | Seconds an image |
|---|---|---|---|---|
| `cgd` as it is (`class guided`) | 0.140 | 0.662 | | 6.7 |
| The text's embeddings, no CLIP | 0.082 | 0.628 | 4 and 11 of 32 | 7.3 |
| The same, CLIP-guided | 0.179 (+0.039 ± 0.008) | 0.707 (+0.045 ± 0.015) | 24 and 23 | 10.2 |
| With the prior's image embedding, no CLIP | 0.098 | 0.638 | 7 and 15 | 7.3 |
| The same, CLIP-guided | 0.182 (+0.042 ± 0.009) | 0.725 (+0.063 ± 0.014) | 25 and 28 | 10.2 |

- **Under CLIP guidance the fine-tuned model beats `cgd`**, by 4 to 5 standard errors on the texts and 3 to 4 on the images, and by more with the prior's image embedding. It draws what the text says beyond its class: the men beside the cannon, the pancakes under the figs, the gazebo under the mountains, the white car and the blueberry muffins, where `cgd` draws the cannon, figs, a red pavilion, a hedge and blue blobs (`vast_guided_32texts_weights.jpg`, rows 2, 4 and 6). It costs half as long again, for the second forward pass of classifier-free guidance.
- **The weights beat their running average.** With the running average the same two guided rows score 0.171 and 0.175 against the texts, 0.693 and 0.700 against the images. With the rate at zero at the end the weights have settled, and the average still holds weights from before the decay.
- **Without CLIP it does not replace `cgd`.** It scores 0.08 to 0.10 against the texts where `cgd` scores 0.14: its images are photographic and well composed, but the text's details are often missing.
- **The guided rows are as harsh in contrast and colour as `cgd`'s.** That is the guidance, at `cgd`'s rate, and not the fine-tune.

32 texts, one judge, one noise each. Not run: other guidance rates or classifier-free guidance scales, and the 12-hour checkpoint against the end one.

### In cgd

`cgd -ft` draws with it. `python -m finetune.export D:/cgd-finetune/cond256-vast` wrote `cgd.pt` beside the run, where `-ft` looks for it by default (`-ftpath` for another): the configuration, the end weights, the embeddings' statistics and `prior.py`'s maps, 1.7 GB where the run's checkpoint is 6.7. `cgd/finetuned.py` reads it, and the model's code moved to `cgd/conditioned.py`, since the wheel holds only `cgd`. It tells the model what the row with the prior above was told, from the command's prompts: each model's mean embedding of the text prompts of positive weight, the image prompts seen whole or the prior's prediction from the text, and the class closest to the text by ViT-B/32.

`reports/evidence/finetune/check_cgd_finetuned.py` drew 16 other held-out texts through `cgd`'s own `generate()`, each from its own seed, against `cgd -size 256` with the same guide. SigLIP2 scored `-ft` at 0.163 against 0.120 on the texts, +0.044 ± 0.014, and 0.678 against 0.619 on the real images, +0.059 ± 0.018, ahead on 12 of 16 for both (`vast_check_cgd_finetuned.txt` and `.jpg`). That is what the evaluation above measured with the run's stored embeddings, so the command tells it what it was trained on. 13.0 s an image against 8.6 on the RTX 4070, and 6.3 GiB at the peak against 5.9: the U-Net's convolutions are kept at half precision, which autocast runs them at anyway and which made the same image as full precision, byte for byte, for 1.9 GiB less. A seed makes the same image every run.


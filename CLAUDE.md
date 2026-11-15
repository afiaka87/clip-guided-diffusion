# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CLIP Guided Diffusion (cgd) is a text-to-image generation tool that uses a CLIP model (OpenAI's by default, or a PE-Core or SigLIP2 model through open_clip) to guide a diffusion process. Originally by Katherine Crowson (@crowsonkb).

## Installation & Setup

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```sh
git clone https://github.com/afiaka87/clip-guided-diffusion.git
cd clip-guided-diffusion
uv sync
```

## Running Tests

```sh
uv run pytest tests/unit  # Pure tests on the CPU, in seconds.
uv run pytest             # Everything, in about three and a half minutes on an RTX 4070.
uv run pytest tests/integration/test_adm.py::test_a_different_seed_changes_the_image  # A single test.
```

- `tests/unit` holds pure tests on small inputs and stand-ins built from real modules, such as Gaussian data whose exact denoiser is known, a linear projection with CLIP's interface, or a small U-Net of ADM's kind with random weights. No mocks.
- `tests/integration` is marked `cuda` and runs the real models and checkpoints. `tests/conftest.py` skips it without a GPU. Give it a free GPU: if another process holds the VRAM it fails with a real out of memory error.
- Keep `tests/integration` to what only the real models can show, each test a contrived run of a few steps: every checkpoint and backbone loads and draws, a seed makes the same image again, an image of a sequence is the one made alone, and the command saves what it says. All of it takes about three minutes, and should stay under four. What a part computes belongs in `tests/unit`, and how good an image is in a report.
- Loading is most of what these tests cost. Only one CLIP model, one ImageNet checkpoint and one thumbnail model stay loaded, and every backbone run loads its backbone from disk, so tests of one model follow each other and an ensemble is used only where it is the point.
- A run too short makes an image that shows nothing. Below four steps the unconditional 256 checkpoint clips nearly every pixel to black or white, and such images are the same whatever made them. Reproducibility of the ImageNet models shows at 128 over six steps, not at 64.
- `tests/integration/test_cli.py` runs the command in this process with `main([...])`, and in a new one where the process matters: `python -m cgd`, and the allocator setting `import cgd` makes.
- Lint with `uv run ruff check`, `uv run ruff format --check` and `uv run pylint cgd tests`. All three are clean, and should stay so.

## CLI Usage

Basic usage: `uv run cgd "Your prompt here"`. `uv run cgd -h` groups the options by pipeline, and an option for the other pipeline is an error.

- `-size`: Image size (64, 128, 256, 512, only 256 or 512 with `-uncond`, or any multiple of 8 with `-bb`)
- `-clip`: CLIP model (default ViT-B/32, or ViT-B/32+PE-Core-B-16 with `-uncond` or `-bb`). `clip_models.MODEL_NAMES` lists them, and the README has the VRAM and time of each. `-clip A+B` guides with an ensemble.
- `-fp16`: Run an open_clip model's vision tower at half precision. Faster and half the CLIP VRAM. Less exact, but scored about the same.
- `-cutn`: Random cutouts CLIP sees each step, both pipelines (default 16, `guide.NUM_CUTOUTS`, or 4 with `-uncond`). 4 scored the same for a quarter of CLIP's work, except with `-thumb glide` and `-pag`, which do better at 16.
- `-aes`: Also reward what people rate highly, over cutouts (default 0, or 0.05 with `-bb`).
- `-best N`: Generate N candidates and keep the one held-out CLIP models prefer.
- A sequence: `"a castle|>a fox|>a sailboat"`. A prompt that starts with `>` begins the next image, each image is the one its prompts make alone, and the GIF or video shows the first being made and then slides from each to the next. `-frames` is how many frames a slide takes (default 30, or 60 with `-bb`). Both pipelines, `-imgs` too. It saves a GIF unless it asks for `-mp4`.
- ImageNet model only: `-uncond` an unconditional checkpoint (256 by default, or 512) with no ImageNet class, and the defaults that made its best images for the time they take (`adm.UNCONDITIONAL_DEFAULTS`: `-clip ViT-B/32+PE-Core-B-16 -cutn 4 -n 60 -recur 2 -thumb clip`), each overridden by its option. `-thumb` the model that sets the composition by reading the noisy image as a 64x64 thumbnail (`clip`, `glide`, `laionide`, `karlo` or `none`, only with `-uncond`), `-pag` perturbed-attention guidance (only with `-uncond` at 256), `-n` steps down the schedule (default 30), `-recur` how many times each early guided step runs (default 1), `-rate` guidance strength (default 0.08), `-gl` guidance limit, `-init` init image, `-skip` how much of the schedule it skips (default half), `-is` LPIPS init scale.
- `-bb`: Draw with a modern one-step model and correct it with CLIP, instead of guiding an ImageNet model (`sd-turbo`, `vega`, `sdxl-lightning`). `backbone.BACKBONES` lists them. The best images this repo makes. `-iters` is its only option of its own.
- Output: `-dir`, `-freq`, `-gif`, `-mp4`, `-up` (a 4x larger copy made with Real-ESRGAN) and `-q`.

The README's "Tried and removed" section has the options that tied or lost, and why they went. The CLIP and TV scales, the cutout layout and the backbone's learning rate are constants, at the values everything was tuned with. Every GIF and number in the README was made at `-cutn 4` unless it says 16, so its commands carry the count and still reproduce their GIFs byte for byte. `-uncond` has 4 as its default.

## Architecture

Records are NamedTuples or frozen dataclasses, and the modules are functions over them. The only classes with behaviour are PyTorch modules, and the only state is `lru_cache` on the model loaders.

### Modules

- **`cgd/cli.py`**: The `cgd` command, also `python -m cgd`. `pipeline()` picks `adm.generate` or `backbone.generate` and its settings from the options, `judged_candidates()` is `-best`, `watched()` gives a sequence its GIF, and `save_outputs()` upscales and makes the GIF or video, and only deletes frames once every requested output exists.

- **`cgd/__init__.py`**: Turns on PyTorch's expandable segments unless the user configured the allocator, or on Windows, where PyTorch has none and would only warn. It sets both names of that variable on purpose, see its comment: renaming it to the one PyTorch's warning asks for turns the setting off.

- **`cgd/adm.py`**: The default pipeline. `generate()` loads OpenAI's ImageNet U-Net (`load_model()`, cached), conditions it on the ImageNet class closest to the prompts (`closest_class()`), or on nothing with an unconditional checkpoint (`UNCONDITIONAL_URLS`), and runs `sampler.sample()` with `guidance_loss()`: CLIP plus TV and range losses, and LPIPS to the init image with `-is`. The options a run leaves unset come from `DEFAULTS`, or `UNCONDITIONAL_DEFAULTS` for an unconditional checkpoint, and the U-Net's noise prediction passes through `pag.denoise()` and `thumbnails.guided_denoise()` when those are on. Yields `(batch_index, path)` of each saved frame, the result last. `sampled_frames()` makes the frames: the first image's model calls, then for each next image of a sequence the frames `between()` it and the one before.

- **`cgd/sampler.py`**: The fast sampler, DPM-Solver++(2M) SDE in uniform logSNR steps with capped guidance. Its docstring explains why each piece exists. `recurring_step()` is `-recur`: it takes a guided step in the first 70% of the schedule (`recurrences()`) several times, with `renoised()` taking the image back up to the step's noise level in between. Also `limit_guidance()` for `-gl`, and for sequences `blend()`, a prediction between two denoisers', and `redraw()`, an image noised partway and denoised again without guidance or fresh noise. Pure functions of tensors and a schedule, so `tests/unit/test_sampler.py` checks them exactly.

- **`cgd/thumbnails.py`**: `-thumb`. A model of noisy 64x64 images reads the noisy image averaged down to a thumbnail, and what it says the text prompts change about the noise is added to the U-Net's prediction (`guided_denoise()`). Its docstring has the derivation. `load()` gives one run's guidance from OpenAI's CLIP of noisy images, GLIDE, LAIONIDE or Karlo's decoder, each at the scale it was measured with (`SCALES`), and `load_model()` keeps only the last one loaded. `tests/unit/test_thumbnails.py` checks it against Gaussian images whose exact denoisers are known.

- **`cgd/pag.py`**: `-pag`. `denoise()` moves the U-Net's prediction away from one made with the self-attention of `LAYERS` replaced by the identity (`perturbed_attention()`), the layers and scale of the PAG paper for the unconditional 256 checkpoint. The second call takes no gradient.

- **`cgd/slide.py`**: What the two pipelines share of a sequence's transitions, and its docstring has what was tried and why it works this way. A frame is a function of its position between two finished images, never of the frame before. `place()` draws each frame halfway between the two neighbours that differ most, so the frames go where the image changes. `fade()` and `noise_timestep()` are what a position means to the ImageNet model, and `slerp()` moves a backbone's noise.

- **`cgd/backbone.py`**: `-bb`. `optimise_noise()` follows the CLIP gradient on a one-step model's initial noise, through the model and a tiny decoder (ReNO), and `generate()` is the counterpart of `adm.generate()`. `corrected_frames()` makes one noise's frames, and `between()` is the frame between two images of a sequence: the model's image of the noise and text embedding partway between theirs. Runs under cuDNN's deterministic flag, because the U-Net's backward pass is nondeterministic without it below about 448px.

- **`cgd/guide.py`**: The single place guidance's loss comes from. `load_guide()` makes a `Guide` of one `Tower` per CLIP model plus the optional aesthetic term, `load_guides()` one per image of a sequence from the same models, and `guide_loss()` is the loss. An ensemble backpropagates each term on its own, so only one tower's activations are in memory at a time, and joins them with `losses.with_gradient()`. `score()` is what `-best`'s judges use.

- **`cgd/clip_models.py`**: CLIP model loading (`load_clip()`, cached), text and image prompt encoding, preprocessing (`preprocess()`, which uses each model's own normalization) and the closest ImageNet class (`closest_imagenet_class()`, text features cached on disk). `OPEN_CLIP_MODELS` holds the open_clip models, lightest first. Those keep their text tower on the CPU and checkpoint their vision tower, so only what guidance needs takes VRAM. Their vision tower also uses plain attention instead of the fused kernels, whose backward pass is nondeterministic and makes a seed produce a different image every run. Their weights are cached at `~/.cache/clip-guided-diffusion/clip/`.

- **`cgd/cutouts.py`**: `make_cutouts()`, random square crops at CLIP's input size, and `adaptive_avg_pool()`. Keep that instead of `tf.adaptive_avg_pool2d`: the latter's CUDA backward pass is nondeterministic, which makes a seed produce a different image every run.

- **`cgd/losses.py`**: `spherical_dist_loss()` for CLIP similarity, `tv_loss()` for smoothness, `range_loss()` for RGB clamping, `finite_gradient()`, and `with_gradient()`, a loss whose gradient was already taken on its own.

- **`cgd/checks.py`**: `ConfigError` and `require()`. `adm.check_parameters()` and `backbone.check_parameters()` reject a configuration before anything slow runs, and the command turns `ConfigError` into a usage error.

- **`cgd/prompts.py`**: `parse_prompt()` and `split_prompts()`, for `text:weight` and `|`, and `scenes()`, the prompts of each image of a sequence. The pipelines take a sequence as a flat list of prompts in which `>a fox` begins the next image, which is what `|>` splits into.

- **`cgd/outputs.py`**: Where frames go (`run_name()`, `frame_path()`, `save_frame()`, `save_batch()`), and `save_gif()` and `save_video()` through ffmpeg.

- **`cgd/download.py`**: Downloads into `~/.cache/clip-guided-diffusion/` with retries, and `fetch()` for image paths or URLs.

- **`cgd/upscale.py`**: Real-ESRGAN x4plus, tiled, for `-up`.

- **`cgd/device.py`** and **`cgd/imagenet_classes.py`**: The default device, `limit_to_vram()` and the random streams (`forked_rng()`, `replayed_rng()`), and the 1000 class names. On Windows NVIDIA's driver backs a CUDA allocation that does not fit in VRAM with system memory instead of failing it, which is many times slower and can take all of it, so both pipelines cap PyTorch at what the GPU has free. Running out then raises, as elsewhere.

### External Dependency

The `guided-diffusion` package (git dependency from crowsonkb) provides only the U-Net and its noise schedule. The flags of each checkpoint are `adm.CHECKPOINT_FLAGS`.

`open-clip-torch` provides the PE-Core and SigLIP2 models. It is imported lazily, because it is slow to import and the default model does not need it. `transformers` and `sentencepiece` are only there for SigLIP2's tokenizer.

`clip-anytorch` imports `pkg_resources`, which setuptools 81 removed, so `setuptools<81` is a dependency. It also reads OpenAI's weights with `torch.jit.load`, which PyTorch warns about on Python 3.14, and `clip_models._load_openai()` silences that.

On Windows `torch` and `torchvision` come from PyTorch's CUDA 13.2 index, since PyPI's Windows wheels are CPU-only. `uv lock --upgrade` moves everything to the latest. A new PyTorch makes a different image from the same seed, so after one the README's GIFs are no longer what their commands make: they were made with PyTorch 2.9.1.

`diffusers`, `accelerate` and `peft` provide the backbones, imported lazily for the same reason. The Hugging Face cache holds their weights, not `~/.cache/clip-guided-diffusion/`.

`glide-text2im` (git dependency from OpenAI) provides GLIDE's 64x64 model and the CLIP of noisy images for `-thumb`, imported lazily. Their weights and LAIONIDE's go to `~/.cache/clip-guided-diffusion/glide/` through `download()`. Karlo's parts come from `diffusers` and `transformers`, into the Hugging Face cache.

### Determinism

The same seed and options give the same image byte for byte, and the README's GIFs depend on it. Every backward pass on the GPU must sum in a fixed order: the hand-written pooling in `cutouts.py`, plain attention in open_clip and the backbones, cuDNN's deterministic flag for both pipelines' U-Nets and for LPIPS's VGG16 (`adm.perceptual_init_loss()`). The ImageNet U-Net needs it at 128 and 256 since PyTorch 2.15, where three runs of one seed made three images without it. `tests/integration/test_determinism.py` and the seed tests in `test_adm.py` and `test_backbone.py` check it. Random numbers are drawn in a fixed order, so moving a `th.randn` changes every image. Loading a thumbnail model must not move that order: each is built with the random stream forked, and Karlo's prior draws its image embedding from a generator of its own, seeded from the run's seed.

Every image of a sequence is byte for byte the image its prompts make alone, and the tests check it. Each starts from the same noise and replays the random stream from there (`device.replayed_rng()`), later images' image prompts are encoded with the stream forked, and the frames between images draw no random numbers at all.

### Prompt Weights

Prompts support weights via `text:weight` syntax (e.g., `"cat:1.0|dog:-0.5"`). Weights are normalized to sum to non-zero.

## Git Commits

When tracking changes with git, NEVER (IMPORTANT!) include attribution to Claude Code or Anthropic. Use 1-2 sentence, per-file commits.

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
uv run pytest             # Everything.
uv run pytest tests/integration/test_adm.py::test_it_draws_the_prompt_without_blowing_out  # A single test.
```

- `tests/unit` holds pure tests on small inputs and stand-ins built from real modules, such as Gaussian data whose exact denoiser is known, or a linear projection with CLIP's interface. No mocks.
- `tests/integration` is marked `cuda` and runs the real models and checkpoints. `tests/conftest.py` skips it without a GPU. Give it a free GPU: if another process holds the VRAM it fails with a real out of memory error.
- `tests/integration/test_cli.py` runs the command in this process with `main([...])`, and in a new one where the process matters: `python -m cgd`, and the allocator setting `import cgd` makes.
- Lint with `uv run ruff check`, `uv run ruff format --check` and `uv run pylint cgd tests`. All three are clean, and should stay so.

## CLI Usage

Basic usage: `uv run cgd "Your prompt here"`. `uv run cgd -h` groups the options by pipeline, and an option for the other pipeline is an error.

- `-size`: Image size (64, 128, 256, 512, only 256 or 512 with `-uncond`, or any multiple of 8 with `-bb`)
- `-clip`: CLIP model (default ViT-B/32, or ViT-B/32+PE-Core-B-16 with `-uncond` or `-bb`). `clip_models.MODEL_NAMES` lists them, and the README has the VRAM and time of each. `-clip A+B` guides with an ensemble.
- `-fp16`: Run an open_clip model's vision tower at half precision. Faster and half the CLIP VRAM. Less exact, but scored about the same.
- `-cutn`: Random cutouts CLIP sees each step, both pipelines (default 16, `guide.NUM_CUTOUTS`, or 4 with `-uncond`). 4 scored the same for a quarter of CLIP's work.
- `-aes`: Also reward what people rate highly, over cutouts (default 0, or 0.05 with `-bb`).
- `-best N`: Generate N candidates and keep the one held-out CLIP models prefer.
- ImageNet model only: `-uncond` an unconditional checkpoint (256 by default, or 512) with no ImageNet class, and the defaults that made its best images for the time they take (`adm.UNCONDITIONAL_DEFAULTS`: `-clip ViT-B/32+PE-Core-B-16 -cutn 4 -n 60 -recur 2 -thumb clip`), each overridden by its option. `-thumb` the model that sets the composition by reading the noisy image as a 64x64 thumbnail (`clip`, `glide`, `laionide`, `karlo` or `none`, only with `-uncond`), `-pag` perturbed-attention guidance (only with `-uncond` at 256), `-n` steps down the schedule (default 30), `-recur` how many times each early guided step runs (default 1), `-rate` guidance strength (default 0.08), `-gl` guidance limit, `-init` init image, `-skip` how much of the schedule it skips (default half), `-is` LPIPS init scale.
- `-bb`: Draw with a modern one-step model and correct it with CLIP, instead of guiding an ImageNet model (`sd-turbo`, `vega`, `sdxl-lightning`). `backbone.BACKBONES` lists them. The best images this repo makes. `-iters` is its only option of its own.
- Output: `-dir`, `-freq`, `-gif`, `-mp4`, `-up` (a 4x larger copy made with Real-ESRGAN) and `-q`.

The README's "Tried and removed" section has the options that tied or lost, and why they went. The CLIP and TV scales, the cutout layout and the backbone's learning rate are constants, at the values everything was tuned with. Every GIF and number in the README was made at `-cutn 4`, so its commands carry it and still reproduce their GIFs byte for byte. `-uncond` has it as its default.

## Architecture

Records are NamedTuples or frozen dataclasses, and the modules are functions over them. The only classes with behaviour are PyTorch modules, and the only state is `lru_cache` on the model loaders.

### Modules

- **`cgd/cli.py`**: The `cgd` command, also `python -m cgd`. `pipeline()` picks `adm.generate` or `backbone.generate` and its settings from the options, `judged_candidates()` is `-best`, and `save_outputs()` upscales and makes the GIF or video, and only deletes frames once every requested output exists.

- **`cgd/__init__.py`**: Turns on PyTorch's expandable segments unless the user configured the allocator. It sets both names of that variable on purpose, see its comment: renaming it to the one PyTorch's warning asks for turns the setting off.

- **`cgd/adm.py`**: The default pipeline. `generate()` loads OpenAI's ImageNet U-Net (`load_model()`, cached), conditions it on the ImageNet class closest to the prompts (`closest_class()`), or on nothing with an unconditional checkpoint (`UNCONDITIONAL_URLS`), and runs `sampler.sample()` with `guidance_loss()`: CLIP plus TV and range losses, and LPIPS to the init image with `-is`. The options a run leaves unset come from `DEFAULTS`, or `UNCONDITIONAL_DEFAULTS` for an unconditional checkpoint, and the U-Net's noise prediction passes through `pag.denoise()` and `thumbnails.guided_denoise()` when those are on. Yields `(batch_index, path)` of each saved frame, the result last.

- **`cgd/sampler.py`**: The fast sampler, DPM-Solver++(2M) SDE in uniform logSNR steps with capped guidance. Its docstring explains why each piece exists. `recurring_step()` is `-recur`: it takes a guided step in the first 70% of the schedule (`recurrences()`) several times, with `renoised()` taking the image back up to the step's noise level in between. Also `limit_guidance()` for `-gl`. Pure functions of tensors and a schedule, so `tests/unit/test_sampler.py` checks them exactly.

- **`cgd/thumbnails.py`**: `-thumb`. A model of noisy 64x64 images reads the noisy image averaged down to a thumbnail, and what it says the text prompts change about the noise is added to the U-Net's prediction (`guided_denoise()`). Its docstring has the derivation. `load()` gives one run's guidance from OpenAI's CLIP of noisy images, GLIDE, LAIONIDE or Karlo's decoder, each at the scale it was measured with (`SCALES`), and `load_model()` keeps only the last one loaded. `tests/unit/test_thumbnails.py` checks it against Gaussian images whose exact denoisers are known.

- **`cgd/pag.py`**: `-pag`. `denoise()` moves the U-Net's prediction away from one made with the self-attention of `LAYERS` replaced by the identity (`perturbed_attention()`), the layers and scale of the PAG paper for the unconditional 256 checkpoint. The second call takes no gradient.

- **`cgd/backbone.py`**: `-bb`. `optimise_noise()` follows the CLIP gradient on a one-step model's initial noise, through the model and a tiny decoder (ReNO), and `generate()` is the counterpart of `adm.generate()`. Runs under cuDNN's deterministic flag, because the U-Net's backward pass is nondeterministic without it below about 448px.

- **`cgd/guide.py`**: The single place guidance's loss comes from. `load_guide()` makes a `Guide` of one `Tower` per CLIP model plus the optional aesthetic term, and `guide_loss()` is the loss. An ensemble backpropagates each term on its own, so only one tower's activations are in memory at a time, and joins them with `losses.with_gradient()`. `score()` is what `-best`'s judges use.

- **`cgd/clip_models.py`**: CLIP model loading (`load_clip()`, cached), text and image prompt encoding, preprocessing (`preprocess()`, which uses each model's own normalization) and the closest ImageNet class (`closest_imagenet_class()`, text features cached on disk). `OPEN_CLIP_MODELS` holds the open_clip models, lightest first. Those keep their text tower on the CPU and checkpoint their vision tower, so only what guidance needs takes VRAM. Their vision tower also uses plain attention instead of the fused kernels, whose backward pass is nondeterministic and makes a seed produce a different image every run. Their weights are cached at `~/.cache/clip-guided-diffusion/clip/`.

- **`cgd/cutouts.py`**: `make_cutouts()`, random square crops at CLIP's input size, and `adaptive_avg_pool()`. Keep that instead of `tf.adaptive_avg_pool2d`: the latter's CUDA backward pass is nondeterministic, which makes a seed produce a different image every run.

- **`cgd/losses.py`**: `spherical_dist_loss()` for CLIP similarity, `tv_loss()` for smoothness, `range_loss()` for RGB clamping, `finite_gradient()`, and `with_gradient()`, a loss whose gradient was already taken on its own.

- **`cgd/checks.py`**: `ConfigError` and `require()`. `adm.check_parameters()` and `backbone.check_parameters()` reject a configuration before anything slow runs, and the command turns `ConfigError` into a usage error.

- **`cgd/prompts.py`**: `parse_prompt()` and `split_prompts()`, for `text:weight` and `|`.

- **`cgd/outputs.py`**: Where frames go (`run_name()`, `frame_path()`, `save_frame()`), and `save_gif()` and `save_video()` through ffmpeg.

- **`cgd/download.py`**: Downloads into `~/.cache/clip-guided-diffusion/` with retries, and `fetch()` for image paths or URLs.

- **`cgd/upscale.py`**: Real-ESRGAN x4plus, tiled, for `-up`.

- **`cgd/device.py`** and **`cgd/imagenet_classes.py`**: The default device, and the 1000 class names.

### External Dependency

The `guided-diffusion` package (git dependency from crowsonkb) provides only the U-Net and its noise schedule. The flags of each checkpoint are `adm.CHECKPOINT_FLAGS`.

`open-clip-torch` provides the PE-Core and SigLIP2 models. It is imported lazily, because it is slow to import and the default model does not need it. `transformers` and `sentencepiece` are only there for SigLIP2's tokenizer.

`diffusers`, `accelerate` and `peft` provide the backbones, imported lazily for the same reason. The Hugging Face cache holds their weights, not `~/.cache/clip-guided-diffusion/`.

`glide-text2im` (git dependency from OpenAI) provides GLIDE's 64x64 model and the CLIP of noisy images for `-thumb`, imported lazily. Their weights and LAIONIDE's go to `~/.cache/clip-guided-diffusion/glide/` through `download()`. Karlo's parts come from `diffusers` and `transformers`, into the Hugging Face cache.

### Determinism

The same seed and options give the same image byte for byte, and the README's GIFs depend on it. Every backward pass on the GPU must sum in a fixed order: the hand-written pooling in `cutouts.py`, plain attention in open_clip and the backbones, cuDNN's deterministic flag for the backbones' U-Net and for LPIPS's VGG16 (`adm.perceptual_init_loss()`). `tests/integration/test_determinism.py` and the seed tests in `test_adm.py` and `test_backbone.py` check it. Random numbers are drawn in a fixed order, so moving a `th.randn` changes every image. Loading a thumbnail model must not move that order: each is built with the random stream forked, and Karlo's prior draws its image embedding from a generator of its own, seeded from the run's seed.

### Prompt Weights

Prompts support weights via `text:weight` syntax (e.g., `"cat:1.0|dog:-0.5"`). Weights are normalized to sum to non-zero.

## Git Commits

When tracking changes with git, NEVER (IMPORTANT!) include attribution to Claude Code or Anthropic. Use 1-2 sentence, per-file commits.

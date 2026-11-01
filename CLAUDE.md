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
uv run pytest
uv run pytest test.py::TestCGD::test_cgd_one_step_succeeds  # single test
```

Note: Most tests need a CUDA GPU and the cached checkpoints, and run against the real models. Give them a free GPU: if another process holds the VRAM they fail with a real out of memory error.

## CLI Usage

Basic usage: `uv run cgd "Your prompt here"`

Key parameters:
- `-size`: Image size (64, 128, 256, 512)
- `-n`: Model calls for the fast sampler (default 30, or 60 with `-uncond`)
- `-rate`: Fast sampler guidance strength (default 0.08)
- `-clip`: CLIP model (default ViT-B/32). `clip_util.CLIP_MODEL_NAMES` lists them; the README has the VRAM and time of each.
- `-fp16`: Run an open_clip model's vision tower at half precision. Faster and half the CLIP VRAM. Less exact, but scored about the same.
- `-cgs`: CLIP guidance scale (default 1000)
- `-tvs`: TV loss scale for smoothness (default 150)
- `-init`: Initial image path
- `-skip`: How much of the schedule an init image skips (default half)
- `-respace`: Use the original guided-diffusion sampler instead, e.g. 1000 or ddim50. Much slower.

## Architecture

### Core Components

- **`cgd/cgd.py`**: Main entry point with `clip_guided_diffusion()` generator function and CLI `main()`. The generator yields `(batch_idx, image_path)` tuples during diffusion.

- **`cgd/__init__.py`**: Turns on PyTorch's expandable segments unless the user configured the allocator. It sets both names of that variable on purpose, see its comment: renaming it to the one PyTorch's warning asks for turns the setting off.

- **`cgd/sampling.py`**: The default fast sampler, DPM-Solver++(2M) SDE in uniform logSNR steps with capped guidance. Its docstring explains why each piece exists.

- **`cgd/script_util.py`**: Utilities for validating a configuration (`check_parameters()`, raising `ConfigError`), downloading checkpoints, loading the guided diffusion model, parsing prompts with weights, and logging images. Model checkpoints cached at `~/.cache/clip-guided-diffusion/`.

- **`cgd/clip_util.py`**: CLIP model loading (`load_clip()`), text/image prompt encoding, preprocessing (`preprocess()`, which uses each model's own normalization) and ImageNet class selection (`imagenet_top_n()`, text features cached on disk). `OPEN_CLIP_MODELS` holds the open_clip models, lightest first. Those keep their text tower on the CPU and checkpoint their vision tower, so only what guidance needs takes VRAM. Their vision tower also uses plain attention instead of the fused kernels, whose backward pass is nondeterministic and makes a seed produce a different image every run. Their weights are cached at `~/.cache/clip-guided-diffusion/clip/`.

- **`cgd/losses.py`**: Loss functions - `spherical_dist_loss()` for CLIP similarity, `tv_loss()` for smoothness, `range_loss()` for RGB clamping.

- **`cgd/modules.py`**: `MakeCutouts` module for random patch extraction with optional augmentations, and `adaptive_avg_pool()`. Keep that instead of `tf.adaptive_avg_pool2d`: the latter's CUDA backward pass is nondeterministic, which makes a seed produce a different image every run.

### External Dependency

The `guided-diffusion` package (git dependency from crowsonkb) provides the diffusion model architecture and sampling loops. Config flags for different model sizes are in `data/diffusion_model_flags.py`.

`open-clip-torch` provides the PE-Core and SigLIP2 models. It is imported lazily, because it is slow to import and the default model does not need it. `transformers` and `sentencepiece` are only there for SigLIP2's tokenizer.

### Diffusion Flow

1. Text/image prompts encoded via CLIP
2. Guided diffusion model loaded based on image size and conditioning
3. With no `timestep_respacing`, `sampling.sample_loop_progressive()` runs, conditioned on the closest ImageNet class. Otherwise guided-diffusion's own loop runs through `cond_fn()`, re-randomising the class each step. Both share `guidance_loss()`: CLIP loss + regularization losses.
4. Diffusion loop yields intermediate samples at specified frequency
5. `main()` builds any GIF or video from the frames this run yielded, and only deletes them once every requested output exists

### Prompt Weights

Prompts support weights via `text:weight` syntax (e.g., `"cat:1.0|dog:-0.5"`). Weights are normalized to sum to non-zero.

## Git Commits

When tracking changes with git, NEVER (IMPORTANT!) include attribution to Claude Code or Anthropic. Use 1-2 sentence, per-file commits.

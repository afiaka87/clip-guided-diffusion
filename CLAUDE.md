# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CLIP Guided Diffusion (cgd) is a text-to-image generation tool that uses OpenAI's CLIP model to guide a diffusion process. Originally by Katherine Crowson (@crowsonkb).

## Installation & Setup

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```sh
git clone https://github.com/afiaka87/clip-guided-diffusion.git
cd clip-guided-diffusion
uv sync
```

## Running Tests

```sh
uv run python -m unittest discover
uv run python -m unittest test.TestCGD.test_cgd_one_step_succeeds  # single test
```

Note: Some tests require a CUDA GPU. Outputs saved to `./outputs/` and `current.png`.

## CLI Usage

Basic usage: `uv run cgd -txt "Your prompt here"`

Key parameters:
- `-size`: Image size (64, 128, 256, 512)
- `-respace`: Timestep respacing (25, 50, 150, 250, 500, 1000, or ddim variants)
- `-cgs`: CLIP guidance scale (default 1000)
- `-tvs`: TV loss scale for smoothness (default 150)
- `-init`: Initial image path
- `-skip`: Skip timesteps for image blending

Performance optimizations:
- `-reduce`: Skip early diffusion steps & reduce CLIP guidance frequency
- `-cutn_skip`: Use fewer cutouts in early steps (4->8->16)
- `-cached_cutn`: Cache cutout coordinates across steps

## Architecture

### Core Components

- **`cgd/cgd.py`**: Main entry point with `clip_guided_diffusion()` generator function and CLI `main()`. The generator yields `(batch_idx, image_path)` tuples during diffusion.

- **`cgd/script_util.py`**: Utilities for downloading checkpoints, loading the guided diffusion model, parsing prompts with weights, and logging images. Model checkpoints cached at `~/.cache/clip-guided-diffusion/`.

- **`cgd/clip_util.py`**: CLIP model loading (`load_clip()`), text/image prompt encoding, and the `MakeCutouts` class for generating random crops.

- **`cgd/losses.py`**: Loss functions - `spherical_dist_loss()` for CLIP similarity, `tv_loss()` for smoothness, `range_loss()` for RGB clamping.

- **`cgd/modules.py`**: `MakeCutouts` module for random patch extraction with optional augmentations.

### External Dependency

The `guided-diffusion` package (git dependency from crowsonkb) provides the diffusion model architecture and sampling loops. Config flags for different model sizes are in `data/diffusion_model_flags.py`.

### Diffusion Flow

1. Text/image prompts encoded via CLIP
2. Guided diffusion model loaded based on image size and conditioning
3. `cond_fn()` computes gradients from CLIP loss + regularization losses
4. Diffusion loop yields intermediate samples at specified frequency
5. Final GIF created from all intermediate outputs

### Prompt Weights

Prompts support weights via `text:weight` syntax (e.g., `"cat:1.0|dog:-0.5"`). Weights are normalized to sum to non-zero.

## Git Commits

When tracking changes with git, NEVER (IMPORTANT!) include attribution to Claude Code or Anthropic. Use 1-2 sentence, per-file commits.

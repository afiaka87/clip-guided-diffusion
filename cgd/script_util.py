import glob
import io
import os
import re
import time
from functools import lru_cache
from pathlib import Path
import requests
import torch as th
import torchvision.transforms.functional as tvf
from tqdm.auto import tqdm
from data.diffusion_model_flags import DIFFUSION_LOOKUP
from PIL import Image

from cgd import clip_util
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

CACHE_PATH = os.path.expanduser("~/.cache/clip-guided-diffusion")
TIMESTEP_RESPACINGS = ("25", "50", "100", "250", "500", "1000",
                       "ddim25", "ddim50", "ddim100", "ddim250", "ddim500", "ddim1000")
DIFFUSION_SCHEDULES = (25, 50, 100, 250, 500, 1000)
IMAGE_SIZES = (64, 128, 256, 512)

def check_parameters(
    prompts: list,
    image_prompts: list,
    image_size: int,
    timestep_respacing: str,
    diffusion_steps: int,
    clip_model_name: str,
    save_frequency: int,
    noise_schedule: str,
):
    if not (len(prompts) > 0 or len(image_prompts) > 0):
        raise ValueError("Must provide at least one prompt, text or image.")
    if not (noise_schedule in ['linear', 'cosine']):
        raise ValueError('Noise schedule should be one of: linear, cosine')
    if not (image_size in IMAGE_SIZES):
        raise ValueError(f"--image size should be one of {IMAGE_SIZES}")
    if not (0 < save_frequency <= int(timestep_respacing.replace('ddim', ''))):
        raise ValueError(
            "--save_frequency must be greater than 0 and less than `timestep_respacing`")
    if not (diffusion_steps in DIFFUSION_SCHEDULES):
        print('(warning) Diffusion steps should be one of:', DIFFUSION_SCHEDULES)
    if not (timestep_respacing in TIMESTEP_RESPACINGS):
        print(
            f"Pausing run. `timestep_respacing` should be one of {TIMESTEP_RESPACINGS}. CTRL-C if this was a mistake.")
        time.sleep(5)
        print("Resuming run.")
    if clip_model_name.endswith('.pt') or clip_model_name.endswith('.pth'):
        assert os.path.isfile(
            clip_model_name), f"{clip_model_name} does not exist"
        print(f"Loading custom model from {clip_model_name}")
    elif not (clip_model_name in clip_util.CLIP_MODEL_NAMES):
        print(
            f"--clip model name should be one of: {clip_util.CLIP_MODEL_NAMES} unless you are trying to use your own checkpoint.")
        print(f"Loading OpenAI CLIP - {clip_model_name}")


def parse_prompt(prompt):  # parse a single prompt in the form "<text||img_url>:<weight>"
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)  # theres two colons, so we grab the 2nd
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)  # grab weight after colon
    vals = vals + ['', '1'][len(vals):]  # if no weight, use 1
    return vals[0], float(vals[1])  # return text, weight


def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')


def alphanumeric_filter(s: str) -> str:
    # regex to remove non-alphanumeric characters
    ALPHANUMERIC_REGEX = r"[^\w\s]"
    return re.sub(ALPHANUMERIC_REGEX, "", s).replace(" ", "_")


def clean_and_combine_prompts(base_path, txts, batch_idx, max_length=255) -> str:
    clean_txt = "_".join([alphanumeric_filter(txt)
                         for txt in txts])[:max_length]
    return os.path.join(base_path, clean_txt, f"{batch_idx:02}")


def log_image(image: th.Tensor, base_path: str, txts: list, current_step: int, batch_idx: int) -> str:
    dirname = clean_and_combine_prompts(base_path, txts, batch_idx)
    os.makedirs(dirname, exist_ok=True)
    stem = f"{current_step:04}"
    filename = os.path.join(dirname, f'{stem}.png')
    pil_image = tvf.to_pil_image(image.add(1).div(2).clamp(0, 1))
    pil_image.save(os.path.join(os.getcwd(), f'current.png'))
    pil_image.save(filename)
    return str(filename)


def create_gif_ffmpeg(base, prompts, batch_idx, fps=10, delete_frames=False):
    """Create a high-quality GIF using ffmpeg with palette optimization."""
    import subprocess

    io_safe_prompts = clean_and_combine_prompts(base, prompts, batch_idx)
    images_glob = os.path.join(io_safe_prompts, "*.png")
    image_files = sorted(glob.glob(images_glob))

    if not image_files:
        print(f"No images found in {io_safe_prompts}")
        return None

    gif_filename = f"{io_safe_prompts}_{batch_idx:02}.gif"
    palette_file = os.path.join(io_safe_prompts, "palette.png")
    input_pattern = os.path.join(io_safe_prompts, "%04d.png")

    # Generate optimized palette for better color accuracy
    palette_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", input_pattern,
        "-vf", "palettegen=max_colors=256:stats_mode=full",
        palette_file
    ]

    # Create GIF using the palette with high-quality dithering
    gif_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", input_pattern,
        "-i", palette_file,
        "-lavfi", "paletteuse=dither=floyd_steinberg:bayer_scale=5:diff_mode=rectangle",
        "-loop", "0",
        gif_filename
    ]

    try:
        subprocess.run(palette_cmd, check=True, capture_output=True)
        subprocess.run(gif_cmd, check=True, capture_output=True)
        print(f"Created GIF: {gif_filename}")

        # Clean up palette file
        if os.path.exists(palette_file):
            os.remove(palette_file)

        if delete_frames:
            for f in image_files:
                os.remove(f)
            # Remove the directory if empty
            if os.path.isdir(io_safe_prompts) and not os.listdir(io_safe_prompts):
                os.rmdir(io_safe_prompts)
            print(f"Deleted {len(image_files)} frame(s)")

        return gif_filename

    except subprocess.CalledProcessError as e:
        print(f"ffmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
        return None
    except FileNotFoundError:
        print("ffmpeg not found. Please install ffmpeg to use this feature.")
        return None


def create_video_ffmpeg(base, prompts, batch_idx, fps=10, delete_frames=False):
    """Create a high-quality MP4 video using ffmpeg with x264 encoding."""
    import subprocess

    io_safe_prompts = clean_and_combine_prompts(base, prompts, batch_idx)
    images_glob = os.path.join(io_safe_prompts, "*.png")
    image_files = sorted(glob.glob(images_glob))

    if not image_files:
        print(f"No images found in {io_safe_prompts}")
        return None

    video_filename = f"{io_safe_prompts}_{batch_idx:02}.mp4"
    input_pattern = os.path.join(io_safe_prompts, "%04d.png")

    # High-quality x264 encoding with settings to avoid artifacts
    video_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", input_pattern,
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        video_filename
    ]

    try:
        subprocess.run(video_cmd, check=True, capture_output=True)
        print(f"Created video: {video_filename}")

        if delete_frames:
            for f in image_files:
                os.remove(f)
            # Remove the directory if empty
            if os.path.isdir(io_safe_prompts) and not os.listdir(io_safe_prompts):
                os.rmdir(io_safe_prompts)
            print(f"Deleted {len(image_files)} frame(s)")

        return video_filename

    except subprocess.CalledProcessError as e:
        print(f"ffmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
        return None
    except FileNotFoundError:
        print("ffmpeg not found. Please install ffmpeg to use this feature.")
        return None


def download(url: str, filename: str, root: str = CACHE_PATH, max_retries: int = 3) -> str:
    os.makedirs(root, exist_ok=True)
    download_target = Path(os.path.join(root, filename))
    download_target_tmp = download_target.with_suffix('.tmp')
    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(
            f"{download_target} exists and is not a regular file")
    if os.path.isfile(download_target):
        return str(download_target)

    for attempt in range(max_retries):
        try:
            # Use longer timeouts: 30s connect, 120s read (for slow connections at end)
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("Content-Length", 0))
                downloaded_size = 0

                with open(download_target_tmp, "wb") as output:
                    with tqdm(total=total_size, ncols=80, unit='B', unit_scale=True, unit_divisor=1024) as loop:
                        # Use larger chunks for efficiency
                        for chunk in response.iter_content(chunk_size=64 * 1024):
                            if chunk:
                                output.write(chunk)
                                downloaded_size += len(chunk)
                                loop.update(len(chunk))
                    # Ensure all data is flushed to disk
                    output.flush()
                    os.fsync(output.fileno())

            # Verify downloaded size matches expected size
            actual_size = download_target_tmp.stat().st_size
            if total_size > 0 and actual_size != total_size:
                raise RuntimeError(
                    f"Download incomplete: expected {total_size} bytes, got {actual_size} bytes")

            os.rename(download_target_tmp, download_target)
            return str(download_target)

        except (requests.exceptions.RequestException, RuntimeError) as e:
            if download_target_tmp.exists():
                download_target_tmp.unlink()
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"Download failed (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"Download failed after {max_retries} attempts: {e}") from e

    return str(download_target)


def download_guided_diffusion(image_size: int, class_cond: bool, checkpoints_dir: str = CACHE_PATH, overwrite: bool = False) -> str:
    cond_key = 'cond' if class_cond else 'uncond'
    diffusion_model_info = DIFFUSION_LOOKUP[cond_key][image_size]
    if not overwrite:
        target_path = Path(checkpoints_dir).joinpath(
            diffusion_model_info["filename"])
        if target_path.exists():
            return str(target_path)
    return download(diffusion_model_info["url"], diffusion_model_info["filename"], checkpoints_dir)


@lru_cache(maxsize=1)
def load_guided_diffusion(
    checkpoint_path: str,
    image_size: int,
    class_cond: bool,
    diffusion_steps: int = None,
    timestep_respacing: str = None,
    use_fp16: bool = True,
    device: str = '',
    noise_schedule: str = 'linear',
    dropout: float = 0.0,
):
    '''
    checkpoint_path: path to the checkpoint to load.
    image_size: size of the images to be used.
    class_cond: whether to condition on the class label
    diffusion_steps: number of diffusion steps
    timestep_respacing: whether to use timestep-respacing or not
    '''
    if not (len(device) > 0):
        raise ValueError("device must be set")
    if not (noise_schedule in ["linear", "cosine"]):
        raise ValueError("linear_or_cosine must be set")

    cond_key = 'cond' if class_cond else 'uncond'
    diffusion_model_info = DIFFUSION_LOOKUP[cond_key][image_size]
    model_config: dict = model_and_diffusion_defaults()
    model_config.update(diffusion_model_info['model_flags'])
    model_config.update(**{  # Custom params from user
        'diffusion_steps': diffusion_steps,
        'timestep_respacing': timestep_respacing,
        "use_fp16": use_fp16,
        "noise_schedule": noise_schedule,
        "dropout": dropout,
    })
    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(th.load(checkpoint_path, map_location='cpu'))
    model.requires_grad_(False).eval().to(device)
    for name, param in model.named_parameters():
        if "qkv" in name or "norm" in name or "proj" in name:
            param.requires_grad_()
    if model_config["use_fp16"]:
        model.convert_to_fp16()
    return model.to(device), diffusion

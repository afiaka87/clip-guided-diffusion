import io
import os
import re
import shutil
import subprocess
import time
from functools import lru_cache
from pathlib import Path
import requests
import torch as th
import torchvision.transforms.functional as tvf
from tqdm.auto import tqdm
from data.diffusion_model_flags import DIFFUSION_LOOKUP

from guided_diffusion.respace import space_timesteps
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

CACHE_PATH = os.path.expanduser("~/.cache/clip-guided-diffusion")
NUM_TRAINING_TIMESTEPS = 1000  # Every checkpoint was trained on this schedule length.


class ConfigError(ValueError):
    """The requested run cannot work. The message says what to change."""


def is_url(path) -> bool:
    return str(path).startswith(('http://', 'https://'))


def check_parameters(
    prompts,
    image_prompts,
    image_size: int,
    class_cond: bool = True,
    timestep_respacing: str = None,
    skip_timesteps: int = 0,
    save_frequency: int = 1,
    sample_steps: int = 1,
    height_offset: int = 0,
    width_offset: int = 0,
    init_image=None,
):
    """Reject a bad configuration before spending time loading anything."""
    if not (prompts or image_prompts):
        raise ConfigError("Provide at least one prompt, text or image.")
    sizes = DIFFUSION_LOOKUP['cond' if class_cond else 'uncond']
    if image_size not in sizes:
        kind = 'class conditional' if class_cond else 'unconditional'
        raise ConfigError(f"No {kind} checkpoint for size {image_size}. Choose one of {sorted(sizes)}.")
    if save_frequency < 1:
        raise ConfigError("--save_frequency must be at least 1.")
    if sample_steps < 1:
        raise ConfigError("--sample_steps must be at least 1.")
    total_steps = NUM_TRAINING_TIMESTEPS
    if timestep_respacing is not None:
        try:
            total_steps = len(space_timesteps(NUM_TRAINING_TIMESTEPS, timestep_respacing))
        except ValueError as error:
            raise ConfigError(f"Bad --timestep_respacing {timestep_respacing!r}: {error}") from error
    if not 0 <= skip_timesteps < total_steps:
        raise ConfigError(f"--skip_timesteps must be between 0 and {total_steps - 1}.")
    stride = image_size // 8  # The U-Net halves the resolution down to 8x8.
    if (image_size + height_offset) % stride or (image_size + width_offset) % stride:
        raise ConfigError(f"At size {image_size}, height and width must be multiples of {stride}. "
                          f"Got {image_size + height_offset}x{image_size + width_offset}.")
    images = [parse_prompt(image_prompt)[0] for image_prompt in image_prompts]
    for image in images + ([init_image] if init_image else []):
        if not is_url(image) and not os.path.isfile(image):
            raise ConfigError(f"No such image: {image}")


def parse_prompt(prompt):  # parse a single prompt in the form "<text||img_url>:<weight>"
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)  # theres two colons, so we grab the 2nd
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)  # grab weight after colon
    vals = vals + ['', '1'][len(vals):]  # if no weight, use 1
    return vals[0], float(vals[1])  # return text, weight


def fetch(url_or_path):
    if is_url(url_or_path):
        r = requests.get(url_or_path, timeout=30)
        r.raise_for_status()
        return io.BytesIO(r.content)
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
    filename = os.path.join(dirname, f'{current_step:04}.png')
    tvf.to_pil_image(image.add(1).div(2).clamp(0, 1)).save(filename)
    shutil.copyfile(filename, os.path.join(os.getcwd(), 'current.png'))
    return filename


def _encode_frames(frames: list, output: str, fps: int, *encoder_args):
    """Pipe `frames` through ffmpeg. Returns `output`, or None if it could not be made."""
    command = ["ffmpeg", "-y", "-f", "image2pipe", "-framerate", str(fps), "-i", "-", *encoder_args, str(output)]
    try:
        subprocess.run(command, input=b"".join(Path(frame).read_bytes() for frame in frames),
                       check=True, capture_output=True)
    except FileNotFoundError:
        print("ffmpeg not found. Please install ffmpeg to use this feature.")
        return None
    except subprocess.CalledProcessError as error:
        print(f"ffmpeg error: {error.stderr.decode()}")
        return None
    print(f"Created {output}")
    return str(output)


def create_gif_ffmpeg(frames: list, output: str, fps: int = 10):
    """Create a high-quality GIF, using a palette generated from the frames themselves."""
    palette = ("split[frames][copy];[copy]palettegen=max_colors=256:stats_mode=full[palette];"
               "[frames][palette]paletteuse=dither=floyd_steinberg:diff_mode=rectangle")
    return _encode_frames(frames, output, fps, "-filter_complex", palette, "-loop", "0")


def create_video_ffmpeg(frames: list, output: str, fps: int = 10):
    """Create a high-quality MP4 video using x264."""
    return _encode_frames(frames, output, fps, "-c:v", "libx264", "-preset", "slow", "-crf", "18",
                          "-pix_fmt", "yuv420p", "-movflags", "+faststart")


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


def download_guided_diffusion(image_size: int, class_cond: bool, checkpoints_dir: str = CACHE_PATH) -> str:
    diffusion_model_info = DIFFUSION_LOOKUP['cond' if class_cond else 'uncond'][image_size]
    return download(diffusion_model_info["url"], diffusion_model_info["filename"], checkpoints_dir)


@lru_cache(maxsize=1)
def load_guided_diffusion(
    checkpoint_path: str,
    image_size: int,
    class_cond: bool,
    timestep_respacing: str = "",
    use_fp16: bool = True,
    device: str = '',
):
    '''
    checkpoint_path: path to the checkpoint to load.
    image_size: size of the images to be used.
    class_cond: whether to condition on the class label
    timestep_respacing: subset of the training timesteps to sample. Empty keeps them all.

    The diffusion schedule always comes from the checkpoint's flags: sampling a model
    on a schedule it was not trained on produces garbage.
    '''
    if not (len(device) > 0):
        raise ValueError("device must be set")

    cond_key = 'cond' if class_cond else 'uncond'
    diffusion_model_info = DIFFUSION_LOOKUP[cond_key][image_size]
    model_config: dict = model_and_diffusion_defaults()
    model_config.update(diffusion_model_info['model_flags'])
    model_config.update(**{  # Custom params from user
        'diffusion_steps': NUM_TRAINING_TIMESTEPS,
        'timestep_respacing': timestep_respacing,
        "use_fp16": use_fp16,
    })
    # Build on the meta device: random-initialising 400M+ parameters only to
    # overwrite them is the slowest part of startup. mmap + assign then adopts
    # the checkpoint's tensors without an extra copy.
    with th.device('meta'):
        model, diffusion = create_model_and_diffusion(**model_config)
    state_dict = th.load(checkpoint_path, map_location='cpu', mmap=True, weights_only=True)
    model.load_state_dict(state_dict, assign=True)
    model.requires_grad_(False).eval()
    # Convert before moving so the 32-bit weights never occupy VRAM.
    if model_config["use_fp16"]:
        model.convert_to_fp16()
    return model.to(device), diffusion

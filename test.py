from clip.model import ModifiedResNet, VisionTransformer
import itertools
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cgd import script_util
from cgd import clip_util
from cgd import losses
from cgd import sampling

import numpy as np
import torch as th
import torchvision.transforms.functional as tvf
from cgd import modules
from cgd.cgd import clip_guided_diffusion
from data.imagenet1000_clsidx_to_labels import IMAGENET_CLASSES
from guided_diffusion import respace
from PIL import Image
from torch.nn import functional as tf


# Integration tests; better than nothing at all.

# TODO results in a StopIteration error; investigate.
# is_model_cuda = lambda x: next(x.parameters()).is_cuda


class TestUtil(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_dir_path = Path(self.test_dir.name)

    def tearDown(self) -> None:
        self.test_dir.cleanup()

    def test_download_returns_target_full_path(self):
        url = 'https://github.com/afiaka87/clip-guided-diffusion/raw/main/images/photon.png'
        result = script_util.download(url, 'photon.png', root=self.test_dir.name)
        expected = self.test_dir_path.joinpath('photon.png')
        self.assertEqual(result, str(expected))
        self.assertTrue(expected.exists())


class TestTorchUtil(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        self.test_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.test_dir.cleanup()

    def test_load_guided_diffusion_cpu(self):
        image_size = 64
        checkpoint_path = Path(script_util.CACHE_PATH).joinpath(
            f"{image_size}x{image_size}_diffusion.pt")
        class_cond = True
        timestep_respacing = '25'
        use_fp16 = False
        device = 'cpu'
        model, diffusion = script_util.load_guided_diffusion(
            checkpoint_path=str(checkpoint_path),
            image_size=image_size,
            class_cond=class_cond,
            timestep_respacing=timestep_respacing,
            use_fp16=use_fp16,
            device=device,
        )
        self.assertIsInstance(model, th.nn.Module)
        self.assertIsInstance(diffusion, respace.SpacedDiffusion)

    def test_load_guided_diffusion_cuda(self):
        image_size = 64
        checkpoint_path = Path(script_util.CACHE_PATH).joinpath(
            f"{image_size}x{image_size}_diffusion.pt")
        class_cond = True
        timestep_respacing = '25'
        use_fp16 = False
        device = 'cuda'
        model, diffusion = script_util.load_guided_diffusion(
            checkpoint_path=str(checkpoint_path),
            image_size=image_size,
            class_cond=class_cond,
            timestep_respacing=timestep_respacing,
            use_fp16=use_fp16,
            device=device,
        )
        def is_model_cuda(x): return next(x.parameters()).is_cuda
        self.assertTrue(is_model_cuda(model))
        self.assertIsInstance(model, th.nn.Module)
        self.assertIsInstance(diffusion, respace.SpacedDiffusion)

    def test_log_image(self):
        image = th.rand(3, 3, 3)
        txts = ['a', 'b', 'c']
        batch_idx = 4
        current_step = 1
        expected_filename = os.path.join(
            self.test_dir.name, "a_b_c/04/0001.png")
        result_filename = script_util.log_image(
            image=image, base_path=self.test_dir.name,
            txts=txts,
            current_step=current_step,
            batch_idx=batch_idx,
        )
        self.assertEqual(result_filename, expected_filename)

    def test_spherical_dist_loss(self):
        x = th.rand(1, 3)
        y = th.rand(1, 3)
        result = losses.spherical_dist_loss(x, y)
        x_norm = tf.normalize(x, dim=-1)
        y_norm = tf.normalize(y, dim=-1)
        expected = (x_norm - y_norm).norm(dim=-1).div(2).arcsin().pow(2).mul(2)
        self.assertEqual(result, expected)


class TestCGD(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_dir_path = str(Path(self.test_dir.name))

    def test_cgd_one_step_succeeds(self):
        samples = clip_guided_diffusion(prompts=["Loose seal."], image_size=64,
                                        num_cutouts=1, clip_model_name="RN50", prefix_path=self.test_dir_path)
        first_yielded_sample = list(itertools.islice(samples, 1))[0]
        self.assertIsNotNone(first_yielded_sample)

    def test_cgd_init_succeeds_with_skip_timesteps(self):
        samples = clip_guided_diffusion(prompts=["Loose seal."], init_image='images/photon.png',
                                        skip_timesteps=500, image_size=64, num_cutouts=1,
                                        clip_model_name="RN50", prefix_path=self.test_dir_path)
        first_yielded_sample = list(itertools.islice(samples, 1))[0]
        self.assertIsNotNone(first_yielded_sample)

    def test_clip_guided_diffusion_yields_batch_idx_path_tuple(self):
        samples = clip_guided_diffusion(prompts=["Loose seal."], image_size=64, batch_size=2,
                                        num_cutouts=1, clip_model_name="RN50", prefix_path=self.test_dir_path, device='cpu')
        first_two_samples = list(itertools.islice(samples, 2))
        first_sample = first_two_samples[0]
        second_sample = first_two_samples[1]
        first_expected_returned_batch_idx = 0
        second_expected_returned_batch_idx = 1
        self.assertEqual(first_sample[0], first_expected_returned_batch_idx)
        self.assertEqual(second_sample[0], second_expected_returned_batch_idx)

class TestClipUtil(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_dir_path = str(Path(self.test_dir.name))
    
    def test_parse_prompt_returns_prompt_weight_tuple(self):
        prompt = "Loose seal.:0.4"
        expected_prompt_weight_tuple = ("Loose seal.", 0.4)
        result = script_util.parse_prompt(prompt)
        self.assertEqual(result, expected_prompt_weight_tuple)

    def test_parse_prompt_negative_weight(self):
        prompt = "Loose seal.:-0.4"
        expected_prompt_weight_tuple = ("Loose seal.", -0.4)
        result = script_util.parse_prompt(prompt)
        self.assertEqual(result, expected_prompt_weight_tuple)

    def test_parse_prompt_no_weight_returns_one(self):
        prompt = "Loose seal."
        expected_prompt_weight_tuple = ("Loose seal.", 1.0)
        result = script_util.parse_prompt(prompt)
        self.assertEqual(result, expected_prompt_weight_tuple)

    def test_imagenet_top_n_runs_on_cuda(self, top_n=100):
        device = "cpu"
        clip_model_name = "RN50"
        text = "Loose seal."
        text, weight = script_util.parse_prompt(text)
        text, _ = clip_util.encode_text_prompt(text, weight, clip_model_name, device)
        result_scores = clip_util.imagenet_top_n(text_encodes=text, device=device, n=100, clip_model_name=clip_model_name)
        print(result_scores)

    def test_load_clip_rn50_cpu(self):
        model_name = "RN50"
        clip_model, clip_size = clip_util.load_clip(model_name=model_name, device="cpu")
        self.assertIsInstance(clip_model.visual, ModifiedResNet)
        self.assertEqual(clip_size, clip_model.visual.input_resolution)
        self.assertEqual(clip_size, 224)

    def test_load_clip_vit_b_16_cpu(self):
        model_name = "ViT-B/16"
        clip_model, clip_size = clip_util.load_clip(model_name=model_name, device="cpu")
        self.assertIsInstance(clip_model.visual, VisionTransformer)
        self.assertEqual(clip_size, clip_model.visual.input_resolution)
        self.assertEqual(clip_size, 224)

    def test_load_clip_rn50_cuda(self):
        model_name = "RN50"
        clip_model, clip_size = clip_util.load_clip(model_name=model_name, device="cuda")
        self.assertIsInstance(clip_model.visual, ModifiedResNet)
        self.assertEqual(clip_size, clip_model.visual.input_resolution)
        self.assertEqual(clip_size, 224)

    def test_load_clip_vit_b_16_cuda(self):
        model_name = "ViT-B/16"
        clip_model, clip_size = clip_util.load_clip(model_name=model_name, device="cuda")
        self.assertIsInstance(clip_model.visual, VisionTransformer)
        self.assertEqual(clip_size, clip_model.visual.input_resolution)
        self.assertEqual(clip_size, 224)

    def test_make_cutouts_to_cpu(self):
        cut_size = 224
        num_cutouts = 8
        cutout_size_power = 0.5
        input = th.rand(1, 3, 512, 512)
        input = clip_util.CLIP_NORMALIZE(input)
        make_cutouts = modules.MakeCutouts(
            cut_size=cut_size,
            num_cutouts=num_cutouts,
            cutout_size_power=cutout_size_power,
        ).to('cpu')
        result = make_cutouts(input)
        print(result.shape)
        self.assertEqual(result.shape[0], num_cutouts)
        self.assertEqual(result.shape[1], 3)
        self.assertEqual(result.shape[2], cut_size)
        self.assertEqual(result.shape[3], cut_size)

    def test_make_cutouts_to_cuda(self):
        cut_size = 224
        num_cutouts = 8
        cutout_size_power = 0.5
        input = th.rand(1, 3, 512, 512)
        input = clip_util.CLIP_NORMALIZE(input)
        make_cutouts = clip_util.MakeCutouts(
            cut_size=cut_size,
            num_cutouts=num_cutouts,
            cutout_size_power=cutout_size_power,
        ).to(th.device('cuda'))
        result = make_cutouts(input)
        self.assertEqual(result.shape[0], num_cutouts)
        self.assertEqual(result.shape[1], 3)
        self.assertEqual(result.shape[2], cut_size)
        self.assertEqual(result.shape[3], cut_size)

    def test_clip_encode_text_cuda(self):
        clip_model_name = "RN50"
        text = "A"
        device = "cuda:0"
        result, weight = clip_util.encode_text_prompt(clip_model_name, text, device=device)
        self.assertEqual(str(result.device), device)
        self.assertIsNotNone(result)

    def test_clip_encode_text_cuda_with_weights(self):
        clip_model_name = "RN50"
        text = "A"
        weight = 0.5
        device = "cuda:0"
        result_encode, result_weight = clip_util.encode_text_prompt(clip_model_name=clip_model_name, txt=text, weight=weight, device=device)
        self.assertEqual(str(result_encode.device), device)
        self.assertEqual(result_weight, weight)


class TestMakeCutouts(unittest.TestCase):
    def test_full_frame_only_collapses_to_one_cutout(self):
        make_cutouts = modules.MakeCutouts(cut_size=224, num_cutouts=16)
        self.assertEqual(make_cutouts.distinct_cutouts(128, 128), 1)
        self.assertEqual(len(set(make_cutouts._generate_coords(128, 128, 16))), 1)

    def test_real_variety_keeps_every_cutout(self):
        self.assertEqual(modules.MakeCutouts(224, 16).distinct_cutouts(256, 256), 16)
        self.assertEqual(modules.MakeCutouts(224, 16, use_augs=True).distinct_cutouts(128, 128), 16)
        self.assertEqual(modules.MakeCutouts(224, 16).distinct_cutouts(128, 192), 16)

    def test_min_cut_fraction_varies_cutouts_below_clip_resolution(self):
        make_cutouts = modules.MakeCutouts(cut_size=224, num_cutouts=4, min_cut_fraction=0.7)
        self.assertEqual(make_cutouts.distinct_cutouts(128, 128), 4)
        sizes = [size for _, _, size in make_cutouts._generate_coords(128, 128, 64)]
        self.assertGreaterEqual(min(sizes), 89)
        self.assertLessEqual(max(sizes), 128)
        self.assertGreater(len(set(sizes)), 1)

    def test_non_square_cutouts_span_the_width(self):
        th.manual_seed(0)
        left_black_right_white = th.zeros(1, 3, 64, 128)
        left_black_right_white[..., 64:] = 1
        cutouts = modules.MakeCutouts(cut_size=64, num_cutouts=32)(left_black_right_white)
        self.assertEqual(cutouts.shape, (32, 3, 64, 64))
        self.assertGreater(cutouts.mean(dim=(1, 2, 3)).max().item(), 0.5)


class TestNoiseLevels(unittest.TestCase):
    def setUp(self):
        self.alphas_cumprod = np.cumprod(1 - np.linspace(1e-4, 0.02, 1000))
        self.levels = sampling.NoiseLevels(self.alphas_cumprod)

    def test_alpha_sigma_matches_the_table_at_integer_timesteps(self):
        for t in (0, 137, 999):
            alpha, sigma = self.levels.alpha_sigma(t)
            self.assertAlmostEqual(alpha ** 2, self.alphas_cumprod[t], places=9)
            self.assertAlmostEqual(alpha ** 2 + sigma ** 2, 1.0, places=9)

    def test_timesteps_are_descending_and_uniform_in_log_snr(self):
        ts = self.levels.timesteps(30)
        self.assertEqual(len(ts), 31)
        self.assertEqual(ts[0], 999)
        self.assertEqual(ts[-1], 0.0)
        self.assertTrue(all(a > b for a, b in zip(ts, ts[1:])))
        log_snr = [np.log(a / s) for a, s in map(self.levels.alpha_sigma, ts)]
        spacing = np.diff(log_snr)
        self.assertLess(spacing.std() / spacing.mean(), 0.01)

    def test_start_joins_the_full_schedule(self):
        full = self.levels.timesteps(30)
        ts = self.levels.timesteps(30, start=400)
        self.assertEqual(ts[0], 400)
        self.assertEqual(ts[1:], [t for t in full if t < 400])
        self.assertTrue(all(a > b for a, b in zip(ts, ts[1:])))


class TestFastSampler(unittest.TestCase):
    PROMPT = "a photograph of a red fox in the snow"

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.prefix = self.test_dir.name
        self.addCleanup(self.test_dir.cleanup)

    def generate(self, **kwargs):
        options = dict(prompts=[self.PROMPT], prefix_path=self.prefix, save_frequency=1000,
                       progress=False, device="cuda")
        options.update(kwargs)
        return list(clip_guided_diffusion(**options))

    def clip_similarity(self, image_path, text):
        clip_model, clip_size = clip_util.load_clip("ViT-B/32", "cuda")
        image = tvf.to_tensor(Image.open(image_path).convert("RGB")).unsqueeze(0).cuda()
        image = tf.interpolate(image, size=clip_size, mode="bicubic", align_corners=False).clamp(0, 1)
        with th.no_grad():
            image_embed = clip_model.encode_image(clip_util.CLIP_NORMALIZE(image)).float()
        text_embed, _ = clip_util.encode_text_prompt(text, 1.0, "ViT-B/32", "cuda")
        return tf.cosine_similarity(image_embed, text_embed).item()

    def test_imagenet_top_n_selects_the_matching_class(self):
        text_embed, _ = clip_util.encode_text_prompt(self.PROMPT, 1.0, "ViT-B/32", "cuda")
        top = clip_util.imagenet_top_n(text_embed, device="cuda", n=5, clip_model_name="ViT-B/32")
        self.assertIn("red fox", [IMAGENET_CLASSES[i] for i in top.tolist()])

    def test_generates_the_prompt_without_blowing_out(self):
        image_path = self.generate(image_size=128)[-1][1]
        self.assertTrue(image_path.endswith("0029.png"))
        pixels = tvf.to_tensor(Image.open(image_path).convert("RGB"))
        clipped = ((pixels <= 1 / 255) | (pixels >= 254 / 255)).float().mean().item()
        self.assertLess(clipped, 0.25)
        self.assertGreater(pixels.std().item(), 0.1)
        self.assertGreater(self.clip_similarity(image_path, self.PROMPT),
                           self.clip_similarity(image_path, "a bowl of ramen on a wooden table"))

    def test_same_seed_reproduces_the_image(self):
        first = self.generate(image_size=64, sample_steps=6, seed=3, prefix_path=f"{self.prefix}/a")[-1][1]
        second = self.generate(image_size=64, sample_steps=6, seed=3, prefix_path=f"{self.prefix}/b")[-1][1]
        difference = tvf.to_tensor(Image.open(first)) - tvf.to_tensor(Image.open(second))
        self.assertLess(difference.abs().max().item(), 2 / 255)

    def test_batch_yields_distinct_images(self):
        results = self.generate(image_size=64, sample_steps=6, batch_size=2)
        first, second = [path for batch_idx, path in results if path.endswith("0005.png")]
        difference = tvf.to_tensor(Image.open(first)) - tvf.to_tensor(Image.open(second))
        self.assertGreater(difference.abs().mean().item(), 0.01)

    def test_init_image_shortens_the_schedule_and_saves_the_final_frame(self):
        results = self.generate(image_size=64, init_image="images/photon.png", skip_timesteps=700, save_frequency=1)
        self.assertLess(len(results), 30)
        result = tvf.to_tensor(Image.open(results[-1][1]).convert("RGB"))
        init = tvf.to_tensor(Image.open("images/photon.png").convert("RGB").resize((64, 64)))
        noise = th.rand_like(init)
        self.assertLess((result - init).abs().mean(), (noise - init).abs().mean())

    def test_skip_timesteps_without_an_init_image_runs_the_whole_schedule(self):
        results = self.generate(image_size=64, sample_steps=6, skip_timesteps=500, save_frequency=1)
        self.assertEqual(len(results), 6)

    def test_batch_with_several_prompts_guides_every_image_with_every_prompt(self):
        results = self.generate(prompts=["a red fox", "a yellow lemon:0.5"], image_size=64, sample_steps=8, batch_size=3)
        self.assertEqual(sorted(batch_idx for batch_idx, _ in results[-3:]), [0, 1, 2])


class TestSkipWeightInit(unittest.TestCase):
    def test_loading_clip_leaves_the_random_stream_and_init_functions_alone(self):
        init_functions = {name: getattr(th.nn.init, name) for name in clip_util._INIT_FUNCTIONS}
        th.manual_seed(0)
        expected = th.rand(4)
        th.manual_seed(0)
        clip_util.load_clip.__wrapped__("RN50", "cpu")
        self.assertTrue(th.equal(th.rand(4), expected))
        for name, function in init_functions.items():
            self.assertIs(getattr(th.nn.init, name), function)


class TestLegacySampler(unittest.TestCase):
    def test_respacing_selects_the_original_sampler_and_saves_the_final_frame_after_skipping(self):
        with tempfile.TemporaryDirectory() as prefix:
            results = list(clip_guided_diffusion(
                prompts=["Loose seal."], image_size=64, timestep_respacing="25", skip_timesteps=5,
                clip_model_name="RN50", prefix_path=prefix, save_frequency=1000, progress=False))
            self.assertEqual(Path(results[-1][1]).name, "0019.png")


class TestCheckpointSchedule(unittest.TestCase):
    def test_loader_keeps_the_schedule_the_checkpoint_was_trained_with(self):
        _, diffusion = script_util.load_guided_diffusion(
            checkpoint_path=str(Path(script_util.CACHE_PATH).joinpath("64x64_diffusion.pt")),
            image_size=64, class_cond=True, timestep_respacing="",
            use_fp16=False, device="cpu")
        self.assertLess(diffusion.betas[0], 1e-4 / 2)  # linear starts at 1e-4, cosine far below it

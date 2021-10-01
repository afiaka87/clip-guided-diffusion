from clip.model import ModifiedResNet, VisionTransformer
import itertools
import os
import tempfile
import unittest
from pathlib import Path

from cgd import script_util
from cgd import clip_util
from cgd import losses

import torch as th
from cgd import modules
from cgd.cgd import clip_guided_diffusion
from guided_diffusion import respace
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
        diffusion_steps = 1000
        timestep_respacing = '25'
        use_fp16 = False
        device = 'cpu'
        noise_schedule = "linear"
        dropout = 0.0
        model, diffusion = script_util.load_guided_diffusion(
            checkpoint_path=str(checkpoint_path),
            image_size=image_size,
            class_cond=class_cond,
            diffusion_steps=diffusion_steps,
            timestep_respacing=timestep_respacing,
            use_fp16=use_fp16,
            device=device,
            noise_schedule=noise_schedule,
            dropout=dropout,
        )
        self.assertIsInstance(model, th.nn.Module)
        self.assertIsInstance(diffusion, respace.SpacedDiffusion)

    def test_load_guided_diffusion_cuda(self):
        image_size = 64
        checkpoint_path = Path(script_util.CACHE_PATH).joinpath(
            f"{image_size}x{image_size}_diffusion.pt")
        class_cond = True
        diffusion_steps = 1000
        timestep_respacing = '25'
        use_fp16 = False
        device = 'cuda'
        noise_schedule = "linear"
        dropout = 0.0
        model, diffusion = script_util.load_guided_diffusion(
            checkpoint_path=str(checkpoint_path),
            image_size=image_size,
            class_cond=class_cond,
            diffusion_steps=diffusion_steps,
            timestep_respacing=timestep_respacing,
            use_fp16=use_fp16,
            device=device,
            noise_schedule=noise_schedule,
            dropout=dropout,
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

    def test_cgd_init_fails_with_default_params(self):
        try:
            samples = clip_guided_diffusion(prompts=["Loose seal."], init_image='images/photon.png',
                                            skip_timesteps=0, image_size=64, num_cutouts=1, clip_model_name="RN50", prefix_path=self.test_dir_path)
        except Exception as assertion_exception:
            self.assertEquals(assertion_exception.__class__, ValueError)

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
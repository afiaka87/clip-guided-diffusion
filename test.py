import itertools
import os
import unittest
from pathlib import Path

import torch as th
from torch.nn import functional as tf

from cgd import util as cgd_util
from cgd.cgd import CLIP_NORMALIZE, clip_guided_diffusion

# Integration tests; better than nothing at all.


class TestUtil(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def test_download_returns_target_full_path(self):
        photon_image_link_from_gh = 'https://github.com/afiaka87/clip-guided-diffusion/raw/main/images/photon.png'
        result = cgd_util.download(
            photon_image_link_from_gh, 'photon.png', root=os.getcwd())
        full_target_path_should_be = os.path.join(os.getcwd(), 'photon.png')
        self.assertEqual(result, full_target_path_should_be)

    def test_download_target_is_downloaded(self):
        photon_image_link_from_gh = 'https://github.com/afiaka87/clip-guided-diffusion/raw/main/images/photon.png'
        cgd_util.download(photon_image_link_from_gh,
                          "photon.png", root=os.getcwd())
        self.assertTrue(os.path.exists("photon.png"))
        os.remove("photon.png")

    def test_alphanumeric_dir_progress_stem(self):
        txt = "A"
        txt_min = "B"
        batch_idx = 0
        current_step = 0
        max_length = 180
        result = cgd_util.alphanumeric_dir_progress_stem(
            txt=txt,
            txt_min=txt_min,
            batch_idx=batch_idx,
            current_step=current_step,
            max_length=max_length,
        )
        expected = "A_MIN_B"
        self.assertEqual(result[0], expected)
        expected = "j_0000_i_0000"
        self.assertEqual(result[1], expected)


class TestTorchUtil(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def test_log_image(self):
        image = th.rand(3, 3, 3)
        txt = "A"
        txt_min = "B"
        batch_idx = 0
        current_step = 0
        expected_filename = os.path.join(
            os.getcwd(), "A_MIN_B/j_0000_i_0000.png")
        result_step, result_batch_idx, result_filename = cgd_util.log_image(
            image=image, base_path=os.getcwd(),
            txt=txt, txt_min=txt_min,
            current_step=current_step,
            batch_idx=batch_idx,
        )
        self.assertEqual(result_step, current_step)
        self.assertEqual(result_batch_idx, batch_idx)
        self.assertEqual(result_filename, expected_filename)

    def test_make_cutouts(self):
        cut_size = 224
        num_cutouts = 8
        cutout_size_power = 0.5
        augment_list = []
        input = th.rand(1, 3, 512, 512)
        input = CLIP_NORMALIZE(input)
        make_cutouts = cgd_util.MakeCutouts(
            cut_size=cut_size,
            num_cutouts=num_cutouts,
            cutout_size_power=cutout_size_power,
            augment_list=augment_list,
        )
        result = make_cutouts(input)
        print(result.shape)
        self.assertEqual(result.shape[0], num_cutouts)
        self.assertEqual(result.shape[1], 3)
        self.assertEqual(result.shape[2], cut_size)
        self.assertEqual(result.shape[3], cut_size)

    def test_spherical_dist_loss(self):
        x = th.rand(1, 3)
        y = th.rand(1, 3)
        result = cgd_util.spherical_dist_loss(x, y)
        x_norm = tf.normalize(x, dim=-1)
        y_norm = tf.normalize(y, dim=-1)
        expected = (x_norm - y_norm).norm(dim=-1).div(2).arcsin().pow(2).mul(2)
        self.assertEqual(result, expected)


class TestCGD(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def test_cgd_one_step_succeeds(self):
        samples = clip_guided_diffusion(
            prompt="Loose seal.", image_size=64, num_cutouts=1, clip_model_name="RN50")
        first_yielded_sample = list(itertools.islice(samples, 1))[0]
        self.assertIsNotNone(first_yielded_sample)

    def test_cgd_init_fails_with_default_params(self):
        try:
            samples = clip_guided_diffusion(prompt="Loose seal.", init_image='images/photon.png',
                                            skip_timesteps=0, image_size=64, num_cutouts=1, clip_model_name="RN50")
            first_yielded_sample = list(itertools.islice(samples, 1))[0]
        except Exception as assertion_exception:
            print(assertion_exception)
            self.assertEquals(assertion_exception.__class__, AssertionError)
        else:
            self.fail(
                "Expected an exception of type AssertionError to be thrown.")

    def test_cgd_init_succeeds_with_skip_timesteps(self):
        samples = clip_guided_diffusion(prompt="Loose seal.", init_image='images/photon.png',
                                        skip_timesteps=500, image_size=64, num_cutouts=1, clip_model_name="RN50")
        first_yielded_sample = list(itertools.islice(samples, 1))[0]
        self.assertIsNotNone(first_yielded_sample)

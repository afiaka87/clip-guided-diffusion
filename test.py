import unittest
from cgd.cgd import clip_guided_diffusion
from cgd import util as cgd_util
import itertools
import os
from pathlib import Path

import torch as th

# Integration tests; better than nothing at all.


class TestUtil(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def test_log_image_txt_with_underscores_as_dir(self):
        result = cgd_util.log_image(
            image=th.randn(3, 256, 256), base_path=os.getcwd(), txt="some text", txt_min="", batch_idx=0, current_step=0
        )
        result_path = Path(result)
        self.assertTrue(result_path.parent.name == "some_text")
        self.assertTrue(result_path.exists())

    def test_log_image_txt_and_min_with_underscores_as_dir(self):
        result = cgd_util.log_image(
            image=th.randn(3, 256, 256), base_path=os.getcwd(), txt="some text", txt_min="txet emos", batch_idx=0, current_step=0
        )
        result_path = Path(result)
        self.assertTrue(result_path.parent.name == "some_text_MIN_txet_emos")
        self.assertTrue(result_path.exists())

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


class TestCGD(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def test_cgd_one_step_succeeds(self):
        samples = clip_guided_diffusion(prompt="Loose seal.", image_size=64, num_cutouts=1, clip_model_name="RN50")
        first_yielded_sample = list(itertools.islice(samples, 1))[0]
        self.assertIsNotNone(first_yielded_sample)


    def test_cgd_init_fails_with_default_params(self):
        try:
            samples = clip_guided_diffusion(prompt="Loose seal.", init_image='images/photon.png', skip_timesteps=0, image_size=64, num_cutouts=1, clip_model_name="RN50")
            first_yielded_sample = list(itertools.islice(samples, 1))[0]
        except Exception as assertion_exception:
            print(assertion_exception)
            self.assertEquals(assertion_exception.__class__, AssertionError)
        else:
            self.fail("Expected an exception of type AssertionError to be thrown.")

    def test_cgd_init_succeeds_with_skip_timesteps(self):
        samples = clip_guided_diffusion(prompt="Loose seal.", init_image='images/photon.png', skip_timesteps=500, image_size=64, num_cutouts=1, clip_model_name="RN50")
        first_yielded_sample = list(itertools.islice(samples, 1))[0]
        self.assertIsNotNone(first_yielded_sample)
import unittest
import os
from pathlib import Path

import torch
from cgd import util as cgd_util

# Integration tests; better than nothing at all.

class TestUtil(unittest.TestCase):
    def test_log_image_saves_image(self):
        input_base_dir = 'test_output'
        txt = "boilerplate"
        txt_min = "minimized"
        combined = "boilerplate_MIN_minimized"
        image_tensor = torch.randn(3, 256, 256)
        current_step = 25
        batch_idx = 25

        os.remove(f'current.png')
        saved_output = cgd_util.log_image(image_tensor, input_base_dir, txt, txt_min, current_step, batch_idx)
        saved_output_path = Path(saved_output)
        self.assertTrue(saved_output_path.is_file())
        self.assertEqual(saved_output_path.suffix, '.png')
        self.assertEqual(saved_output_path.stem, f'j_0025_i_0025')
        self.assertEqual(saved_output_path.parent.name, combined)
        self.assertEqual(saved_output_path.parent.parent.name, input_base_dir)
        self.assertTrue(saved_output_path.exists())
        self.assertTrue(Path('current.png').exists())
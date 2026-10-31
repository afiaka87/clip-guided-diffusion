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

    def test_pooling_matches_adaptive_avg_pool2d(self):
        for side in (44, 64, 89, 128, 224, 256, 301, 512):
            image = th.rand(2, 3, side, side)
            pooled = modules.adaptive_avg_pool(image, 224)
            self.assertLess((pooled - tf.adaptive_avg_pool2d(image, 224)).abs().max().item(), 1e-5)

    @unittest.skipUnless(th.cuda.is_available(), "The rounding differences come from the CUDA backward pass.")
    def test_pooling_gradient_is_reproducible(self):
        image = th.rand(4, 3, 128, 128, device="cuda").requires_grad_()
        direction = th.randn(4, 3, 224, 224, device="cuda")
        gradients = [th.autograd.grad((modules.adaptive_avg_pool(image, 224) * direction).sum(), image)[0]
                     for _ in range(10)]
        self.assertEqual(len({gradient.cpu().numpy().tobytes() for gradient in gradients}), 1)


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

    def test_same_seed_reproduces_the_image_exactly(self):
        for name, options in (("fast", dict(sample_steps=12)), ("original", dict(timestep_respacing="ddim10"))):
            with self.subTest(sampler=name):
                first = self.generate(image_size=128, seed=3, prefix_path=f"{self.prefix}/{name}a", **options)[-1][1]
                second = self.generate(image_size=128, seed=3, prefix_path=f"{self.prefix}/{name}b", **options)[-1][1]
                self.assertEqual(Path(first).read_bytes(), Path(second).read_bytes())

    def test_512_runs_with_guidance(self):  # Needs gradient checkpointing to fit in 8 GB.
        self.assertTrue(self.generate(image_size=512, sample_steps=6)[-1][1].endswith("0005.png"))

    def test_gradient_checkpointing_keeps_the_gradient(self):
        checkpoint_path = script_util.download_guided_diffusion(image_size=64, class_cond=True)
        image = th.randn(1, 3, 64, 64, device="cuda")
        timestep = th.tensor([500.0], device="cuda")
        label = th.tensor([278], device="cuda")
        gradients = []
        for use_checkpoint in (False, True):
            model, _ = script_util.load_guided_diffusion(checkpoint_path, 64, True, device="cuda",
                                                         use_checkpoint=use_checkpoint)
            pixels = image.clone().requires_grad_()
            gradients.append(th.autograd.grad(model(pixels, timestep, y=label)[:, :3].square().sum(), pixels)[0])
        # 16 bit weights set the tolerance, as the two paths round in a different order.
        self.assertLess((gradients[0] - gradients[1]).abs().max() / gradients[0].abs().max(), 0.01)

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

    def test_init_image_without_skip_timesteps_starts_halfway(self):
        results = self.generate(image_size=64, init_image="images/photon.png", save_frequency=1)
        self.assertLess(len(results), 30)
        self.assertGreater(len(results), 5)

    def test_init_image_follows_the_requested_shape(self):
        results = self.generate(image_size=64, init_image="images/photon.png", sample_steps=6, width_offset=32)
        self.assertEqual(Image.open(results[-1][1]).size, (96, 64))

    def test_batch_with_several_prompts_guides_every_image_with_every_prompt(self):
        results = self.generate(prompts=["a red fox", "a yellow lemon:0.5"], image_size=64, sample_steps=8, batch_size=3)
        self.assertEqual(sorted(batch_idx for batch_idx, _ in results[-3:]), [0, 1, 2])

    def test_image_prompt_embeds_like_its_content(self):
        clip_model, clip_size = clip_util.load_clip("ViT-B/32", "cuda")
        fox_path = self.generate(image_size=128)[-1][1]
        embed, _ = clip_util.encode_image_prompt(fox_path, 1.0, 128, 4, "ViT-B/32", "cuda")
        top = clip_util.imagenet_top_n(embed.mean(0, keepdim=True), device="cuda", n=10, clip_model_name="ViT-B/32")
        labels = " ".join(IMAGENET_CLASSES[i] for i in top.tolist())
        self.assertIn("fox", labels)

    def test_image_prompt_accepts_non_square_and_small_images(self):
        path = f"{self.prefix}/source.png"
        for size in ((300, 150), (40, 40)):
            Image.open("images/photon.png").convert("RGB").resize(size).save(path)
            embed, weights = clip_util.encode_image_prompt(path, 1.0, 128, 4, "ViT-B/32", "cuda")
            self.assertEqual(embed.shape[0], len(weights))
            self.assertAlmostEqual(sum(weights), 1.0)

    def test_image_prompt_alone_steers_generation(self):
        fox_path = self.generate(image_size=128)[-1][1]
        result = self.generate(prompts=[], image_prompts=[fox_path], image_size=128)[-1][1]
        self.assertGreater(self.clip_similarity(result, "a photo of a fox"),
                           self.clip_similarity(result, "a photo of a car"))


class TestOutOfMemory(unittest.TestCase):
    def test_out_of_memory_raises_instead_of_ending_the_run_early(self):
        with tempfile.TemporaryDirectory() as prefix:
            with self.assertRaises(th.cuda.OutOfMemoryError):
                list(clip_guided_diffusion(prompts=["Loose seal."], image_size=64, batch_size=4096,
                                           prefix_path=prefix, save_frequency=1, progress=False))
        th.cuda.empty_cache()


class TestParsePrompt(unittest.TestCase):
    def test_a_colon_in_the_text_is_not_a_weight(self):
        self.assertEqual(script_util.parse_prompt("a sign that says: hello"), ("a sign that says: hello", 1.0))

    def test_urls_keep_their_colons(self):
        url = "https://example.com:8080/cat.png"
        self.assertEqual(script_util.parse_prompt(url), (url, 1.0))
        self.assertEqual(script_util.parse_prompt(f"{url}:0.5"), (url, 0.5))

    def test_non_finite_weights_are_text(self):
        self.assertEqual(script_util.parse_prompt("cat:nan"), ("cat:nan", 1.0))


class TestCheckParameters(unittest.TestCase):
    def assertRejected(self, message, prompts=("cat",), image_prompts=(), image_size=128, **kwargs):
        with self.assertRaisesRegex(script_util.ConfigError, message):
            script_util.check_parameters(prompts, image_prompts, image_size, **kwargs)

    def test_needs_a_prompt(self):
        self.assertRejected("at least one prompt", prompts=())

    def test_rejects_a_size_with_no_checkpoint(self):
        self.assertRejected("Choose one of", image_size=128, class_cond=False)
        self.assertRejected("Choose one of", image_size=100)

    def test_rejects_shapes_the_unet_cannot_process(self):
        self.assertRejected("multiples of 16", height_offset=8)

    def test_rejects_missing_images(self):
        self.assertRejected("No such image", image_prompts=("missing.png:0.5",))
        self.assertRejected("No such image", init_image="missing.png")

    def test_rejects_impossible_respacing(self):
        self.assertRejected("Bad --timestep_respacing", timestep_respacing="ddim999")
        self.assertRejected("Bad --timestep_respacing", timestep_respacing="fast")

    def test_rejects_skipping_the_whole_schedule(self):
        self.assertRejected("between 0 and 999", skip_timesteps=1000)
        self.assertRejected("between 0 and 49", timestep_respacing="50", skip_timesteps=50)

    def test_accepts_a_good_configuration(self):
        script_util.check_parameters(("cat",), ("images/photon.png:2",), 256, class_cond=False,
                                     timestep_respacing="ddim50", skip_timesteps=25, width_offset=32)


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


class TestCommandLine(unittest.TestCase):
    COMMAND = [sys.executable, "-W", "ignore", "-m", "cgd.cgd"]
    OPTIONS = ["-size", "64", "-n", "6", "-clip", "RN50", "-q"]

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cwd = Path(self.temp.name)
        self.repo = Path(__file__).parent

    def run_cgd(self, *args, env=None):
        environment = {**os.environ, "PYTHONPATH": str(self.repo), **(env or {})}
        return subprocess.run([*self.COMMAND, *args], cwd=self.cwd, env=environment, capture_output=True, text=True)

    def frames(self, directory="Loose_seal"):
        return sorted((self.cwd / "outputs" / directory / "00").glob("*.png"))

    def test_positional_prompt_reports_the_saved_image(self):
        result = self.run_cgd("Loose seal.", *self.OPTIONS)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Saved outputs/Loose_seal/00/0005.png", result.stdout)
        self.assertTrue((self.cwd / "current.png").exists())

    def test_without_a_prompt_it_explains_itself(self):
        result = self.run_cgd("-q")
        self.assertEqual(result.returncode, 2)
        self.assertIn("at least one prompt", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_weights_that_cancel_out_are_explained(self):
        result = self.run_cgd("cat:1|dog:-1", *self.OPTIONS)
        self.assertEqual(result.returncode, 2)
        self.assertIn("must not sum to 0", result.stderr)

    def test_prefix_can_be_a_new_nested_directory(self):
        result = self.run_cgd("Loose seal.", *self.OPTIONS, "-dir", "a/b/c")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.cwd / "a/b/c/Loose_seal/00/0005.png").exists())

    def test_out_of_memory_fails_the_process_with_advice(self):
        result = self.run_cgd("Loose seal.", *self.OPTIONS, "-bs", "4096")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Ran out of GPU memory", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    @unittest.skipUnless(shutil.which("ffprobe"), "needs ffmpeg")
    def test_gif_holds_every_saved_frame_even_with_gaps_and_stale_frames(self):
        stale = self.run_cgd("Loose seal.", *self.OPTIONS[:2], "-n", "8", "-clip", "RN50", "-q", "-freq", "1")
        self.assertEqual(stale.returncode, 0, stale.stderr)
        self.assertEqual(len(self.frames()), 8)

        result = self.run_cgd("Loose seal.", *self.OPTIONS, "-freq", "2", "-gif")
        self.assertEqual(result.returncode, 0, result.stderr)
        gif = self.cwd / "outputs/Loose_seal/00.gif"
        count = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                                "stream=nb_read_frames", "-of", "csv=p=0", str(gif)], capture_output=True, text=True)
        self.assertEqual(count.stdout.strip(), "4")  # Steps 0, 2, 4 and the final step 5.
        self.assertEqual(len(self.frames()), 4)  # The older run's frames were not this run's to delete.

    @unittest.skipUnless(shutil.which("ffmpeg"), "needs ffmpeg")
    def test_video_replaces_the_frames(self):
        result = self.run_cgd("Loose seal.", *self.OPTIONS, "-mp4")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertGreater((self.cwd / "outputs/Loose_seal/00.mp4").stat().st_size, 0)
        self.assertEqual(self.frames(), [])

    def test_frames_survive_when_ffmpeg_is_unavailable(self):
        bin_dir = Path(sys.executable).parent
        result = self.run_cgd("Loose seal.", *self.OPTIONS, "-gif", env={"PATH": str(bin_dir)})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ffmpeg not found", result.stdout)
        self.assertEqual(len(self.frames()), 6)
        self.assertIn("Saved outputs/Loose_seal/00/0005.png", result.stdout)

    def test_wandb_logging_runs_offline(self):
        result = self.run_cgd("Loose seal.", *self.OPTIONS, "-bs", "2", "-proj", "cgd_test",
                              env={"WANDB_MODE": "offline", "WANDB_DIR": str(self.cwd)})
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_image_prompt_runs_are_named_after_the_image(self):
        image = self.repo / "images/photon.png"
        result = self.run_cgd("-imgs", f"{image}:2", *self.OPTIONS)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.frames("photon")), 2)  # Steps 0 and the final step 5.


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

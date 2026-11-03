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
from cgd import cgd
from cgd import modern
from cgd import modules
from cgd import secondary
from cgd import upscale
from cgd.cgd import clip_guided_diffusion, judged_candidates
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
        for name, options in (("fast", dict(sample_steps=12)), ("original", dict(timestep_respacing="ddim10")),
                              ("ensemble", dict(sample_steps=12, clip_model_name="ViT-B/32+PE-Core-B-16")),
                              ("secondary", dict(sample_steps=12, secondary_guidance=True)),
                              ("er-sde", dict(sample_steps=12, solver="er-sde")), ("limit", dict(sample_steps=12, guidance_limit=0.99)),
                              ("pag", dict(sample_steps=12, pag_scale=1.0)), ("aesthetic", dict(sample_steps=12, aesthetic_scale=0.2))):
            with self.subTest(sampler=name):
                first = self.generate(image_size=128, seed=3, prefix_path=f"{self.prefix}/{name}a", **options)[-1][1]
                second = self.generate(image_size=128, seed=3, prefix_path=f"{self.prefix}/{name}b", **options)[-1][1]
                self.assertEqual(Path(first).read_bytes(), Path(second).read_bytes())

    def test_512_runs_with_guidance(self):  # Needs gradient checkpointing to fit in 8 GB.
        self.assertTrue(self.generate(image_size=512, sample_steps=6)[-1][1].endswith("0005.png"))

    def test_unconditional_checkpoints_get_their_own_defaults(self):
        self.assertTrue(self.generate(image_size=256, class_cond=False)[-1][1].endswith("0059.png"))

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

    def test_perceptual_init_loss_runs(self):
        results = self.generate(image_size=64, init_image="images/32K_HUHD_Mushroom.png", init_scale=1000, sample_steps=6)
        self.assertTrue(results[-1][1].endswith(".png"))
        self.assertTrue(os.path.isfile(script_util.download_vgg16()))

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


class TestOpenClip(unittest.TestCase):
    """Real open_clip checkpoints: the smallest of each family, which differ in tower and tokenizer."""
    PERCEPTION_ENCODER = "PE-Core-B-16"
    SIGLIP = "ViT-L-16-SigLIP2-256"
    PROMPT = "a photograph of a red fox in the snow"

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.prefix = self.test_dir.name
        self.addCleanup(self.test_dir.cleanup)

    def generate(self, **kwargs):
        options = dict(prompts=[self.PROMPT], prefix_path=self.prefix, save_frequency=1000, progress=False,
                       device="cuda", image_size=64, sample_steps=6)
        options.update(kwargs)
        return list(clip_guided_diffusion(**options))

    def test_every_listed_model_is_published_by_open_clip(self):
        import open_clip
        published = set(open_clip.list_pretrained())
        for model_name, weights in clip_util.OPEN_CLIP_MODELS.items():
            self.assertIn((model_name, weights), published)

    def test_only_the_vision_tower_is_on_the_gpu(self):
        clip_model, _ = clip_util.load_clip(self.PERCEPTION_ENCODER, "cuda")
        for name, parameter in clip_model.named_parameters():
            self.assertEqual(parameter.device.type, "cuda" if name.startswith("visual") else "cpu", name)

    def test_skipping_random_init_still_loads_every_weight(self):
        import open_clip
        expected = open_clip.create_model(self.PERCEPTION_ENCODER, pretrained="meta",
                                          cache_dir=clip_util.CLIP_CACHE_PATH).state_dict()
        loaded = clip_util.load_clip.__wrapped__(self.PERCEPTION_ENCODER, "cpu")[0].state_dict()
        self.assertEqual(expected.keys(), loaded.keys())
        for name, tensor in expected.items():
            self.assertTrue(th.equal(tensor, loaded[name]), name)

    def test_preprocess_uses_the_normalization_each_model_was_trained_with(self):
        white = th.ones(1, 3, 8, 8)
        perception_encoder = clip_util.load_clip.__wrapped__(self.PERCEPTION_ENCODER, "cpu")[0]
        self.assertTrue(th.equal(clip_util.preprocess(perception_encoder, white), white))
        openai = clip_util.load_clip.__wrapped__("RN50", "cpu")[0]
        self.assertTrue(th.equal(clip_util.preprocess(openai, white), clip_util.CLIP_NORMALIZE(white)))

    def test_text_and_images_land_in_the_same_space(self):
        fox_path = self.generate(image_size=128, sample_steps=30)[-1][1]
        for model_name in (self.PERCEPTION_ENCODER, self.SIGLIP):
            for name, embed in (("text", clip_util.encode_text_prompt(self.PROMPT, 1.0, model_name, "cuda")[0]),
                                ("image", clip_util.encode_image_prompt(fox_path, 1.0, 128, 4, model_name, "cuda")[0])):
                top = clip_util.imagenet_top_n(embed.mean(0, keepdim=True), device="cuda", n=10, clip_model_name=model_name)
                self.assertIn("fox", " ".join(IMAGENET_CLASSES[i] for i in top.tolist()), (model_name, name))

    def test_gradient_checkpointing_keeps_the_gradient(self):
        clip_model, clip_size = clip_util.load_clip(self.PERCEPTION_ENCODER, "cuda")

        def gradient():
            image = th.linspace(0, 1, 3 * clip_size ** 2, device="cuda").view(1, 3, clip_size, clip_size).requires_grad_()
            embed = clip_model.encode_image(clip_util.preprocess(clip_model, image))
            return th.autograd.grad(embed.square().sum(), image)[0]

        checkpointed = gradient()
        self.assertGreater(checkpointed.abs().max().item(), 0)
        clip_model.visual.set_grad_checkpointing(False)
        self.addCleanup(clip_model.visual.set_grad_checkpointing, True)
        self.assertLess((checkpointed - gradient()).abs().max().item(), 1e-4 * checkpointed.abs().max().item())

    def test_same_seed_reproduces_the_image_exactly(self):
        for model_name in (self.PERCEPTION_ENCODER, self.SIGLIP):
            first = self.generate(clip_model_name=model_name, seed=3, prefix_path=f"{self.prefix}/a")[-1][1]
            second = self.generate(clip_model_name=model_name, seed=3, prefix_path=f"{self.prefix}/b")[-1][1]
            self.assertEqual(Path(first).read_bytes(), Path(second).read_bytes(), model_name)

    def test_backward_pass_is_reproducible(self):
        # The fused attention kernels are not. This is the model that shows it fastest.
        clip_model, clip_size = clip_util.load_clip("PE-Core-L-14-336", "cuda")
        generator = th.Generator("cuda").manual_seed(0)
        images = th.rand(4, 3, clip_size, clip_size, device="cuda", generator=generator)

        def gradient():
            inputs = images.clone().requires_grad_()
            embed = clip_model.encode_image(clip_util.preprocess(clip_model, inputs))
            return th.autograd.grad(embed.square().sum() * 100, inputs)[0]

        first = gradient()
        for _ in range(3):
            self.assertTrue(th.equal(first, gradient()))

    def test_a_gradient_that_overflows_fp16_does_not_black_out_the_image(self):
        for options in ({}, {"clip_model_name": self.PERCEPTION_ENCODER},
                        {"timestep_respacing": "10"}, {"timestep_respacing": "10", "image_size": 128},
                        {"timestep_respacing": "10", "clip_guidance_scale": 1e30}):  # All of it overflows.
            with self.subTest(options):
                path = self.generate(**{"clip_guidance_scale": 1e7, **options})[-1][1]
                self.assertGreater(np.asarray(Image.open(path)).std(), 5)

    def test_clip_fp16_only_changes_the_vision_tower(self):
        self.addCleanup(clip_util.load_clip.cache_clear)
        results = self.generate(clip_model_name=self.PERCEPTION_ENCODER, clip_fp16=True)
        self.assertTrue(Path(results[-1][1]).exists())
        clip_model, _ = clip_util.load_clip(self.PERCEPTION_ENCODER, "cuda", True)
        for name, parameter in clip_model.named_parameters():
            self.assertEqual(parameter.dtype, th.float16 if name.startswith("visual") else th.float32, name)

    def test_unknown_model_is_rejected_and_every_listed_one_is_not(self):
        with self.assertRaisesRegex(script_util.ConfigError, "Unknown CLIP model"):
            clip_util.check_clip_model("ViT-Z/99")
        for model_name in clip_util.CLIP_MODEL_NAMES:
            clip_util.check_clip_model(model_name)
        clip_util.check_clip_model(__file__)  # Any checkpoint file on disk.


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

    def environment_without_allocator_setting(self):
        names = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
        return {name: value for name, value in os.environ.items() if name not in names}

    def test_cuda_allocator_setting_leaves_the_users_own_alone(self):
        code = "import os, cgd; print(os.environ.get('PYTORCH_ALLOC_CONF'), os.environ.get('PYTORCH_CUDA_ALLOC_CONF'))"
        own = "max_split_size_mb:64"
        for given, expected in (({"PYTORCH_ALLOC_CONF": own}, f"{own} None"), ({"PYTORCH_CUDA_ALLOC_CONF": own}, f"None {own}")):
            result = subprocess.run([sys.executable, "-c", code], env={**self.environment_without_allocator_setting(), **given},
                                    capture_output=True, text=True)
            self.assertEqual(result.stdout.strip(), expected, result.stderr)

    def test_cuda_allocator_setting_reaches_the_allocator_without_a_warning(self):
        # Only expandable segments can serve the third block from the memory of the first two.
        code = ("import cgd, torch as th\n"
                "first, second = (th.empty(100 << 20, dtype=th.uint8, device='cuda') for _ in range(2))\n"
                "del first, second\n"
                "third = th.empty(200 << 20, dtype=th.uint8, device='cuda')\n"
                "print(th.cuda.memory_reserved() >> 20)\n")
        result = subprocess.run([sys.executable, "-c", code], env=self.environment_without_allocator_setting(), capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), "200", result.stderr)
        self.assertNotIn("deprecated", result.stderr)

    def test_unknown_clip_model_is_explained(self):
        result = self.run_cgd("Loose seal.", "-clip", "ViT-Z/99", "-q")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Unknown CLIP model", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_open_clip_model_runs_without_noise_from_its_libraries(self):
        result = self.run_cgd("Loose seal.", *self.OPTIONS, "-clip", "PE-Core-B-16", "-fp16")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Saved outputs/Loose_seal/00/0005.png", result.stdout)
        self.assertNotIn("HF Hub", result.stderr)

    def test_prefix_can_be_a_new_nested_directory(self):
        result = self.run_cgd("Loose seal.", *self.OPTIONS, "-dir", "a/b/c")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.cwd / "a/b/c/Loose_seal/00/0005.png").exists())

    def test_out_of_memory_fails_the_process_with_advice(self):
        result = self.run_cgd("Loose seal.", *self.OPTIONS, "-bs", "4096")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Ran out of GPU memory", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_both_of_pytorchs_memory_errors_count_as_out_of_memory(self):
        self.assertTrue(cgd.is_out_of_memory(th.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 14.00 MiB.")))
        self.assertTrue(cgd.is_out_of_memory(th.AcceleratorError("CUDA error: out of memory", 2)))
        self.assertFalse(cgd.is_out_of_memory(RuntimeError("mat1 and mat2 shapes cannot be multiplied")))

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

    def test_upscale_saves_a_larger_copy_beside_the_image(self):
        result = self.run_cgd("Loose seal.", *self.OPTIONS, "-up")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Saved outputs/Loose_seal/00/0005_x4.png", result.stdout)
        self.assertEqual(Image.open(self.frames()[-1]).size, (256, 256))

    def test_upscaled_copy_survives_an_animation_replacing_the_frames(self):
        result = self.run_cgd("Loose seal.", *self.OPTIONS, "-up", "-gif")
        if "ffmpeg not found" in result.stdout:
            self.skipTest("needs ffmpeg")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([path.name for path in self.frames()], ["0005_x4.png"])

    def test_candidates_report_the_winning_seed(self):
        result = self.run_cgd("Loose seal.", *self.OPTIONS, "-best", "2", "-seed", "7")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(result.stdout, r"Saved outputs/seed_[78]/Loose_seal/00/0005.png")

    def test_a_bad_candidate_count_is_explained(self):
        result = self.run_cgd("Loose seal.", *self.OPTIONS, "-best", "0")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--candidates", result.stderr)

    def test_fast_sampler_options_reach_the_sampler(self):
        result = self.run_cgd("Loose seal.", *self.OPTIONS, "-dt", "0.995", "-gl", "0.99", "-gs", "sine", "-loss", "contrastive", "-view", "detached",
                              "-pag", "0.5", "-contrast", "0.5", "-class", "1", "-sec", "-solver", "er-sde", "-aes", "0.2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Saved outputs/Loose_seal/00/0005.png", result.stdout)

    def test_a_size_with_no_checkpoint_is_explained(self):
        result = self.run_cgd("Loose seal.", "-size", "100", "-q")
        self.assertEqual(result.returncode, 2)
        self.assertIn("No class conditional checkpoint", result.stderr)

    def test_backbone_runs_at_any_multiple_of_eight(self):
        result = self.run_cgd("Loose seal.", "-bb", "sd-turbo", "-size", "136", "-iters", "2", "-clip", "ViT-B/32", "-q")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Saved outputs/Loose_seal/00/0002.png", result.stdout)
        self.assertEqual(Image.open(self.frames()[-1]).size, (136, 136))

    def test_unknown_backbone_is_rejected(self):
        result = self.run_cgd("Loose seal.", "-bb", "dall-e", "-q")
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid choice", result.stderr)


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


class TestGuidanceTricks(unittest.TestCase):
    PROMPT = "a photograph of a red fox in the snow"

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.prefix = self.test_dir.name
        self.addCleanup(self.test_dir.cleanup)

    def generate(self, **kwargs):
        options = dict(prompts=[self.PROMPT], image_size=128, prefix_path=self.prefix, save_frequency=1000, progress=False, device="cuda")
        return list(clip_guided_diffusion(**{**options, **kwargs}))[-1][1]

    def test_dynamic_threshold_leaves_an_in_range_image_alone(self):
        image = th.rand(2, 3, 16, 16) * 2 - 1
        self.assertTrue(th.equal(sampling.dynamic_threshold(image, 0.995), image))

    def test_dynamic_threshold_rescales_only_the_image_past_the_range(self):
        fine, loud = th.rand(1, 3, 32, 32) - 0.5, (th.rand(1, 3, 32, 32) - 0.5) * 8
        result = sampling.dynamic_threshold(th.cat([fine, loud]), 0.99)
        self.assertTrue(th.equal(result[0], fine[0]))
        self.assertLessEqual(result[1].abs().max().item(), 1.0)
        self.assertGreater(result[1].abs().max().item(), 0.99)  # Rescaled rather than flattened.

    def test_limit_guidance_keeps_an_update_that_stays_in_range(self):
        x0, update = th.full((1, 3, 8, 8), 0.2), th.full((1, 3, 8, 8), 0.3)
        self.assertTrue(th.equal(sampling.limit_guidance(x0, update, 0.99), x0 + update))

    def test_limit_guidance_holds_back_only_as_much_as_it_must(self):
        x0, update = th.full((2, 3, 8, 8), 0.5), th.full((2, 3, 8, 8), 1.0)
        update[1] = 0.1
        result = sampling.limit_guidance(x0, update, 0.99)
        self.assertAlmostEqual(result[0].max().item(), 1.0, delta=0.01)
        self.assertLessEqual(result[0].max().item(), 1.0)
        self.assertAlmostEqual(result[1].max().item(), 0.6, places=5)  # The other image of the batch is untouched.

    def test_guidance_schedules_keep_the_total_guidance(self):
        for schedule in sampling.GUIDANCE_SCHEDULES:
            weights = [sampling.schedule_weight(schedule, (i + 0.5) / 1000) for i in range(1000)]
            self.assertAlmostEqual(sum(weights) / 1000, 1.0, places=3)

    def test_clip_view_x0_changes_the_image(self):
        blend, view = self.generate(sample_steps=10), self.generate(sample_steps=10, clip_view="x0", prefix_path=f"{self.prefix}/x0")
        self.assertNotEqual(Path(blend).read_bytes(), Path(view).read_bytes())

    def test_unknown_options_are_explained(self):
        for options in (dict(clip_view="sideways"), dict(clip_loss="hinge"), dict(solver="euler"), dict(aesthetic_scale=-1)):
            with self.assertRaises(script_util.ConfigError):
                self.generate(image_size=64, sample_steps=2, **options)

    def test_perturbed_attention_changes_the_prediction_only_while_active(self):
        for size in (64, 128):
            model, _ = script_util.load_guided_diffusion(script_util.download_guided_diffusion(size, True), size, True, device="cuda")
            x, t, y = th.randn(1, 3, size, size, device="cuda"), th.tensor([500.0], device="cuda"), th.tensor([278], device="cuda")
            with th.no_grad():
                before = model(x, t, y=y)
                with sampling.perturbed_attention(model):
                    perturbed = model(x, t, y=y)
                after = model(x, t, y=y)
            self.assertTrue(th.equal(before, after))
            self.assertGreater((before - perturbed).abs().mean().item(), 1e-4)

    def test_contrastive_loss_prefers_the_prompt_to_the_negatives(self):
        prompt, negatives = tf.normalize(th.randn(1, 512), dim=-1), tf.normalize(th.randn(50, 512), dim=-1)
        on_prompt = losses.contrastive_loss(prompt.view(1, 1, 1, 512), prompt, negatives)
        on_negative = losses.contrastive_loss(negatives[:1].view(1, 1, 1, 512), prompt, negatives, exclude=1)
        self.assertLess(on_prompt.item(), on_negative.item())

    def test_ensemble_gradient_is_the_mean_of_its_members(self):
        members = ("ViT-B/32", "PE-Core-B-16")
        images = th.randn(1, 3, 224, 224, device="cuda").mul(0.5).clamp(-1, 1).requires_grad_()
        gradients = [th.autograd.grad(clip_util.ClipGuide(name, [self.PROMPT], (), 224, 1, "cuda").loss(images), images)[0] for name in members]
        guide = clip_util.ClipGuide("+".join(members), [self.PROMPT], (), 224, 1, "cuda")
        ensemble = th.autograd.grad(guide.loss(images), images)[0]
        # Not exact: halving the loss changes what underflows on the way back through a 16 bit tower.
        expected = (gradients[0] + gradients[1]) / 2
        self.assertGreater(tf.cosine_similarity(ensemble.flatten(), expected.flatten(), dim=0).item(), 0.999)

    def test_aesthetic_rating_matches_the_published_head(self):
        guide = clip_util.ClipGuide("ViT-B/32", [self.PROMPT], (), 224, 1, "cuda", aesthetic_scale=1.0)
        images = th.rand(2, 3, 224, 224, device="cuda")
        embeds = tf.normalize(guide.aesthetic_model.encode_image(clip_util.preprocess(guide.aesthetic_model, images)).float(), dim=-1)
        expected = embeds @ guide.aesthetic_head["linear.weight"].T * 512 ** 0.5 + guide.aesthetic_head["linear.bias"]
        self.assertEqual(guide.aesthetic(images).shape, (2,))
        self.assertLess((guide.aesthetic(images) - expected[:, 0]).abs().max().item(), 1e-2)

    def test_following_the_aesthetic_gradient_raises_the_rating(self):
        guide = clip_util.ClipGuide("ViT-B/32", [self.PROMPT], (), 224, 1, "cuda", aesthetic_scale=1.0)
        images = th.rand(1, 3, 224, 224, device="cuda").mul(2).sub(1).requires_grad_()
        pixels = images.detach().add(1).div(2)
        before = guide.aesthetic(pixels).item()
        for _ in range(10):
            gradient = th.autograd.grad(-guide.aesthetic(images.add(1).div(2)).sum(), images)[0]
            images = (images - 0.02 * gradient.sign()).clamp(-1, 1).detach().requires_grad_()
        self.assertGreater(guide.aesthetic(images.detach().add(1).div(2)).item(), before + 0.5)

    def test_aesthetic_term_adds_to_the_loss_and_its_gradient(self):
        images = th.rand(1, 3, 224, 224, device="cuda").mul(2).sub(1).requires_grad_()
        for model in ("ViT-B/32", "ViT-B/32+PE-Core-B-16"):
            th.manual_seed(0)
            plain = clip_util.ClipGuide(model, [self.PROMPT], (), 224, 1, "cuda").loss(images)
            th.manual_seed(0)
            guide = clip_util.ClipGuide(model, [self.PROMPT], (), 224, 1, "cuda", aesthetic_scale=0.5)
            combined = guide.loss(images)
            rating = guide.aesthetic(images.add(1).div(2)).sum()
            self.assertLess(abs(combined.item() - (plain.item() - 0.5 * rating.item())), 1e-2)
            self.assertGreater(th.autograd.grad(combined, images)[0].abs().sum().item(), 0)

    def test_cutout_rating_keeps_the_images_of_a_batch_apart(self):
        guide = clip_util.ClipGuide("ViT-B/32", [self.PROMPT], (), 224, 3, "cuda", aesthetic_scale=1.0)
        images = th.rand(2, 3, 224, 224, device="cuda") * th.tensor([1.0, 0.2], device="cuda").view(2, 1, 1, 1)
        rating = guide.aesthetic(images)  # A cutout of a 224px image at this fraction is the whole image.
        self.assertGreater((rating[0] - rating[1]).abs().item(), 0.01)
        self.assertLess((guide._cutout_rating(images) - rating).abs().max().item(), 1e-3)

    def test_cutout_rating_averages_crops_of_a_larger_image(self):
        guide = clip_util.ClipGuide("ViT-B/32", [self.PROMPT], (), 512, 4, "cuda", min_cut_fraction=0.7, aesthetic_scale=1.0)
        images = th.rand(2, 3, 512, 512, device="cuda")
        th.manual_seed(0)
        crops = guide.aesthetic_cutouts(images)
        self.assertEqual(crops.shape, (8, 3, 224, 224))
        th.manual_seed(0)
        rating = guide._cutout_rating(images)
        self.assertEqual(rating.shape, (2,))
        self.assertLess((rating - guide.aesthetic(crops).view(4, 2).mean(0)).abs().max().item(), 1e-3)

    def test_aesthetic_scale_changes_the_image(self):
        plain, rated = self.generate(sample_steps=10), self.generate(sample_steps=10, aesthetic_scale=1.0, prefix_path=f"{self.prefix}/rated")
        self.assertNotEqual(Path(plain).read_bytes(), Path(rated).read_bytes())

    def test_ensemble_runs_through_the_pipeline(self):
        self.assertTrue(self.generate(clip_model_name="ViT-B/32+PE-Core-B-16", sample_steps=6).endswith("0005.png"))

    def test_score_ranks_the_matching_prompt_first(self):
        image = tvf.to_tensor(Image.open(self.generate()).convert("RGB")).unsqueeze(0).mul(2).sub(1)
        fox = clip_util.ClipGuide(cgd.JUDGE_MODELS, [self.PROMPT], device="cpu").score(image)
        ramen = clip_util.ClipGuide(cgd.JUDGE_MODELS, ["a bowl of ramen on a wooden table"], device="cpu").score(image)
        self.assertGreater(fox.item(), ramen.item())

    def test_candidates_keep_the_image_the_judges_prefer(self):
        fox = self.generate()

        def run(seed, prefix):
            Path(prefix).mkdir(parents=True)
            path = str(Path(prefix) / "0000.png")
            if seed == 1:
                shutil.copy(fox, path)
            else:
                Image.new("RGB", (128, 128), "gray").save(path)
            yield 0, path

        winner = judged_candidates(run, 3, 0, self.prefix, [self.PROMPT], ())
        self.assertEqual(winner, {0: [str(Path(self.prefix) / "seed_1" / "0000.png")]})

    def test_er_sde_step_keeps_the_marginal_of_a_known_image(self):
        image, noise = th.full((1, 1000, 1000), 0.5), th.randn(1, 1000, 1000)
        alpha, sigma, alpha_next, sigma_next = 0.6, 0.8, 0.8, 0.6
        result = sampling.er_sde_step(alpha * image + sigma * noise, image, image, None, alpha, sigma, alpha_next, sigma_next)
        self.assertAlmostEqual(result.mean().item(), alpha_next * 0.5, delta=0.01)
        self.assertAlmostEqual(result.std().item(), sigma_next, delta=0.01)

    def test_er_sde_generates_the_prompt(self):
        image = tvf.to_tensor(Image.open(self.generate(solver="er-sde")).convert("RGB")).unsqueeze(0).mul(2).sub(1)
        self.assertGreater(image.std().item(), 0.1)
        fox = clip_util.ClipGuide("ViT-B/32", [self.PROMPT], device="cuda").score(image.cuda())
        ramen = clip_util.ClipGuide("ViT-B/32", ["a bowl of ramen on a wooden table"], device="cuda").score(image.cuda())
        self.assertGreater(fox.item(), ramen.item())


class TestSecondaryModel(unittest.TestCase):
    def test_denoises_a_real_image(self):
        model = secondary.load_secondary("cuda")
        image = tvf.to_tensor(Image.open("images/photon.png").convert("RGB").resize((128, 128))).unsqueeze(0).cuda().mul(2).sub(1)
        alpha, sigma = 0.8, 0.6
        noisy = alpha * image + sigma * th.randn_like(image)
        with th.no_grad():
            denoised = model(noisy, alpha, sigma)
        self.assertLess((denoised - image).square().mean(), 0.5 * (noisy / alpha - image).square().mean())

    def test_rejects_weights_that_do_not_match_the_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / secondary.WEIGHTS_FILENAME).write_bytes(b"not the weights")
            with self.assertRaisesRegex(RuntimeError, "sha256"):
                secondary.load_secondary("cpu", directory)

    def test_guides_the_sampler_without_a_gradient_through_the_u_net(self):
        with tempfile.TemporaryDirectory() as prefix:
            th.cuda.reset_peak_memory_stats()
            results = list(clip_guided_diffusion(prompts=["a red fox"], image_size=256, sample_steps=8, secondary_guidance=True,
                                                 prefix_path=prefix, save_frequency=1000, progress=False, device="cuda"))
            self.assertTrue(results[-1][1].endswith("0007.png"))
            self.assertLess(th.cuda.max_memory_allocated(), 3 * 2 ** 30)  # The U-Net path takes over 5 GiB.


class TestUpscale(unittest.TestCase):
    def image(self, size):
        pixels = tvf.to_tensor(Image.open("images/photon.png").convert("RGB").resize((size, size)))
        return pixels.unsqueeze(0).cuda().mul(2).sub(1)

    def test_output_is_four_times_larger_and_downsamples_to_the_input(self):
        image = self.image(128)
        large = upscale.upscale(upscale.load_upscaler("cuda"), image)
        self.assertEqual(large.shape[-2:], (512, 512))
        self.assertLess((tf.avg_pool2d(large, upscale.SCALE) - image).abs().mean().item(), 0.1)

    def test_tiling_matches_the_whole_image(self):
        image = self.image(300)
        model = upscale.load_upscaler("cuda")
        with th.no_grad():
            whole = model(image.add(1).div(2).half()).float().mul(2).sub(1).clamp(-1, 1)
        tiled = upscale.upscale(model, image)
        self.assertLess((whole - tiled).abs().mean().item(), 0.01)
        seam = upscale.TILE_SIZE * upscale.SCALE
        self.assertLess((whole - tiled)[..., seam - 2:seam + 2, :].abs().mean().item(), 0.02)

    def test_upscale_file_writes_beside_the_image(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "0001.png")
            Image.open("images/photon.png").convert("RGB").resize((64, 64)).save(path)
            output = upscale.upscale_file(upscale.load_upscaler("cuda"), path, "cuda")
            self.assertEqual(Path(output).name, "0001_x4.png")
            self.assertEqual(Image.open(output).size, (256, 256))


class TestModernBackbones(unittest.TestCase):
    PROMPT = "a photograph of a red fox in the snow"

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.prefix = self.test_dir.name
        self.addCleanup(self.test_dir.cleanup)

    def generate(self, **kwargs):
        options = dict(prompts=[self.PROMPT], image_size=256, clip_model_name="ViT-B/32", iterations=4, prefix_path=self.prefix,
                       save_frequency=1000, progress=False, device="cuda")
        return list(modern.generate(**{**options, **kwargs}))

    def test_noise_penalty_is_lowest_on_the_prior(self):
        z = th.randn(1, 4, 64, 64)
        z = z / z.norm() * (z.numel() - 1) ** 0.5
        self.assertLess(modern.noise_penalty(z), modern.noise_penalty(z * 1.1))
        self.assertLess(modern.noise_penalty(z), modern.noise_penalty(z * 0.9))

    def test_optimising_the_noise_lowers_the_clip_loss_and_keeps_the_best(self):
        model = modern.load_backbone("sd-turbo", self.PROMPT, "cuda")
        guide = clip_util.ClipGuide("ViT-B/32", [self.PROMPT], (), 256, 4, "cuda")
        th.manual_seed(0)
        shape = model.latent_shape(256, 256)
        start = th.randn(shape, device="cuda")
        with th.backends.cudnn.flags(enabled=True, deterministic=True):  # As generate() runs it. Without this, this size varies run to run.
            steps = list(modern.optimise_noise(model, guide, start, iterations=10, progress=False))
        values = [step["loss"] for step in steps]
        self.assertLess(min(values[1:]), values[0])
        self.assertAlmostEqual(steps[-1]["best_z"].norm().item(), start.norm().item(), delta=0.1 * start.norm().item())
        best = steps[values.index(min(values))]
        self.assertTrue(th.equal(steps[-1]["best_z"], best["best_z"]))

    def test_same_seed_reproduces_the_image_exactly(self):
        for name, options in (("tiny decoder", dict()), ("full decoder", dict(tiny_decoder=False)), ("aesthetic", dict(aesthetic_scale=0.2))):
            with self.subTest(name):
                first = self.generate(seed=3, prefix_path=f"{self.prefix}/{name}a", **options)[-1][1]
                second = self.generate(seed=3, prefix_path=f"{self.prefix}/{name}b", **options)[-1][1]
                self.assertEqual(Path(first).read_bytes(), Path(second).read_bytes())

    def test_a_different_seed_changes_the_image(self):
        first = self.generate(seed=3, prefix_path=f"{self.prefix}/a")[-1][1]
        second = self.generate(seed=4, prefix_path=f"{self.prefix}/b")[-1][1]
        self.assertNotEqual(Path(first).read_bytes(), Path(second).read_bytes())

    def test_zero_iterations_is_the_plain_model(self):
        results = self.generate(iterations=0)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0][1].endswith("0000.png"))

    def test_saves_the_frames_it_was_asked_for_and_the_final_image(self):
        names = [Path(path).name for _, path in self.generate(save_frequency=2)]
        self.assertEqual(names, ["0000.png", "0002.png", "0004.png"])

    def test_final_image_follows_the_requested_shape(self):
        path = self.generate(width_offset=64, iterations=1)[-1][1]
        self.assertEqual(Image.open(path).size, (320, 256))

    def test_batch_yields_distinct_images(self):
        results = self.generate(batch_size=2, iterations=1)
        first, second = [path for _, path in results if path.endswith("0001.png")]
        self.assertNotEqual(Path(first).read_bytes(), Path(second).read_bytes())

    def test_full_decoder_runs(self):
        self.assertTrue(self.generate(tiny_decoder=False, iterations=2)[-1][1].endswith("0002.png"))

    def test_several_render_steps_run(self):
        self.assertTrue(self.generate(render_steps=4, iterations=1)[-1][1].endswith("0001.png"))

    def test_image_prompt_alone_runs(self):
        results = self.generate(prompts=[], image_prompts=["images/photon.png"], iterations=1)
        self.assertTrue(results[-1][1].endswith("0001.png"))

    def test_the_model_is_only_told_what_to_draw(self):
        self.assertEqual(modern.conditioning_text(["a fox:2", "a hat", "blurry:-1"]), "a fox, a hat")

    def test_bad_configurations_are_explained(self):
        for options in (dict(backbone="dall-e"), dict(image_size=100), dict(iterations=-1), dict(render_steps=0),
                        dict(save_frequency=0), dict(prompts=[]), dict(clip_model_name="ViT-Z/99")):
            with self.subTest(options=options), self.assertRaises(script_util.ConfigError):
                self.generate(**options)

    def test_sdxl_backbones_run_with_the_tiny_decoder(self):
        for backbone in ("vega", "sdxl-lightning"):
            with self.subTest(backbone=backbone):
                path = self.generate(backbone=backbone, image_size=512, iterations=1, prefix_path=f"{self.prefix}/{backbone}")[-1][1]
                self.assertEqual(Image.open(path).size, (512, 512))
                self.assertGreater(tvf.to_tensor(Image.open(path).convert("RGB")).std().item(), 0.05)

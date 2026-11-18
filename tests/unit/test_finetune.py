"""The fine-tune's model, noise levels and loader, on a small U-Net of the real blocks and shards written here.

The U-Net is `guided_diffusion`'s, two levels deep with attention at both, and its zero-initialised layers are
given random weights as a trained checkpoint has. The datasets are a few shards of generated images in the
layout of the real ones, read through the same index and embedding files.
"""

import io
import json
import sys
import tarfile
import time
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch as th
from guided_diffusion.unet import AttentionBlock, QKVAttentionLegacy, UNetModel
from PIL import Image

from cgd import sampler
from cgd.adm import denoise, guidance_loss
from cgd.conditioned import AdapterConfig, ConditionedUNet, FusedAttention, adapter_checkpoint, load_adapter, trainable
from cgd.finetuned import predict_image
from finetune import cond, data, distill, noise, parallel, prior, train
from finetune.index import build_index
from tests.unit import team_worker

ALPHAS_CUMPROD = np.cumprod(1 - np.linspace(1e-4, 0.02, 1000))
EMBED_DIMS = {"ViT-B/32": 12, "PE-Core-B-16": 20}
PER_SHARD = 6


def small_unet(seed: int = 0) -> UNetModel:
    """An unconditional U-Net of the 256 checkpoint's kind of blocks, with no layer left at zero."""
    th.manual_seed(seed)
    unet = UNetModel(
        image_size=16,
        in_channels=3,
        model_channels=32,
        out_channels=6,
        num_res_blocks=1,
        attention_resolutions=(1, 2),
        channel_mult=(1, 2),
        num_head_channels=16,
        use_scale_shift_norm=True,
        resblock_updown=True,
    )
    for parameter in unet.parameters():
        if not parameter.any():
            th.nn.init.normal_(parameter, std=0.05)
    return unet.eval()


def inputs(batch: int = 3) -> tuple[th.Tensor, th.Tensor, dict[str, th.Tensor]]:
    """Noisy images, their timesteps and an embedding from each CLIP model."""
    generator = th.Generator().manual_seed(1)
    embeds = {name: th.randn(batch, dim, generator=generator) for name, dim in EMBED_DIMS.items()}
    return th.randn(batch, 3, 16, 16, generator=generator), th.tensor([10.0, 500.5, 990.0])[:batch], embeds


def randomise(model: ConditionedUNet) -> None:
    """Move every trained weight off its starting value, as training would."""
    generator = th.Generator().manual_seed(2)
    for parameter in trainable(model).values():
        parameter.data.add_(th.randn(parameter.shape, generator=generator) * 0.1)


@pytest.fixture(name="log_alpha_bar", scope="module")
def fixture_log_alpha_bar() -> np.ndarray:
    """The log signal level of ADM's linear schedule, at each of its timesteps."""
    return sampler.schedule(ALPHAS_CUMPROD)


def test_fused_attention_is_the_unets_attention():
    qkv = th.randn(2, 4 * 3 * 16, 25, generator=th.Generator().manual_seed(0))
    assert th.allclose(FusedAttention(4)(qkv), QKVAttentionLegacy(4, spatial=(5, 5))(qkv), atol=1e-5)


def test_a_new_model_is_the_pretrained_one():
    x, t, embeds = inputs()
    expected = small_unet()(x, t)
    model = ConditionedUNet(small_unet(), AdapterConfig(EMBED_DIMS, rank=4, tokens=2, token_dim=8, hidden=16))
    assert th.allclose(model(x, t, embeds), expected, atol=1e-5)
    assert th.allclose(model(x, t), expected, atol=1e-5)


@pytest.mark.parametrize("train_encoder", [False, True])
def test_the_condition_changes_the_output_once_trained(train_encoder: bool):
    x, t, embeds = inputs()
    config = AdapterConfig(EMBED_DIMS, rank=4, tokens=2, token_dim=8, hidden=16, train_encoder=train_encoder)
    model = ConditionedUNet(small_unet(), config)
    randomise(model)
    other = {name: -embedding for name, embedding in embeds.items()}
    assert not th.allclose(model(x, t, embeds), model(x, t, other), atol=1e-3)


def test_dropping_every_embedding_is_the_unconditional_model():
    x, t, embeds = inputs()
    model = ConditionedUNet(small_unet(), AdapterConfig(EMBED_DIMS, rank=4, tokens=2, token_dim=8, hidden=16))
    randomise(model)
    keep = th.tensor([[True, True], [False, False], [True, False]])
    mixed, unconditional = model(x, t, embeds, keep), model(x, t)
    assert th.allclose(mixed[1], unconditional[1], atol=1e-5)
    assert not th.allclose(mixed[0], unconditional[0], atol=1e-3)
    assert not th.allclose(mixed[2], mixed[0], atol=1e-3)


def test_a_dropped_models_embedding_is_ignored():
    x, t, embeds = inputs()
    model = ConditionedUNet(small_unet(), AdapterConfig(EMBED_DIMS, rank=4, tokens=2, token_dim=8, hidden=16))
    randomise(model)
    keep = th.tensor([[True, False]] * 3)
    changed = embeds | {"PE-Core-B-16": th.zeros(3, 20)}
    assert th.allclose(model(x, t, embeds, keep), model(x, t, changed, keep), atol=1e-6)


def test_only_what_was_added_is_trained_and_the_encoder_gets_no_gradient():
    x, t, embeds = inputs()
    model = ConditionedUNet(small_unet(), AdapterConfig(EMBED_DIMS, rank=4, tokens=2, token_dim=8, hidden=16))
    randomise(model)
    before = {name: value.clone() for name, value in model.unet.state_dict().items() if ".base." in name}
    model(x, t, embeds).square().mean().backward()
    names = set(trainable(model))
    assert all(".down." in n or ".up." in n or n.startswith(("conditioner.", "token_attention.")) for n in names)
    assert not any(name.startswith("unet.input_blocks") for name in names)
    assert all(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in trainable(model).values())
    assert all(parameter.grad is None for parameter in model.unet.input_blocks.parameters())
    assert all(th.equal(model.unet.state_dict()[name], value) for name, value in before.items())


def test_training_the_encoder_adapts_it_too():
    config = AdapterConfig(EMBED_DIMS, rank=4, tokens=2, token_dim=8, hidden=16, train_encoder=True)
    names = set(trainable(ConditionedUNet(small_unet(), config)))
    assert any(name.startswith("unet.input_blocks") for name in names)
    assert any(name.startswith("token_attention.input_") for name in names)


def test_a_checkpoint_rebuilds_the_model():
    x, t, embeds = inputs()
    model = ConditionedUNet(small_unet(), AdapterConfig(EMBED_DIMS, rank=4, tokens=2, token_dim=8, hidden=16))
    randomise(model)
    buffer = io.BytesIO()
    th.save(adapter_checkpoint(model), buffer)
    buffer.seek(0)
    loaded = load_adapter(small_unet(), th.load(buffer, weights_only=True))
    assert th.equal(loaded(x, t, embeds), model(x, t, embeds))


def test_a_checkpoint_of_another_configuration_is_rejected():
    model = ConditionedUNet(small_unet(), AdapterConfig(EMBED_DIMS, rank=4, tokens=2, token_dim=8, hidden=16))
    checkpoint = adapter_checkpoint(model)
    checkpoint["weights"].pop(next(iter(checkpoint["weights"])))
    with pytest.raises(ValueError, match="not this configuration"):
        load_adapter(small_unet(), checkpoint)


def test_without_the_fused_kernel_the_original_attention_stays():
    config = AdapterConfig(EMBED_DIMS, rank=4, tokens=0)
    model = ConditionedUNet(small_unet(), config, fused=False)
    blocks = [module for module in model.modules() if isinstance(module, AttentionBlock)]
    assert blocks
    assert all(isinstance(block.attention, QKVAttentionLegacy) for block in blocks)
    assert not model.token_attention


def test_training_a_blocks_own_weights_takes_no_update_beside_them():
    with pytest.raises(ValueError, match="own weights"):
        ConditionedUNet(small_unet(), AdapterConfig(EMBED_DIMS, full=True, film=True))


def test_noise_levels_cover_the_schedule_and_lean_to_high_noise(log_alpha_bar: np.ndarray):
    table = sampler.log_snr(log_alpha_bar)
    uniform = noise.sample_log_snr(log_alpha_bar, 20000, 1.0, th.Generator().manual_seed(0))
    biased = noise.sample_log_snr(log_alpha_bar, 20000, 2.0, th.Generator().manual_seed(0))
    assert uniform.min() >= table[-1]
    assert uniform.max() <= table[0]
    assert uniform.mean().item() == pytest.approx((table[0] + table[-1]) / 2, abs=0.1)
    middle = (table[0] + table[-1]) / 2
    assert (biased < middle).double().mean().item() == pytest.approx(0.5**0.5, abs=0.02)


def test_timesteps_are_the_samplers(log_alpha_bar: np.ndarray):
    table = sampler.log_snr(log_alpha_bar)
    log_snrs = th.from_numpy(np.linspace(table[-1], table[0], 31))
    theirs = sampler.timesteps(log_alpha_bar, 30)
    ours = noise.timestep_of(log_alpha_bar, log_snrs)
    assert ours.tolist() == pytest.approx(theirs, abs=1e-6)
    alpha, sigma = noise.level_of(log_snrs)
    for t, a, s in zip(theirs, alpha.tolist(), sigma.tolist(), strict=True):
        level = sampler.level(log_alpha_bar, t)
        assert (a, s) == pytest.approx((level.alpha, level.sigma), abs=1e-9)


def test_min_snr_caps_the_weight_of_clean_levels():
    log_snr = th.tensor([-2.0, 0.0, 0.5 * np.log(5.0), 2.0], dtype=th.float64)
    weight = noise.min_snr_weight(log_snr, 5.0)
    assert weight.tolist() == pytest.approx([1.0, 1.0, 1.0, 5.0 / np.exp(4.0)])


def test_embedding_noise_keeps_unit_length_and_grows_with_its_level():
    generator = th.Generator().manual_seed(0)
    embedding = th.nn.functional.normalize(th.randn(500, 64, generator=generator), dim=-1)
    levels = th.tensor([0.0, 0.25, 0.75])
    similarity = []
    for level in levels:
        noisy = noise.noisy_embedding(embedding, level.expand(500, 1), generator)
        assert noisy.norm(dim=-1).tolist() == pytest.approx([1.0] * 500, abs=1e-5)
        similarity.append((noisy * embedding).sum(-1).mean().item())
    assert similarity[0] == pytest.approx(1.0, abs=1e-5)
    assert similarity[0] > similarity[1] > similarity[2] > 0.5


def test_mixing_gives_each_dataset_its_share_of_every_batch():
    indices = list(islice(data.mixed_indices([10, 1000], [1, 1], seed=0), 64))
    first = [index < 10 for index in indices]
    assert all(sum(first[i : i + 8]) == 4 for i in range(0, 64, 8))
    indices = list(islice(data.mixed_indices([10, 1000], [1, 3], seed=0), 64))
    assert all(sum(index < 10 for index in indices[i : i + 8]) == 2 for i in range(0, 64, 8))


def test_mixing_visits_every_image_of_a_dataset_before_any_repeats():
    indices = list(islice(data.mixed_indices([10, 1000], [1, 1], seed=0), 80))
    small = [index for index in indices if index < 10]
    large = [index for index in indices if index >= 10]
    assert all(sorted(small[i : i + 10]) == list(range(10)) for i in range(0, 40, 10))
    assert small[:10] != small[10:20]
    assert len(set(large)) == len(large)


def test_mixing_is_reproducible_and_resumes():
    sampler_ = data.MixedSampler([10, 1000], [1, 1], seed=3)
    again = data.MixedSampler([10, 1000], [1, 1], seed=3, start=20)
    other = data.MixedSampler([10, 1000], [1, 1], seed=4)
    order = list(islice(sampler_, 50))
    assert order == list(islice(data.MixedSampler([10, 1000], [1, 1], seed=3), 50))
    assert order[20:] == list(islice(again, 30))
    assert order != list(islice(other, 50))


def test_texts_drop_the_shared_opening_and_keep_a_first_sentence():
    index = pd.DataFrame(
        {
            "generated_caption": ["The image shows a red fox in the snow. It looks left.", "A plain caption"],
            "alt_text": ["fox photo", " "],
        }
    )
    assert data.texts(index) == {
        "caption": ["A red fox in the snow. It looks left.", "A plain caption"],
        "first": ["A red fox in the snow.", "A plain caption"],
        "short": ["fox photo", "A plain caption"],
    }
    classes = pd.DataFrame({"generated_caption": ["The image shows a newt."], "class_name": ["smooth newt"]})
    assert data.texts(classes)["short"] == ["a photo of a smooth newt"]
    assert data.texts(pd.DataFrame({"alt_text": ["only alt text"]}))["caption"] == ["only alt text"]


def sample(n: int, imagenet: bool) -> dict[str, bytes]:
    """The members of sample n, by extension. Its image is a flat colour whose red value is n."""
    image = io.BytesIO()
    Image.new("RGB", (16, 16), (n, 100, 200)).save(image, "PNG")  # Lossless, so the value survives.
    meta = {"key": f"{n:09d}", "width": 16, "generated_caption": f"The image shows thing {n}. More."}
    meta |= {"cgd_filter": {"quality": n / 10, "wm_yolo": 0.0}}
    if imagenet:
        meta |= {"label": n % 3, "class_name": f"class {n % 3}"}
    else:
        meta |= {"caption": f"alt {n}", "aesthetic": 5 + n / 10, "status": "success"}
    return {"jpg": image.getvalue(), "json": json.dumps(meta).encode(), "txt": b""}


def write_dataset(root: Path, shards: int, imagenet: bool) -> None:
    """Shards in the layout of the real datasets, their index, and embeddings that say which row they are."""
    root.mkdir()
    for shard in range(shards):
        with tarfile.open(root / f"{shard:05d}.tar", "w") as tar:
            for n in range(shard * PER_SHARD, (shard + 1) * PER_SHARD):
                for extension, payload in sample(n, imagenet).items():
                    member = tarfile.TarInfo(f"{n:09d}.{extension}")
                    member.size = len(payload)
                    tar.addfile(member, io.BytesIO(payload))
    rows = np.arange(len(build_index(root, workers=1)), dtype=np.float16)[:, None]
    for name, dim in EMBED_DIMS.items():
        for number, kind in enumerate(data.EMBED_KINDS):
            np.save(data.embed_path(root, name, kind), np.full((len(rows), dim), number, np.float16) + rows)


@pytest.fixture(name="roots", scope="module")
def fixture_roots(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """A LAION-like dataset of 3 shards and an ImageNet-like one of 2."""
    base = tmp_path_factory.mktemp("datasets")
    write_dataset(base / "laion", 3, imagenet=False)
    write_dataset(base / "imagenet", 2, imagenet=True)
    return base / "laion", base / "imagenet"


def test_the_index_has_a_row_per_image_with_its_metadata(roots: tuple[Path, Path]):
    index = pd.read_parquet(roots[0] / data.INDEX_NAME)
    assert len(index) == 3 * PER_SHARD
    assert index["shard"].tolist() == [n // PER_SHARD for n in range(len(index))]
    assert index["key"].tolist() == [f"{n:09d}" for n in range(len(index))]
    assert {"alt_text", "aesthetic", "quality", "wm_yolo", "generated_caption", "offset", "size"} <= set(index.columns)
    assert not {"status", "width", "cgd_filter", "caption"} & set(index.columns)
    assert index["quality"].tolist() == pytest.approx([n / 10 for n in range(len(index))])


def test_the_last_shard_is_held_out(roots: tuple[Path, Path]):
    index = pd.read_parquet(roots[0] / data.INDEX_NAME)
    kept, held_out = data.select(index), data.select(index, split="holdout")
    assert kept["row"].tolist() == list(range(2 * PER_SHARD))
    assert held_out["row"].tolist() == list(range(2 * PER_SHARD, 3 * PER_SHARD))
    assert len(data.select(index, split="all")) == len(index)


def test_a_source_is_filtered_on_its_metadata(roots: tuple[Path, Path]):
    laion = data.open_source("laion", roots[0], where="aesthetic >= 5.5 and quality < 0.9")
    assert laion.index["row"].tolist() == [5, 6, 7, 8]
    imagenet = data.open_source("imagenet", roots[1], where="label == 2 or class_name == 'class 0'", split="all")
    assert set(imagenet.index["label"]) == {0, 2}
    with pytest.raises(SystemExit, match="No rows"):
        data.open_source("laion", roots[0], where="aesthetic > 100")


def test_images_captioned_as_watermarked_are_left_out_unless_asked_for():
    index = pd.DataFrame(
        {
            "shard": [0, 0, 0, 1],
            "generated_caption": [
                "A fox.",
                "A fox with a Watermark in the corner.",
                "A dog with watermarks.",
                "A cat.",
            ],
        }
    )
    assert data.select(index, split="all")["row"].tolist() == [0, 3]
    assert data.select(index)["row"].tolist() == [0]
    assert data.select(index, split="all", keep_watermarked=True)["row"].tolist() == [0, 1, 2, 3]
    assert data.select(index, "row > 0", "all")["row"].tolist() == [3]
    without_captions = pd.DataFrame({"shard": [0, 1], "alt_text": ["a watermark", "b"]})
    assert data.select(without_captions, split="all")["row"].tolist() == [0, 1]


def test_a_source_needs_its_index_and_embeddings(roots: tuple[Path, Path], tmp_path: Path):
    with pytest.raises(SystemExit, match=r"finetune\.index"):
        data.open_source("none", tmp_path)
    with pytest.raises(SystemExit, match=r"finetune\.embed"):
        data.open_source("laion", roots[0], ["ViT-L/14"])


def test_batches_pair_each_image_with_its_own_texts_and_embeddings(roots: tuple[Path, Path]):
    models = list(EMBED_DIMS)
    sources = [
        data.open_source("laion", roots[0], models, where="aesthetic >= 5.3"),
        data.open_source("imagenet", roots[1], models),
    ]
    batch = next(data.batches(sources, batch_size=4, seed=0, workers=0))
    assert batch["image"].shape == (4, 3, 16, 16)
    assert batch["image"].dtype == th.uint8
    assert batch["source"].tolist() == [0, 1, 0, 1]
    assert batch["label"][0] == -1
    assert 0 <= batch["label"][1] <= 2
    for i in range(4):
        source = sources[batch["source"][i]]
        row = int(source.index["row"][int(batch["position"][i])])
        assert batch["image"][i, 0, 0, 0] == row
        assert batch["texts"]["caption"][i] == f"Thing {row}. More."
        assert batch["texts"]["first"][i] == f"Thing {row}."
        assert batch["texts"]["short"][i] == (
            f"alt {row}" if batch["source"][i] == 0 else f"a photo of a class {row % 3}"
        )
        for name, dim in EMBED_DIMS.items():
            for number, kind in enumerate(data.EMBED_KINDS):
                assert batch["embeds"][name][kind][i].tolist() == [row + number] * dim
    assert all(row >= 3 for row in sources[0].index["row"])


def test_batches_resume_where_they_stopped(roots: tuple[Path, Path]):
    sources = [data.open_source("laion", roots[0]), data.open_source("imagenet", roots[1])]
    whole = list(islice(data.batches(sources, batch_size=2, seed=1, workers=0), 4))
    rest = list(islice(data.batches(sources, batch_size=2, seed=1, start=4, workers=0), 2))
    assert [batch["position"].tolist() for batch in whole[2:]] == [batch["position"].tolist() for batch in rest]


def test_a_step_draws_the_same_numbers_however_the_run_reached_it():
    first, again, next_step = (train.step_generator(0, step) for step in (7, 7, 8))
    drawn = th.rand(4, generator=first)
    assert th.equal(drawn, th.rand(4, generator=again))
    assert not th.equal(drawn, th.rand(4, generator=next_step))


def test_resuming_drops_what_was_logged_past_the_checkpoint(tmp_path: Path):
    log = tmp_path / "log.jsonl"
    records = [{"step": 1, "loss": 0.5}, {"step": 2, "loss": 0.4}, {"step": 2, "gap": 0.1}, {"step": 3, "loss": 0.3}]
    log.write_text("".join(json.dumps(record) + chr(10) for record in records))
    train.trim_log(log, 2)
    assert [json.loads(line) for line in log.read_text().splitlines()] == records[:2]


def test_a_floor_counts_the_noisiest_levels_for_more():
    log_snr = th.tensor([-3.0, 0.5 * np.log(0.1), 0.0, 2.0], dtype=th.float64)
    weight = noise.min_snr_weight(log_snr, 5.0, floor=0.1)
    assert weight.tolist() == pytest.approx([0.1 * np.exp(6.0), 1.0, 1.0, 5.0 / np.exp(4.0)])


FULL = {"standardised": True, "film": True, "layout": 2, "layout_channels": 4, "layout_blocks": 2, "conv_rank": 2}


def full_model(seed: int = 0) -> ConditionedUNet:
    """A model with everything the distillation's student adds, and embedding statistics that are not the defaults."""
    config = AdapterConfig(EMBED_DIMS, rank=4, tokens=2, token_dim=8, hidden=16, **FULL)
    model = ConditionedUNet(small_unet(seed), config)
    generator = th.Generator().manual_seed(3)
    statistics = {
        name: (th.randn(dim, generator=generator), th.rand(dim, generator=generator) + 0.5)
        for name, dim in EMBED_DIMS.items()
    }
    model.conditioner.set_statistics(statistics)
    return model


def test_a_new_model_with_every_addition_is_still_the_pretrained_one():
    x, t, embeds = inputs()
    expected = small_unet()(x, t)
    model = full_model()
    assert th.allclose(model(x, t, embeds), expected, atol=1e-5)
    assert set(model.layout_proj) == {"middle", "output_0", "output_1"}
    assert len(model.film) == sum(
        1 for name in trainable(model) if name.startswith("film.") and name.endswith(".weight")
    )


def test_every_addition_is_trained_and_changes_what_the_condition_does():
    x, t, embeds = inputs()
    model = full_model()
    randomise(model)
    other = {name: -embedding for name, embedding in embeds.items()}
    assert not th.allclose(model(x, t, embeds), model(x, t, other), atol=1e-3)
    model(x, t, embeds).square().mean().backward()
    names = set(trainable(model))
    for prefix in ("film.", "layout_proj.", "to_layout.", "conditioner.mix."):
        assert any(name.startswith(prefix) for name in names)
    assert any(".out_layers." in name and ".down." in name for name in names)
    assert all(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in trainable(model).values())
    assert all(target.extra is None for target in model.modules() if hasattr(target, "extra"))


def test_embeddings_are_centred_and_scaled_by_statistics_that_a_checkpoint_keeps():
    x, t, embeds = inputs()
    model = full_model()
    randomise(model)
    conditioner = model.conditioner
    mean, scale = conditioner.mean_ViT_B_32, conditioner.scale_ViT_B_32
    assert th.allclose(conditioner.standardise("ViT-B/32", embeds["ViT-B/32"]), (embeds["ViT-B/32"] - mean) / scale)
    shifted = {name: embedding + 1 for name, embedding in embeds.items()}
    assert not th.allclose(model(x, t, embeds), model(x, t, shifted), atol=1e-3)
    restored = load_adapter(small_unet(), adapter_checkpoint(model))
    assert th.equal(restored.conditioner.mean_ViT_B_32, mean)
    assert th.allclose(restored(x, t, embeds), model(x, t, embeds), atol=1e-6)


def test_a_student_that_is_the_base_model_misses_the_whole_move_and_the_teacher_none():
    generator = th.Generator().manual_seed(0)
    base, shift = th.randn(2, 3, 8, 8, generator=generator), th.randn(2, 3, 8, 8, generator=generator)
    missed, cosine = distill.how_much_missed(base.clone(), base, shift)
    assert missed.tolist() == pytest.approx([1.0, 1.0])
    missed, cosine = distill.how_much_missed(base + shift, base, shift)
    assert missed.tolist() == pytest.approx([0.0, 0.0], abs=1e-12)
    assert cosine.tolist() == pytest.approx([1.0, 1.0])
    missed, cosine = distill.how_much_missed(base + 0.5 * shift, base, shift)
    assert missed.tolist() == pytest.approx([0.25, 0.25])


def test_training_timesteps_stay_where_cgd_guides(log_alpha_bar: np.ndarray):
    generator = th.Generator().manual_seed(0)
    drawn = np.array([distill.guided_timestep(log_alpha_bar, generator) for _ in range(500)])
    last = len(log_alpha_bar) - 1
    assert drawn.min() >= distill.WINDOW[0] * last - 1e-6
    assert drawn.max() <= distill.WINDOW[1] * last + 1e-6
    levels = np.interp(drawn, np.arange(last + 1), sampler.log_snr(log_alpha_bar))
    assert abs(np.median(levels) - (levels.min() + levels.max()) / 2) < 0.4  # Uniform in the noise level.


class LinearClip(th.nn.Module):  # pylint: disable=abstract-method  # Guidance never calls CLIP's forward().
    """CLIP's image interface over a random projection of the pixels, as tests/unit/test_guide.py has it."""

    def __init__(self, size: int, features: int) -> None:
        super().__init__()
        self.visual = th.nn.Linear(3 * size * size, features)
        self.visual.input_resolution = size

    def encode_image(self, images: th.Tensor) -> th.Tensor:
        """One feature vector per image."""
        return self.visual(images.flatten(1))


def test_the_teachers_move_is_cgds_guided_step(log_alpha_bar: np.ndarray):
    th.manual_seed(5)
    unet, clip = small_unet(), LinearClip(8, 12)
    teacher = distill.Teacher(unet, (("stand-in", clip, 8),), 4, 0.08)
    x, t = th.randn(1, 3, 16, 16), 640.5
    targets = {"stand-in": th.randn(1, 12)}
    th.manual_seed(7)
    eps, shift = distill.guided_shift(teacher, log_alpha_bar, x, t, targets)
    th.manual_seed(7)
    guidance = sampler.Guidance(lambda *a: guidance_loss(distill.guide_for(teacher, targets), *a), teacher.rate)
    now = sampler.level(log_alpha_bar, t)
    expected = sampler.predict(lambda x, ts: denoise(unet, None, x, ts), x, t, now, guidance)
    assert shift.abs().max() > 0
    assert th.allclose((x - now.sigma * eps) / now.alpha, expected.x0, atol=1e-5)
    moved = (x - now.sigma * (eps + shift)) / now.alpha
    assert th.allclose(moved.clamp(-sampler.X0_CLAMP, sampler.X0_CLAMP), expected.guided_x0, atol=1e-4)


def test_text_statistics_are_of_every_kind_of_text(roots: tuple[Path, Path]):
    statistics = distill.text_statistics(roots, list(EMBED_DIMS))
    for name, dim in EMBED_DIMS.items():
        stores = [np.load(data.embed_path(root, name, kind)) for root in roots for kind in data.TEXT_KINDS]
        every = np.concatenate(stores).astype(np.float32)
        assert statistics[name][0].shape == (dim,)
        assert np.allclose(statistics[name][0].numpy(), every.mean(0), atol=1e-5)
        assert np.allclose(statistics[name][1].numpy(), every.std(0, ddof=1), atol=1e-5)


def test_samples_use_the_averaged_weights_and_training_gets_its_own_back():
    args = distill.parse_args(["--device", "cpu", "--accumulate", "1", "--warmup", "1", "--ema", "0.5"])
    run = train.Run(args, full_model(), None)
    x, t, embeds = inputs()
    for _ in range(3):
        run.step(run.model(x, t, embeds).square().mean())
    trained = {name: weight.detach().clone() for name, weight in run.weights.items()}
    assert any(not th.equal(trained[name], run.average[name]) for name in trained)
    with distill.averaged(run):
        assert all(th.equal(weight, run.average[name]) for name, weight in run.weights.items())
    assert all(th.equal(weight, trained[name]) for name, weight in run.weights.items())


def test_a_wide_layer_gets_proportionally_less_of_the_learning_rate():
    weights = trainable(full_model())
    assert [group["rate"] for group in train.rate_groups(weights, None)] == [1.0]
    groups = train.rate_groups(weights, 8)
    assert sum(len(group["params"]) for group in groups) == len(weights)
    for group in groups:
        for weight in group["params"]:
            fan_in = weight[0].numel() if weight.ndim > 1 else 1
            assert group["rate"] == min(1.0, 8 / fan_in)
    assert {group["rate"] for group in groups} >= {1.0, 0.5}  # Biases, and the 16 hidden units the layout is made from.
    args = distill.parse_args(
        ["--device", "cpu", "--accumulate", "1", "--warmup", "4", "--lr", "0.01", "--widest", "8"]
    )
    run = train.Run(args, full_model(), None)
    x, t, embeds = inputs()
    run.step(run.model(x, t, embeds).square().mean())
    assert all(group["lr"] == pytest.approx(0.01 * group["rate"] / 4) for group in run.optimizer.param_groups)


def test_a_move_is_scaled_down_to_the_cap_and_no_further():
    generator = th.Generator().manual_seed(0)
    base = th.randn(2, 3, 8, 8, generator=generator)
    direction = th.randn(2, 3, 8, 8, generator=generator)
    direction = direction / sampler.rms(direction) * sampler.rms(base)
    scales = th.tensor([0.04, 0.4]).reshape(2, 1, 1, 1)
    move, size = distill.capped_move(base + scales * direction, base, 0.08)
    assert size.flatten().tolist() == pytest.approx([0.5, 5.0], rel=1e-4)
    assert th.allclose(move[0], scales[0] * direction[0], atol=1e-6)
    assert th.allclose(move[1], 0.08 * direction[1], atol=1e-5)
    still = base.clone().requires_grad_()
    move, size = distill.capped_move(still, base, 0.08)
    (move.sum() + size.sum()).backward()
    assert th.isfinite(still.grad).all()  # A student that has not moved yet, as every one starts.


def test_each_quantity_is_averaged_over_the_examples_that_measured_it():
    window = [{"size": 1.0, "gain": 4.0}, {"size": 3.0}]
    assert distill.averages(window) == {"size": 2.0, "gain": 4.0}


def test_training_on_the_guidance_loss_itself_lowers_it_within_the_cap(log_alpha_bar: np.ndarray):
    args = distill.parse_args(["--device", "cpu", "--accumulate", "1", "--warmup", "1", "--lr", "3e-3", "--direct"])
    th.manual_seed(0)
    towers = tuple((name, LinearClip(8, dim), 8) for name, dim in EMBED_DIMS.items())
    teacher = distill.Teacher(small_unet(), towers, 4, 0.08)
    run = train.Run(args, full_model(), None)
    generator = th.Generator().manual_seed(4)
    image = th.randint(0, 256, (3, 16, 16), generator=generator, dtype=th.uint8)
    targets = {name: th.randn(1, dim, generator=generator) for name, dim in EMBED_DIMS.items()}
    gains, sizes = [], []
    for _ in range(60):
        th.manual_seed(run.steps)
        result = distill.reward_example(run, teacher, log_alpha_bar, (image, targets), th.Generator().manual_seed(9))
        sizes.append(result["size"])
        gains += [result["gain"]] if "gain" in result else []
    assert len(gains) == 60 // distill.GAIN_EVERY
    assert gains[0] == pytest.approx(0.0, abs=1e-6)  # It starts as the base model.
    assert gains[-1] > 0.001  # And ends with a lower loss than the base model's, on the same cutouts.
    assert max(sizes[-10:]) < 1.5  # Held near the cap.


def test_the_direct_loss_is_cgds_guidance_loss_with_a_share_of_its_smoothness_term(log_alpha_bar: np.ndarray):
    th.manual_seed(0)
    towers = tuple((name, LinearClip(8, dim), 8) for name, dim in EMBED_DIMS.items())
    teacher = distill.Teacher(small_unet(), towers, 4, 0.08)
    generator = th.Generator().manual_seed(4)
    targets = {name: th.randn(1, dim, generator=generator) for name, dim in EMBED_DIMS.items()}
    x, eps = th.randn(1, 3, 16, 16, generator=generator), th.randn(1, 3, 16, 16, generator=generator)
    now = sampler.level(log_alpha_bar, 600.0)
    guide = distill.guide_for(teacher, targets)

    def loss(tv: float) -> float:
        th.manual_seed(1)
        return distill.guided_loss(guide, x, eps, now, tv).item()

    th.manual_seed(1)
    whole = guidance_loss(guide, x, (x - now.sigma * eps) / now.alpha, now.sigma).item() / 1000
    assert loss(1.0) == pytest.approx(whole, rel=1e-6)
    assert loss(0.0) < loss(0.5) < loss(1.0)
    assert loss(0.5) == pytest.approx((loss(0.0) + loss(1.0)) / 2, rel=1e-6)


def test_a_gradient_reaches_the_noisy_image_through_the_whole_model():
    x, t, embeds = inputs()
    x = x.requires_grad_()
    expected = th.autograd.grad(small_unet()(x, t).square().sum(), x)[0]
    gradient = th.autograd.grad(full_model()(x, t, embeds).square().sum(), x)[0]
    assert expected.abs().max() > 0
    assert th.allclose(
        gradient, expected, rtol=1e-3, atol=1e-5
    )  # As guidance takes it. A new model is the pretrained one.


CLASSES = 10
SLOT_DIMS = {name: dim for model, dim in EMBED_DIMS.items() for name in (f"{model} image", f"{model} text")}


def class_unet(seed: int = 0) -> UNetModel:
    """`small_unet` as a class-conditional model, as the checkpoint `cgd` draws with by default is."""
    th.manual_seed(seed)
    unet = UNetModel(
        image_size=16,
        in_channels=3,
        model_channels=32,
        out_channels=6,
        num_res_blocks=1,
        attention_resolutions=(1, 2),
        channel_mult=(1, 2),
        num_classes=CLASSES,
        num_head_channels=16,
        use_scale_shift_norm=True,
        resblock_updown=True,
    )
    for parameter in unet.parameters():
        if not parameter.any():
            th.nn.init.normal_(parameter, std=0.05)
    return unet.eval()


def anchored_model(seed: int = 0, align_dim: int = 6) -> ConditionedUNet:
    """The model `finetune.cond` trains: the class kept, both conditions of each CLIP model, its own weights trained."""
    config = AdapterConfig(
        SLOT_DIMS, tokens=2, hidden=24, token_dim=8, standardised=True, full=True, align_dim=align_dim
    )
    return ConditionedUNet(class_unet(seed), config)


def anchored_inputs(batch: int = 3) -> tuple[th.Tensor, th.Tensor, cond.Condition]:
    """Noisy images, their timesteps, and every condition with a class each."""
    x, t, _ = inputs(batch)
    generator = th.Generator().manual_seed(3)
    embeds = {name: th.randn(batch, dim, generator=generator) for name, dim in SLOT_DIMS.items()}
    keep = th.ones(batch, len(SLOT_DIMS), dtype=th.bool)
    return x, t, cond.Condition(embeds, keep, th.tensor([1, 4, 7])[:batch])


def test_a_new_class_anchored_model_is_the_pretrained_one_told_the_class():
    x, t, condition = anchored_inputs()
    expected = class_unet()(x, t, y=condition.labels)
    output, aligned = anchored_model()(x, t, condition.embeds, condition.keep, labels=condition.labels, middle=True)
    assert th.allclose(output, expected, atol=1e-5)
    assert aligned.shape == (3, 6, 8, 8)  # The middle block's own grid, at the width asked for.


def test_the_class_anchored_model_trains_the_middle_and_decoder_whole_and_leaves_the_encoder():
    model = anchored_model()
    names = set(trainable(model))
    frozen = ("unet.input_blocks", "unet.time_embed", "unet.label_emb")
    assert not [name for name in names if name.startswith(frozen)]
    for prefix in ("unet.middle_block", "unet.output_blocks", "unet.out", "conditioner", "token_attention", "align"):
        assert [name for name in names if name.startswith(prefix)], prefix
    assert not [name for name in names if ".up." in name or ".down." in name]  # No low-rank updates beside them.
    x, t, condition = anchored_inputs()
    model(x, t, condition.embeds, condition.keep, labels=condition.labels).square().sum().backward()
    assert all(parameter.grad is None for parameter in model.unet.input_blocks.parameters())
    assert model.unet.out[-1].weight.grad.abs().max() > 0


def test_the_embeddings_change_what_a_trained_class_anchored_model_draws_and_dropping_them_does_not():
    model = anchored_model()
    randomise(model)
    x, t, condition = anchored_inputs()
    told = cond.noise_given(model, False, x, t, condition)
    alone = cond.noise_given(model, False, x, t, condition.alone())
    other = condition._replace(embeds={name: -value for name, value in condition.embeds.items()})
    assert not th.allclose(told, alone, atol=1e-3)
    assert th.allclose(cond.noise_given(model, False, x, t, other.alone()), alone, atol=1e-6)
    assert th.allclose(cond.class_denoiser(model, False, condition)(x, t), alone)


def test_classifier_free_guidance_scales_the_embeddings_move_and_keeps_one_gradient():
    model = anchored_model()
    randomise(model)
    x, t, condition = anchored_inputs()
    told = cond.noise_given(model, False, x, t, condition)
    alone = cond.noise_given(model, False, x, t, condition.alone())
    assert th.allclose(cond.cfg_denoiser(model, False, condition, 1.0)(x, t), told, atol=1e-6)
    assert th.allclose(cond.cfg_denoiser(model, False, condition, 3.0)(x, t), alone + 3 * (told - alone), atol=1e-5)
    x = x.requires_grad_()
    expected = th.autograd.grad(cond.noise_given(model, False, x, t, condition).sum(), x)[0]
    # The gradient is the conditioned pass's alone, whatever the scale adds to its output.
    gradient = th.autograd.grad(cond.cfg_denoiser(model, False, condition, 3.0)(x, t).sum(), x)[0]
    assert expected.abs().max() > 0
    assert th.allclose(gradient, expected, rtol=1e-4, atol=1e-7)


def test_the_anchor_is_the_class_closest_to_the_text():
    classes = th.nn.functional.normalize(th.randn(CLASSES, 12, generator=th.Generator().manual_seed(0)), dim=-1)
    texts = 5 * classes[[7, 2]] + 0.01
    assert cond.anchor_classes(texts, classes).tolist() == [7, 2]


def conditions(count: int = 400, **options: float) -> tuple[cond.Condition, dict, th.Tensor]:
    """Conditions drawn for `count` images with unit embeddings, the embeddings, and the classes' own."""
    generator = th.Generator().manual_seed(0)

    def unit(dim: int) -> th.Tensor:
        return th.nn.functional.normalize(th.randn(count, dim, generator=generator), dim=-1)

    embeds = {name: {kind: unit(dim) for kind in data.EMBED_KINDS} for name, dim in EMBED_DIMS.items()}
    classes = unit(12)[:CLASSES]
    defaults = {"embed_noise": 0.5, "p_image_only": 0.3, "p_text_only": 0.3, "p_drop_model": 0.0, "p_uncond": 0.1}
    args = cond.parse_args([])
    for name, value in (defaults | options).items():
        setattr(args, name, value)
    return cond.choose_condition(embeds, classes, th.Generator().manual_seed(1), args), embeds, classes


def test_a_condition_keeps_the_images_embedding_the_texts_both_or_neither():
    condition, embeds, classes = conditions()
    assert list(condition.embeds) == cond.slots(list(EMBED_DIMS)) == list(SLOT_DIMS)
    image, text = condition.keep[:, 0], condition.keep[:, 1]
    assert th.equal(condition.keep[:, 2:], condition.keep[:, :2])  # Both CLIP models alike, with none dropped.
    shares = [(image & text).float().mean(), (image & ~text).float().mean(), (~image & text).float().mean()]
    assert [share.item() for share in shares] == pytest.approx([0.36, 0.27, 0.27], abs=0.07)
    assert (~image & ~text).float().mean().item() == pytest.approx(0.1, abs=0.05)
    # The text is one of the image's three, the same one for both models, and the anchor is its class.
    first = th.stack([embeds["ViT-B/32"][kind] for kind in data.TEXT_KINDS], dim=1)
    second = th.stack([embeds["PE-Core-B-16"][kind] for kind in data.TEXT_KINDS], dim=1)
    which = (first == condition.embeds["ViT-B/32 text"][:, None]).all(dim=2)
    assert th.equal(which.sum(dim=1), th.ones(400, dtype=th.long))
    assert th.equal(second[which], condition.embeds["PE-Core-B-16 text"])
    assert which.float().mean(dim=0).tolist() == pytest.approx([0.2, 0.4, 0.4], abs=0.08)
    assert th.equal(condition.labels, cond.anchor_classes(condition.embeds["ViT-B/32 text"], classes))
    # The image's embedding is moved off its direction, and still of unit length.
    moved = condition.embeds["ViT-B/32 image"]
    assert th.allclose(moved.norm(dim=1), th.ones(400), atol=1e-5)
    assert (moved * embeds["ViT-B/32"]["image"]).sum(dim=1).mean() < 0.999


def test_the_options_drop_every_embedding_or_one_models():
    assert not conditions(p_uncond=1.0)[0].keep.any()
    assert conditions(p_uncond=0.0, p_drop_model=0.0)[0].keep.any(dim=1).all()
    keep = conditions(p_uncond=0.0, p_drop_model=0.5)[0].keep
    assert (keep[:, :2].any(dim=1) != keep[:, 2:].any(dim=1)).float().mean().item() == pytest.approx(0.5, abs=0.08)
    text_only = conditions(p_uncond=0.0, p_text_only=1.0, p_image_only=0.0)[0].keep
    assert not text_only[:, 0::2].any()
    assert text_only[:, 1::2].all()


def test_patch_tokens_become_a_grid_of_unit_features_without_the_prefix():
    tokens = th.randn(2, 1 + 16, 5, generator=th.Generator().manual_seed(0))
    grid = cond.patch_grid(tokens, 1, 2)
    assert grid.shape == (2, 5, 2, 2)
    assert th.allclose(grid.norm(dim=1), th.ones(2, 2, 2), atol=1e-5)
    # The top left of the pooled grid is the mean of the top left four patches, in rows of four.
    corner = tokens[:, 1:].reshape(2, 4, 4, 5)[:, :2, :2].mean(dim=(1, 2))
    assert th.allclose(grid[:, :, 0, 0], th.nn.functional.normalize(corner, dim=1), atol=1e-5)
    assert cond.align_miss(grid, grid).tolist() == pytest.approx([0, 0], abs=1e-6)
    assert cond.align_miss(grid, -grid).tolist() == pytest.approx([2, 2], abs=1e-6)
    # An image that is half noise counts half, and one that is nearly all noise nearly whole.
    assert cond.counted_miss(th.tensor([2.0]), th.tensor([0.0], dtype=th.float64)).item() == pytest.approx(1.0)
    assert cond.counted_miss(th.tensor([2.0]), th.tensor([-5.0], dtype=th.float64)).item() == pytest.approx(2, abs=1e-3)


def test_the_pretrained_weights_get_their_own_share_of_the_learning_rate():
    model = anchored_model()
    groups = train.rate_groups(trainable(model), 64, 0.1)
    own = sum(weight.numel() for name, weight in trainable(model).items() if name.startswith("unet."))
    shares = {group["rate"]: sum(weight.numel() for weight in group["params"]) for group in groups}
    assert shares[0.1] == own
    assert sum(shares.values()) == sum(weight.numel() for weight in trainable(model).values())
    # Without it they are grouped by their width like the rest.
    assert sum(len(group["params"]) for group in train.rate_groups(trainable(model), 64)) == len(trainable(model))


def test_a_run_without_a_running_average_keeps_none_and_steps():
    args = cond.parse_args(["--ema", "0", "--device", "cpu", "--accumulate", "1", "--warmup", "1"])
    model = anchored_model()
    run = train.Run(args, model, None)
    assert not run.average
    before = model.unet.out[-1].weight.detach().clone()
    x, t, condition = anchored_inputs()
    run.step(model(x, t, condition.embeds, condition.keep, labels=condition.labels).square().mean())
    assert run.steps == 1
    assert not th.equal(model.unet.out[-1].weight, before)
    assert train.Run(cond.parse_args(["--device", "cpu"]), anchored_model(), None).average


def test_the_learning_rate_falls_to_zero_over_the_last_batches_before_the_end():
    options = ["--device", "cpu", "--accumulate", "1", "--warmup", "1", "--lr", "0.01", "--ema", "0"]
    args = cond.parse_args([*options, "--max-steps", "8", "--decay", "4"])
    model = anchored_model()
    run = train.Run(args, model, None)
    x, t, condition = anchored_inputs()
    shares = []
    for _ in range(args.max_steps):
        before = model.unet.out[-1].weight.detach().clone()
        run.step(model(x, t, condition.embeds, condition.keep, labels=condition.labels).square().mean())
        shares.append(run.optimizer.param_groups[0]["lr"] / (0.01 * run.optimizer.param_groups[0]["rate"]))
    assert shares == pytest.approx([1, 1, 1, 1, 0.75, 0.5, 0.25, 0])
    assert th.equal(model.unet.out[-1].weight, before)  # The last batch moves nothing.
    assert train.rate_share(cond.parse_args(options), 10**6) == 1  # Without --decay the rate never falls.
    with pytest.raises(SystemExit):
        cond.parse_args([*options, "--decay", "4"])  # Nothing to count back from.


def test_a_class_anchored_checkpoint_rebuilds_the_model():
    model = anchored_model()
    randomise(model)
    x, t, condition = anchored_inputs()
    told = cond.noise_given(model, False, x, t, condition)
    # The frozen half comes from the pretrained checkpoint, which the same seed stands in for.
    again = load_adapter(class_unet(), adapter_checkpoint(model))
    assert th.allclose(cond.noise_given(again, False, x, t, condition), told, atol=1e-6)
    other = load_adapter(class_unet(seed=5), adapter_checkpoint(model))
    assert not th.allclose(cond.noise_given(other, False, x, t, condition), told, atol=1e-3)


def test_the_prior_recovers_a_linear_map_from_texts_to_images():
    rng = np.random.default_rng(0)
    texts = rng.normal(size=(500, 6))
    weight, bias = rng.normal(size=(6, 6)), rng.normal(size=6)
    images = texts @ weight + bias
    fitted, offset = prior.fit(texts, images, ridge=1e-9)
    assert np.allclose(fitted, weight, atol=1e-4)
    assert np.allclose(offset, bias, atol=1e-4)
    maps = {"weight": th.from_numpy(fitted).float(), "bias": th.from_numpy(offset).float()}
    predicted = predict_image(maps, th.from_numpy(texts[:4]).float())
    assert th.allclose(predicted, th.nn.functional.normalize(th.from_numpy(images[:4]).float(), dim=1), atol=1e-3)
    # A penalty that outweighs the data leaves the mean image embedding for every text.
    flat, mean = prior.fit(texts, images, ridge=1e9)
    assert np.allclose(flat, 0, atol=1e-6)
    assert np.allclose(mean, images.mean(0), atol=1e-4)


def test_each_process_draws_its_own_batches_and_together_they_draw_what_one_would():
    whole = list(islice(data.MixedSampler([10, 1000], [1, 1], seed=3, start=24), 48))
    parts = [
        list(islice(data.MixedSampler([10, 1000], [1, 1], seed=3, start=24, share=(member, 4), block=2), 12))
        for member in range(4)
    ]
    assert parts[1][:4] == whole[2:4] + whole[10:12]  # Its batch of every four, two draws each.
    dealt = [index for batch in range(6) for part in parts for index in part[2 * batch : 2 * batch + 2]]
    assert dealt == whole


linux = pytest.mark.skipif(sys.platform != "linux", reason="PyTorch for Windows cannot connect processes.")


@linux
def test_two_processes_end_with_the_weights_one_gets_from_both_their_images(tmp_path):
    assert parallel.launch(["cpu", "cpu"], "tests.unit.team_worker", ["train", str(tmp_path)]) == 0
    first, second = (th.load(tmp_path / f"{rank}.pt", weights_only=True) for rank in range(2))
    for name, weight in first["weights"].items():
        assert th.equal(weight, second["weights"][name]), name
        assert th.equal(first["average"][name], second["average"][name]), name
    assert first["record"] == second["record"] == {"rank": 0.5}
    assert first["agreed"]
    assert second["agreed"]  # The first process's answer, not its own.

    model = anchored_model(align_dim=0)
    run = train.Run(cond.parse_args(team_worker.OPTIONS), model, None)
    for step in range(team_worker.STEPS):
        both = [team_worker.images(step, rank) for rank in range(2)]
        # Two images each, so the mean of the two losses is the loss over all four.
        losses = [model(x, t, c.embeds, c.keep, labels=c.labels).square().mean() for x, t, c in both]
        run.step(sum(losses) / 2)
    for name, weight in run.weights.items():
        assert th.allclose(first["weights"][name], weight, atol=1e-5), name
    moved = [
        name
        for name, weight in trainable(anchored_model(align_dim=0)).items()
        if not th.equal(weight, run.weights[name])
    ]
    assert len(moved) > len(run.weights) // 2


@linux
def test_one_process_failing_stops_the_rest():
    started = time.monotonic()
    assert parallel.launch(["cpu", "cpu"], "tests.unit.team_worker", ["fail"]) == 3
    assert time.monotonic() - started < 30  # The first one would have waited a minute.


def test_a_run_started_by_itself_is_a_team_of_one():
    team = parallel.join("cpu")
    model = anchored_model()
    assert team == parallel.Team()
    assert team.lead
    assert parallel.shared(team, model) is model
    assert parallel.unshared(model) is model
    with parallel.quiet(model, sending=False):  # Nothing to hold back without other processes.
        pass
    assert parallel.agreed(team, False) is False
    assert parallel.averaged(team, {"loss": 2.0}) == {"loss": 2.0}
    with parallel.lead_first(team):
        parallel.leave(team)

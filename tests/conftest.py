"""Fixtures every test can use.

tests/unit holds pure tests that run on the CPU in seconds. tests/integration runs the
real models, and its tests are marked cuda: they need a GPU and the cached checkpoints,
and are skipped without a GPU.
"""

from pathlib import Path

import pytest
import torch as th

IMAGES = Path(__file__).resolve().parent.parent / "images"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip the tests that need a GPU when there is none, instead of failing them."""
    if th.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="needs a CUDA GPU")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty working directory, since every saved frame is also copied to current.png in it."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def photon() -> Path:
    """A photograph in the repository, for image prompts and init images."""
    return IMAGES / "photon.png"


@pytest.fixture
def goose() -> Path:
    """A photograph of a goose in the repository, one of ImageNet's classes."""
    return IMAGES / "goose_on_the_run.png"


@pytest.fixture
def mushroom() -> Path:
    """An image of a mushroom in the repository, for init images."""
    return IMAGES / "32K_HUHD_Mushroom.png"

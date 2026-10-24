import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_model: needs a real HF model (slow, downloads weights)"
    )


@pytest.fixture(scope="session")
def cfg():
    from gcg_defense.config import Config

    return Config.load(os.path.join(ROOT, "configs", "default.yaml"))


@pytest.fixture(scope="session")
def small_model():
    """A small instruct model on CPU in float32, for tokeniser-level tests."""
    from gcg_defense.model_utils import MoveModel

    name = os.environ.get("GCG_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    return MoveModel(name, device="cpu", dtype="float32", batch_size=8)

import os
from pathlib import Path

import pytest

os.environ.setdefault("DEADEYE_CACHE", str(Path(__file__).resolve().parents[1] / ".cache"))


@pytest.fixture(scope="session")
def tiny_model_path():
    from deadeye.models.tiny import ensure_tiny_model
    return ensure_tiny_model()


@pytest.fixture(scope="session")
def tiny_model(tiny_model_path):
    from deadeye.models.hf_backend import HFModel
    return HFModel(str(tiny_model_path))

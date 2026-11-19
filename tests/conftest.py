import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _cpu_only():
    """Run the tests on CPU.

    `accelerator: auto` picks Apple MPS when it is available, and MPS has no
    float64. The tests use small CPU-sized data anyway.
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    try:
        import torch

        torch.backends.mps.is_available = lambda: False
    except ImportError:
        pass
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

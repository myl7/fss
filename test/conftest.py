import pytest
import torch


@pytest.fixture
def cuda_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

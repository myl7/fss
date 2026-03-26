import pytest
import torch
import fss_crypto


@pytest.fixture
def dpf():
    return fss_crypto.Dpf(in_bits=16, group="bytes", prg="chacha")


@pytest.fixture
def s0s():
    return torch.randint(-2**31, 2**31, (2, 4), dtype=torch.int32)


@pytest.fixture
def beta():
    return torch.tensor([0, 0, 0, 604], dtype=torch.int32)


class TestDpfGenShape:
    def test_cws_shape(self, dpf, s0s, beta):
        cws = dpf.gen(s0s, alpha=107, beta=beta)
        assert cws.shape == (17, 8)
        assert cws.dtype == torch.int32

    def test_cws_device_is_cpu(self, dpf, s0s, beta):
        cws = dpf.gen(s0s, alpha=107, beta=beta)
        assert cws.device.type == "cpu"


class TestDpfEvalShape:
    def test_output_shape_cpu(self, dpf, s0s, beta):
        cws = dpf.gen(s0s, alpha=107, beta=beta)
        out = dpf.eval(party=0, s0=s0s[0], cws=cws, x=50)
        assert out.shape == (4,)
        assert out.dtype == torch.int32
        assert out.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_output_shape_cuda(self, dpf, s0s, beta):
        cws = dpf.gen(s0s, alpha=107, beta=beta)
        out = dpf.eval(
            party=0,
            s0=s0s[0].to("cuda"),
            cws=cws.to("cuda"),
            x=50,
        )
        assert out.shape == (4,)
        assert out.dtype == torch.int32
        assert out.device.type == "cuda"


class TestDpfEvalAllShape:
    def test_output_shape(self, dpf, s0s, beta):
        cws = dpf.gen(s0s, alpha=107, beta=beta)
        out = dpf.eval_all(party=0, s0=s0s[0], cws=cws)
        assert out.shape == (2**16, 4)
        assert out.dtype == torch.int32
        assert out.device.type == "cpu"

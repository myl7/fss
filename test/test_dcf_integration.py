import pytest
import torch
import fss_crypto


@pytest.fixture
def dcf():
    return fss_crypto.Dcf(in_bits=16, group="bytes", prg="chacha", pred="lt")


@pytest.fixture
def s0s():
    return torch.randint(-2**31, 2**31, (2, 4), dtype=torch.int32)


@pytest.fixture
def beta():
    return torch.tensor([0, 0, 0, 604], dtype=torch.int32)


class TestDcfGenShape:
    def test_cws_shape(self, dcf, s0s, beta):
        cws = dcf.gen(s0s, alpha=107, beta=beta)
        assert cws.shape == (17, 8)
        assert cws.dtype == torch.int32

    def test_cws_device_is_cpu(self, dcf, s0s, beta):
        cws = dcf.gen(s0s, alpha=107, beta=beta)
        assert cws.device.type == "cpu"


class TestDcfEvalShape:
    def test_output_shape_cpu(self, dcf, s0s, beta):
        cws = dcf.gen(s0s, alpha=107, beta=beta)
        out = dcf.eval(party=0, s0=s0s[0], cws=cws, x=50)
        assert out.shape == (4,)
        assert out.dtype == torch.int32
        assert out.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_output_shape_cuda(self, dcf, s0s, beta):
        cws = dcf.gen(s0s, alpha=107, beta=beta)
        out = dcf.eval(
            party=0,
            s0=s0s[0].to("cuda"),
            cws=cws.to("cuda"),
            x=50,
        )
        assert out.shape == (4,)
        assert out.dtype == torch.int32
        assert out.device.type == "cuda"


class TestDcfEvalAllShape:
    def test_output_shape(self, dcf, s0s, beta):
        cws = dcf.gen(s0s, alpha=107, beta=beta)
        out = dcf.eval_all(party=0, s0=s0s[0], cws=cws)
        assert out.shape == (2**16, 4)
        assert out.dtype == torch.int32
        assert out.device.type == "cpu"


class TestDcfPredVariant:
    def test_gt_variant(self, s0s, beta):
        dcf_gt = fss_crypto.Dcf(
            in_bits=16, group="bytes", prg="chacha", pred="gt"
        )
        cws = dcf_gt.gen(s0s, alpha=107, beta=beta)
        assert cws.shape == (17, 8)

import pytest
import torch
import fss_crypto


@pytest.fixture
def dpf():
    return fss_crypto.HalfTreeDpf(in_bits=16, group="bytes", prg="chacha")


@pytest.fixture
def s0s():
    return torch.randint(-2**31, 2**31, (2, 4), dtype=torch.int32)


@pytest.fixture
def beta():
    return torch.tensor([0, 0, 0, 604], dtype=torch.int32)


class TestHalfTreeDpfGenShape:
    def test_key_shapes(self, dpf, s0s, beta):
        cws, ocw = dpf.gen(s0s, alpha=107, beta=beta)
        assert cws.shape == (16, 8)
        assert cws.dtype == torch.int32
        assert ocw.shape == (4,)
        assert ocw.dtype == torch.int32

    def test_key_device_is_cpu(self, dpf, s0s, beta):
        cws, ocw = dpf.gen(s0s, alpha=107, beta=beta)
        assert cws.device.type == "cpu"
        assert ocw.device.type == "cpu"


class TestHalfTreeDpfEvalShape:
    def test_output_shape_cpu(self, dpf, s0s, beta):
        cws, ocw = dpf.gen(s0s, alpha=107, beta=beta)
        out = dpf.eval(party=0, s0=s0s[0], cws=cws, ocw=ocw, x=50)
        assert out.shape == (4,)
        assert out.dtype == torch.int32
        assert out.device.type == "cpu"

    def test_rejects_out_of_domain_x(self, dpf, s0s, beta):
        cws, ocw = dpf.gen(s0s, alpha=107, beta=beta)
        with pytest.raises(ValueError, match="x must be"):
            dpf.eval(party=0, s0=s0s[0], cws=cws, ocw=ocw, x=2**16)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_output_shape_cuda(self, dpf, s0s, beta):
        cws, ocw = dpf.gen(s0s, alpha=107, beta=beta)
        out = dpf.eval(
            party=0,
            s0=s0s[0].to("cuda"),
            cws=cws.to("cuda"),
            ocw=ocw.to("cuda"),
            x=50,
        )
        assert out.shape == (4,)
        assert out.dtype == torch.int32
        assert out.device.type == "cuda"


class TestHalfTreeDpfEvalAllShape:
    def test_output_shape(self, dpf, s0s, beta):
        cws, ocw = dpf.gen(s0s, alpha=107, beta=beta)
        out = dpf.eval_all(party=0, s0=s0s[0], cws=cws, ocw=ocw)
        assert out.shape == (2**16, 4)
        assert out.dtype == torch.int32
        assert out.device.type == "cpu"


class TestHalfTreeDpfCuda:

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_eval_matches_cpu(self, dpf, s0s, beta):
        cws, ocw = dpf.gen(s0s, alpha=107, beta=beta)
        for party in (0, 1):
            cpu = dpf.eval(party=party, s0=s0s[party], cws=cws, ocw=ocw, x=107)
            gpu = dpf.eval(
                party=party,
                s0=s0s[party].to("cuda"),
                cws=cws.to("cuda"),
                ocw=ocw.to("cuda"),
                x=107,
            )
            assert gpu.device.type == "cuda"
            assert torch.equal(cpu, gpu.cpu())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_eval_all_matches_cpu_cuda(self, dpf, s0s, beta):
        cws, ocw = dpf.gen(s0s, alpha=107, beta=beta)
        for party in (0, 1):
            cpu = dpf.eval_all(party=party, s0=s0s[party], cws=cws, ocw=ocw)
            gpu = dpf.eval_all(
                party=party,
                s0=s0s[party].to("cuda"),
                cws=cws.to("cuda"),
                ocw=ocw.to("cuda"),
            )
            assert gpu.device.type == "cuda"
            assert torch.equal(cpu, gpu.cpu())
        first = dpf.eval_all(
            party=0, s0=s0s[0].to("cuda"), cws=cws.to("cuda"), ocw=ocw.to("cuda")
        ).cpu()
        second = dpf.eval_all(
            party=1, s0=s0s[1].to("cuda"), cws=cws.to("cuda"), ocw=ocw.to("cuda")
        ).cpu()
        # XOR reconstruction (Bytes group): beta at alpha, zero everywhere else.
        diff = torch.bitwise_xor(first, second)
        assert torch.equal(diff[107], beta)
        mask = torch.ones(diff.shape[0], dtype=torch.bool)
        mask[107] = False
        assert not torch.any(diff[mask])

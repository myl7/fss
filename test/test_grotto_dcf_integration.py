import pytest
import torch
import fss_crypto


@pytest.fixture
def dcf():
    return fss_crypto.GrottoDcf(in_bits=16, prg="chacha")


@pytest.fixture
def s0s():
    return torch.randint(-2**31, 2**31, (2, 4), dtype=torch.int32)


class TestGrottoDcfGenShape:
    def test_cws_shape(self, dcf, s0s):
        cws = dcf.gen(s0s, alpha=107)
        assert cws.shape == (17, 8)
        assert cws.dtype == torch.int32

    def test_cws_device_is_cpu(self, dcf, s0s):
        cws = dcf.gen(s0s, alpha=107)
        assert cws.device.type == "cpu"


class TestGrottoDcfEvalAllShape:
    def test_output_shape(self, dcf, s0s):
        cws = dcf.gen(s0s, alpha=107)
        out = dcf.eval_all(party=0, s0=s0s[0], cws=cws)
        assert out.shape == (2**16,)
        assert out.dtype == torch.bool
        assert out.device.type == "cpu"


class TestGrottoDcfReconstruction:
    def test_reconstruction_is_comparison(self, dcf, s0s):
        alpha = 107
        cws = dcf.gen(s0s, alpha=alpha)
        first = dcf.eval_all(party=0, s0=s0s[0], cws=cws)
        second = dcf.eval_all(party=1, s0=s0s[1], cws=cws)
        # XOR reconstruction: 1[alpha <= x] for every x in the domain.
        recon = torch.bitwise_xor(first, second)
        expected = torch.arange(2**16, dtype=torch.int64) >= alpha
        assert torch.equal(recon, expected)


class TestGrottoDcfPreprocessEval:
    def test_tree_shape(self, dcf, s0s):
        cws = dcf.gen(s0s, alpha=107)
        tree = dcf.preprocess(party=0, s0=s0s[0], cws=cws)
        assert tree.shape == (2 * 2**16 - 1,)
        assert tree.dtype == torch.bool
        assert tree.device.type == "cpu"

    def test_eval_matches_eval_all(self, dcf, s0s):
        alpha = 40000
        cws = dcf.gen(s0s, alpha=alpha)
        for party in (0, 1):
            full = dcf.eval_all(party=party, s0=s0s[party], cws=cws)
            tree = dcf.preprocess(party=party, s0=s0s[party], cws=cws)
            for x in (0, 1, alpha - 1, alpha, alpha + 1, 2**16 - 1):
                assert dcf.eval(tree, x, party=party) == bool(full[x])

    def test_rejects_out_of_domain_x(self, dcf, s0s):
        cws = dcf.gen(s0s, alpha=107)
        tree = dcf.preprocess(party=0, s0=s0s[0], cws=cws)
        with pytest.raises(ValueError, match="x must be"):
            dcf.eval(tree, 2**16, party=0)


class TestGrottoDcfCuda:

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_preprocess_gpu_matches_cpu(self, dcf, s0s):
        cws = dcf.gen(s0s, alpha=107)
        for party in (0, 1):
            cpu = dcf.preprocess(party=party, s0=s0s[party], cws=cws)
            gpu = dcf.preprocess(
                party=party, s0=s0s[party].to("cuda"), cws=cws.to("cuda")
            )
            assert gpu.device.type == "cuda"
            assert torch.equal(cpu, gpu.cpu())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_eval_all_gpu_matches_cpu(self, dcf, s0s):
        cws = dcf.gen(s0s, alpha=107)
        for party in (0, 1):
            cpu = dcf.eval_all(party=party, s0=s0s[party], cws=cws)
            gpu = dcf.eval_all(
                party=party, s0=s0s[party].to("cuda"), cws=cws.to("cuda")
            )
            assert gpu.device.type == "cuda"
            assert torch.equal(cpu, gpu.cpu())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_above_24_bits_matches_cpu(self):
        dcf = fss_crypto.GrottoDcf(in_bits=25, prg="chacha")
        s0s = torch.randint(-2**31, 2**31, (2, 4), dtype=torch.int32)
        cws = dcf.gen(s0s, alpha=107)
        for party in (0, 1):
            tree_cpu = dcf.preprocess(party=party, s0=s0s[party], cws=cws)
            tree_gpu = dcf.preprocess(
                party=party, s0=s0s[party].to("cuda"), cws=cws.to("cuda")
            )
            assert tree_gpu.device.type == "cuda"
            assert torch.equal(tree_cpu, tree_gpu.cpu())
            cpu = dcf.eval_all(party=party, s0=s0s[party], cws=cws)
            gpu = dcf.eval_all(
                party=party, s0=s0s[party].to("cuda"), cws=cws.to("cuda")
            )
            assert gpu.device.type == "cuda"
            assert torch.equal(cpu, gpu.cpu())

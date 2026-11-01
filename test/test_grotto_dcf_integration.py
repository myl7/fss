"""Grotto comparison shares and caller-owned parity-tree integration."""

import pytest
import torch
import fss_crypto
from conftest import evaluate_all, generate_point, make_point_scheme


@pytest.fixture
def dcf():
    return make_point_scheme("GrottoDcf")


@pytest.fixture
def s0s():
    return torch.randint(-2**31, 2**31, (2, 4), dtype=torch.int32)


def parity_tree(scheme, party, seed, corrections, **kwargs):
    p = torch.empty((2**(scheme.in_bits + 1) - 1,), dtype=torch.bool,
                    device=seed.device)
    scheme.preprocess(bool(party), seed, corrections, p, **kwargs)
    return fss_crypto.ParityTreeView(p)


class TestGrottoDcfGenShape:
    def test_cws_shape(self, dcf, s0s):
        cws = generate_point(dcf, s0s, alpha=107)
        assert cws.shape == dcf.cws_shape
        assert cws.dtype == torch.uint8

    def test_cws_device_is_cpu(self, dcf, s0s):
        cws = generate_point(dcf, s0s, alpha=107)
        assert cws.device.type == "cpu"


class TestGrottoDcfEvalAllShape:
    def test_output_shape(self, dcf, s0s):
        cws = generate_point(dcf, s0s, alpha=107)
        out = evaluate_all(dcf, party=0, s0=s0s[0], cws=cws)
        assert out.shape == (2**16,)
        assert out.dtype == torch.bool
        assert out.device.type == "cpu"


class TestGrottoDcfReconstruction:
    def test_reconstruction_is_comparison(self, dcf, s0s):
        alpha = 107
        cws = generate_point(dcf, s0s, alpha=alpha)
        outputs = [evaluate_all(dcf, party, s0s[party], cws) for party in (0, 1)]
        expected = torch.arange(2**16, dtype=torch.int64) >= alpha
        assert torch.equal(torch.bitwise_xor(*outputs), expected)


class TestGrottoDcfPreprocessEval:
    def test_tree_shape(self, dcf, s0s):
        cws = generate_point(dcf, s0s, alpha=107)
        tree = parity_tree(dcf, 0, s0s[0], cws)
        assert tree.p.shape == (2 * 2**16 - 1,)
        assert tree.p.dtype == torch.bool
        assert tree.p.device.type == "cpu"

    def test_eval_matches_eval_all(self, dcf, s0s):
        alpha = 40000
        cws = generate_point(dcf, s0s, alpha=alpha)
        for party in (0, 1):
            full = evaluate_all(dcf, party, s0s[party], cws)
            tree = parity_tree(dcf, party, s0s[party], cws)
            for x in (0, 1, alpha - 1, alpha, alpha + 1, 2**16 - 1):
                assert dcf.eval_tree(tree, x) == bool(full[x])

    def test_rejects_out_of_domain_x(self, dcf, s0s):
        cws = generate_point(dcf, s0s, alpha=107)
        tree = parity_tree(dcf, 0, s0s[0], cws)
        with pytest.raises(RuntimeError, match="outside"):
            dcf.eval_tree(tree, 2**16)


class TestGrottoDcfCuda:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_preprocess_gpu_matches_cpu(self, dcf, s0s):
        cws = generate_point(dcf, s0s, alpha=107)
        for party in (0, 1):
            cpu = parity_tree(dcf, party, s0s[party], cws)
            gpu = parity_tree(dcf, party, s0s[party].to("cuda"), cws.to("cuda"))
            assert gpu.p.device.type == "cuda"
            assert torch.equal(cpu.p, gpu.p.cpu())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_eval_all_gpu_matches_cpu(self, dcf, s0s):
        cws = generate_point(dcf, s0s, alpha=107)
        for party in (0, 1):
            cpu = evaluate_all(dcf, party, s0s[party], cws)
            gpu = evaluate_all(dcf, party, s0s[party].to("cuda"), cws.to("cuda"))
            assert gpu.device.type == "cuda"
            assert torch.equal(cpu, gpu.cpu())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_above_24_bits_matches_cpu(self):
        dcf = make_point_scheme("GrottoDcf", bits=25)
        s0s = torch.randint(-2**31, 2**31, (2, 4), dtype=torch.int32)
        cws = generate_point(dcf, s0s, alpha=107)
        for party in (0, 1):
            tree_cpu = parity_tree(dcf, party, s0s[party], cws)
            tree_gpu = parity_tree(dcf, party, s0s[party].to("cuda"), cws.to("cuda"),
                                   frontier_depth=17)
            assert tree_gpu.p.device.type == "cuda"
            assert torch.equal(tree_cpu.p, tree_gpu.p.cpu())
            cpu = evaluate_all(dcf, party, s0s[party], cws)
            gpu = evaluate_all(dcf, party, s0s[party].to("cuda"), cws.to("cuda"),
                               frontier_depth=17)
            assert gpu.device.type == "cuda"
            assert torch.equal(cpu, gpu.cpu())

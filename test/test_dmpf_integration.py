"""DMPF component buffers, reconstruction, and execution-space boundaries."""

import pytest
import torch

from conftest import (evaluate_multi, evaluate_multi_all, generate_multi,
                      make_multi_scheme)


IN_BITS = 12
MAX_POINTS = 30
BUCKET_BITS = 10
T = 30


@pytest.fixture
def dmpf():
    return make_multi_scheme(False, bits=IN_BITS, max_points=MAX_POINTS,
                             bucket_bits=BUCKET_BITS)


@pytest.fixture
def alphas():
    return torch.randperm(2**IN_BITS, dtype=torch.int64)[:T]


@pytest.fixture
def betas():
    values = torch.randint(-2**31, 2**31, (T, 4), dtype=torch.int32)
    values[:, 3] &= ~1
    return values


@pytest.fixture
def material(dmpf, alphas, betas):
    return generate_multi(dmpf, alphas, betas)


class TestDmpfParams:
    def test_allocated_and_runtime_parameters(self, dmpf):
        assert dmpf.kBucketCount == 53
        assert dmpf.kMinPoints == T
        assert dmpf.kDomainSize == 2**IN_BITS
        assert dmpf.kBucketSize == (3 * 2**IN_BITS + 52) // 53
        assert dmpf.cws_shape[:2] == (53, BUCKET_BITS + 1)

    def test_wide_input_domain_is_supported(self):
        scheme = make_multi_scheme(False, bits=64, max_points=MAX_POINTS,
                                   bucket_bits=64)
        assert scheme.in_bits == 64
        assert scheme.kDomainSize == 2**64

    def test_rejects_too_small_bucket_bits(self):
        with pytest.raises((ValueError, RuntimeError)):
            make_multi_scheme(False, bits=IN_BITS, max_points=MAX_POINTS,
                              bucket_bits=5)

    def test_rejects_small_max_points(self):
        with pytest.raises(ValueError, match="max_points"):
            make_multi_scheme(False, bits=IN_BITS, max_points=29,
                              bucket_bits=BUCKET_BITS)


class TestDmpfGenBuffers:
    def test_shapes_and_devices(self, dmpf, material):
        assert tuple(material.cws.shape) == dmpf.cws_shape
        assert material.cws.dtype == torch.uint8
        assert material.seeds.shape == (2, dmpf.kBucketCount, 4)
        assert material.seeds.dtype == torch.int32
        assert material.seeds[0].is_contiguous()
        assert material.seeds[1].is_contiguous()
        assert material.sigma.shape == (4,)
        for tensor in (material.cws, material.seeds, material.sigma):
            assert tensor.device.type == "cpu"

    def test_runtime_parameters(self, dmpf, material):
        assert material.m_rt == dmpf.kBucketCount
        assert material.b_size_rt == dmpf.kBucketSize


class TestDmpfEvaluation:
    def test_batch_shape_and_reconstruction(self, dmpf, material, alphas, betas):
        outputs = [evaluate_multi(dmpf, material, party, alphas)[0]
                   for party in (False, True)]
        for output in outputs:
            assert output.shape == (T, 4)
            assert output.dtype == torch.int32
            assert output.device.type == "cpu"
        assert torch.equal(outputs[0] ^ outputs[1], betas)

    def test_zero_away_from_secret_points(self, dmpf, material, alphas):
        mask = torch.ones(2**IN_BITS, dtype=torch.bool)
        mask[alphas] = False
        xs = torch.arange(2**IN_BITS, dtype=torch.int64)[mask][:128]
        outputs = [evaluate_multi(dmpf, material, party, xs)[0]
                   for party in (False, True)]
        assert not torch.any(outputs[0] ^ outputs[1])

    def test_full_domain_reconstruction(self, dmpf, material, alphas, betas):
        outputs = [evaluate_multi_all(dmpf, material, party)[0]
                   for party in (False, True)]
        expected = torch.zeros((2**IN_BITS, 4), dtype=torch.int32)
        expected[alphas] = betas
        assert torch.equal(outputs[0] ^ outputs[1], expected)


class TestDmpfValidation:
    @pytest.mark.parametrize("wrong_layout", [False, True])
    def test_rejects_bad_seed_buffer(self, dmpf, material, alphas, betas, wrong_layout):
        shape = ((dmpf.kBucketCount, 2, 4) if wrong_layout
                 else (2, dmpf.kBucketCount, 4))
        wrong = torch.zeros(shape, dtype=torch.int32 if wrong_layout else torch.int64)
        with pytest.raises(RuntimeError, match="s0s must have shape|int32"):
            dmpf.gen(material.cws, material.sigma, wrong, alphas, betas, T)

    def test_rejects_bad_sigma_shape(self, dmpf, material, alphas, betas):
        with pytest.raises(RuntimeError, match="four|4"):
            dmpf.gen(material.cws, torch.zeros(5, dtype=torch.int32),
                     material.seeds, alphas, betas, T)

    @pytest.mark.parametrize("t", [29, MAX_POINTS + 1])
    def test_rejects_unsupported_point_count(self, dmpf, material, alphas, betas, t):
        with pytest.raises(RuntimeError, match="t must be between"):
            dmpf.gen(material.cws, material.sigma, material.seeds,
                     alphas, betas, t)

    def test_rejects_alpha_out_of_domain(self, dmpf, material, alphas, betas):
        invalid = alphas.clone()
        invalid[0] = 2**IN_BITS
        with pytest.raises(RuntimeError, match="outside the input domain"):
            dmpf.gen(material.cws, material.sigma, material.seeds,
                     invalid, betas, T)

    def test_rejects_duplicate_alphas(self, dmpf, material, alphas, betas):
        invalid = alphas.clone()
        invalid[1] = invalid[0]
        with pytest.raises(RuntimeError, match="distinct"):
            dmpf.gen(material.cws, material.sigma, material.seeds,
                     invalid, betas, T)

    def test_rejects_bad_query_dtype(self, dmpf, material, alphas):
        with pytest.raises(RuntimeError, match="64-bit"):
            evaluate_multi(dmpf, material, False, alphas.to(torch.int32))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_generation_requires_cpu(self, dmpf, material, alphas, betas):
        with pytest.raises(RuntimeError, match="CPU storage"):
            dmpf.gen(material.cws.cuda(), material.sigma.cuda(),
                     material.seeds.cuda(), alphas.cuda(), betas.cuda(), T)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestDmpfGpu:
    def test_batch_matches_cpu_and_reconstructs(self, dmpf, material, alphas, betas):
        gpu = [evaluate_multi(dmpf, material, party, alphas, device="cuda:0")[0]
               for party in (False, True)]
        cpu = evaluate_multi(dmpf, material, False, alphas)[0]
        assert gpu[0].device == torch.device("cuda:0")
        assert torch.equal(cpu, gpu[0].cpu())
        assert torch.equal((gpu[0] ^ gpu[1]).cpu(), betas)

    def test_full_domain_matches_cpu(self, dmpf, material, alphas, betas):
        gpu = [evaluate_multi_all(dmpf, material, party, device="cuda:0")[0]
               for party in (False, True)]
        cpu = evaluate_multi_all(dmpf, material, False)[0]
        assert torch.equal(cpu, gpu[0].cpu())
        expected = torch.zeros_like(cpu)
        expected[alphas] = betas
        assert torch.equal((gpu[0] ^ gpu[1]).cpu(), expected)

    def test_host_only_prp_rejects_cuda(self, alphas, betas):
        scheme = make_multi_scheme(False, bits=IN_BITS, max_points=MAX_POINTS,
                                   bucket_bits=BUCKET_BITS, host_only=True)
        material = generate_multi(scheme, alphas, betas)
        with pytest.raises((TypeError, RuntimeError), match="CUDA"):
            evaluate_multi(scheme, material, False, alphas, device="cuda:0")

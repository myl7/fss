"""VDMPF component buffers and explicit chain/Merkle proof protocols."""

import pytest
import torch

from conftest import (evaluate_multi, evaluate_multi_all, generate_multi,
                      make_multi_scheme)


IN_BITS = 12
MAX_POINTS = 30
BUCKET_BITS = 10
T = 30


@pytest.fixture
def vdmpf():
    return make_multi_scheme(True, bits=IN_BITS, max_points=MAX_POINTS,
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
def material(vdmpf, alphas, betas):
    return generate_multi(vdmpf, alphas, betas)


class TestVdmpfParams:
    def test_allocated_and_runtime_parameters(self, vdmpf):
        assert vdmpf.kBucketCount == 53
        assert vdmpf.kMinPoints == T
        assert vdmpf.kDomainSize == 2**IN_BITS
        assert vdmpf.kBucketSize == (3 * 2**IN_BITS + 52) // 53

    def test_wide_input_domain_is_supported(self):
        scheme = make_multi_scheme(True, bits=64, max_points=MAX_POINTS,
                                   bucket_bits=64)
        assert scheme.in_bits == 64
        assert scheme.kDomainSize == 2**64

    def test_rejects_inappropriate_hash(self):
        from fss_crypto import Vdmpf, group, hash, prg, prp
        with pytest.raises(ValueError, match="appropriate explicit hash"):
            Vdmpf(IN_BITS, MAX_POINTS, BUCKET_BITS, group.Bytes(),
                  prg.ChaCha((1, 2), mul=2), hash.Blake3(bytes(32)),
                  hash.Aes128Ccr(prg.Aes128Soft(1, (bytes(16),))),
                  prp.ChaChaFeistel())


class TestVdmpfGenBuffers:
    def test_shapes_and_devices(self, vdmpf, material):
        assert tuple(material.cws.shape) == vdmpf.cws_shape
        assert material.cws.dtype == torch.uint8
        assert material.cs.shape == (vdmpf.kBucketCount, 4, 4)
        assert material.cs.dtype == torch.int32
        assert material.seeds.shape == (2, vdmpf.kBucketCount, 4)
        assert material.seeds[0].is_contiguous()
        assert material.seeds[1].is_contiguous()
        assert material.sigma.shape == (4,)
        for tensor in (material.cws, material.cs, material.seeds, material.sigma):
            assert tensor.device.type == "cpu"

    def test_runtime_parameters(self, vdmpf, material):
        assert material.m_rt == vdmpf.kBucketCount
        assert material.b_size_rt == vdmpf.kBucketSize


class TestVdmpfEvaluation:
    @pytest.mark.parametrize("merkle", [False, True])
    def test_batch_reconstruction_and_proofs(self, vdmpf, material, alphas, betas, merkle):
        results = [evaluate_multi(vdmpf, material, party, alphas, merkle=merkle)
                   for party in (False, True)]
        for outputs, proof in results:
            assert outputs.shape == (T, 4)
            assert outputs.dtype == torch.int32
            assert proof.shape == (4, 4)
            assert proof.dtype == torch.int32
        assert torch.equal(results[0][0] ^ results[1][0], betas)
        assert vdmpf.verify(results[0][1], results[1][1])

    def test_zero_away_from_secret_points(self, vdmpf, material, alphas):
        mask = torch.ones(2**IN_BITS, dtype=torch.bool)
        mask[alphas] = False
        xs = torch.arange(2**IN_BITS, dtype=torch.int64)[mask][:128]
        first, proof0 = evaluate_multi(vdmpf, material, False, xs)
        second, proof1 = evaluate_multi(vdmpf, material, True, xs)
        assert not torch.any(first ^ second)
        assert vdmpf.verify(proof0, proof1)

    @pytest.mark.parametrize("merkle", [False, True])
    def test_full_domain_reconstruction(self, vdmpf, material, alphas, betas, merkle):
        results = [evaluate_multi_all(vdmpf, material, party, merkle=merkle)
                   for party in (False, True)]
        expected = torch.zeros((2**IN_BITS, 4), dtype=torch.int32)
        expected[alphas] = betas
        assert torch.equal(results[0][0] ^ results[1][0], expected)
        assert vdmpf.verify(results[0][1], results[1][1])

    def test_rejects_tampered_proof(self, vdmpf, material, alphas):
        _, first = evaluate_multi(vdmpf, material, False, alphas)
        _, second = evaluate_multi(vdmpf, material, True, alphas)
        second[0, 0] ^= 1
        assert not vdmpf.verify(first, second)


class TestVdmpfValidation:
    def test_rejects_bad_seed_layout(self, vdmpf, material, alphas, betas):
        wrong = torch.zeros((vdmpf.kBucketCount, 2, 4), dtype=torch.int32)
        with pytest.raises(RuntimeError, match="s0s must have shape"):
            vdmpf.gen(material.cws, material.cs, material.sigma,
                      wrong, alphas, betas, T)

    def test_rejects_bad_payload_shape(self, vdmpf, material, alphas):
        with pytest.raises(RuntimeError, match="four|4"):
            vdmpf.gen(material.cws, material.cs, material.sigma,
                      material.seeds, alphas, torch.zeros((T, 5), dtype=torch.int32), T)

    def test_rejects_unsupported_point_count(self, vdmpf, material, alphas, betas):
        with pytest.raises(RuntimeError, match="t must be between"):
            vdmpf.gen(material.cws, material.cs, material.sigma,
                      material.seeds, alphas[:29], betas[:29], 29)

    def test_rejects_alpha_out_of_domain(self, vdmpf, material, alphas, betas):
        invalid = alphas.clone()
        invalid[0] = -1
        with pytest.raises(RuntimeError, match="outside the input domain"):
            vdmpf.gen(material.cws, material.cs, material.sigma,
                      material.seeds, invalid, betas, T)

    def test_rejects_bad_check_buffer(self, vdmpf, material, alphas):
        with pytest.raises(RuntimeError):
            vdmpf.batch_eval(False, material.sigma, material.m_rt,
                             material.b_size_rt, material.seeds[0], material.cws,
                             torch.zeros((vdmpf.kBucketCount, 4), dtype=torch.int32),
                             alphas, torch.empty((T, 4), dtype=torch.int32),
                             torch.empty((4, 4), dtype=torch.int32))

    def test_rejects_bad_proof_shape(self, vdmpf, material, alphas):
        _, first = evaluate_multi(vdmpf, material, False, alphas)
        _, second = evaluate_multi(vdmpf, material, True, alphas)
        with pytest.raises(RuntimeError, match="four|4"):
            vdmpf.verify(first, second.flatten())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_rejects_mixed_devices(self, vdmpf, material, alphas):
        with pytest.raises(RuntimeError, match="same device"):
            vdmpf.batch_eval(False, material.sigma, material.m_rt,
                             material.b_size_rt, material.seeds[0], material.cws,
                             material.cs, alphas.cuda(),
                             torch.empty((T, 4), dtype=torch.int32),
                             torch.empty((4, 4), dtype=torch.int32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestVdmpfGpu:
    def test_merkle_batch_matches_cpu_and_reconstructs(self, vdmpf, material, alphas, betas):
        cpu = [evaluate_multi(vdmpf, material, party, alphas, merkle=True)
               for party in (False, True)]
        gpu = [evaluate_multi(vdmpf, material, party, alphas, merkle=True, device="cuda:0")
               for party in (False, True)]
        for party in (0, 1):
            assert gpu[party][0].device == torch.device("cuda:0")
            assert torch.equal(cpu[party][0], gpu[party][0].cpu())
            assert torch.equal(cpu[party][1], gpu[party][1].cpu())
        assert torch.equal((gpu[0][0] ^ gpu[1][0]).cpu(), betas)
        assert vdmpf.verify(gpu[0][1], gpu[1][1])
        tampered = gpu[1][1].clone()
        tampered[0, 0] ^= 1
        assert not vdmpf.verify(gpu[0][1], tampered)

    def test_merkle_full_domain_matches_cpu(self, vdmpf, material):
        cpu = evaluate_multi_all(vdmpf, material, False, merkle=True)
        gpu = evaluate_multi_all(vdmpf, material, False, merkle=True, device="cuda:0")
        assert torch.equal(cpu[0], gpu[0].cpu())
        assert torch.equal(cpu[1], gpu[1].cpu())

    def test_chain_cuda_requires_explicit_merkle_method(self, vdmpf, material, alphas):
        with pytest.raises(RuntimeError, match="chain.*CPU"):
            evaluate_multi(vdmpf, material, False, alphas, device="cuda:0")

    def test_host_only_prp_rejects_cuda(self, alphas, betas):
        scheme = make_multi_scheme(True, bits=IN_BITS, max_points=MAX_POINTS,
                                   bucket_bits=BUCKET_BITS, host_only=True)
        material = generate_multi(scheme, alphas, betas)
        with pytest.raises((TypeError, RuntimeError), match="CUDA"):
            evaluate_multi(scheme, material, False, alphas, merkle=True, device="cuda:0")

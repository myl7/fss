"""VDPF component storage, reconstruction, proof protocols, and devices."""

import pytest
import torch

from conftest import (evaluate_vdpf, evaluate_vdpf_all, generate_vdpf,
                      make_point_scheme, random_blocks)


@pytest.fixture
def vdpf():
    return make_point_scheme("Vdpf")


@pytest.fixture
def beta():
    return torch.tensor([0, 0, 0, 604], dtype=torch.int32)


@pytest.fixture
def state(vdpf, beta):
    return generate_vdpf(vdpf, random_blocks((2, 4)), 107, beta)


class TestVdpfGen:
    def test_shapes(self, vdpf, state):
        assert state.seeds.shape == (2, 4)
        assert state.cws.shape == vdpf.cws_shape
        assert state.cws.dtype == torch.uint8
        assert state.cs.shape == (4, 4)
        assert state.cs.dtype == torch.int32
        assert all(t.device.type == "cpu" for t in (state.seeds, state.cws, state.cs))

    def test_single_attempt_preserves_input_seeds(self, vdpf, beta):
        seeds = random_blocks((2, 4))
        original = seeds.clone()
        cws = torch.empty(vdpf.cws_shape, dtype=torch.uint8)
        cs = torch.empty((4, 4), dtype=torch.int32)
        assert vdpf.gen(cws, cs, seeds, 5, beta) in (0, 1)
        assert torch.equal(seeds, original)


class TestVdpfEval:
    def test_output_shapes(self, vdpf, state):
        y, corrected = evaluate_vdpf(vdpf, state, False, 50)
        assert y.shape == (4,) and y.dtype == torch.int32
        assert corrected.shape == (4, 4) and corrected.dtype == torch.int32

    def test_rejects_out_of_domain_x(self, vdpf, state):
        with pytest.raises(RuntimeError, match="outside.*domain"):
            evaluate_vdpf(vdpf, state, False, 2**16)


class TestVdpfEvalAll:
    def test_merkle_matches_chain_outputs(self, vdpf, state):
        for party in (0, 1):
            chain, _ = evaluate_vdpf_all(vdpf, state, party)
            merkle, proof = evaluate_vdpf_all(vdpf, state, party, merkle=True)
            assert chain.shape == (2**16, 4)
            assert proof.shape == (4, 4)
            assert torch.equal(chain, merkle)

    def test_reconstruction(self, vdpf, state, beta):
        first, pi0 = evaluate_vdpf_all(vdpf, state, False)
        second, pi1 = evaluate_vdpf_all(vdpf, state, True)
        expected = torch.zeros_like(first)
        expected[107] = beta
        assert torch.equal(first ^ second, expected)
        assert vdpf.verify(pi0, pi1)


class TestVdpfProofs:
    @staticmethod
    def corrected_hashes(vdpf, state):
        return [torch.stack([evaluate_vdpf(vdpf, state, party, x)[1]
                             for x in (0, 1, 106, 107, 108, 65535)])
                for party in (0, 1)]

    @pytest.mark.parametrize("merkle", [False, True])
    def test_prove_and_verify(self, vdpf, state, merkle):
        corrected = self.corrected_hashes(vdpf, state)
        proofs = [torch.empty((4, 4), dtype=torch.int32) for _ in range(2)]
        prove = vdpf.prove_merkle if merkle else vdpf.prove
        for party in (0, 1):
            prove(corrected[party], state.cs, proofs[party])
        assert vdpf.verify(*proofs) is True

    def test_tampered_proof_rejected(self, vdpf, state):
        corrected = self.corrected_hashes(vdpf, state)
        proofs = [torch.empty((4, 4), dtype=torch.int32) for _ in range(2)]
        for party in (0, 1):
            vdpf.prove(corrected[party], state.cs, proofs[party])
        assert vdpf.verify(*proofs)
        bad = proofs[1].clone()
        bad[2, 1] ^= 1
        assert vdpf.verify(proofs[0], bad) is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestVdpfCuda:
    def test_eval_matches_cpu(self, vdpf, state):
        for party in (0, 1):
            cpu = evaluate_vdpf(vdpf, state, party, 12345)
            gpu = evaluate_vdpf(vdpf, state, party, 12345, device="cuda")
            assert all(torch.equal(left, right.cpu()) for left, right in zip(cpu, gpu))

    def test_merkle_full_matches_cpu(self, vdpf, state):
        for party in (0, 1):
            cpu = evaluate_vdpf_all(vdpf, state, party, merkle=True)
            gpu = evaluate_vdpf_all(vdpf, state, party, merkle=True, device="cuda")
            assert all(torch.equal(left, right.cpu()) for left, right in zip(cpu, gpu))

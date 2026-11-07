"""VDPF sparse batch evaluation over an ordered query sequence."""

import pytest
import torch

from conftest import generate_vdpf, make_point_scheme, random_blocks


IN_BITS = 8
# Sparse, unsorted, duplicated, odd-length sequence; includes ALPHA.
QUERIES = [3, 200, 3, 107, 255]
ALPHA = 107
BETA = torch.tensor([0, 0, 0, 604], dtype=torch.int32)


@pytest.fixture(scope="module")
def vdpf():
    return make_point_scheme("Vdpf", bits=IN_BITS)


@pytest.fixture(scope="module")
def state(vdpf):
    return generate_vdpf(vdpf, random_blocks((2, 4)), ALPHA, BETA)


def scalar_party(vdpf, state, party):
    """Return stacked ys and corrected hashes from per-input eval calls."""
    ys, hashes = [], []
    for x in QUERIES:
        y = torch.empty((4,), dtype=torch.int32)
        hashes.append(vdpf.eval(party, state.seeds[int(party)], state.cws,
                                state.cs, x, y))
        ys.append(y)
    return torch.stack(ys), torch.stack(hashes)


def batch_party(vdpf, state, party, merkle):
    ys = torch.empty((len(QUERIES), 4), dtype=torch.int32)
    pi = torch.empty((4, 4), dtype=torch.int32)
    method = vdpf.batch_eval_merkle if merkle else vdpf.batch_eval
    result = method(party, state.seeds[int(party)], state.cws, state.cs,
                    torch.tensor(QUERIES, dtype=torch.uint8), ys, pi)
    return result, ys, pi


class TestVdpfBatchChain:
    def test_matches_scalar_eval_and_prove(self, vdpf, state):
        for party in (False, True):
            scalar_ys, hashes = scalar_party(vdpf, state, party)
            pi = torch.empty((4, 4), dtype=torch.int32)
            vdpf.prove(hashes, state.cs, pi)
            _, ys, batch_pi = batch_party(vdpf, state, party, merkle=False)
            assert torch.equal(ys, scalar_ys)
            assert torch.equal(batch_pi, pi)

    def test_reconstructs_and_verifies(self, vdpf, state):
        results = [batch_party(vdpf, state, party, merkle=False)[1:] for party in (False, True)]
        combined = results[0][0] ^ results[1][0]
        expected = torch.zeros((len(QUERIES), 4), dtype=torch.int32)
        expected[QUERIES.index(ALPHA)] = BETA
        assert torch.equal(combined, expected)
        assert vdpf.verify(results[0][1], results[1][1])
        tampered = results[1][1].clone()
        tampered[0, 0] ^= 1
        assert not vdpf.verify(results[0][1], tampered)


class TestVdpfBatchMerkle:
    def test_matches_scalar_eval_and_prove_merkle(self, vdpf, state):
        assert len(QUERIES) % 2 == 1
        for party in (False, True):
            scalar_ys, hashes = scalar_party(vdpf, state, party)
            pi = torch.empty((4, 4), dtype=torch.int32)
            vdpf.prove_merkle(hashes, state.cs, pi)
            _, ys, batch_pi = batch_party(vdpf, state, party, merkle=True)
            assert torch.equal(ys, scalar_ys)
            assert torch.equal(batch_pi, pi)

    def test_reconstructs_and_verifies(self, vdpf, state):
        results = [batch_party(vdpf, state, party, merkle=True)[1:] for party in (False, True)]
        combined = results[0][0] ^ results[1][0]
        expected = torch.zeros((len(QUERIES), 4), dtype=torch.int32)
        expected[QUERIES.index(ALPHA)] = BETA
        assert torch.equal(combined, expected)
        assert vdpf.verify(results[0][1], results[1][1])


class TestVdpfBatchOutputs:
    def test_writes_caller_buffers_and_returns_none(self, vdpf, state):
        ys = torch.full((len(QUERIES), 4), 7, dtype=torch.int32)
        pi = torch.empty((4, 4), dtype=torch.int32)
        ys_ptr, pi_ptr = ys.data_ptr(), pi.data_ptr()
        for method in (vdpf.batch_eval, vdpf.batch_eval_merkle):
            assert method(False, state.seeds[0], state.cws, state.cs,
                          torch.tensor(QUERIES, dtype=torch.uint8), ys, pi) is None
            assert ys.data_ptr() == ys_ptr and pi.data_ptr() == pi_ptr
        scalar_ys, _ = scalar_party(vdpf, state, False)
        assert torch.equal(ys, scalar_ys)

    def test_chain_and_merkle_proofs_differ(self, vdpf, state):
        proofs = [batch_party(vdpf, state, False, merkle=merkle)[2]
                  for merkle in (False, True)]
        assert not torch.equal(proofs[0], proofs[1])


class TestVdpfBatchValidation:
    def queries(self, dtype=torch.uint8):
        return torch.tensor(QUERIES, dtype=dtype)

    def test_rejects_empty_xs(self, vdpf, state):
        with pytest.raises(RuntimeError, match="nonempty"):
            vdpf.batch_eval(False, state.seeds[0], state.cws, state.cs,
                            torch.empty(0, dtype=torch.uint8),
                            torch.empty((0, 4), dtype=torch.int32),
                            torch.empty((4, 4), dtype=torch.int32))

    @pytest.mark.parametrize("count", [len(QUERIES) - 1, len(QUERIES) + 1])
    def test_rejects_ys_count_mismatch(self, vdpf, state, count):
        with pytest.raises(RuntimeError, match="wrong number of blocks"):
            vdpf.batch_eval(False, state.seeds[0], state.cws, state.cs,
                            self.queries(), torch.empty((count, 4), dtype=torch.int32),
                            torch.empty((4, 4), dtype=torch.int32))

    def test_rejects_wrong_input_dtype(self, vdpf, state):
        with pytest.raises(RuntimeError, match="8-bit words"):
            vdpf.batch_eval(False, state.seeds[0], state.cws, state.cs,
                            self.queries(torch.int64),
                            torch.empty((len(QUERIES), 4), dtype=torch.int32),
                            torch.empty((4, 4), dtype=torch.int32))

    def test_rejects_noncontiguous_xs(self, vdpf, state):
        xs = torch.zeros(10, dtype=torch.uint8)[::2]
        with pytest.raises(RuntimeError, match="contiguous"):
            vdpf.batch_eval(False, state.seeds[0], state.cws, state.cs, xs,
                            torch.empty((5, 4), dtype=torch.int32),
                            torch.empty((4, 4), dtype=torch.int32))

    def test_rejects_bad_pi_shape(self, vdpf, state):
        with pytest.raises(RuntimeError, match="wrong number of blocks"):
            vdpf.batch_eval_merkle(False, state.seeds[0], state.cws, state.cs,
                                   self.queries(),
                                   torch.empty((len(QUERIES), 4), dtype=torch.int32),
                                   torch.empty((5, 4), dtype=torch.int32))

    def test_rejects_bad_cws_layout(self, vdpf, state):
        with pytest.raises(ValueError, match="correction words"):
            vdpf.batch_eval(False, state.seeds[0],
                            torch.empty((3,), dtype=torch.uint8),
                            state.cs, self.queries(),
                            torch.empty((len(QUERIES), 4), dtype=torch.int32),
                            torch.empty((4, 4), dtype=torch.int32))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_rejects_mixed_devices(self, vdpf, state):
        with pytest.raises(RuntimeError, match="same device"):
            vdpf.batch_eval(False, state.seeds[0], state.cws, state.cs,
                            self.queries().cuda(),
                            torch.empty((len(QUERIES), 4), dtype=torch.int32),
                            torch.empty((4, 4), dtype=torch.int32))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_rejects_chain_on_cuda(self, vdpf, state):
        ys = torch.empty((len(QUERIES), 4), dtype=torch.int32, device="cuda")
        pi = torch.empty((4, 4), dtype=torch.int32, device="cuda")
        with pytest.raises(RuntimeError, match="chain.*CPU"):
            vdpf.batch_eval(False, state.seeds[0].cuda(), state.cws.cuda(),
                            state.cs.cuda(), self.queries().cuda(), ys, pi)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestVdpfBatchGpu:
    def cuda_party(self, vdpf, state, party, threads_per_block=256, stream=None):
        ys = torch.empty((len(QUERIES), 4), dtype=torch.int32, device="cuda")
        pi = torch.empty((4, 4), dtype=torch.int32, device="cuda")
        kwargs = {} if threads_per_block is None else {
            "threads_per_block": threads_per_block}
        if stream is None:
            vdpf.batch_eval_merkle(party, state.seeds[int(party)].cuda(),
                                   state.cws.cuda(), state.cs.cuda(),
                                   torch.tensor(QUERIES, dtype=torch.uint8,
                                                device="cuda"),
                                   ys, pi, **kwargs)
        else:
            with torch.cuda.stream(stream):
                vdpf.batch_eval_merkle(party, state.seeds[int(party)].cuda(),
                                       state.cws.cuda(), state.cs.cuda(),
                                       torch.tensor(QUERIES, dtype=torch.uint8,
                                                    device="cuda"),
                                       ys, pi, **kwargs)
        torch.cuda.synchronize()
        return ys.cpu(), pi.cpu()

    def test_merkle_matches_cpu_and_reconstructs(self, vdpf, state):
        for party in (False, True):
            _, cpu_ys, cpu_pi = batch_party(vdpf, state, party, merkle=True)
            gpu_ys, gpu_pi = self.cuda_party(vdpf, state, party,
                                             threads_per_block=512)
            assert torch.equal(cpu_ys, gpu_ys)
            assert torch.equal(cpu_pi, gpu_pi)
        first, pi0 = self.cuda_party(vdpf, state, False)
        second, pi1 = self.cuda_party(vdpf, state, True)
        expected = torch.zeros((len(QUERIES), 4), dtype=torch.int32)
        expected[QUERIES.index(ALPHA)] = BETA
        assert torch.equal(first ^ second, expected)
        assert vdpf.verify(pi0, pi1)

    def test_merkle_runs_on_a_nondefault_stream(self, vdpf, state):
        stream = torch.cuda.Stream()
        _, pi = self.cuda_party(vdpf, state, False, stream=stream)
        _, _, cpu_pi = batch_party(vdpf, state, False, merkle=True)
        assert torch.equal(pi, cpu_pi)


WIDE_BITS = 100


@pytest.fixture(scope="module")
def wide_scheme():
    return make_point_scheme("Vdpf", bits=WIDE_BITS)


@pytest.fixture(scope="module")
def wide_key(wide_scheme):
    beta = torch.tensor([1, 2, 3, 604], dtype=torch.int32)
    return generate_vdpf(wide_scheme, random_blocks((2, 4)), (1 << 99) + 5, beta)


class TestVdpfBatchIn128:
    """Batch queries over a 100-bit domain use [N, 2] limbs, low limb first."""

    def points(self):
        return [(1 << 99) + 5, 0, (1 << WIDE_BITS) - 1, 7, 0]

    def limbs(self, values):
        # Write each 128-bit value as two int64 limbs, low limb first, keeping
        # the low limb inside the signed range.
        mask = (1 << 64) - 1
        lows = [v & mask for v in values]
        signed = [low - (1 << 64) if low >= (1 << 63) else low for low in lows]
        low = torch.tensor(signed, dtype=torch.int64)
        high = torch.tensor([v >> 64 for v in values], dtype=torch.int64)
        return torch.stack((low, high), dim=1)

    def test_matches_scalar_eval_for_batch_and_merkle(self, wide_scheme, wide_key):
        for merkle in (False, True):
            xs = self.limbs(self.points())
            batch_ys = torch.empty((5, 4), dtype=torch.int32)
            pi = torch.empty((4, 4), dtype=torch.int32)
            method = wide_scheme.batch_eval_merkle if merkle else wide_scheme.batch_eval
            method(False, wide_key.seeds[0], wide_key.cws, wide_key.cs, xs, batch_ys, pi)
            scalar_ys = []
            for x in self.points():
                y = torch.empty((4,), dtype=torch.int32)
                wide_scheme.eval(False, wide_key.seeds[0], wide_key.cws,
                                 wide_key.cs, x, y)
                scalar_ys.append(y)
            assert torch.equal(batch_ys, torch.stack(scalar_ys))
            assert torch.any(pi != 0)

    def test_rejects_scalar_and_batch_inputs_out_of_domain(self, wide_scheme, wide_key):
        y = torch.empty((4,), dtype=torch.int32)
        with pytest.raises(RuntimeError, match="outside the configured domain"):
            wide_scheme.eval(False, wide_key.seeds[0], wide_key.cws, wide_key.cs,
                            1 << WIDE_BITS, y)
        xs = self.limbs([0] * 4 + [1 << WIDE_BITS])
        with pytest.raises(RuntimeError, match="outside the input domain"):
            wide_scheme.batch_eval(False, wide_key.seeds[0], wide_key.cws,
                                   wide_key.cs, xs,
                                   torch.empty((5, 4), dtype=torch.int32),
                                   torch.empty((4, 4), dtype=torch.int32))

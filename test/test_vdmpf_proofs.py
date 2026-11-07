"""VDMPF per-input evaluation and split chain/Merkle proof accumulation."""

import pytest
import torch

from conftest import generate_multi, make_multi_scheme, make_prg
from fss_crypto import Vdmpf, group, hash, prp


IN_BITS = 12
MAX_POINTS = 30
BUCKET_BITS = 10
T = 30
# Sparse, unsorted, duplicated, odd-length sequence.
QUERIES = [0, 173, 173, 5, 4095]


@pytest.fixture(scope="module")
def vdmpf():
    return make_multi_scheme(True, bits=IN_BITS, max_points=MAX_POINTS,
                             bucket_bits=BUCKET_BITS)


@pytest.fixture(scope="module")
def material(vdmpf):
    alphas = torch.randperm(2**IN_BITS, dtype=torch.int64)[:T]
    betas = torch.randint(-2**31, 2**31, (T, 4), dtype=torch.int32)
    betas[:, 3] &= ~1
    return generate_multi(vdmpf, alphas, betas), alphas, betas


def scalar_party(vdmpf, key, party, b_size_rt=None):
    """Return stacked ys and proof values from per-input eval calls."""
    size = key.b_size_rt if b_size_rt is None else b_size_rt
    ys, values = [], []
    for x in QUERIES:
        y = torch.empty((4,), dtype=torch.int32)
        values.append(vdmpf.eval(party, key.sigma, size, key.seeds[int(party)],
                                 key.cws, key.cs, x, y))
        ys.append(y)
    return torch.stack(ys), torch.stack(values)


def fused_party(vdmpf, key, party, merkle, xs=None):
    xs = torch.tensor(QUERIES, dtype=torch.int64) if xs is None else xs
    ys = torch.empty((xs.shape[0], 4), dtype=torch.int32)
    pi = torch.empty((4, 4), dtype=torch.int32)
    method = vdmpf.batch_eval_merkle if merkle else vdmpf.batch_eval
    method(party, key.sigma, key.b_size_rt, key.seeds[int(party)], key.cws,
           key.cs, xs, ys, pi)
    return ys, pi


class TestVdmpfEval:
    def test_returns_candidate_axis_proof_and_writes_y(self, vdmpf, material):
        key, _, _ = material
        y = torch.full((4,), 7, dtype=torch.int32)
        y_ptr = y.data_ptr()
        proof = vdmpf.eval(False, key.sigma, key.b_size_rt, key.seeds[0],
                           key.cws, key.cs, QUERIES[0], y)
        assert proof.shape == (3, 4, 4)
        assert proof.dtype == torch.int32
        assert y.data_ptr() == y_ptr
        assert not torch.equal(y, torch.full((4,), 7, dtype=torch.int32))
        assert not torch.equal(proof[0], proof[1])
        assert not torch.equal(proof[0], proof[2])
        again = vdmpf.eval(False, key.sigma, key.b_size_rt, key.seeds[0],
                            key.cws, key.cs, QUERIES[0], y)
        assert torch.equal(proof, again)
        assert again.data_ptr() != y_ptr

    @pytest.mark.parametrize("merkle", [False, True])
    def test_split_path_matches_fused_batch(self, vdmpf, material, merkle):
        key, _, _ = material
        prove = vdmpf.prove_merkle if merkle else vdmpf.prove
        fused = [fused_party(vdmpf, key, party, merkle) for party in (False, True)]
        proofs = []
        for party in (False, True):
            ys, values = scalar_party(vdmpf, key, party)
            pi = torch.empty((4, 4), dtype=torch.int32)
            assert prove(key.sigma, key.b_size_rt, key.cs,
                         torch.tensor(QUERIES, dtype=torch.int64), values,
                         pi) is None
            assert torch.equal(ys, fused[int(party)][0])
            assert torch.equal(pi, fused[int(party)][1])
            proofs.append(pi)
        assert vdmpf.verify(*proofs)
        tampered = proofs[1].clone()
        tampered[0, 0] ^= 1
        assert not vdmpf.verify(proofs[0], tampered)

    @pytest.mark.parametrize("merkle", [False, True])
    def test_split_path_reconstructs_secret_shares(self, vdmpf, material, merkle):
        key, alphas, betas = material
        shares = []
        for party in (False, True):
            ys, _ = scalar_party(vdmpf, key, party)
            shares.append(ys)
        combined = shares[0] ^ shares[1]
        expected = torch.zeros((len(QUERIES), 4), dtype=torch.int32)
        for position, x in enumerate(QUERIES):
            hits = (alphas == x).nonzero()
            if hits.numel():
                expected[position] = betas[hits[0, 0]]
        assert torch.equal(combined, expected)

    @pytest.mark.parametrize("merkle", [False, True])
    def test_prove_pairs_xs_order_with_proof_values(self, vdmpf, material, merkle):
        key, _, _ = material
        prove = vdmpf.prove_merkle if merkle else vdmpf.prove
        xs = torch.tensor([11, 12, 13], dtype=torch.int64)
        values = torch.stack([
            vdmpf.eval(False, key.sigma, key.b_size_rt, key.seeds[0],
                       key.cws, key.cs, int(x),
                       torch.empty((4,), dtype=torch.int32))
            for x in xs])
        pi = torch.empty((4, 4), dtype=torch.int32)
        prove(key.sigma, key.b_size_rt, key.cs, xs, values, pi)
        repeated = torch.empty((4, 4), dtype=torch.int32)
        prove(key.sigma, key.b_size_rt, key.cs, xs, values, repeated)
        assert torch.equal(pi, repeated)
        for wrong_xs, wrong_values in ((xs.flip(0), values), (xs, values.flip(0))):
            mismatched = torch.empty((4, 4), dtype=torch.int32)
            prove(key.sigma, key.b_size_rt, key.cs, wrong_xs, wrong_values,
                  mismatched)
            assert not torch.equal(pi, mismatched)

    def test_chain_and_merkle_proofs_differ(self, vdmpf, material):
        key, _, _ = material
        _, values = scalar_party(vdmpf, key, False)
        xs = torch.tensor(QUERIES, dtype=torch.int64)
        proofs = []
        for prove in (vdmpf.prove, vdmpf.prove_merkle):
            pi = torch.empty((4, 4), dtype=torch.int32)
            prove(key.sigma, key.b_size_rt, key.cs, xs, values, pi)
            proofs.append(pi)
        assert not torch.equal(proofs[0], proofs[1])


class TestVdmpfEvalValidation:
    def test_rejects_scalar_input_out_of_domain(self, vdmpf, material):
        key, _, _ = material
        y = torch.empty((4,), dtype=torch.int32)
        with pytest.raises(RuntimeError, match="outside the configured domain"):
            vdmpf.eval(False, key.sigma, key.b_size_rt, key.seeds[0],
                       key.cws, key.cs, 1 << IN_BITS, y)

    def test_rejects_bool_scalar_input_and_size(self, vdmpf, material):
        key, _, _ = material
        y = torch.empty((4,), dtype=torch.int32)
        with pytest.raises(RuntimeError, match="not bool"):
            vdmpf.eval(False, key.sigma, key.b_size_rt, key.seeds[0],
                       key.cws, key.cs, True, y)
        with pytest.raises(RuntimeError, match="not bool"):
            vdmpf.eval(False, key.sigma, True, key.seeds[0],
                       key.cws, key.cs, 5, y)

    @pytest.mark.parametrize("size", [0, -1, 1, 1 << (BUCKET_BITS + 1)])
    def test_rejects_sizes_outside_bucket_domain(self, vdmpf, material, size):
        key, _, _ = material
        y = torch.empty((4,), dtype=torch.int32)
        with pytest.raises(RuntimeError, match="nonnegative|bucket"):
            vdmpf.eval(False, key.sigma, size, key.seeds[0], key.cws,
                       key.cs, 5, y)

    @pytest.mark.parametrize("seeds,cs,y", [
        (None, None, torch.empty((5, 4), dtype=torch.int32)),
        (torch.empty((4, 4), dtype=torch.int32), None,
         torch.empty((4,), dtype=torch.int32)),
        (None, torch.empty((4, 4, 4), dtype=torch.int32),
         torch.empty((4,), dtype=torch.int32)),
    ])
    def test_rejects_bad_block_counts(self, vdmpf, material, seeds, cs, y):
        key, _, _ = material
        if seeds is None:
            seeds = key.seeds[0]
        if cs is None:
            cs = key.cs
        with pytest.raises(RuntimeError, match="wrong number of blocks"):
            vdmpf.eval(False, key.sigma, key.b_size_rt, seeds, key.cws, cs, 5, y)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_rejects_cuda_buffers(self, vdmpf, material):
        key, _, _ = material
        y = torch.empty((4,), dtype=torch.int32, device="cuda")
        with pytest.raises(RuntimeError, match="CPU storage"):
            vdmpf.eval(False, key.sigma, key.b_size_rt, key.seeds[0],
                       key.cws, key.cs, 5, y)
        pi = torch.empty((4, 4), dtype=torch.int32, device="cuda")
        with pytest.raises(RuntimeError, match="CPU storage"):
            vdmpf.prove(key.sigma, key.b_size_rt, key.cs,
                        torch.tensor([5], dtype=torch.int64),
                        torch.zeros((1, 3, 4, 4), dtype=torch.int32), pi)


class TestVdmpfProveValidation:
    @staticmethod
    def split(vdmpf, key, count=3):
        xs = torch.tensor([9, 10, 11][:count], dtype=torch.int64)
        values = torch.stack([
            vdmpf.eval(False, key.sigma, key.b_size_rt, key.seeds[0],
                       key.cws, key.cs, int(x),
                       torch.empty((4,), dtype=torch.int32))
            for x in xs])
        return xs, values

    def test_rejects_empty_xs(self, vdmpf, material):
        key, _, _ = material
        with pytest.raises(RuntimeError, match="nonempty"):
            vdmpf.prove(key.sigma, key.b_size_rt, key.cs,
                        torch.empty(0, dtype=torch.int64),
                        torch.zeros((0, 3, 4, 4), dtype=torch.int32),
                        torch.empty((4, 4), dtype=torch.int32))

    @pytest.mark.parametrize("count", [2, 4])
    def test_rejects_proof_count_mismatch(self, vdmpf, material, count):
        key, _, _ = material
        xs, values = self.split(vdmpf, key)
        wrong = torch.zeros((count, 3, 4, 4), dtype=torch.int32)
        shared = min(count, values.shape[0])
        wrong[:shared] = values[:shared]
        with pytest.raises(RuntimeError, match=r"shape \[n, 3, 4, 4\]"):
            vdmpf.prove(key.sigma, key.b_size_rt, key.cs, xs, wrong,
                        torch.empty((4, 4), dtype=torch.int32))

    def test_rejects_wrong_candidate_axis(self, vdmpf, material):
        key, _, _ = material
        xs, values = self.split(vdmpf, key)
        with pytest.raises(RuntimeError, match=r"shape \[n, 3, 4, 4\]"):
            vdmpf.prove_merkle(key.sigma, key.b_size_rt, key.cs, xs,
                               torch.zeros((3, 4, 4, 4), dtype=torch.int32),
                               torch.empty((4, 4), dtype=torch.int32))
        with pytest.raises(RuntimeError, match=r"shape \[n, 3, 4, 4\]"):
            vdmpf.prove_merkle(key.sigma, key.b_size_rt, key.cs, xs,
                               values.reshape(3, 4, 3, 4),
                               torch.empty((4, 4), dtype=torch.int32))

    def test_rejects_wrong_proof_dtype(self, vdmpf, material):
        key, _, _ = material
        xs, values = self.split(vdmpf, key)
        with pytest.raises(RuntimeError, match="int32"):
            vdmpf.prove(key.sigma, key.b_size_rt, key.cs, xs, values.long(),
                        torch.empty((4, 4), dtype=torch.int32))

    def test_rejects_noncontiguous_proof_values(self, vdmpf, material):
        key, _, _ = material
        xs, _ = self.split(vdmpf, key)
        strided = torch.zeros((6, 3, 4, 4), dtype=torch.int32)[::2]
        with pytest.raises(RuntimeError, match="contiguous"):
            vdmpf.prove(key.sigma, key.b_size_rt, key.cs, xs, strided,
                        torch.empty((4, 4), dtype=torch.int32))

    def test_rejects_xs_out_of_domain(self, vdmpf, material):
        key, _, _ = material
        xs, values = self.split(vdmpf, key)
        xs = xs.clone()
        xs[0] = 1 << IN_BITS
        with pytest.raises(RuntimeError, match="outside the input domain"):
            vdmpf.prove(key.sigma, key.b_size_rt, key.cs, xs, values,
                        torch.empty((4, 4), dtype=torch.int32))

    def test_rejects_wrong_xs_dtype(self, vdmpf, material):
        key, _, _ = material
        xs, values = self.split(vdmpf, key)
        with pytest.raises(RuntimeError, match="64-bit words"):
            vdmpf.prove(key.sigma, key.b_size_rt, key.cs, xs.int(), values,
                        torch.empty((4, 4), dtype=torch.int32))

    def test_rejects_bad_pi_shape(self, vdmpf, material):
        key, _, _ = material
        xs, values = self.split(vdmpf, key)
        with pytest.raises(RuntimeError, match="wrong number of blocks"):
            vdmpf.prove(key.sigma, key.b_size_rt, key.cs, xs, values,
                        torch.empty((5, 4), dtype=torch.int32))

    @pytest.mark.parametrize("size", [0, 1, 1 << (BUCKET_BITS + 1)])
    def test_rejects_sizes_outside_bucket_domain(self, vdmpf, material, size):
        key, _, _ = material
        xs, values = self.split(vdmpf, key)
        with pytest.raises(RuntimeError, match="bucket"):
            vdmpf.prove(key.sigma, size, key.cs, xs, values,
                        torch.empty((4, 4), dtype=torch.int32))


WIDE_IN_BITS = 70
WIDE_BUCKET_BITS = 66
WIDE_T = 30


@pytest.fixture(scope="module")
def wide_scheme():
    # The shared fixture pins in_type=uint64; the wide domain needs uint128.
    return Vdmpf(WIDE_IN_BITS, MAX_POINTS, WIDE_BUCKET_BITS, group.Bytes(),
                 make_prg(2), hash.Blake3(bytes(32)),
                 hash.Blake3(bytes(range(32))), prp.ChaChaFeistel())


@pytest.fixture(scope="module")
def wide_key(wide_scheme):
    points = limbs([i * ((1 << WIDE_IN_BITS) // (WIDE_T + 1))
                    for i in range(1, WIDE_T + 1)])
    betas = torch.randint(-2**31, 2**31, (WIDE_T, 4), dtype=torch.int32)
    betas[:, 3] &= ~1
    return generate_multi(wide_scheme, points, betas)


def limbs(values):
    # Two int64 limbs per value, low limb first, low limb kept signed.
    mask = (1 << 64) - 1
    lows = [v & mask for v in values]
    signed = [low - (1 << 64) if low >= (1 << 63) else low for low in lows]
    low = torch.tensor(signed, dtype=torch.int64)
    high = torch.tensor([v >> 64 for v in values], dtype=torch.int64)
    return torch.stack((low, high), dim=1)


class TestVdmpfWideDomain:
    """A 70-bit domain drives a 128-bit runtime size and uint128 inputs."""


    def test_runtime_size_exceeds_64_bits(self, wide_key):
        assert wide_key.b_size_rt > 1 << 64
        assert wide_key.b_size_rt <= 1 << WIDE_BUCKET_BITS

    def test_split_path_matches_fused_batch(self, wide_scheme, wide_key):
        queries = [0, (1 << 68) + 7, (1 << WIDE_IN_BITS) - 1]
        ys = torch.empty((len(queries), 4), dtype=torch.int32)
        values = []
        for position, x in enumerate(queries):
            y = torch.empty((4,), dtype=torch.int32)
            values.append(wide_scheme.eval(False, wide_key.sigma, wide_key.b_size_rt,
                                           wide_key.seeds[0], wide_key.cws,
                                           wide_key.cs, x, y))
            ys[position] = y
        pi = torch.empty((4, 4), dtype=torch.int32)
        wide_scheme.prove(wide_key.sigma, wide_key.b_size_rt, wide_key.cs,
                          limbs(queries),
                     torch.stack(values), pi)
        fused_ys, fused_pi = fused_party(
            wide_scheme, wide_key, False, merkle=False, xs=limbs(queries))
        assert torch.equal(ys, fused_ys)
        assert torch.equal(pi, fused_pi)

    def test_scalar_and_batch_inputs_out_of_domain(self, wide_scheme, wide_key):
        y = torch.empty((4,), dtype=torch.int32)
        with pytest.raises(RuntimeError, match="outside the configured domain"):
            wide_scheme.eval(False, wide_key.sigma, wide_key.b_size_rt,
                            wide_key.seeds[0], wide_key.cws, wide_key.cs,
                            1 << WIDE_IN_BITS, y)
        xs = limbs([0, 1 << WIDE_IN_BITS])
        with pytest.raises(RuntimeError, match="outside the input domain"):
            wide_scheme.prove(wide_key.sigma, wide_key.b_size_rt, wide_key.cs,
                             xs,
                         torch.zeros((2, 3, 4, 4), dtype=torch.int32),
                         torch.empty((4, 4), dtype=torch.int32))

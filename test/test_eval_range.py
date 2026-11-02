"""Closed-range evaluation through caller-owned component buffers."""

from types import SimpleNamespace

import pytest
import torch
from conftest import evaluate_all, generate_point, generate_vdpf, make_point_scheme, make_prg, random_blocks

import fss_crypto
from fss_crypto import group, hash

_CASES = ["Dpf", "DcfLT", "DcfGT", "HalfTreeDpf", "PackedHalfTreeDpf", "Vdpf"]
_ERRORS = (RuntimeError, ValueError, TypeError, OverflowError)


def _scheme(name, bits):
    if bits == 8:
        if name.startswith("Dcf"):
            pred = fss_crypto.DcfPred.LT if name.endswith("LT") else fss_crypto.DcfPred.GT
            return make_point_scheme("Dcf", bits=bits, pred=pred)
        return make_point_scheme(name, bits=bits)
    options = {"in_type": "uint128"}
    if name in ("HalfTreeDpf", "PackedHalfTreeDpf"):
        ccr = hash.Aes128Ccr(make_prg(1, "aes128_soft"))
        if name == "PackedHalfTreeDpf":
            return fss_crypto.PackedHalfTreeDpf(bits, 8, ccr, (1, 2, 3, 4), **options)
        return fss_crypto.HalfTreeDpf(bits, group.Bytes(), ccr, (1, 2, 3, 4), **options)
    if name == "Vdpf":
        return fss_crypto.Vdpf(bits, group.Bytes(), make_prg(2),
                               hash.Blake3(bytes(32)), hash.Blake3(bytes(range(32))),
                               **options)
    if name.startswith("Dcf"):
        pred = fss_crypto.DcfPred.LT if name.endswith("LT") else fss_crypto.DcfPred.GT
        return fss_crypto.Dcf(bits, group.Bytes(), make_prg(4), pred, **options)
    return fss_crypto.Dpf(bits, group.Bytes(), make_prg(2), **options)


@pytest.fixture(scope="module", params=_CASES)
def small(request):
    return _state(request.param, 8, 107)


@pytest.fixture(scope="module", params=[name for name in _CASES if name != "DcfGT"])
def wide(request):
    return _state(request.param, 128, (1 << 128) - 3)


def _state(name, bits, alpha):
    scheme = _scheme(name, bits)
    seeds = random_blocks((2, 4))
    beta = torch.tensor([0, 0, 0, 604], dtype=torch.int32)
    if name == "Vdpf":
        state = generate_vdpf(scheme, seeds, alpha, beta)
    else:
        cws = generate_point(scheme, seeds, alpha, 171 if name == "PackedHalfTreeDpf" else beta)
        state = SimpleNamespace(seeds=seeds, cws=cws)
    state.scheme, state.name, state.alpha, state.beta = scheme, name, alpha, beta
    return state


def _count(state, first, last):
    size = last - first + 1
    return ((size + state.scheme.kLanes - 1) // state.scheme.kLanes
            if state.name == "PackedHalfTreeDpf" else size)


def _call(state, party, first, last, *, device="cpu", ys=None, pi=None,
          s0=None, cws=None, cs=None, **kwargs):
    scheme = state.scheme
    if ys is None:
        ys = torch.empty((_count(state, first, last), 4), dtype=torch.int32, device=device)
    if s0 is None:
        s0 = state.seeds[party].to(device)
    if cws is None:
        cws = state.cws.to(device)
    if state.name == "Vdpf":
        if pi is None:
            pi = torch.empty((4, 4), dtype=torch.int32, device=device)
        if cs is None:
            cs = state.cs.to(device)
        result = scheme.eval_range_merkle(bool(party), first, last, s0, cws, cs, ys, pi, **kwargs)
    else:
        result = scheme.eval_range(bool(party), first, last, s0, cws, ys, **kwargs)
    assert result is None
    return ys, pi


def _scalar(state, party, first, last):
    scheme = state.scheme
    outputs, hashes = [], []
    for x in range(first, last + 1):
        if state.name == "Vdpf":
            y = torch.empty(4, dtype=torch.int32)
            hashes.append(scheme.eval(bool(party), state.seeds[party], state.cws,
                                      state.cs, x, y))
        else:
            y = scheme.eval(bool(party), state.seeds[party], state.cws, x)
        outputs.append(y)
    return torch.stack(outputs), hashes


def _check(state, first, last, device="cpu", full=None):
    scheme = state.scheme
    results, proofs = [], []
    for party in (0, 1):
        actual, proof = _call(state, party, first, last, device=device)
        actual = actual.cpu()
        scalar, hashes = _scalar(state, party, first, last)
        if state.name == "PackedHalfTreeDpf":
            for j, x in enumerate(range(first, last + 1)):
                value = scheme.extract(actual[j // scheme.kLanes], j)
                assert value == scheme.extract(scalar[j], x)
                if full is not None:
                    assert value == scheme.extract(full[party][x // scheme.kLanes], x)
            for j in range(last - first + 1, actual.shape[0] * scheme.kLanes):
                assert scheme.extract(actual[j // scheme.kLanes], j) == 0
        else:
            assert torch.equal(actual, scalar)
            if full is not None:
                assert torch.equal(actual, full[party][first:last + 1])
        if state.name == "Vdpf":
            expected = torch.empty((4, 4), dtype=torch.int32)
            scheme.prove_merkle(torch.stack(hashes), state.cs, expected)
            assert torch.equal(proof.cpu(), expected)
            chain_outputs = torch.empty_like(actual)
            chain_proof = torch.empty((4, 4), dtype=torch.int32)
            scheme.eval_range(bool(party), first, last, state.seeds[party],
                              state.cws, state.cs, chain_outputs, chain_proof)
            assert torch.equal(chain_outputs, actual)
            scheme.prove(torch.stack(hashes), state.cs, expected)
            assert torch.equal(chain_proof, expected)
            proofs.append(proof.cpu())
        results.append(actual)
    combined = results[0] ^ results[1]
    for j, x in enumerate(range(first, last + 1)):
        active = (x < state.alpha if state.name == "DcfLT" else
                  x > state.alpha if state.name == "DcfGT" else x == state.alpha)
        if state.name == "PackedHalfTreeDpf":
            assert scheme.extract(combined[j // scheme.kLanes], j) == (171 if active else 0)
        else:
            assert torch.equal(combined[j], state.beta if active else torch.zeros_like(state.beta))
    if proofs:
        assert scheme.verify(*proofs)


def test_closed_range_matches_scalar_and_full(small):
    full = None
    if small.name != "Vdpf":
        full = [evaluate_all(small.scheme, party, small.seeds[party], small.cws)
                for party in (0, 1)]
    else:
        full = []
        for party in (0, 1):
            ys = torch.empty((256, 4), dtype=torch.int32)
            pi = torch.empty((4, 4), dtype=torch.int32)
            small.scheme.eval_all_merkle(bool(party), small.seeds[party], small.cws,
                                        small.cs, ys, pi)
            full.append(ys)
    for first, last in [(0, 0), (3, 19), (101, 109), (253, 255), (0, 255)]:
        _check(small, first, last, full=full)
    if small.name != "PackedHalfTreeDpf":
        actual, _ = _call(small, 0, 3, 19, frontier_depth=-1, threads_per_block=-1)
        assert torch.equal(actual, full[0][3:20])


def test_range_preserves_output_capacity_tail(small):
    count = _count(small, 3, 19)
    storage = torch.full((count + 2, 4), 12345, dtype=torch.int32)
    expected, _ = _call(small, 0, 3, 19)
    actual, _ = _call(small, 0, 3, 19, ys=storage)
    assert torch.equal(actual[:count], expected)
    assert torch.all(actual[count:] == 12345)


def test_narrow_range_in_uint128_domain(wide):
    maximum = (1 << 128) - 1
    for first, last in [(0, 0), ((1 << 127) + 5, (1 << 127) + 7),
                        (maximum - 5, maximum)]:
        _check(wide, first, last)
    with pytest.raises(_ERRORS):
        _call(wide, 0, 0, maximum, ys=torch.empty((1, 4), dtype=torch.int32))


def test_rejects_invalid_range_and_buffers(small):
    valid = torch.empty((3, 4), dtype=torch.int32)
    for first, last in [(True, 2), (0, False), (-1, 2), (0, -1), (3, 2),
                        (0, 256), (1 << 128, 1 << 128), (1.0, 2)]:
        with pytest.raises(_ERRORS):
            _call(small, 0, first, last, ys=valid)
    required = _count(small, 5, 7)
    for output in [torch.empty((required - 1, 4), dtype=torch.int32),
                   torch.empty((required, 4), dtype=torch.int64),
                   torch.empty((required, 8), dtype=torch.int32)[:, ::2]]:
        with pytest.raises(_ERRORS):
            _call(small, 0, 5, 7, ys=output)
    for seed in [small.seeds[0].to(torch.int64), random_blocks((8,))[::2]]:
        with pytest.raises(_ERRORS):
            _call(small, 0, 5, 7, s0=seed)
    with pytest.raises(_ERRORS):
        _call(small, 0, 5, 7, cws=small.cws.to(torch.int32))
    with pytest.raises(_ERRORS):
        _call(small, 0, 5, 7, cws=torch.empty((*small.cws.shape, 2), dtype=torch.uint8)[..., 0])
    misaligned = torch.empty(small.cws.numel() + 1, dtype=torch.uint8)[1:].view(small.cws.shape)
    with pytest.raises(_ERRORS):
        _call(small, 0, 5, 7, cws=misaligned)
    misaligned_output = torch.empty((required * 4 + 1,), dtype=torch.int32)[1:].view(required, 4)
    with pytest.raises(_ERRORS):
        _call(small, 0, 5, 7, ys=misaligned_output)
    with pytest.raises(_ERRORS):
        _call(small, 0, 5, 7, par_depth=-1)
    if small.name == "Vdpf":
        for proof in [torch.empty((3, 4), dtype=torch.int32),
                      torch.empty((4, 4), dtype=torch.int64),
                      torch.empty((4, 8), dtype=torch.int32)[:, ::2]]:
            with pytest.raises(_ERRORS):
                _call(small, 0, 5, 7, pi=proof)
        with pytest.raises(_ERRORS):
            _call(small, 0, 5, 7, cs=small.cs.to(torch.int64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_matches_cpu_on_current_stream(small):
    if small.name == "PackedHalfTreeDpf":
        with pytest.raises(_ERRORS):
            _call(small, 0, 3, 19, device="cuda")
        return
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        _check(small, 3, 19, device="cuda")
        actual, proof = _call(small, 0, 101, 109, device="cuda", par_depth=-1)
        capacity = torch.full((_count(small, 3, 19) + 2, 4), 12345,
                              dtype=torch.int32, device="cuda")
        filled, _ = _call(small, 0, 3, 19, device="cuda", ys=capacity)
    stream.synchronize()
    assert torch.all(filled[_count(small, 3, 19):].cpu() == 12345)
    expected, expected_proof = _call(small, 0, 101, 109)
    assert torch.equal(actual.cpu(), expected)
    if proof is not None:
        assert torch.equal(proof.cpu(), expected_proof)
    with pytest.raises(_ERRORS):
        _call(small, 0, 3, 19, device="cuda", cws=small.cws)
    with pytest.raises(_ERRORS):
        _call(small, 0, 3, 19, device="cuda", s0=small.seeds[0])
    if small.name == "Vdpf":
        with pytest.raises(_ERRORS):
            _call(small, 0, 3, 19, device="cuda", cs=small.cs)
        with pytest.raises(_ERRORS):
            _call(small, 0, 3, 19, device="cuda", pi=torch.empty((4, 4), dtype=torch.int32))
        with pytest.raises(_ERRORS):
            small.scheme.eval_range(False, 3, 19, small.seeds[0].cuda(), small.cws.cuda(),
                                    small.cs.cuda(), torch.empty((17, 4), dtype=torch.int32, device="cuda"),
                                    torch.empty((4, 4), dtype=torch.int32, device="cuda"))
    if torch.cuda.device_count() > 1:
        with pytest.raises(_ERRORS):
            _call(small, 0, 3, 19, device="cuda:0", cws=small.cws.to("cuda:1"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_narrow_uint128_range(wide):
    if wide.name == "PackedHalfTreeDpf":
        pytest.skip("packed range is CPU only")
    _check(wide, (1 << 128) - 6, (1 << 128) - 1, device="cuda")

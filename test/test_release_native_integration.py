"""Native acceptance for configured backends, execution spaces, and wide inputs."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import subprocess
import sys

import pytest
import torch

import fss_crypto
from conftest import (evaluate_all, evaluate_multi, evaluate_multi_all,
                      evaluate_vdpf, evaluate_vdpf_all, generate_multi,
                      generate_point, generate_vdpf, make_multi_scheme,
                      make_point_scheme, random_blocks)


def _multipoint(verifiable, **kwargs):
    scheme = make_multi_scheme(verifiable, bits=8, bucket_bits=4, **kwargs)
    points = torch.arange(30, dtype=torch.int64)
    payloads = random_blocks((30, 4))
    payloads[:, 3] &= -2
    return scheme, generate_multi(scheme, points, payloads), points, payloads


@pytest.mark.parametrize("scheme_name", ["Dpf", "Dcf", "HalfTreeDpf", "PackedHalfTreeDpf",
                                          "GrottoDcf", "Vdpf", "Dmpf", "Vdmpf"])
def test_every_advertised_aes_scheme_cpu(scheme_name):
    if scheme_name in ("Dmpf", "Vdmpf"):
        scheme, material, points, payloads = _multipoint(
            scheme_name == "Vdmpf", backend="aes128_mmo")
        results = [evaluate_multi(scheme, material, party, points) for party in (0, 1)]
        assert torch.equal(results[0][0] ^ results[1][0], payloads)
        if scheme_name == "Vdmpf":
            assert scheme.verify(results[0][1], results[1][1])
        else:
            full = [evaluate_multi_all(scheme, material, party)[0] for party in (0, 1)]
            expected = torch.zeros_like(full[0])
            expected[points] = payloads
            assert torch.equal(full[0] ^ full[1], expected)
        return

    scheme = make_point_scheme(scheme_name, bits=8, out_bits=64, backend="aes128_mmo")
    seeds = random_blocks((2, 4))
    payload = random_blocks((4,))
    payload[3] &= -2
    if scheme_name == "Vdpf":
        material = generate_vdpf(scheme, seeds, 7, payload)
        full = [evaluate_vdpf_all(scheme, material, party)[0] for party in (0, 1)]
    else:
        beta = (1 << 64) - 1 if scheme_name == "PackedHalfTreeDpf" else payload
        cws = generate_point(scheme, seeds, 7, beta)
        full = [evaluate_all(scheme, party, seeds[party], cws) for party in (0, 1)]
    reconstructed = full[0] ^ full[1]
    if scheme_name == "PackedHalfTreeDpf":
        assert scheme.extract(reconstructed[7 // scheme.kLanes], 7) == (1 << 64) - 1
    elif scheme_name == "GrottoDcf":
        assert torch.equal(reconstructed, torch.arange(256) >= 7)
    else:
        expected = torch.zeros_like(reconstructed)
        if scheme_name == "Dcf":
            expected[:7] = payload
        else:
            expected[7] = payload
        assert torch.equal(reconstructed, expected)


@pytest.mark.parametrize("backend", ["chacha", "aes128_mmo"])
def test_parallel_backend_policy_and_concurrent_calls(backend):
    scheme = make_point_scheme("Dpf", bits=8, backend=backend)
    seeds = random_blocks((2, 4))
    payload = torch.tensor([11, 12, 13, 14], dtype=torch.int32)
    cws = generate_point(scheme, seeds, 7, payload)
    sequential = evaluate_all(scheme, False, seeds[0], cws)
    try:
        parallel = evaluate_all(scheme, False, seeds[0], cws, par_depth=2)
    except RuntimeError as error:
        if backend == "chacha":
            assert "OpenMP" in str(error)
            pytest.skip("parallel execution requires OpenMP support")
        assert "OpenMP" in str(error) or "shared mutable contexts" in str(error)
    else:
        assert torch.equal(parallel, sequential)
    assert torch.equal(sequential[7], scheme.eval(False, seeds[0], cws, 7))
    if backend == "chacha":
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(evaluate_all, scheme, False, seeds[0], cws)
                       for _ in range(8)]
            assert all(torch.equal(future.result(), sequential) for future in futures)


@pytest.mark.parametrize("scheme_name", ["Dpf", "Dcf", "HalfTreeDpf", "PackedHalfTreeDpf",
                                          "GrottoDcf", "Vdpf", "Dmpf"])
def test_every_parallel_scheme_matches_point_evaluation(scheme_name):
    if scheme_name == "Dmpf":
        scheme, material, _, _ = _multipoint(False)
        points = torch.arange(256, dtype=torch.int64)
        for party in (0, 1):
            try:
                full = evaluate_multi_all(scheme, material, party, par_depth=2)[0]
            except RuntimeError as error:
                if "OpenMP" in str(error):
                    pytest.skip("parallel execution requires OpenMP support")
                raise
            assert torch.equal(full, evaluate_multi(scheme, material, party, points)[0])
        return
    scheme = make_point_scheme(scheme_name, bits=8, out_bits=64)
    seeds = random_blocks((2, 4))
    payload = torch.tensor([11, 12, 13, 14], dtype=torch.int32)
    if scheme_name == "Vdpf":
        material = generate_vdpf(scheme, seeds, 7, payload)
        for party in (0, 1):
            try:
                full, _ = evaluate_vdpf_all(scheme, material, party, par_depth=2)
            except RuntimeError as error:
                if "OpenMP" in str(error):
                    pytest.skip("parallel execution requires OpenMP support")
                raise
            for x in range(256):
                assert torch.equal(full[x], evaluate_vdpf(scheme, material, party, x)[0])
        return
    beta = (1 << 64) - 1 if scheme_name == "PackedHalfTreeDpf" else payload
    cws = generate_point(scheme, seeds, 7, beta)
    for party in (0, 1):
        try:
            full = evaluate_all(scheme, party, seeds[party], cws, par_depth=2)
        except RuntimeError as error:
            if "OpenMP" in str(error):
                pytest.skip("parallel execution requires OpenMP support")
            raise
        if scheme_name == "GrottoDcf":
            owned = scheme.preprocess_owned(bool(party), seeds[party], cws)
            actual = torch.tensor([scheme.eval_tree(owned.view(), x) for x in range(256)])
            assert torch.equal(full, actual)
        else:
            for x in range(256):
                point = scheme.eval(bool(party), seeds[party], cws, x)
                index = x // scheme.kLanes if scheme_name == "PackedHalfTreeDpf" else x
                assert torch.equal(full[index], point)


@pytest.mark.parametrize("beta", [0, (1 << 63) - 1, 1 << 63, (1 << 64) - 1])
def test_packed_complete_uint64_beta(beta):
    scheme = make_point_scheme("PackedHalfTreeDpf", bits=4, out_bits=64)
    seeds = random_blocks((2, 4))
    cws = generate_point(scheme, seeds, 3, beta)
    point = scheme.eval(False, seeds[0], cws, 3) ^ scheme.eval(True, seeds[1], cws, 3)
    full = evaluate_all(scheme, False, seeds[0], cws) ^ evaluate_all(scheme, True, seeds[1], cws)
    assert scheme.extract(point, 3) == beta
    assert scheme.extract(full[3 // scheme.kLanes], 3) == beta
    for invalid in (-1, 1 << 64):
        with pytest.raises((TypeError, ValueError, OverflowError)):
            scheme.gen(cws, seeds, 3, invalid)


@pytest.mark.parametrize("xs", [torch.empty(0, dtype=torch.int64), torch.tensor([0, 1, 1, 255])])
def test_explicit_vdmpf_merkle_cpu_and_cuda(xs):
    scheme, material, _, _ = _multipoint(True)
    cpu = [evaluate_multi(scheme, material, party, xs, merkle=True) for party in (0, 1)]
    assert cpu[0][0].shape == (xs.numel(), 4)
    assert scheme.verify(cpu[0][1], cpu[1][1])
    if torch.cuda.is_available():
        for party in (0, 1):
            gpu = evaluate_multi(scheme, material, party, xs, merkle=True, device="cuda")
            assert all(torch.equal(left, right.cpu()) for left, right in zip(cpu[party], gpu))
        assert evaluate_multi(scheme, material, False, torch.tensor([1]),
                              merkle=True, device="cuda")[0].shape == (1, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("xor_hash_backend,hash_backend", [("sha256", "blake3"), ("blake3", "sha256"), ("sha256", "sha256")])
def test_vdmpf_cuda_hash_rejection_preserves_context(xor_hash_backend, hash_backend):
    scheme, material, points, _ = _multipoint(True, xor_hash_backend=xor_hash_backend,
                                             hash_backend=hash_backend)
    cpu = [evaluate_multi(scheme, material, party, points) for party in (0, 1)]
    assert scheme.verify(cpu[0][1], cpu[1][1])
    with pytest.raises((RuntimeError, TypeError), match="CUDA|backend"):
        evaluate_multi(scheme, material, False, points, merkle=True, device="cuda")
    assert torch.equal(torch.ones(1, device="cuda").cpu(), torch.ones(1))


@pytest.mark.parametrize("verifiable", [False, True])
def test_direct_native_query_rejection_is_an_exception(verifiable):
    script = f'''import sys
sys.path.insert(0, {str(Path(__file__).parent)!r})
import torch
from conftest import make_multi_scheme, generate_multi, random_blocks
scheme = make_multi_scheme({verifiable!r}, bits=8, bucket_bits=4)
points = torch.arange(30, dtype=torch.int64)
payloads = random_blocks((30, 4))
payloads[:, 3] &= -2
material = generate_multi(scheme, points, payloads)
for value in (-1, 256):
    args = (False, material.sigma, material.b_size_rt,
            material.seeds[0], material.cws)
    if {verifiable!r}:
        args += (material.cs,)
    args += (torch.tensor([value], dtype=torch.int64), torch.empty((1, 4), dtype=torch.int32))
    if {verifiable!r}:
        args += (torch.empty((4, 4), dtype=torch.int32),)
    try:
        scheme._native.batch_eval(*args)
    except RuntimeError as error:
        assert "domain" in str(error), str(error)
    else:
        raise AssertionError("native query bounds were not checked")
valid = list(args)
valid[-3 if {verifiable!r} else -2] = torch.tensor([0], dtype=torch.int64)
scheme._native.batch_eval(*valid)
'''
    result = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True,
                            env=os.environ.copy(), timeout=600, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("bits", [64, 128])
def test_wide_vdpf_point_and_full_domain_rejection(bits):
    scheme = make_point_scheme("Vdpf", bits=bits)
    material = generate_vdpf(scheme, random_blocks((2, 4)), (1 << (bits - 1)) + 1,
                             torch.zeros(4, dtype=torch.int32))
    evaluate_vdpf(scheme, material, False, (1 << (bits - 1)) + 1)
    output = torch.empty((0, 4), dtype=torch.int32)
    proof = torch.empty((4, 4), dtype=torch.int32)
    for method in (scheme.eval_all, scheme.eval_all_merkle):
        with pytest.raises((RuntimeError, ValueError), match="address|overflow|fit|represent"):
            method(False, material.seeds[0], material.cws, material.cs, output, proof)
    for method in (scheme._native.eval_all, scheme._native.eval_all_merkle):
        with pytest.raises((RuntimeError, ValueError), match="address|overflow|fit|represent"):
            method(False, material.seeds[0], material.cws, material.cs, output, proof, 0)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices required")
def test_tensor_device_guard_restores_caller_device():
    torch.cuda.set_device(0)
    scheme, material, points, _ = _multipoint(True)
    cpu = evaluate_multi(scheme, material, False, points, merkle=True)
    gpu = evaluate_multi(scheme, material, False, points, merkle=True, device="cuda:1")
    assert torch.cuda.current_device() == 0
    assert all(torch.equal(left, right.cpu()) for left, right in zip(cpu, gpu))
    with pytest.raises(RuntimeError, match="same device"):
        scheme._native.batch_eval_merkle(
            False, material.sigma.to("cuda:1"), material.b_size_rt,
            material.seeds[0].to("cuda:1"), material.cws.to("cuda:1"),
            material.cs.to("cuda:1"), points.to("cuda:0"),
            torch.empty((30, 4), dtype=torch.int32, device="cuda:1"),
            torch.empty((4, 4), dtype=torch.int32, device="cuda:1"))
    assert torch.cuda.current_device() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_aes_native_cuda_rejection_preserves_context():
    scheme = make_point_scheme("Dpf", bits=8, backend="aes128_mmo")
    seeds = random_blocks((2, 4))
    cws = generate_point(scheme, seeds, 1, torch.zeros(4, dtype=torch.int32))
    with pytest.raises((RuntimeError, TypeError), match="CUDA|backend"):
        scheme._native.eval(False, seeds[0].cuda(), cws.cuda(), 1)
    assert torch.equal(torch.ones(1, device="cuda").cpu(), torch.ones(1))


@pytest.mark.parametrize("bits", [64, 128])
@pytest.mark.parametrize("scheme_name", ["Dpf", "Dcf", "HalfTreeDpf", "PackedHalfTreeDpf", "Vdpf", "GrottoDcf"])
def test_high_bit_domain_values_roundtrip_through_native(bits, scheme_name):
    scheme = make_point_scheme(scheme_name, bits=bits, out_bits=64)
    seeds = random_blocks((2, 4))
    alpha = (1 << (bits - 1)) + (1 << 63 if bits > 64 else 0) + 1
    payload = torch.tensor([11, 12, 13, 14], dtype=torch.int32)
    if scheme_name == "Vdpf":
        material = generate_vdpf(scheme, seeds, alpha, payload)
    else:
        beta = (1 << 63) + 3 if scheme_name == "PackedHalfTreeDpf" else payload
        cws = generate_point(scheme, seeds, alpha, beta)
    for x in (alpha - 1, alpha):
        if scheme_name == "Vdpf":
            shares = [evaluate_vdpf(scheme, material, party, x)[0] for party in (0, 1)]
        else:
            shares = [scheme.eval(bool(party), seeds[party], cws, x) for party in (0, 1)]
        if scheme_name == "GrottoDcf":
            assert (shares[0] ^ shares[1]) == (x >= alpha)
        else:
            reconstructed = shares[0] ^ shares[1]
            selected = x < alpha if scheme_name == "Dcf" else x == alpha
            if scheme_name == "PackedHalfTreeDpf":
                assert scheme.extract(reconstructed, x) == (beta if selected else 0)
            else:
                assert torch.equal(reconstructed, payload if selected else torch.zeros_like(payload))
        if torch.cuda.is_available():
            for party in (0, 1):
                if scheme_name == "Vdpf":
                    gpu = evaluate_vdpf(scheme, material, party, x, device="cuda")[0]
                else:
                    gpu = scheme.eval(bool(party), seeds[party].cuda(), cws.cuda(), x)
                if isinstance(gpu, bool):
                    assert gpu == shares[party]
                else:
                    assert torch.equal(gpu.cpu(), shares[party])
    if scheme_name != "GrottoDcf":
        with pytest.raises((RuntimeError, ValueError), match="address|overflow|fit|represent"):
            if scheme_name == "Vdpf":
                scheme.eval_all(False, material.seeds[0], material.cws, material.cs,
                                torch.empty((0, 4), dtype=torch.int32),
                                torch.empty((4, 4), dtype=torch.int32))
            else:
                scheme.eval_all(False, seeds[0], cws, torch.empty((0, 4), dtype=torch.int32))
    else:
        with pytest.raises((RuntimeError, ValueError), match="address|overflow|fit|represent"):
            scheme.preprocess(False, seeds[0], cws, torch.empty(0, dtype=torch.bool))


_GROUP_DEVICE_PARAMS = [
    pytest.param("cpu", id="cpu"),
    pytest.param("cuda", id="cuda",
                 marks=pytest.mark.skipif(not torch.cuda.is_available(),
                                          reason="CUDA required")),
]


def _run_group_operations(device, operations):
    target = torch.device(device)
    if target.type == "cpu":
        return operations()
    with torch.cuda.stream(torch.cuda.Stream(device=target)):
        return operations()


def _signed_group_word(word):
    word &= 0xFFFFFFFF
    return word if word < (1 << 31) else word - (1 << 32)


def _uint_block_words(value, bits):
    if bits == 128:
        words = (value & 0xFFFFFFFF, (value >> 32) & 0xFFFFFFFF,
                 (value >> 64) & 0xFFFFFFFF,
                 ((value >> 96) & 0x7FFFFFFF) << 1)
    else:
        words = (value & 0xFFFFFFFF, (value >> 32) & 0xFFFFFFFF, 0, 0)
    return [_signed_group_word(word) for word in words]


def _uint_block(value, bits, device):
    return torch.tensor(_uint_block_words(value, bits), dtype=torch.int32,
                        device=device)


def _uint_boundary_payloads(bits, mod):
    width = 127 if bits == 128 else bits
    limit = 1 << width
    mixed = {
        8: 0xA5,
        16: 0xA5C3,
        32: 0xA5C3F00D,
        64: 0xA5C3F00DDEADBEEF,
        128: ((0x5A5A5A5A << 96) | (0xDEADBEEF << 64) |
              (0x89ABCDEF << 32) | 0x76543210),
    }[bits]
    payloads = [0, 1, 1 << (width - 1), limit - 1, mixed]
    if mod:
        payloads.extend(candidate for candidate in (mod - 1, mod, mod + 1)
                        if candidate < limit)
    return list(dict.fromkeys(payloads))


@pytest.mark.parametrize("bits,mod", [
    pytest.param(8, 0, id="u8"),
    pytest.param(8, 131, id="u8mod"),
    pytest.param(16, 0, id="u16"),
    pytest.param(16, 32771, id="u16mod"),
    pytest.param(32, 0, id="u32"),
    pytest.param(32, 2147483659, id="u32mod"),
    pytest.param(64, 0, id="u64"),
    pytest.param(64, 9223372036854775837, id="u64mod"),
    pytest.param(128, 1 << 127, id="u127"),
    pytest.param(128, (1 << 126) + 17, id="u127mod"),
])
@pytest.mark.parametrize("device", _GROUP_DEVICE_PARAMS)
def test_group_value_uint_boundaries(bits, mod, device):
    group = fss_crypto.group.Uint(bits=bits, mod=mod)

    def run():
        for payload in _uint_boundary_payloads(bits, mod):
            expected = payload if mod == 0 else payload % mod
            block = _uint_block(payload, bits, device)
            value = group.from_(block)

            assert value.value() == expected
            assert int(value) == expected

            into = value.into()
            assert into.dtype == torch.int32
            assert tuple(into.shape) == (4,)
            assert into.device == block.device
            into_words = into.cpu().tolist()
            assert into_words == _uint_block_words(expected, bits)
            assert into_words[0] == _signed_group_word(expected & 0xFFFFFFFF)
            if bits >= 64:
                assert into_words[1] == _signed_group_word((expected >> 32) & 0xFFFFFFFF)
            else:
                assert into_words[1] == 0
            if bits == 128:
                assert into_words[2] == _signed_group_word((expected >> 64) & 0xFFFFFFFF)
                high31 = (expected >> 96) & 0x7FFFFFFF
                assert (into_words[3] & 0xFFFFFFFF) == high31 << 1
                assert (into_words[3] & 0xFFFFFFFF) % 2 == 0
            else:
                assert into_words[2:] == [0, 0]

            zero = group.zero(device=block.device)
            inverse_sum = value + (-value)
            assert zero.value() == 0
            assert int(zero) == 0
            assert zero.into().cpu().tolist() == [0, 0, 0, 0]
            assert inverse_sum.value() == 0
            assert int(inverse_sum) == 0
            assert inverse_sum.into().cpu().tolist() == [0, 0, 0, 0]

    _run_group_operations(device, run)


@pytest.mark.parametrize("device", _GROUP_DEVICE_PARAMS)
def test_group_value_bytes_boundaries(device):
    group = fss_crypto.group.Bytes()
    unsigned_words = [0x80000000, 0xFFFFFFFF, 0xDEADBEEF, 0xFFFFFFFE]
    words = [_signed_group_word(word) for word in unsigned_words]
    expected_bytes = b"".join(word.to_bytes(4, "little")
                              for word in unsigned_words)

    def run():
        block = torch.tensor(words, dtype=torch.int32, device=device)
        value = group.from_(block)

        for result in (value.value(), value.into()):
            assert result.dtype == torch.int32
            assert tuple(result.shape) == (4,)
            assert result.device == block.device
            assert result.cpu().tolist() == words
        assert bytes(value) == expected_bytes

        zero = group.zero(device=block.device)
        for result in (zero.value(), zero.into()):
            assert result.dtype == torch.int32
            assert tuple(result.shape) == (4,)
            assert result.device == block.device
            assert result.cpu().tolist() == [0, 0, 0, 0]
        assert bytes(zero) == bytes(16)

        doubled = value + value
        for result in (doubled.value(), doubled.into()):
            assert result.dtype == torch.int32
            assert tuple(result.shape) == (4,)
            assert result.device == block.device
            assert result.cpu().tolist() == [0, 0, 0, 0]

        negated = -value
        for result in (negated.value(), negated.into()):
            assert result.dtype == torch.int32
            assert tuple(result.shape) == (4,)
            assert result.device == block.device
            assert result.cpu().tolist() == words

    _run_group_operations(device, run)

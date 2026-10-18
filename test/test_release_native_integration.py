"""Native acceptance for the repaired public Python options and boundaries."""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

import fss_crypto


def _seeds(shape=(2, 4)):
    return torch.randint(-(1 << 31), 1 << 31, shape, dtype=torch.int32)


def _multipoint(scheme_cls, **kwargs):
    scheme = scheme_cls(8, 30, 4, prp="chacha_feistel", **kwargs)
    alphas = torch.arange(30, dtype=torch.int64)
    betas = _seeds((30, 4))
    betas[:, 3] &= ~1
    key = scheme.gen(_seeds((scheme.m, 2, 4)), _seeds((4,)), alphas, betas, 30)
    return scheme, key, alphas, betas


@pytest.mark.parametrize("scheme_name", ["Dpf", "Dcf", "HalfTreeDpf", "PackedHalfTreeDpf",
                                          "GrottoDcf", "Vdpf", "Dmpf", "Vdmpf"])
def test_every_advertised_aes_scheme_cpu(scheme_name):
    cls = getattr(fss_crypto, scheme_name)
    if scheme_name in ("Dmpf", "Vdmpf"):
        scheme, key, xs, betas = _multipoint(cls, prg="aes128_mmo")
        results = [scheme.batch_eval(party, key, xs) for party in (0, 1)]
        if scheme_name == "Vdmpf":
            assert scheme.verify(results[0][1], results[1][1])
            results = [result[0] for result in results]
        assert torch.equal(results[0] ^ results[1], betas)
        if scheme_name == "Dmpf":
            assert torch.equal(scheme.eval_all(0, key) ^ scheme.eval_all(1, key),
                               torch.cat([betas, torch.zeros((226, 4), dtype=torch.int32)]))
        return

    scheme = cls(8, 64, prg="aes128_mmo") if scheme_name == "PackedHalfTreeDpf" else cls(8, prg="aes128_mmo")
    seeds = _seeds()
    beta = _seeds((4,))
    beta[3] &= ~1
    if scheme_name == "GrottoDcf":
        cws = scheme.gen(seeds, 7)
        outputs = [scheme.eval_all(party, seeds[party], cws) for party in (0, 1)]
        assert torch.equal(outputs[0] ^ outputs[1], torch.arange(256) >= 7)
        for party in (0, 1):
            tree = scheme.preprocess(party, seeds[party], cws)
            assert scheme.eval(party, tree, 7) == bool(outputs[party][7])
        return
    if scheme_name == "Vdpf":
        seeds, cws, cs, ocw = scheme.gen(seeds, 7, beta)
        args = (cws, cs, ocw)
        outputs = [scheme.eval_all(party, seeds[party], *args)[0] for party in (0, 1)]
    elif scheme_name in ("HalfTreeDpf", "PackedHalfTreeDpf"):
        args = scheme.gen(seeds, 7, (1 << 64) - 1 if scheme_name == "PackedHalfTreeDpf" else beta)
        outputs = [scheme.eval_all(party, seeds[party], *args) for party in (0, 1)]
    else:
        args = (scheme.gen(seeds, 7, beta),)
        outputs = [scheme.eval_all(party, seeds[party], *args) for party in (0, 1)]
    reconstructed = outputs[0] ^ outputs[1]
    if scheme_name == "PackedHalfTreeDpf":
        assert scheme.extract(reconstructed[7 // scheme.lanes], 7) == (1 << 64) - 1
    elif scheme_name == "Dcf":
        assert torch.equal(reconstructed[:7], beta.expand(7, 4))
        assert not torch.any(reconstructed[7:])
    else:
        assert torch.equal(reconstructed[7], beta)
        assert not torch.any(torch.cat((reconstructed[:7], reconstructed[8:])))


@pytest.mark.parametrize("prg", ["chacha", "aes128_mmo"])
def test_parallel_and_concurrent_aes_calls(prg):
    sequential = fss_crypto.Dpf(8, prg=prg, par_depth=0)
    parallel = fss_crypto.Dpf(8, prg=prg, par_depth=2)
    # Extension modules intentionally own different public keys/nonces. Compare
    # each module with its own point evaluator, then reuse a shared module from
    # several Python threads. The C++ PRG factory owns per-thread AES contexts.
    seeds, beta = _seeds(), _seeds((4,))
    beta[3] &= ~1
    for scheme in (sequential, parallel):
        cws = scheme.gen(seeds, 7, beta)
        full = scheme.eval_all(0, seeds[0], cws)
        assert torch.equal(full[7], scheme.eval(0, seeds[0], cws, 7))
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(scheme.eval_all, 0, seeds[0], cws) for _ in range(8)]
            assert all(torch.equal(future.result(), full) for future in futures)


@pytest.mark.parametrize("scheme_name", ["Dpf", "Dcf", "HalfTreeDpf", "PackedHalfTreeDpf",
                                          "GrottoDcf", "Vdpf", "Dmpf"])
def test_every_parallel_scheme_matches_point_evaluation(scheme_name):
    cls = getattr(fss_crypto, scheme_name)
    domain = torch.arange(256)
    if scheme_name == "Dmpf":
        scheme, key, _, betas = _multipoint(cls, par_depth=2)
        full = [scheme.eval_all(party, key) for party in (0, 1)]
        for party in (0, 1):
            assert torch.equal(full[party], scheme.batch_eval(party, key, domain))
        expected = torch.zeros((256, 4), dtype=torch.int32)
        expected[:30] = betas
        assert torch.equal(full[0] ^ full[1], expected)
        return

    scheme = cls(8, 64, par_depth=2) if scheme_name == "PackedHalfTreeDpf" else cls(8, par_depth=2)
    seeds = _seeds()
    alpha = 7
    beta = torch.tensor([11, 12, 13, 14], dtype=torch.int32)
    if scheme_name == "GrottoDcf":
        cws = scheme.gen(seeds, alpha)
        full = [scheme.eval_all(party, seeds[party], cws) for party in (0, 1)]
        for party in (0, 1):
            tree = scheme.preprocess(party, seeds[party], cws)
            points = torch.tensor([scheme.eval(party, tree, x) for x in range(256)])
            assert torch.equal(points, full[party])
        assert torch.equal(full[0] ^ full[1], domain >= alpha)
        return

    if scheme_name == "Vdpf":
        seeds, cws, cs, ocw = scheme.gen(seeds, alpha, beta)
        args = (cws, cs, ocw)
    elif scheme_name in ("HalfTreeDpf", "PackedHalfTreeDpf"):
        args = scheme.gen(seeds, alpha, (1 << 64) - 1 if scheme_name == "PackedHalfTreeDpf" else beta)
    else:
        args = (scheme.gen(seeds, alpha, beta),)
    results = [scheme.eval_all(party, seeds[party], *args) for party in (0, 1)]
    if scheme_name == "Vdpf":
        assert scheme.verify(results[0][1], results[1][1])
        full = [result[0] for result in results]
        merkle = [scheme.eval_all_merkle(party, seeds[party], *args) for party in (0, 1)]
        assert scheme.verify(merkle[0][1], merkle[1][1])
        assert all(torch.equal(merkle[party][0], full[party]) for party in (0, 1))
    else:
        full = results
    for party in (0, 1):
        for x in range(256):
            point = scheme.eval(party, seeds[party], *args, x)
            if scheme_name == "Vdpf":
                point = point[0]
            index = x // scheme.lanes if scheme_name == "PackedHalfTreeDpf" else x
            assert torch.equal(point, full[party][index])
    reconstructed = full[0] ^ full[1]
    if scheme_name == "PackedHalfTreeDpf":
        for x in range(256):
            assert scheme.extract(reconstructed[x // scheme.lanes], x) == ((1 << 64) - 1 if x == alpha else 0)
    else:
        expected = torch.zeros((256, 4), dtype=torch.int32)
        if scheme_name == "Dcf":
            expected[:alpha] = beta
        else:
            expected[alpha] = beta
        assert torch.equal(reconstructed, expected)


@pytest.mark.parametrize("beta", [0, (1 << 63) - 1, 1 << 63, (1 << 64) - 1])
def test_packed_complete_uint64_beta(beta):
    scheme = fss_crypto.PackedHalfTreeDpf(4, 64)
    seeds = _seeds()
    cws, fcw = scheme.gen(seeds, 3, beta)
    point = scheme.eval(0, seeds[0], cws, fcw, 3) ^ scheme.eval(1, seeds[1], cws, fcw, 3)
    full = scheme.eval_all(0, seeds[0], cws, fcw) ^ scheme.eval_all(1, seeds[1], cws, fcw)
    assert scheme.extract(point, 3) == beta
    assert scheme.extract(full[3 // scheme.lanes], 3) == beta
    for invalid in (-1, 1 << 64):
        with pytest.raises(ValueError, match="beta must be"):
            scheme.gen(seeds, 3, invalid)


@pytest.mark.parametrize("xs", [torch.empty(0, dtype=torch.int64), torch.tensor([0, 1, 1, 255])])
def test_explicit_vdmpf_merkle_cpu_and_cuda(xs):
    scheme, key, _, _ = _multipoint(fss_crypto.Vdmpf)
    cpu = [scheme.batch_eval_merkle(party, key, xs) for party in (0, 1)]
    assert cpu[0][0].shape == (xs.numel(), 4)
    assert scheme.verify(cpu[0][1], cpu[1][1])
    if not torch.cuda.is_available():
        return
    gpu_key, gpu_xs = key.to("cuda"), xs.to("cuda")
    for party in (0, 1):
        gpu = scheme.batch_eval_merkle(party, gpu_key, gpu_xs)
        assert torch.equal(cpu[party][0], gpu[0].cpu())
        assert torch.equal(cpu[party][1], gpu[1].cpu())
    # Empty batches must leave a usable context and produce the same nonempty
    # result after their zero-point launch path.
    followup = torch.tensor([1], device="cuda")
    assert scheme.batch_eval_merkle(0, gpu_key, followup)[0].shape == (1, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("xor_hash,hash", [("sha256", "blake3"), ("blake3", "sha256"), ("sha256", "sha256")])
def test_vdmpf_cuda_hash_rejection_preserves_context(xor_hash, hash):
    scheme, key, xs, _ = _multipoint(fss_crypto.Vdmpf, xor_hash=xor_hash, hash=hash)
    # Host SHA256 remains supported.
    _, pi0 = scheme.batch_eval(0, key, xs)
    _, pi1 = scheme.batch_eval(1, key, xs)
    assert scheme.verify(pi0, pi1)
    gpu_key, gpu_xs = key.to("cuda"), xs.to("cuda")
    with pytest.raises(RuntimeError, match="sha256.*CPU-only"):
        scheme.batch_eval_merkle(0, gpu_key, gpu_xs)
    # Native callers cannot bypass the guard either.
    with pytest.raises(RuntimeError, match="sha256.*CPU-only"):
        scheme._ext.batch_eval_merkle(0, gpu_key.cws, gpu_key.cs, gpu_key.ocw,
                                      gpu_key.s0s[0], gpu_key.sigma, gpu_key.meta, gpu_xs)
    assert torch.equal(torch.ones(1, device="cuda").cpu(), torch.ones(1))


@pytest.mark.parametrize("scheme_name", ["Dmpf", "Vdmpf"])
def test_direct_native_query_rejection_is_an_exception(scheme_name):
    script = f'''import torch, fss_crypto
scheme = fss_crypto.{scheme_name}(8, 30, 4, prp="chacha_feistel")
s0s = torch.randint(-(1 << 31), 1 << 31, (scheme.m, 2, 4), dtype=torch.int32)
sigma = torch.zeros(4, dtype=torch.int32)
betas = torch.zeros((30, 4), dtype=torch.int32)
key = scheme.gen(s0s, sigma, torch.arange(30), betas, 30)
args = [key.cws] + ([key.cs, key.ocw] if "{scheme_name}" == "Vdmpf" else []) + [key.s0s[0], key.sigma, key.meta]
for value in (-1, 256):
    try:
        scheme._ext.batch_eval(0, *args, torch.tensor([value]))
    except RuntimeError:
        pass
    else:
        raise AssertionError("native query validation was bypassed")
scheme._ext.batch_eval(0, *args, torch.tensor([0]))
'''
    result = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True,
                            env=os.environ.copy(), timeout=600, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("bits", [64, 128])
def test_wide_vdpf_point_and_full_domain_rejection(bits):
    scheme = fss_crypto.Vdpf(bits)
    seeds, cws, cs, ocw = scheme.gen(_seeds(), (1 << (bits - 1)) + 1, torch.zeros(4, dtype=torch.int32))
    scheme.eval(0, seeds[0], cws, cs, ocw, (1 << (bits - 1)) + 1)
    for method in (scheme.eval_all, scheme.eval_all_merkle):
        with pytest.raises(RuntimeError, match="cannot represent"):
            method(0, seeds[0], cws, cs, ocw)
    # Direct extension callers get the same check before shifts or allocations.
    for method in (scheme._ext.eval_all, scheme._ext.eval_all_merkle):
        with pytest.raises(RuntimeError, match="cannot represent"):
            method(0, seeds[0], cws, cs, ocw)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices required")
def test_tensor_device_guard_restores_caller_device():
    torch.cuda.set_device(0)
    scheme, cpu_key, xs, _ = _multipoint(fss_crypto.Vdmpf)
    cpu = scheme.batch_eval_merkle(0, cpu_key, xs)
    key = cpu_key.to("cuda:1")
    gpu_xs = xs.to("cuda:1")
    assert torch.cuda.current_device() == 0
    gpu = scheme.batch_eval_merkle(0, key, gpu_xs)
    assert torch.cuda.current_device() == 0
    assert torch.equal(cpu[0], gpu[0].cpu())
    assert torch.equal(cpu[1], gpu[1].cpu())
    # Direct calls cover restoration on exceptions from the native guard.
    with pytest.raises(RuntimeError, match="nonnegative"):
        scheme._ext.batch_eval_merkle(0, key.cws, key.cs, key.ocw, key.s0s[0],
                                      key.sigma, key.meta, torch.tensor([-1], device="cuda:1"))
    assert torch.cuda.current_device() == 0
    assert torch.equal(scheme.batch_eval_merkle(0, key, gpu_xs)[0].cpu(), cpu[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_vdpf_native_cuda_resource_guard_precedes_output_allocation():
    scheme = fss_crypto.Vdpf(23)
    seeds = torch.zeros(4, dtype=torch.int32, device="cuda")
    cws = torch.zeros((23, 8), dtype=torch.int32, device="cuda")
    cs = torch.zeros((4, 4), dtype=torch.int32, device="cuda")
    ocw = torch.zeros(4, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="supports in_bits <= 22"):
        scheme._ext.eval_all_merkle(0, seeds, cws, cs, ocw)
    assert torch.equal(torch.ones(1, device="cuda").cpu(), torch.ones(1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_aes_native_cuda_rejection_preserves_context():
    scheme = fss_crypto.Dpf(8, prg="aes128_mmo")
    seeds = _seeds()
    cws = scheme.gen(seeds, 1, torch.zeros(4, dtype=torch.int32))
    with pytest.raises(RuntimeError, match="aes128_mmo is CPU-only"):
        scheme._ext.eval(0, seeds[0].to("cuda"), cws.to("cuda"), 1, 0)
    assert torch.equal(torch.ones(1, device="cuda").cpu(), torch.ones(1))


@pytest.mark.parametrize("bits", [64, 128])
@pytest.mark.parametrize("scheme_name", ["Dpf", "Dcf", "HalfTreeDpf", "PackedHalfTreeDpf", "Vdpf", "GrottoDcf"])
def test_high_bit_domain_values_roundtrip_through_native(bits, scheme_name):
    cls = getattr(fss_crypto, scheme_name)
    scheme = cls(bits, 64) if scheme_name == "PackedHalfTreeDpf" else cls(bits)
    seeds = _seeds()
    alpha = (1 << (bits - 1)) + (1 << 63 if bits > 64 else 0) + 1
    if scheme_name == "GrottoDcf":
        cws = scheme.gen(seeds, alpha)
        assert cws.shape == (bits + 1, 8)
        for method in (scheme.preprocess, scheme.eval_all):
            with pytest.raises(RuntimeError, match="cannot represent"):
                method(0, seeds[0], cws)
        if torch.cuda.is_available():
            for method in (scheme._ext.preprocess, scheme._ext.eval_all):
                with pytest.raises(RuntimeError, match="cannot represent"):
                    method(0, seeds[0].cuda(), cws.cuda())
        return
    beta = (1 << 63) + 3 if scheme_name == "PackedHalfTreeDpf" else torch.tensor([11, 12, 13, 14], dtype=torch.int32)
    if scheme_name == "Vdpf":
        seeds, cws, cs, ocw = scheme.gen(seeds, alpha, beta)
        args = (cws, cs, ocw)
    elif scheme_name in ("HalfTreeDpf", "PackedHalfTreeDpf"):
        args = scheme.gen(seeds, alpha, beta)
    else:
        args = (scheme.gen(seeds, alpha, beta),)
    for x in (alpha - 1, alpha):
        shares = [scheme.eval(party, seeds[party], *args, x) for party in (0, 1)]
        ys = [share[0] if scheme_name == "Vdpf" else share for share in shares]
        reconstructed = ys[0] ^ ys[1]
        expected = beta if ((x < alpha) if scheme_name == "Dcf" else x == alpha) else 0
        if scheme_name == "PackedHalfTreeDpf":
            assert scheme.extract(reconstructed, x) == expected
        elif isinstance(expected, int):
            assert not torch.any(reconstructed)
        else:
            assert torch.equal(reconstructed, expected)
        if torch.cuda.is_available():
            for party in (0, 1):
                gpu = scheme.eval(party, seeds[party].cuda(), *(arg.cuda() for arg in args), x)
                if scheme_name == "Vdpf":
                    assert torch.equal(gpu[0].cpu(), shares[party][0])
                    assert torch.equal(gpu[1].cpu(), shares[party][1])
                else:
                    assert torch.equal(gpu.cpu(), shares[party])
    for method in (scheme.eval_all, scheme._ext.eval_all):
        with pytest.raises(RuntimeError, match="cannot represent"):
            method(0, seeds[0], *args)
    if torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="cannot represent"):
            scheme._ext.eval_all(0, seeds[0].cuda(), *(arg.cuda() for arg in args))

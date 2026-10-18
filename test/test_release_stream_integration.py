"""Native bindings must read scalar inputs after their current-stream producers."""

import dataclasses

import pytest
import torch

import fss_crypto


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _same(actual, expected):
    if isinstance(actual, tuple):
        return all(_same(a, e) for a, e in zip(actual, expected))
    return torch.equal(actual.cpu(), expected)


@pytest.mark.parametrize("scalar", ["dpf_seed", "vdpf_cs", "dmpf_meta", "half_ocw", "packed_fcw"])
def test_scalar_copy_follows_current_stream(scalar):
    generator = torch.Generator().manual_seed(713)
    seeds = torch.randint(-2**31, 2**31, (2, 4), generator=generator, dtype=torch.int32)
    beta = torch.tensor([6, 0, 0, 0], dtype=torch.int32)

    if scalar == "dpf_seed":
        scheme = fss_crypto.Dpf(8)
        cws = scheme.gen(seeds, 42, beta)
        expected = scheme.eval_all(0, seeds[0], cws)
        target, delayed, cws_gpu = seeds[0].cuda(), torch.zeros(4, dtype=torch.int32, device="cuda"), cws.cuda()
        evaluate = lambda: scheme.eval_all(0, delayed, cws_gpu)
    elif scalar == "vdpf_cs":
        scheme = fss_crypto.Vdpf(8)
        effective, cws, cs, ocw = scheme.gen(seeds, 42, beta)
        expected = scheme.eval_all_merkle(0, effective[0], cws, cs, ocw)
        seed_gpu, cws_gpu, ocw_gpu = effective[0].cuda(), cws.cuda(), ocw.cuda()
        target, delayed = cs.cuda(), torch.zeros_like(cs, device="cuda")
        evaluate = lambda: scheme.eval_all_merkle(0, seed_gpu, cws_gpu, delayed, ocw_gpu)
    elif scalar == "dmpf_meta":
        scheme = fss_crypto.Dmpf(8, 30, 4, prp="chacha_feistel")
        bucket_seeds = torch.randint(-2**31, 2**31, (scheme.m, 2, 4),
                                     generator=generator, dtype=torch.int32)
        alphas = torch.arange(30, dtype=torch.int64)
        betas = beta.repeat(30, 1)
        key = scheme.gen(bucket_seeds, seeds[0], alphas, betas, 30)
        expected = scheme.eval_all(0, key)
        gpu_key = key.to("cuda")
        wrong_meta = key.meta.clone()
        wrong_meta[1] += 1  # Both capacities fit the four-bit bucket domain.
        target, delayed = gpu_key.meta, wrong_meta.cuda()
        gpu_key = dataclasses.replace(gpu_key, meta=delayed)
        evaluate = lambda: scheme.eval_all(0, gpu_key)
    elif scalar == "half_ocw":
        scheme = fss_crypto.HalfTreeDpf(8)
        cws, ocw = scheme.gen(seeds, 42, beta)
        expected = scheme.eval_all(0, seeds[0], cws, ocw)
        seed_gpu, cws_gpu = seeds[0].cuda(), cws.cuda()
        target, delayed = ocw.cuda(), torch.zeros_like(ocw, device="cuda")
        evaluate = lambda: scheme.eval_all(0, seed_gpu, cws_gpu, delayed)
    else:
        scheme = fss_crypto.PackedHalfTreeDpf(8, 8)
        cws, fcw = scheme.gen(seeds, 42, 171)
        expected = scheme.eval_all(0, seeds[0], cws, fcw)
        seed_gpu, cws_gpu = seeds[0].cuda(), cws.cuda()
        target, delayed = fcw.cuda(), torch.zeros_like(fcw, device="cuda")
        evaluate = lambda: scheme.eval_all(0, seed_gpu, cws_gpu, delayed)

    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(200_000_000)
        delayed.copy_(target)
        actual = evaluate()
    stream.synchronize()
    assert _same(actual, expected), f"{scalar} was read before its current-stream producer"

"""Bindings must read component inputs after current-stream producers."""

import pytest
import torch

from conftest import (evaluate_all, evaluate_multi_all, evaluate_vdpf_all,
                      generate_multi, generate_point, generate_vdpf,
                      make_multi_scheme, make_point_scheme, random_blocks)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("component", ["dpf_seed", "vdpf_cs", "dmpf_sigma",
                                       "half_terminal_cws", "packed_terminal_cws"])
def test_scalar_copy_follows_current_stream(component):
    seeds = random_blocks((2, 4))
    payload = torch.tensor([6, 0, 0, 0], dtype=torch.int32)
    if component == "vdpf_cs":
        scheme = make_point_scheme("Vdpf", bits=8)
        material = generate_vdpf(scheme, seeds, 42, payload)
        expected = evaluate_vdpf_all(scheme, material, False, merkle=True)
        target = material.cs.cuda()
        delayed = torch.zeros_like(target)
        seed_gpu, cws_gpu = material.seeds[0].cuda(), material.cws.cuda()
        output = torch.empty_like(expected[0], device="cuda")
        proof = torch.empty_like(expected[1], device="cuda")

        def evaluate():
            scheme.eval_all_merkle(False, seed_gpu, cws_gpu, delayed, output, proof)
            return output, proof
    elif component == "dmpf_sigma":
        scheme = make_multi_scheme(False, bits=8, bucket_bits=4)
        material = generate_multi(scheme, torch.arange(30, dtype=torch.int64),
                                  payload.repeat(30, 1))
        expected = (evaluate_multi_all(scheme, material, False)[0],)
        target = material.sigma.cuda()
        delayed = torch.zeros_like(target)
        seeds_gpu, cws_gpu = material.seeds[0].cuda(), material.cws.cuda()
        output = torch.empty_like(expected[0], device="cuda")

        def evaluate():
            scheme.eval_all(False, delayed, material.b_size_rt,
                            seeds_gpu, cws_gpu, output)
            return (output,)
    else:
        name = {"dpf_seed": "Dpf", "half_terminal_cws": "HalfTreeDpf",
                "packed_terminal_cws": "PackedHalfTreeDpf"}[component]
        scheme = make_point_scheme(name, bits=8)
        beta = 171 if component == "packed_terminal_cws" else payload
        cws = generate_point(scheme, seeds, 42, beta)
        expected = (evaluate_all(scheme, False, seeds[0], cws),)
        seed_gpu, cws_gpu = seeds[0].cuda(), cws.cuda()
        if component == "dpf_seed":
            target = seed_gpu
            delayed = torch.zeros_like(target)
            seed_gpu = delayed
        else:
            # The terminal correction is now the last native Cw record.
            target = cws_gpu[-1].clone()
            delayed = cws_gpu[-1]
            delayed.zero_()

        def evaluate():
            return (evaluate_all(scheme, False, seed_gpu, cws_gpu),)

    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(200_000_000)
        delayed.copy_(target)
        actual = evaluate()
    stream.synchronize()
    assert all(torch.equal(left.cpu(), right) for left, right in zip(actual, expected))

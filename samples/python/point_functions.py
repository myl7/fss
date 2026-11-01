"""Point/comparison functions on CPU or CUDA, including GPU key batches.

Run with ``python samples/python/point_functions.py --device cpu`` or
``--device cuda:0`` after installing fss_crypto and its JIT prerequisites.
The dealer creates both parties' keys here only to demonstrate reconstruction.
"""

import argparse
import secrets

import torch

from fss_crypto import Dcf, DcfPred, Dpf, HalfTreeDpf, PackedHalfTreeDpf
from fss_crypto import group, hash, prg


def random_blocks(count, device):
    """Sample independent 128-bit seeds with a cryptographic RNG."""
    words = [secrets.randbits(32) - 2**31 for _ in range(count * 4)]
    return torch.tensor(words, dtype=torch.int32, device=device).reshape(count, 4)


def make_schemes():
    output_group = group.Bytes()
    ccr = hash.Aes128Ccr(prg.Aes128Soft(mul=1, keys=(secrets.token_bytes(16),)))
    instance_seed = tuple(secrets.randbits(32) for _ in range(4))
    return (
        Dpf(8, output_group, prg.ChaCha(nonce=(1, 2), mul=2)),
        Dcf(8, output_group, prg.ChaCha(nonce=(3, 4), mul=4), pred=DcfPred.LT),
        HalfTreeDpf(8, output_group, ccr, instance_seed),
        PackedHalfTreeDpf(8, 8, ccr, instance_seed),
    )


def run_single(scheme, device):
    point = 42
    beta = 7
    packed = isinstance(scheme, PackedHalfTreeDpf)
    seeds = random_blocks(2, device)
    cws = torch.empty(scheme.cws_shape, dtype=torch.uint8, device=device)
    payload = torch.tensor([beta, 0, 0, 0], dtype=torch.int32, device=device)
    if packed:
        scheme.gen(cws, seeds, point, beta)
    else:
        scheme.gen(cws, seeds, point, payload)

    # Bytes and packed outputs reconstruct with XOR, not integer addition.
    for x in (point - 1, point, point + 1):
        result = torch.bitwise_xor(
            scheme.eval(False, seeds[0], cws, x),
            scheme.eval(True, seeds[1], cws, x),
        )
        selected = x < point if isinstance(scheme, Dcf) else x == point
        if packed:
            assert scheme.extract(result, x) == (beta if selected else 0)
        else:
            expected = payload if selected else torch.zeros_like(payload)
            assert torch.equal(result, expected)

    output_count = 2**scheme.kDepth if packed else 2**scheme.in_bits
    outputs = [torch.empty((output_count, 4), dtype=torch.int32, device=device)
               for _ in range(2)]
    for party in (0, 1):
        scheme.eval_all(bool(party), seeds[party], cws, outputs[party])
    combined = torch.bitwise_xor(*outputs)
    if packed:
        for x in range(2**scheme.in_bits):
            block = combined[x // scheme.kLanes]
            assert scheme.extract(block, x) == (beta if x == point else 0)
    else:
        expected = torch.zeros_like(combined)
        if isinstance(scheme, Dcf):
            expected[:point] = payload
        else:
            expected[point] = payload
        assert torch.equal(combined, expected)


def run_gpu_batch(scheme, device):
    """Two independently generated keys, each evaluated over its whole domain."""
    packed = isinstance(scheme, PackedHalfTreeDpf)
    points = (42, 43)
    pairs = random_blocks(4, device).reshape(2, 2, 4)
    corrections = torch.empty((2, *scheme.cws_shape), dtype=torch.uint8, device=device)
    payload = torch.tensor([7, 0, 0, 0], dtype=torch.int32, device=device)
    for i, point in enumerate(points):
        if packed:
            scheme.gen(corrections[i], pairs[i], point, 7)
        else:
            scheme.gen(corrections[i], pairs[i], point, payload)
    count = 2**scheme.kDepth if packed else 2**scheme.in_bits
    outputs = [torch.empty((2, count, 4), dtype=torch.int32, device=device)
               for _ in range(2)]
    for party in (0, 1):
        # Selecting one party from [key, party, word] is a strided view.
        party_seeds = pairs[:, party].contiguous()
        scheme.eval_all_gpu_batch(bool(party), party_seeds, corrections, outputs[party])
    combined = torch.bitwise_xor(*outputs)
    for i, point in enumerate(points):
        if packed:
            for x in range(2**scheme.in_bits):
                assert scheme.extract(combined[i, x // scheme.kLanes], x) == (7 if x == point else 0)
        else:
            expected = torch.zeros_like(combined[i])
            if isinstance(scheme, Dcf):
                expected[:point] = payload
            else:
                expected[point] = payload
            assert torch.equal(combined[i], expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    device = torch.device(parser.parse_args().device)
    payload = torch.tensor([7, 0, 0, 0], dtype=torch.int32, device=device)
    assert int(group.Uint(32).from_(payload)) == 7
    assert bytes(group.Bytes().from_(payload)) == (7).to_bytes(16, "little")
    for scheme in make_schemes():
        run_single(scheme, device)
        if device.type == "cuda":
            run_gpu_batch(scheme, device)
        print(f"{type(scheme).__name__}: reconstructed outputs match")


if __name__ == "__main__":
    main()

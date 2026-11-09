"""Grotto preprocessing with caller-owned tensors and CPU-owned trees.

Run with ``python samples/python/grotto.py --device cpu`` or ``--device cuda:0``.
The CPU-owned path is also shown during the CUDA example because
``preprocess_owned`` deliberately allocates CPU storage only.
"""

import argparse
import secrets

import torch

from fss_crypto import GrottoDcf, HalfTreeGrottoDcf, PackedParityTreeView, ParityTreeView
from fss_crypto import hash, prg


def random_seeds(device):
    words = [secrets.randbits(32) - 2**31 for _ in range(8)]
    return torch.tensor(words, dtype=torch.int32, device=device).reshape(2, 4)


def run(scheme, device):
    threshold = 42
    seeds = random_seeds(device)
    cws = torch.empty(scheme.cws_shape, dtype=torch.uint8, device=device)
    scheme.gen(cws, seeds, threshold)
    packed = isinstance(scheme, HalfTreeGrottoDcf)
    leaf_count = 2**scheme.kDepth if packed else 2**scheme.in_bits
    views = []
    for party in (0, 1):
        parity = torch.empty(2 * leaf_count - 1, dtype=torch.bool, device=device)
        if packed:
            blocks = torch.empty((leaf_count, 4), dtype=torch.int32, device=device)
            scheme.preprocess(bool(party), seeds[party], cws, blocks, parity)
            views.append(PackedParityTreeView(blocks, parity))
        else:
            scheme.preprocess(bool(party), seeds[party], cws, parity)
            views.append(ParityTreeView(parity))

    for x in (threshold - 1, threshold, threshold + 1):
        if (scheme.eval(False, seeds[0], cws, x)
                ^ scheme.eval(True, seeds[1], cws, x)) != (x >= threshold):
            raise AssertionError("raw share reconstruction mismatch")
        if (scheme.eval_tree(views[0], x)
                ^ scheme.eval_tree(views[1], x)) != (x >= threshold):
            raise AssertionError("tree share reconstruction mismatch")

    count = 2**scheme.in_bits
    outputs = [torch.empty(count, dtype=torch.bool, device=device) for _ in range(2)]
    for party in (0, 1):
        scheme.eval_all_tree(views[party], outputs[party])
    expected = torch.arange(count, device=device) >= threshold
    if not torch.equal(torch.bitwise_xor(*outputs), expected):
        raise AssertionError("full-domain reconstruction mismatch")

    # Do not compare individual raw-Eval shares against tree shares: Core's
    # prefix-parity route may use different shares with the same reconstruction.
    for party in (0, 1):
        scheme.eval_all(bool(party), seeds[party], cws, outputs[party])
    if not torch.equal(torch.bitwise_xor(*outputs), expected):
        raise AssertionError("full-domain reconstruction mismatch")

    host_seeds, host_cws = seeds.cpu(), cws.cpu()
    owners = [scheme.preprocess_owned(bool(party), host_seeds[party], host_cws)
              for party in (0, 1)]
    owned_views = [owner.view() for owner in owners]
    for x in (threshold - 1, threshold, threshold + 1):
        if (scheme.eval_tree(owned_views[0], x)
                ^ scheme.eval_tree(owned_views[1], x)) != (x >= threshold):
            raise AssertionError("owned tree share reconstruction mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    device = torch.device(parser.parse_args().device)
    ccr = hash.Aes128Ccr(prg.Aes128Soft(mul=1, keys=(secrets.token_bytes(16),)))
    instance_seed = tuple(secrets.randbits(32) for _ in range(4))
    schemes = (
        GrottoDcf(8, prg.ChaCha(nonce=(1, 2), mul=2)),
        HalfTreeGrottoDcf(8, ccr, instance_seed),
    )
    for scheme in schemes:
        run(scheme, device)
        print(f"{type(scheme).__name__}: raw and tree reconstructions match")


if __name__ == "__main__":
    main()

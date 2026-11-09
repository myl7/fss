"""VDPF evaluation and proof aggregation on CPU and optionally CUDA.

Run with ``python samples/python/verifiable.py [--cuda]`` after installing
fss_crypto and its CUDA build dependencies. Generation runs on CPU in this
example. The two-party execution in one process is only a demonstration.
"""

import argparse
import secrets

import torch

from fss_crypto import Vdpf, group, hash, prg


def seed_pair():
    return torch.tensor(
        [[int.from_bytes(secrets.token_bytes(4), "little", signed=True)
          for _ in range(4)] for _ in range(2)], dtype=torch.int32,
    )


def main(cuda=False):
    scheme = Vdpf(
        in_bits=8,
        group=group.Bytes(),
        prg=prg.ChaCha(nonce=(secrets.randbits(32), secrets.randbits(32)), mul=2),
        xor_hash=hash.Blake3(iv=secrets.token_bytes(32)),
        hash=hash.Blake3(iv=secrets.token_bytes(32)),
    )
    alpha = 42
    payload = torch.tensor([7, 0, 0, 0], dtype=torch.int32)
    cws = torch.empty(scheme.cws_shape, dtype=torch.uint8)
    cs = torch.empty((4, 4), dtype=torch.int32)
    for _ in range(128):
        seeds = seed_pair()
        status = scheme.gen(cws, cs, seeds, alpha, payload)
        if status == 0:
            break
        if status != 1:
            raise RuntimeError(f"unexpected VDPF generation status {status}")
    else:
        raise RuntimeError("VDPF generation exhausted the sample retry budget")

    queries = [0, alpha, 255]
    query_tensor = torch.tensor(queries, dtype=torch.uint8)
    expected = torch.zeros((len(queries), 4), dtype=torch.int32)
    expected[queries.index(alpha)] = payload

    # Entry path: one batch_eval call evaluates the query set and folds the
    # chain proof; both parties pass the same ordered sequence.
    shares = [torch.empty_like(expected) for _ in range(2)]
    chain_proofs = [torch.empty((4, 4), dtype=torch.int32) for _ in range(2)]
    for party in (False, True):
        scheme.batch_eval(party, seeds[int(party)], cws, cs,
                          query_tensor, shares[int(party)], chain_proofs[int(party)])
    if not torch.equal(shares[0] ^ shares[1], expected):
        raise AssertionError("point reconstruction mismatch")
    if not scheme.verify(*chain_proofs):
        raise AssertionError("proof verification failed")

    # Staged path: scalar Eval returns each corrected hash, and Prove then
    # folds the collected hashes in query order. Chain and Merkle are distinct.
    corrected = []
    for party in (False, True):
        party_shares = torch.empty_like(expected)
        party_hashes = []
        for row, x in enumerate(queries):
            party_hashes.append(scheme.eval(party, seeds[int(party)], cws, cs, x,
                                            party_shares[row]))
        shares[int(party)] = party_shares
        corrected.append(torch.stack(party_hashes))
    if not torch.equal(shares[0] ^ shares[1], expected):
        raise AssertionError("point reconstruction mismatch")

    for prove in (scheme.prove, scheme.prove_merkle):
        proofs = [torch.empty((4, 4), dtype=torch.int32) for _ in range(2)]
        for party in (False, True):
            prove(corrected[int(party)], cs, proofs[int(party)])
        if not scheme.verify(*proofs):
            raise AssertionError("proof verification failed")

    full_expected = torch.zeros((2**scheme.in_bits, 4), dtype=torch.int32)
    full_expected[alpha] = payload
    full_shares = [torch.empty_like(full_expected) for _ in range(2)]
    proofs = [torch.empty((4, 4), dtype=torch.int32) for _ in range(2)]
    for party in (False, True):
        scheme.eval_all(party, seeds[int(party)], cws, cs,
                        full_shares[int(party)], proofs[int(party)])
    if not torch.equal(full_shares[0] ^ full_shares[1], full_expected):
        raise AssertionError("full-domain reconstruction mismatch")
    if not scheme.verify(*proofs):
        raise AssertionError("proof verification failed")

    # Range endpoints are inclusive. Range evaluation uses CPU chain proofs.
    left, right = 40, 44
    range_shares = [torch.empty((right - left + 1, 4), dtype=torch.int32)
                    for _ in range(2)]
    for party in (False, True):
        scheme.eval_range(party, left, right, seeds[int(party)], cws, cs,
                          range_shares[int(party)], proofs[int(party)])
    if not torch.equal(range_shares[0] ^ range_shares[1], full_expected[left:right + 1]):
        raise AssertionError("range reconstruction mismatch")
    if not scheme.verify(*proofs):
        raise AssertionError("proof verification failed")

    devices = [torch.device("cpu")]
    if cuda:
        devices.append(torch.device("cuda:0"))
    for device in devices:
        device_cws, device_cs, device_seeds = (t.to(device) for t in (cws, cs, seeds))
        merkle_proofs = [torch.empty((4, 4), dtype=torch.int32, device=device)
                         for _ in range(2)]
        device_corrected = [t.to(device) for t in corrected]
        for party in (False, True):
            scheme.prove_merkle(device_corrected[int(party)], device_cs,
                                merkle_proofs[int(party)])
        if not scheme.verify(*merkle_proofs):
            raise AssertionError("merkle proof verification failed")

        outputs = [torch.empty(full_expected.shape, dtype=torch.int32, device=device)
                   for _ in range(2)]
        for party in (False, True):
            scheme.eval_all_merkle(party, device_seeds[int(party)], device_cws,
                                   device_cs, outputs[int(party)],
                                   merkle_proofs[int(party)])
        if not scheme.verify(*merkle_proofs):
            raise AssertionError("merkle proof verification failed")
        if not torch.equal((outputs[0] ^ outputs[1]).cpu(), full_expected):
            raise AssertionError("full-domain reconstruction mismatch")
    print("VDPF shares and matching-protocol proofs agree")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true", help="also evaluate on cuda:0")
    main(parser.parse_args().cuda)

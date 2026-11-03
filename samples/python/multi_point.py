"""DMPF/VDMPF component buffers, phase-specific retries, and CPU/CUDA queries.

Run with ``python samples/python/multi_point.py [--cuda]`` after installing
fss_crypto and its CUDA build dependencies. Both schemes generate on CPU.
The two-party execution in one process is only a demonstration.
"""

import argparse
import secrets

import torch

from fss_crypto import Dmpf, Vdmpf, group, hash, prg, prp


def random_blocks(count):
    return torch.tensor(
        [[int.from_bytes(secrets.token_bytes(4), "little", signed=True)
          for _ in range(4)] for _ in range(count)], dtype=torch.int32,
    )


def run(verifiable, cuda):
    configuration = dict(
        in_bits=8, max_points=32, bucket_bits=8,
        group=group.Bytes(),
        prg=prg.ChaCha(nonce=(secrets.randbits(32), secrets.randbits(32)), mul=2),
        prp=prp.ChaChaFeistel(),
        in_type="uint8",
    )
    if verifiable:
        scheme = Vdmpf(**configuration,
                       xor_hash=hash.Blake3(iv=secrets.token_bytes(32)),
                       hash=hash.Blake3(iv=secrets.token_bytes(32)))
    else:
        scheme = Dmpf(**configuration)

    # The current Cuckoo bucket estimate requires at least 30 secret points.
    points = torch.tensor(secrets.SystemRandom().sample(range(256), 30), dtype=torch.uint8)
    payloads = torch.zeros((len(points), 4), dtype=torch.int32)
    payloads[:, 0] = torch.arange(1, len(points) + 1, dtype=torch.int32)
    cws = torch.empty(scheme.cws_shape, dtype=torch.uint8)
    cs = (torch.empty((scheme.kBucketCount, 4, 4), dtype=torch.int32)
          if verifiable else None)
    seed_pairs = random_blocks(scheme.kBucketCount * 2).view(2, scheme.kBucketCount, 4)
    sigma = random_blocks(1)[0]
    for _ in range(128):
        if verifiable:
            status, m_rt, bucket_size = scheme.gen(
                cws, cs, sigma, seed_pairs, points, payloads, len(points))
        else:
            status, m_rt, bucket_size = scheme.gen(
                cws, sigma, seed_pairs, points, payloads, len(points))
        if status == 0:
            break
        if status == 1:
            sigma = random_blocks(1)[0]
        elif verifiable and status == 2:
            seed_pairs = random_blocks(scheme.kBucketCount * 2).view(
                2, scheme.kBucketCount, 4)
        else:
            raise RuntimeError(f"unexpected generation status {status}")
    else:
        raise RuntimeError("generation exhausted the sample retry budget")

    # m_rt is the runtime bucket count of the generated key. This sample does
    # not read it; evaluation takes only bucket_size and derives its bucket
    # bound from it.
    # Party-first storage makes each party's seed matrix a contiguous view.
    party_seeds = [seed_pairs[party] for party in range(2)]
    queries = torch.tensor([0, 42, 255], dtype=torch.uint8)
    full_expected = torch.zeros((2**scheme.in_bits, 4), dtype=torch.int32)
    full_expected[points.to(torch.int64)] = payloads
    query_expected = full_expected[queries.to(torch.int64)]

    devices = [torch.device("cpu")]
    if cuda:
        devices.append(torch.device("cuda:0"))
    for device in devices:
        device_sigma, device_cws, device_queries = (t.to(device) for t in (sigma, cws, queries))
        device_seeds = [t.to(device) for t in party_seeds]
        shares = [torch.empty((len(queries), 4), dtype=torch.int32, device=device)
                  for _ in range(2)]
        full_shares = [torch.empty(full_expected.shape, dtype=torch.int32, device=device)
                       for _ in range(2)]
        if verifiable:
            device_cs = cs.to(device)
            proofs = [torch.empty((4, 4), dtype=torch.int32, device=device)
                      for _ in range(2)]
            for party in (False, True):
                scheme.batch_eval_merkle(
                    party, device_sigma, bucket_size,
                    device_seeds[int(party)], device_cws, device_cs,
                    device_queries, shares[int(party)], proofs[int(party)])
            assert scheme.verify(*proofs)
            for party in (False, True):
                scheme.eval_all_merkle(
                    party, device_sigma, bucket_size,
                    device_seeds[int(party)], device_cws, device_cs,
                    full_shares[int(party)], proofs[int(party)])
            assert scheme.verify(*proofs)
            if device.type == "cpu":
                # Chain proofs are CPU-only and must not be mixed with Merkle proofs.
                for party in (False, True):
                    scheme.batch_eval(
                        party, device_sigma, bucket_size,
                        device_seeds[int(party)], device_cws, device_cs,
                        device_queries, shares[int(party)], proofs[int(party)])
                assert scheme.verify(*proofs)
                for party in (False, True):
                    scheme.eval_all(
                        party, device_sigma, bucket_size,
                        device_seeds[int(party)], device_cws, device_cs,
                        full_shares[int(party)], proofs[int(party)])
                assert scheme.verify(*proofs)
        else:
            for party in (False, True):
                scheme.batch_eval(party, device_sigma, bucket_size,
                                  device_seeds[int(party)], device_cws, device_queries,
                                  shares[int(party)])
                scheme.eval_all(party, device_sigma, bucket_size,
                                device_seeds[int(party)], device_cws,
                                full_shares[int(party)])
        assert torch.equal((shares[0] ^ shares[1]).cpu(), query_expected)
        assert torch.equal((full_shares[0] ^ full_shares[1]).cpu(), full_expected)
    print(f"{'VDMPF' if verifiable else 'DMPF'} shares agree")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true", help="also evaluate on cuda:0")
    arguments = parser.parse_args()
    run(verifiable=False, cuda=arguments.cuda)
    run(verifiable=True, cuda=arguments.cuda)

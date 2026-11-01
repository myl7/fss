"""Standalone C++ primitive bindings on an explicitly selected device.

Run with ``python samples/python/primitives.py --device cpu`` or
``--device cuda:0`` after installing fss_crypto and its JIT prerequisites.
The CUDA example uses only device-capable backends, with no CPU fallback.
"""

import argparse
import secrets

import torch

from fss_crypto import group, hash, prg, prp, util


def random_blocks(count, device):
    """Sample independent blocks using a cryptographic RNG."""
    words = [secrets.randbits(32) - 2**31 for _ in range(4 * count)]
    return torch.tensor(words, dtype=torch.int32, device=device).reshape(count, 4)


def run_groups(device):
    payload = torch.tensor([7, 0, 0, 0], dtype=torch.int32, device=device)
    for numbers in (group.Uint(32), group.Uint(64, mod=11)):
        value = numbers.from_(payload)
        assert int(value) == 7
        assert int(value + -value) == 0
        assert torch.equal(value.into(), payload)
    blocks = group.Bytes()
    value = blocks.from_(payload)
    assert bytes(value) == (7).to_bytes(16, "little")
    assert torch.equal((value + value).into(), blocks.zero(device=device).into())


def run_prgs(device):
    keys = tuple(secrets.token_bytes(16) for _ in range(2))
    generators = [prg.ChaCha(nonce=(1, 2), mul=2),
                  prg.Aes128Soft(mul=2, keys=keys)]
    if device.type == "cpu":
        # OpenSSL and AES-NI backends do not accept CUDA tensors.
        generators += [prg.Aes128Mmo(mul=2, keys=keys),
                       prg.Aes128MmoRaw(mul=2, keys=keys)]
    seed = random_blocks(1, device)[0]
    original = seed.clone()
    for generator in generators:
        output = generator.gen(seed)
        assert tuple(output.shape) == (2, 4)
        assert output.device == device
        assert torch.equal(output, generator.gen(seed))
    assert torch.equal(seed, original)


def run_hashes(device):
    hashes = [hash.Blake3(iv=secrets.token_bytes(32))]
    if device.type == "cpu":
        hashes.append(hash.Sha256(key=secrets.token_bytes(16)))
    message = random_blocks(4, device)
    for hasher in hashes:
        # These shapes select the two fixed-length C++ Hash overloads.
        assert tuple(hasher.hash(message).shape) == (2, 4)
        assert tuple(hasher.hash(message[:2]).shape) == (4, 4)

    aes = prg.Aes128Soft(mul=1, keys=(secrets.token_bytes(16),))
    ccr = hash.Aes128Ccr(aes)
    assert tuple(ccr.hash(message[0]).shape) == (4,)


def run_prps(device):
    permutations = [prp.ChaChaFeistel()]
    if device.type == "cpu":
        permutations += [prp.Aes128Feistel(), prp.Aes128FeistelRaw()]
    seed = random_blocks(1, device)[0]
    domain_size = 7
    for permutation in permutations:
        # Python integer results require a device-to-host transfer on CUDA.
        outputs = [permutation.permu(seed, x, domain_size)
                   for x in range(domain_size)]
        assert sorted(outputs) == list(range(domain_size))


def run_seed_util(device):
    seed = random_blocks(1, device)[0]
    original = seed.clone()
    for bit in (False, True):
        modified = util.set_seed_control_bit(seed, bit)
        assert util.get_seed_control_bit(modified) is bit
    assert torch.equal(seed, original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    device = torch.device(parser.parse_args().device)
    if device.type not in ("cpu", "cuda"):
        parser.error("device must be cpu or cuda")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    for operation in (run_groups, run_prgs, run_hashes, run_prps, run_seed_util):
        operation(device)
    print("primitive shapes, group conversions and permutation results match")


if __name__ == "__main__":
    main()

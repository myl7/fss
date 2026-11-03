import torch
from types import SimpleNamespace

import fss_crypto
from fss_crypto import group, hash, prg, prp


def random_blocks(shape):
    return torch.randint(-(1 << 31), 1 << 31, shape, dtype=torch.int32)


def make_prg(mul, backend="chacha"):
    if backend == "chacha":
        return prg.ChaCha(nonce=(1, 2), mul=mul)
    cls = {"aes128_mmo": prg.Aes128Mmo,
           "aes128_mmo_raw": prg.Aes128MmoRaw,
           "aes128_soft": prg.Aes128Soft}[backend]
    return cls(mul=mul, keys=tuple(bytes([i]) * 16 for i in range(mul)))


def make_point_scheme(name, bits=16, out_bits=8, backend="chacha", pred=None):
    if name in ("HalfTreeDpf", "PackedHalfTreeDpf", "HalfTreeGrottoDcf"):
        aes_backend = "aes128_soft" if backend == "chacha" else backend
        ccr = hash.Aes128Ccr(make_prg(1, aes_backend))
        if name == "PackedHalfTreeDpf":
            return fss_crypto.PackedHalfTreeDpf(bits, out_bits, ccr, (1, 2, 3, 4))
        if name == "HalfTreeGrottoDcf":
            return fss_crypto.HalfTreeGrottoDcf(bits, ccr, (1, 2, 3, 4))
        return fss_crypto.HalfTreeDpf(bits, group.Bytes(), ccr, (1, 2, 3, 4))
    if name == "GrottoDcf":
        return fss_crypto.GrottoDcf(bits, make_prg(2, backend))
    if name == "Vdpf":
        return fss_crypto.Vdpf(bits, group.Bytes(), make_prg(2, backend),
                               hash.Blake3(bytes(32)), hash.Blake3(bytes(range(32))))
    if name == "Dcf":
        return fss_crypto.Dcf(bits, group.Bytes(), make_prg(4, backend),
                              pred=pred or fss_crypto.DcfPred.LT)
    return fss_crypto.Dpf(bits, group.Bytes(), make_prg(2, backend))


def generate_point(scheme, seeds, alpha, beta=None):
    cws = torch.empty(scheme.cws_shape, dtype=torch.uint8, device=seeds.device)
    if isinstance(scheme, (fss_crypto.GrottoDcf, fss_crypto.HalfTreeGrottoDcf)):
        scheme.gen(cws, seeds, alpha)
    else:
        scheme.gen(cws, seeds, alpha, beta)
    return cws


def evaluate_all(scheme, party, s0, cws, **kwargs):
    boolean = isinstance(scheme, (fss_crypto.GrottoDcf, fss_crypto.HalfTreeGrottoDcf))
    packed = isinstance(scheme, fss_crypto.PackedHalfTreeDpf)
    count = 1 << (scheme.kDepth if packed else scheme.in_bits)
    shape = (count,) if boolean else (count, 4)
    output = torch.empty(shape, dtype=torch.bool if boolean else torch.int32,
                         device=s0.device)
    scheme.eval_all(bool(party), s0, cws, output, **kwargs)
    return output


def generate_vdpf(scheme, seeds, alpha, beta):
    cws = torch.empty(scheme.cws_shape, dtype=torch.uint8, device=seeds.device)
    cs = torch.empty((4, 4), dtype=torch.int32, device=seeds.device)
    for _ in range(128):
        status = scheme.gen(cws, cs, seeds, alpha, beta)
        if status == 0:
            return SimpleNamespace(cws=cws, cs=cs, seeds=seeds)
        assert status == 1
        seeds = random_blocks((2, 4)).to(beta.device)
    raise AssertionError("VDPF fixture exhausted its retry budget")


def evaluate_vdpf(scheme, state, party, x, device="cpu"):
    device = torch.device(device)
    y = torch.empty((4,), dtype=torch.int32, device=device)
    proof = scheme.eval(bool(party), state.seeds[int(party)].to(device),
                        state.cws.to(device), state.cs.to(device), x, y)
    return y, proof


def evaluate_vdpf_all(scheme, state, party, *, merkle=False, device="cpu", **kwargs):
    device = torch.device(device)
    output = torch.empty((1 << scheme.in_bits, 4), dtype=torch.int32, device=device)
    proof = torch.empty((4, 4), dtype=torch.int32, device=device)
    method = scheme.eval_all_merkle if merkle else scheme.eval_all
    method(bool(party), state.seeds[int(party)].to(device), state.cws.to(device),
           state.cs.to(device), output, proof, **kwargs)
    return output, proof


def make_multi_scheme(verifiable, bits=12, max_points=30, bucket_bits=10,
                      host_only=False, backend="chacha", xor_hash_backend="blake3",
                      hash_backend="blake3"):
    options = dict(in_bits=bits, max_points=max_points, bucket_bits=bucket_bits,
                   group=group.Bytes(), prg=make_prg(2, backend),
                   prp=prp.Aes128Feistel() if host_only else prp.ChaChaFeistel(),
                   in_type="uint64")
    if verifiable:
        hashes = {"blake3": lambda: hash.Blake3(bytes(32)),
                  "sha256": lambda: hash.Sha256(bytes(16))}
        return fss_crypto.Vdmpf(**options, xor_hash=hashes[xor_hash_backend](),
                                hash=hashes[hash_backend]())
    return fss_crypto.Dmpf(**options)


def generate_multi(scheme, points, payloads):
    verifiable = isinstance(scheme, fss_crypto.Vdmpf)
    cws = torch.empty(scheme.cws_shape, dtype=torch.uint8)
    cs = (torch.empty((scheme.kBucketCount, 4, 4), dtype=torch.int32)
          if verifiable else None)
    seeds = random_blocks((2, scheme.kBucketCount, 4))
    sigma = random_blocks((4,))
    for _ in range(128):
        args = (cws, cs) if verifiable else (cws,)
        status, rows, size = scheme.gen(*args, sigma, seeds, points, payloads, len(points))
        if status == 0:
            return SimpleNamespace(cws=cws, cs=cs, seeds=seeds, sigma=sigma,
                                   m_rt=rows, b_size_rt=size)
        if status == 1:
            sigma = random_blocks((4,))
        elif verifiable and status == 2:
            seeds = random_blocks((2, scheme.kBucketCount, 4))
        else:
            raise AssertionError(f"unexpected generation status {status}")
    raise AssertionError("multi-point fixture exhausted its retry budget")


def evaluate_multi(scheme, state, party, xs, *, merkle=False, device="cpu"):
    device = torch.device(device)
    output = torch.empty((xs.shape[0], 4), dtype=torch.int32, device=device)
    args = (bool(party), state.sigma.to(device), state.b_size_rt,
            state.seeds[int(party)].to(device), state.cws.to(device))
    if isinstance(scheme, fss_crypto.Vdmpf):
        proof = torch.empty((4, 4), dtype=torch.int32, device=device)
        method = scheme.batch_eval_merkle if merkle else scheme.batch_eval
        method(*args, state.cs.to(device), xs.to(device), output, proof)
        return output, proof
    scheme.batch_eval(*args, xs.to(device), output)
    return output, None


def evaluate_multi_all(scheme, state, party, *, merkle=False, device="cpu", **kwargs):
    device = torch.device(device)
    output = torch.empty((1 << scheme.in_bits, 4), dtype=torch.int32, device=device)
    args = (bool(party), state.sigma.to(device), state.b_size_rt,
            state.seeds[int(party)].to(device), state.cws.to(device))
    if isinstance(scheme, fss_crypto.Vdmpf):
        proof = torch.empty((4, 4), dtype=torch.int32, device=device)
        method = scheme.eval_all_merkle if merkle else scheme.eval_all
        method(*args, state.cs.to(device), output, proof, **kwargs)
        return output, proof
    scheme.eval_all(*args, output, **kwargs)
    return output, None

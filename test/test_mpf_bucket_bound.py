"""Boundary of the runtime bucket size in the evaluation entries.

The largest bucket index a candidate can locate into is
ceil(3*2^in_bits/b_size_rt) - 1, so a runtime bucket size is usable exactly
when it is at least B = ceil(3*2^in_bits/kBucketCount). One below B lets the
located bucket index leave the allocated rows.

The evaluation entries take only b_size_rt. The base API generation checked
only the ranges of the runtime bucket count and bucket size in CheckRuntime, so
the accept case asserts the same behaviour on both generations while the reject
case asserts rejection on the generation that carries the bound and acceptance
on the base generation, where the missing clause is exactly what this file
exposes.
"""

import inspect

import pytest
import torch

from conftest import generate_multi, make_multi_scheme


BITS = 12
MAX_POINTS = 30
BUCKET_BITS = 10


def make_scheme(verifiable):
    return make_multi_scheme(verifiable, bits=BITS, max_points=MAX_POINTS,
                             bucket_bits=BUCKET_BITS)


def bucket_bound(scheme):
    """Smallest accepted bucket size, ceil(3*2^in_bits/kBucketCount)."""
    return -(-3 * scheme.kDomainSize // scheme.kBucketCount)


def takes_m_rt(scheme):
    """Whether the installed wrapper still passes the runtime bucket count."""
    return "m_rt" in inspect.signature(scheme.eval_all).parameters


def eval_all(scheme, material, party, output, proof, b_size_rt):
    """Positional eval_all call for either wrapper generation."""
    args = [bool(party), material.sigma]
    if takes_m_rt(scheme):
        args.append(material.m_rt)
    args.append(b_size_rt)
    args.append(material.seeds[int(party)])
    args.append(material.cws)
    if proof is not None:
        args.append(material.cs)
    args.append(output)
    if proof is not None:
        args.append(proof)
    scheme.eval_all(*args)


def fixture(scheme):
    """One key generated at t = max_points, so b_size_rt lands on the bound."""
    points = torch.arange(MAX_POINTS, dtype=torch.int64)
    payloads = torch.arange(1, MAX_POINTS + 1, dtype=torch.int32).reshape(-1, 1).repeat(1, 4)
    payloads[:, 3] &= ~1
    material = generate_multi(scheme, points, payloads)
    assert material.m_rt == scheme.kBucketCount
    assert material.b_size_rt == bucket_bound(scheme)
    expected = torch.zeros((1 << BITS, 4), dtype=torch.int32)
    expected[points] = payloads
    return material, expected


def accept(verifiable):
    """A bucket size at the bound is accepted and still reconstructs."""
    scheme = make_scheme(verifiable)
    material, expected = fixture(scheme)
    bound = bucket_bound(scheme)
    assert (scheme.kBucketCount, bound) == (53, 232)
    results = []
    for party in (False, True):
        output = torch.empty((1 << BITS, 4), dtype=torch.int32)
        proof = torch.empty((4, 4), dtype=torch.int32) if verifiable else None
        eval_all(scheme, material, party, output, proof, bound)
        results.append((output, proof))
    assert torch.equal(results[0][0] ^ results[1][0], expected)
    if verifiable:
        assert scheme.verify(results[0][1], results[1][1])


def reject(verifiable):
    """A bucket size one below the bound cannot be used."""
    scheme = make_scheme(verifiable)
    material, _ = fixture(scheme)
    bound = bucket_bound(scheme)
    output = torch.empty((1 << BITS, 4), dtype=torch.int32)
    proof = torch.empty((4, 4), dtype=torch.int32) if verifiable else None
    if takes_m_rt(scheme):
        # The base CheckRuntime has no lower bound on the bucket size, so the
        # call runs there. Only the generation with the bound must reject it.
        eval_all(scheme, material, False, output, proof, bound - 1)
    else:
        with pytest.raises(RuntimeError,
                           match="too small for the allocated bucket count"):
            eval_all(scheme, material, False, output, proof, bound - 1)


def test_dmpf_accepts_bucket_size_at_bound():
    accept(False)


def test_dmpf_rejects_bucket_size_below_bound():
    reject(False)


def test_vdmpf_accepts_bucket_size_at_bound():
    accept(True)


def test_vdmpf_rejects_bucket_size_below_bound():
    reject(True)

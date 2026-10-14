"""Python wrapper for the VDMPF (Verifiable Distributed Multi-Point Function) scheme."""

import dataclasses

import torch

from fss_crypto._jit import load
from fss_crypto._validate import (
    _ch_bucket,
    validate_bucket_bits,
    validate_cpu_only,
    validate_group,
    validate_hash,
    validate_in_bits,
    validate_max_points,
    validate_party,
    validate_prp,
    validate_prg,
    validate_retry,
    validate_t,
)


def _validate_s0s(s0s: torch.Tensor, m: int) -> None:
    if s0s.shape != (m, 2, 4) or s0s.dtype != torch.int32:
        raise TypeError(
            f"s0s must be a ({m}, 2, 4) int32 tensor, "
            f"got shape {tuple(s0s.shape)} dtype {s0s.dtype}"
        )


def _validate_sigma(sigma: torch.Tensor) -> None:
    if sigma.shape != (4,) or sigma.dtype != torch.int32:
        raise TypeError(
            f"sigma must be a (4,) int32 tensor, "
            f"got shape {tuple(sigma.shape)} dtype {sigma.dtype}"
        )


def _validate_alphas(alphas: torch.Tensor, t: int, in_bits: int) -> None:
    if alphas.shape != (t,) or alphas.dtype != torch.int64:
        raise TypeError(
            f"alphas must be a ({t},) int64 tensor, "
            f"got shape {tuple(alphas.shape)} dtype {alphas.dtype}"
        )
    lo = int(alphas.min())
    hi = int(alphas.max())
    if lo < 0 or hi >= (1 << in_bits):
        raise ValueError(f"alphas must be in [0, 2^{in_bits}), got [{lo}, {hi}]")


def _validate_betas(betas: torch.Tensor, t: int) -> None:
    if betas.shape != (t, 4) or betas.dtype != torch.int32:
        raise TypeError(
            f"betas must be a ({t}, 4) int32 tensor, "
            f"got shape {tuple(betas.shape)} dtype {betas.dtype}"
        )


def _validate_meta(meta: torch.Tensor) -> None:
    if meta.shape != (2,) or meta.dtype != torch.int32:
        raise TypeError(
            f"meta must be a (2,) int32 tensor, "
            f"got shape {tuple(meta.shape)} dtype {meta.dtype}"
        )


def _validate_xs(xs: torch.Tensor) -> None:
    if xs.dim() != 1 or xs.dtype != torch.int64:
        raise TypeError(
            f"xs must be a 1-D int64 tensor, "
            f"got shape {tuple(xs.shape)} dtype {xs.dtype}"
        )


def _validate_pi(pi: torch.Tensor) -> None:
    if pi.shape != (4, 4) or pi.dtype != torch.int32:
        raise TypeError(
            f"pi must be a (4, 4) int32 tensor, "
            f"got shape {tuple(pi.shape)} dtype {pi.dtype}"
        )


@dataclasses.dataclass(frozen=True)
class VdmpfKey:
    """Key material returned by Vdmpf.gen().

    Attributes:
        cws: (m, bucket_bits, 8) int32 correction words, shared by both
            parties.
        cs: (m, 4, 4) int32 correction seeds, shared by both parties.
        ocw: (m, 4) int32 output correction words, shared by both parties.
        s0s: (2, m, 4) int32 initial seeds; row i is party i's key share.
        sigma: (4,) int32 effective PRP seed for Cuckoo hashing, written back
            after Gen's internal retries.
        meta: (2,) int32 runtime parameters {m_rt, b_size_rt} used by eval.
    """

    cws: torch.Tensor
    cs: torch.Tensor
    ocw: torch.Tensor
    s0s: torch.Tensor
    sigma: torch.Tensor
    meta: torch.Tensor


class Vdmpf:
    """2-party verifiable distributed multi-point function (VDMPF).

    Host-only scheme: all tensors must live on CPU. This binding release keeps
    alphas and xs in int64 tensors, so in_bits is limited to 63.

    Args:
        in_bits: Input domain bit size (1..63).
        max_points: Maximum number of point functions (>= 30).
        bucket_bits: Bit size of the inner VDPF domain (per bucket).
        group: Output group type, "bytes" or "uint".
        prg: PRG type, "chacha" or "aes128_mmo".
        xor_hash: XorHash type (the paper's H), "blake3" or "sha256".
        hash: Hash type (the paper's H'); defaults to xor_hash.
        prp: PRP type for Cuckoo hashing, "aes128_feistel".
    """

    # Verify() is key- and knob-independent: every VDMPF extension exposes the
    # same check, so any compiled module works.
    _verify_ext: object | None = None

    def __init__(self, in_bits: int, max_points: int, bucket_bits: int,
                 group: str = "bytes", prg: str = "chacha",
                 xor_hash: str = "blake3", hash: str | None = None,
                 prp: str = "aes128_feistel"):
        validate_in_bits(in_bits)
        if in_bits > 63:
            raise ValueError(
                f"in_bits must be at most 63 in this binding release "
                f"(alphas and xs are int64 tensors), got {in_bits}"
            )
        validate_max_points(max_points)
        validate_bucket_bits(in_bits, max_points, bucket_bits)
        validate_group(group)
        validate_prg(prg, "vdmpf")
        validate_hash(xor_hash)
        if hash is None:
            hash = xor_hash
        validate_hash(hash)
        validate_prp(prp)
        self.in_bits = in_bits
        self.max_points = max_points
        self.bucket_bits = bucket_bits
        self.group = group
        self.prg = prg
        self.xor_hash = xor_hash
        self.hash = hash
        # Number of Cuckoo-hash buckets, mirroring fss::cuckoo_hash::ChBucket.
        self.m = _ch_bucket(max_points)
        self._ext = load("vdmpf", in_bits, group, prg, max_points=max_points,
                         bucket_bits=bucket_bits, xor_hash=xor_hash,
                         hash=hash, prp=prp)
        Vdmpf._verify_ext = self._ext

    def gen(self, s0s: torch.Tensor, sigma: torch.Tensor,
            alphas: torch.Tensor, betas: torch.Tensor, t: int,
            retry: bool = True) -> VdmpfKey:
        """Generate VDMPF keys for t point functions.

        Args:
            s0s: (m, 2, 4) int32 tensor of initial seeds for the inner VDPFs.
            sigma: (4,) int32 tensor, the PRP seed for Cuckoo hashing.
            alphas: (t,) int64 tensor of distinct point inputs in
                [0, 2^in_bits).
            betas: (t, 4) int32 tensor of point outputs. The scheme clamps the
                last word's LSB to 0 (lambda = 127), so that bit of beta never
                reconstructs.
            t: Number of points, 30 <= t <= max_points.
            retry: Whether to retry with perturbed sigma and seeds when Cuckoo
                hashing fails.

        Returns:
            VdmpfKey holding the correction words, correction seeds, output
            correction words, the effective seeds and sigma of the successful
            attempt, and the runtime parameters.
        """
        validate_t(t, self.max_points)
        validate_retry(retry)
        _validate_s0s(s0s, self.m)
        _validate_sigma(sigma)
        _validate_alphas(alphas, t, self.in_bits)
        _validate_betas(betas, t)
        if torch.unique(alphas).numel() != t:
            raise ValueError(f"alphas must be distinct, got {t} points with duplicates")
        validate_cpu_only(s0s, sigma, alphas, betas, fn_name="gen")

        cws, cs, ocw, s0s, sigma, meta = self._ext.gen(s0s, sigma, alphas, betas, t, retry)
        return VdmpfKey(cws=cws, cs=cs, ocw=ocw, s0s=s0s, sigma=sigma, meta=meta)

    def _check_key(self, key: VdmpfKey) -> None:
        if not isinstance(key, VdmpfKey):
            raise TypeError(f"key must be a VdmpfKey, got {type(key).__name__}")
        expected_s0s = (2, self.m, 4)
        if key.s0s.shape != expected_s0s or key.s0s.dtype != torch.int32:
            raise TypeError(
                f"key.s0s must be a {expected_s0s} int32 tensor, "
                f"got shape {tuple(key.s0s.shape)} dtype {key.s0s.dtype}"
            )
        _validate_sigma(key.sigma)
        _validate_meta(key.meta)
        cws_expected = (self.m, self.bucket_bits, 8)
        if key.cws.shape != cws_expected or key.cws.dtype != torch.int32:
            raise TypeError(
                f"cws must be a {cws_expected} int32 tensor, "
                f"got shape {tuple(key.cws.shape)} dtype {key.cws.dtype}"
            )
        if key.cs.shape != (self.m, 4, 4) or key.cs.dtype != torch.int32:
            raise TypeError(
                f"cs must be a ({self.m}, 4, 4) int32 tensor, "
                f"got shape {tuple(key.cs.shape)} dtype {key.cs.dtype}"
            )
        if key.ocw.shape != (self.m, 4) or key.ocw.dtype != torch.int32:
            raise TypeError(
                f"ocw must be a ({self.m}, 4) int32 tensor, "
                f"got shape {tuple(key.ocw.shape)} dtype {key.ocw.dtype}"
            )

    def batch_eval(self, party: int, key: VdmpfKey,
                   xs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the VDMPF on a batch of inputs and produce a proof.

        Args:
            party: Party index, 0 or 1.
            key: This party's key from gen().
            xs: 1-D int64 tensor of inputs in [0, 2^in_bits). Duplicates are
                allowed; each occurrence gets its own share.

        Returns:
            (ys, pi): a (t, 4) int32 tensor of output shares and a (4, 4)
            int32 proof. Verification compares the proofs of both parties.
        """
        validate_party(party)
        self._check_key(key)
        _validate_xs(xs)
        validate_cpu_only(key.cws, key.cs, key.ocw, key.s0s, key.sigma, key.meta, xs,
                          fn_name="batch_eval")

        return self._ext.batch_eval(party, key.cws, key.cs, key.ocw,
                                    key.s0s[party], key.sigma, key.meta, xs)

    @staticmethod
    def verify(pi0: torch.Tensor, pi1: torch.Tensor) -> bool:
        """Verify a pair of proofs from batch_eval().

        Args:
            pi0: (4, 4) int32 proof from party 0.
            pi1: (4, 4) int32 proof from party 1.

        Returns:
            True if the proofs match (Accept), False otherwise (Reject).
        """
        _validate_pi(pi0)
        _validate_pi(pi1)
        validate_cpu_only(pi0, pi1, fn_name="verify")
        if Vdmpf._verify_ext is None:
            # No Vdmpf instance has been built yet; compile a minimal module.
            # Verify itself is knob-independent.
            Vdmpf._verify_ext = load("vdmpf", 1, "bytes", "chacha", max_points=30,
                                     bucket_bits=1, xor_hash="blake3",
                                     hash="blake3", prp="aes128_feistel")
        return bool(Vdmpf._verify_ext.verify(pi0, pi1))

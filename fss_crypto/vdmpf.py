"""Thin bindings over the C++ Vdmpf key buffers."""

from fss_crypto._scheme import _Scheme

class Vdmpf(_Scheme):
    """Configure a multi-point scheme without owning an alternative Key API.

    Cws have native opaque shape [M, bucket_bits + 1, sizeof(Cw)]. Seed pairs
    have party-first shape [2, M, 4], party seeds [M, 4], and sigma [4].
    M is kBucketCount. All native block buffers are contiguous int32 tensors.
    cs has shape [M, 4, 4], and each proof has shape [4, 4].
    Runtime bucket sizes are Python integers, including 128-bit values.
    Evaluation takes only the runtime bucket size; the runtime bucket count from gen is key metadata.
    Output group width and modulus come from group, independently of in_bits.
    """

    def __init__(self, in_bits, max_points, bucket_bits, group, prg, xor_hash, hash, prp, *,
                 in_type=None, ch_lambda=80):
        super().__init__("vdmpf", in_bits=in_bits, max_points=max_points,
                         bucket_bits=bucket_bits, group=group, prg=prg,
                         xor_hash=xor_hash, hash=hash, prp=prp, in_type=in_type,
                         ch_lambda=ch_lambda)

    @classmethod
    def precompile(cls, in_bits, max_points, bucket_bits, group, prg, xor_hash, hash, prp, *,
                   in_type=None, ch_lambda=80,
                   threads_per_block: int = 256) -> None:
        """Prepare the compiled configuration without creating native state."""
        cls._precompile_config("vdmpf", in_bits=in_bits, max_points=max_points,
                               bucket_bits=bucket_bits, group=group, prg=prg,
                               xor_hash=xor_hash, hash=hash, prp=prp,
                               in_type=in_type, ch_lambda=ch_lambda,
                               threads_per_block=threads_per_block)

    @property
    def kMinPoints(self) -> int:
        """Smallest supported point count for the configured bucket width."""
        return self._native.kMinPoints

    @property
    def kBucketCount(self) -> int:
        """Allocated bucket count from Core, used to size seed and CS rows."""
        return self._native.kBucketCount

    @property
    def kDomainSize(self) -> int:
        """Input-domain cardinality from Core, returned as a Python integer."""
        return self._native.kDomainSize

    @property
    def kBucketSize(self) -> int:
        """Core bucket size at max_points. Use Gen's b_size_rt for actual keys."""
        return self._native.kBucketSize

    def gen(self, cws, cs, sigma, s0s, as_, b_bufs, t, max_evictions=1000):
        """Make one CPU Core attempt, preserving sigma and seeds.

        t must be between kMinPoints and the configured max_points.
        s0s is contiguous int32 [2, kBucketCount, 4]. Each s0s[party] is
        contiguous and can be passed directly to this party's evaluation.
        Return (status, m_rt, b_size_rt). Only status 0 makes outputs usable.
        Status 1 means Cuckoo placement failed: draw a fresh uniform sigma
        from a cryptographically secure RNG and preserve s0s.
        Status 2 means an inner VDPF failed: preserve sigma and draw fresh
        independent uniform seed pairs for every allocated bucket from a
        cryptographically secure RNG. For either failure, discard all
        correction words and checks and rerun the whole gen with the same points and
        payloads. The caller owns retry limits.
        max_evictions bounds Cuckoo insertion work within this one call.
        as_ holds at least t distinct inputs and b_bufs at least t output blocks.
        Core clears bit 0 of b_bufs[j, 3] before Group::From conversion.
        In128 inputs use int64
        or uint64 [t, 2], with the low limb first.
        """
        return self._native.gen(self._cws(cws), cs, sigma, s0s, as_, b_bufs,
                                t, max_evictions)

    def batch_eval(self, b, sigma, b_size_rt, s0s, cws, cs, xs, ys, pi):
        """Write output shares and a CPU chain proof, rejecting CUDA buffers."""
        self._native.batch_eval(b, sigma, b_size_rt, s0s,
                                self._cws(cws), cs, xs, ys, pi)

    def batch_eval_merkle(self, b, sigma, b_size_rt, s0s, cws, cs,
                          xs, ys, pi):
        """Write shares and the canonical Core Merkle proof on CPU or CUDA.

        Leaves retain (query position, PRP candidate) order within each bucket.
        Adjacent pairs fold with odd-node carry, then bucket roots fold in
        bucket order. Both parties must use the same ordered xs and protocol.
        """
        self._native.batch_eval_merkle(b, sigma, b_size_rt, s0s,
                                       self._cws(cws), cs, xs, ys, pi)

    def eval_all(self, b, sigma, b_size_rt, s0s, cws, cs, ys, pi):
        """Write the full CPU output domain and its Core chain proof.

        All buffers must use CPU storage. CUDA buffers are rejected.
        Outputs and proof match
        ``batch_eval`` over all domain inputs in increasing order.
        The caller allocates ``ys`` and ``pi``.
        """
        self._native.eval_all(b, sigma, b_size_rt, s0s,
                              self._cws(cws), cs, ys, pi)

    def eval_all_merkle(self, b, sigma, b_size_rt, s0s, cws, cs, ys, pi, *,
                        threads_per_block: int = 256):
        """Write the full output domain and its Core Merkle proof on CPU or CUDA.

        CUDA uses threads_per_block. CPU ignores this value.
        All buffers must use the same device. Outputs and proof match
        ``batch_eval_merkle`` over all domain inputs in increasing order.
        The caller allocates ``ys`` and ``pi``.
        """
        self._execution_native(ys, threads_per_block=threads_per_block).eval_all_merkle(b, sigma, b_size_rt, s0s,
                                     self._cws(cws), cs, ys, pi)

    def verify(self, pi0, pi1):
        """Compare proofs. CUDA proofs require a stream-local scalar transfer."""
        return self._native.verify(pi0, pi1)


__all__ = ['Vdmpf']

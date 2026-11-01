"""Thin component-buffer bindings for C++ Dmpf."""

from fss_crypto._scheme import _Scheme, _parallel_depth

class Dmpf(_Scheme):
    """Configure a multi-point scheme without owning an alternative Key API.

    Cws have native opaque shape [M, bucket_bits + 1, sizeof(Cw)]. Seed pairs
    have party-first shape [2, M, 4], party seeds [M, 4], and sigma [4].
    M is kBucketCount. All native block buffers are contiguous int32 tensors.
    Runtime bucket sizes are Python integers, including 128-bit values.
    Output group width and modulus come from group, independently of in_bits.
    """

    def __init__(self, in_bits, max_points, bucket_bits, group, prg, prp, *,
                 in_type=None, ch_lambda=80):
        super().__init__("dmpf", in_bits=in_bits, max_points=max_points,
                         bucket_bits=bucket_bits, group=group, prg=prg,
                         prp=prp, in_type=in_type,
                         ch_lambda=ch_lambda)

    @property
    def kMinPoints(self) -> int:
        """Smallest supported point count for the configured bucket width."""
        return self._native.kMinPoints

    @property
    def kBucketCount(self) -> int:
        """Allocated bucket count from Core, used to size seed and correction rows."""
        return self._native.kBucketCount

    @property
    def kDomainSize(self) -> int:
        """Input-domain cardinality from Core, returned as a Python integer."""
        return self._native.kDomainSize

    @property
    def kBucketSize(self) -> int:
        """Core bucket size at max_points. Use Gen's b_size_rt for actual keys."""
        return self._native.kBucketSize

    def gen(self, cws, sigma, s0s, as_, b_bufs, t, max_evictions=1000):
        """Make one CPU Core attempt, preserving sigma and seeds.

        t must be between kMinPoints and the configured max_points.
        s0s is contiguous int32 [2, kBucketCount, 4]. Each s0s[party] is
        contiguous and can be passed directly to this party's evaluation.
        Return (status, m_rt, b_size_rt). Only status 0 makes outputs usable.
        After status 1, discard corrections, draw fresh uniform sigma from a
        cryptographically secure RNG, and call gen again with the same inputs.
        Existing independently sampled bucket seeds may be retained.
        max_evictions bounds Cuckoo insertion work within this one call.
        as_ holds at least t distinct inputs and b_bufs at least t output blocks.
        Core clears bit 0 of b_bufs[j, 3] before Group::From conversion.
        In128 inputs use int64
        or uint64 [t, 2], with the low limb first.
        """
        return self._native.gen(self._cws(cws), sigma, s0s, as_, b_bufs,
                                t, max_evictions)

    def batch_eval(self, b, sigma, m_rt, b_size_rt, s0s, cws, xs, ys):
        """Write caller-owned output shares."""
        self._native.batch_eval(b, sigma, m_rt, b_size_rt, s0s,
                                self._cws(cws), xs, ys)

    def eval_all(self, b, sigma, m_rt, b_size_rt, s0s, cws, ys, *, par_depth: int = 0,
                 frontier_depth: int = 16, threads_per_block: int = 256):
        """Write all domain shares to caller-owned storage.

        par_depth controls CPU partitioning; 0 is sequential.
        Positive values require OpenMP and a parallel-safe backend.
        CUDA calls ignore par_depth and use frontier_depth/threads_per_block.
        CPU calls ignore frontier_depth/threads_per_block.
        """
        self._execution_native(ys, frontier_depth, threads_per_block).eval_all(b, sigma, m_rt, b_size_rt, s0s,
                              self._cws(cws), ys, _parallel_depth(par_depth, ys))


__all__ = ['Dmpf']

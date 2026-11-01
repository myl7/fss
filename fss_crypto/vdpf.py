"""Thin component-buffer bindings for the C++ verifiable point function."""

from fss_crypto._scheme import _Scheme, _parallel_depth

class Vdpf(_Scheme):
    """Configure Core VDPF with explicit PRG and verification hashes.

    Corrections include the final output correction. Native block buffers use
    int32 [..., 4], and cs and proofs use int32 [4, 4].
    Output group width and modulus come from group, independently of in_bits.
    Chain and Merkle proofs are distinct protocols. Both parties must use the
    same protocol and the same ordered query sequence.
    """

    def __init__(self, in_bits, group, prg, xor_hash, hash, *, in_type=None):
        super().__init__("vdpf", in_bits=in_bits, group=group, prg=prg,
                         xor_hash=xor_hash, hash=hash, in_type=in_type)

    def gen(self, cws, cs, s0s, a, b_buf):
        """Write corrections once, preserving s0s. Return 0 on success or 1.

        Outputs are unspecified on failure. s0s has shape [2, 4].
        To repeat after status 1, draw two fresh independent uniform seed
        blocks from a cryptographically secure RNG and call gen again with
        the same a and b_buf. Retry limits belong to the caller.
        b_buf is an int32 [4] native block. Core clears bit 0 of b_buf[3]
        before Group::From conversion. This does not mutate b_buf.
        """
        return self._native.gen(self._cws(cws), cs, s0s, a, b_buf)

    def eval(self, b, s0, cws, cs, x, y):
        """Write one output share to y and return its corrected hash [4, 4]."""
        return self._native.eval(b, s0, self._cws(cws), cs, x, y)

    def prove(self, pi_tildes, cs, pi):
        """Write the Core chain proof on CPU from corrected hashes [N, 4, 4]."""
        self._native.prove(pi_tildes, cs, pi)

    def prove_merkle(self, pi_tildes, cs, pi, *, par_depth: int = 0):
        """Write the canonical Merkle proof on CPU or through its GPU helper.

        pi_tildes is a nonempty int32 [N, 4, 4] tensor in query order.
        Adjacent nodes combine, carrying odd trailing nodes unchanged.
        Execution geometry does not change the proof tree.

        par_depth controls CPU partitioning; 0 is sequential.
        Positive values require OpenMP and a parallel-safe backend.
        CUDA calls ignore par_depth.
        """
        self._native.prove_merkle(pi_tildes, cs, pi, _parallel_depth(par_depth, pi))

    def eval_all(self, b, s0, cws, cs, ys, pi, *, par_depth: int = 0):
        """Write all shares and the chain proof on CPU into caller storage.

        par_depth controls CPU partitioning; 0 is sequential.
        Positive values require OpenMP and a parallel-safe backend.
        This chain-proof operation requires CPU buffers.
        """
        self._native.eval_all(b, s0, self._cws(cws), cs, ys, pi, _parallel_depth(par_depth, pi))

    def eval_range(self, b, L, R, s0, cws, cs, ys, pi, *, par_depth: int = 0):
        """Write shares for Core's inclusive range [L, R] and a CPU chain proof.

        Endpoints may use the full configured input width.
        The range's output count and byte size must fit addressable storage.

        par_depth controls CPU partitioning; 0 is sequential.
        Positive values require OpenMP and a parallel-safe backend.
        This chain-proof operation requires CPU buffers.
        """
        self._native.eval_range(b, L, R, s0, self._cws(cws), cs, ys, pi, _parallel_depth(par_depth))

    def eval_all_merkle(self, b, s0, cws, cs, ys, pi, *, par_depth: int = 0,
                        frontier_depth: int = 16, threads_per_block: int = 256):
        """Write full-domain shares and a Merkle proof on the buffers' device.

        CUDA uses frontier_depth/threads_per_block. CPU ignores these values.

        par_depth controls CPU partitioning; 0 is sequential.
        Positive values require OpenMP and a parallel-safe backend.
        CUDA calls ignore par_depth.
        """
        self._execution_native(ys, frontier_depth, threads_per_block).eval_all_merkle(b, s0, self._cws(cws), cs, ys, pi, _parallel_depth(par_depth, pi))

    def verify(self, pi0, pi1):
        """Compare proofs. CUDA proofs require a stream-local scalar transfer."""
        return self._native.verify(pi0, pi1)


__all__ = ['Vdpf']

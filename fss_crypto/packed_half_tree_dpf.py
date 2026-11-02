"""Thin binding of ``fss::PackedHalfTreeDpf``."""

import torch

from fss_crypto._scheme import _Scheme, _parallel_depth


class PackedHalfTreeDpf(_Scheme):
    """Configure packed outputs, a CCR hash and public instance seed S.

    ``instance_seed`` supplies the four 32-bit words of S. int4 storage uses
    contiguous ``torch.int32`` tensors with a final dimension of four words.
    ``s0s`` contains two blocks and ``s0`` contains one. Allocate ``cws`` as a
    contiguous uint8 tensor with ``cws_shape``, including the terminal correction word.
    Full-domain output retains exactly ``2**kDepth`` packed int4 blocks.
    """

    def __init__(self, in_bits: int, out_bits: int, hash, instance_seed, *,
                 in_type=None):
        super().__init__("packed_half_tree_dpf", in_bits=in_bits, out_bits=out_bits,
                         hash=hash, instance_seed=instance_seed, in_type=in_type)

    @classmethod
    def precompile(cls, in_bits: int, out_bits: int, hash, *, in_type=None,
                   frontier_depth: int = 16, threads_per_block: int = 256) -> None:
        """Prepare the compiled configuration without creating native state."""
        cls._precompile_config("packed_half_tree_dpf", in_bits=in_bits, out_bits=out_bits, hash=hash,
                               in_type=in_type,
                               frontier_depth=frontier_depth,
                               threads_per_block=threads_per_block)

    @property
    def kDepth(self) -> int:
        """Number of tree levels expanded before packed output lanes."""
        return self._native.kDepth

    @property
    def kLanes(self) -> int:
        """Number of consecutive output shares in one int4 block."""
        return self._native.kLanes

    def gen(self, cws, s0s: torch.Tensor, a: int, beta: int) -> None:
        """Fill correction words once for ``0 <= beta < 2**out_bits``."""
        self._native.gen(self._cws(cws), s0s, a, beta)

    def eval(self, b: bool, s0: torch.Tensor, cws, x: int) -> torch.Tensor:
        """Return the packed int4 block containing x's output lane.

        The result contains kLanes shares. Use ``extract(result, x)`` to read
        x's share, passing the original input x.
        """
        return self._native.eval(b, s0, self._cws(cws), x)

    def extract(self, block: torch.Tensor, x: int) -> int:
        """Return x's lane through Core ``Extract``.

        For full-domain output, use ``extract(ys[x // kLanes], x)``.
        For point and full-domain output, x is the original input and the lane
        is ``x % kLanes``. For range output, use the range-relative offset j
        with ``extract(ys[j // kLanes], j)``.
        For in_bits=8 and out_bits=8, kLanes=16, so x=42 uses
        ``extract(ys[2], 42)`` and reads lane 10.
        Returning a Python integer synchronizes when ``block`` is on CUDA.
        """
        return self._native.extract(block, x)

    def eval_range(self, b: bool, first: int, last: int, s0: torch.Tensor,
                   cws, ys: torch.Tensor, *, par_depth: int = 0) -> None:
        """Fill packed shares for the closed interval [first, last] on CPU.

        first and last are Python integers in the configured input domain.
        Allocate ceil((last - first + 1) / kLanes) int4 blocks in ys.
        Range-relative lane zero corresponds to first.
        Read offset j with ``extract(ys[j // kLanes], j)``.
        Unused lanes in the final block are zero.
        All tensors must be contiguous CPU buffers.
        par_depth controls CPU partitioning, and 0 is sequential.
        """
        self._native.eval_range(b, first, last, s0, self._cws(cws), ys,
                                _parallel_depth(par_depth))

    def eval_all(self, b: bool, s0: torch.Tensor, cws, ys: torch.Tensor, *, par_depth: int = 0,
                 frontier_depth: int = 16, threads_per_block: int = 256) -> None:
        """Fill ``2**kDepth`` packed blocks in input order, without unpacking.

        Allocate contiguous int32 ys with shape ``(B, 4)``, where
        ``B = 2**kDepth = 2**in_bits // kLanes``.
        Its tensor payload is ``16 * B`` bytes per party.
        Two parties' outputs use twice that amount. Key buffers and temporary
        storage are separate. Counts and bytes must fit tensor address limits.
        For in_bits=8 and out_bits=8, ys has shape (16, 4), not (256, 4).
        Read input x with ``extract(ys[x // kLanes], x)``.

        par_depth controls CPU partitioning; 0 is sequential.
        Positive values require OpenMP and a parallel-safe backend.
        CUDA calls ignore par_depth and use frontier_depth/threads_per_block.
        CPU calls ignore frontier_depth/threads_per_block.
        """
        self._execution_native(ys, frontier_depth, threads_per_block).eval_all(b, s0, self._cws(cws), ys, _parallel_depth(par_depth, ys))


    def eval_all_gpu_batch(self, b: bool, s0s: torch.Tensor, cws: torch.Tensor,
                           ys: torch.Tensor, *, frontier_depth: int = 16,
                           threads_per_block: int = 256) -> None:
        """Fill full-domain shares for a key-major batch on one CUDA device.

        s0s has shape (key_count, 4), cws has shape
        (key_count, *self.cws_shape), and ys has shape
        (key_count, outputs_per_key, 4). All tensors must be contiguous.
        outputs_per_key is 2**self.kDepth packed blocks.
        All keys share this instance's backends and the same party b.
        The first call on a device synchronizes to prepare its host launch view.
        Later launches are asynchronous on the current PyTorch stream.
        frontier_depth and threads_per_block select the Core GPU launch specialization.
        """
        self._execution_native(ys, frontier_depth, threads_per_block).eval_all_gpu_batch(b, s0s, cws, ys)


__all__ = ['PackedHalfTreeDpf']

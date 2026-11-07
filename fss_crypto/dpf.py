"""Thin binding of ``fss::Dpf`` and its explicit GPU Helpers."""

import torch

from fss_crypto._scheme import _Scheme, _parallel_depth


class Dpf(_Scheme):
    """Configure DPF with explicit output-group and PRG constructor values.

    ``in_type`` controls the C++ input type independently of ``group``.

    int4 storage uses contiguous ``torch.int32`` tensors with a final dimension
    of four words. ``s0s`` contains two blocks, ``s0`` and ``b_buf`` contain one,
    and ``eval_all`` output ``ys`` contains exactly ``2**in_bits`` blocks. Allocate ``cws`` as a
    contiguous uint8 tensor with ``cws_shape``: 16-byte component rows holding
    the seed corrections, including the terminal, and the packed control.
    """

    def __init__(self, in_bits: int, group, prg, *, in_type=None):
        super().__init__("dpf", in_bits=in_bits, group=group, prg=prg,
                         in_type=in_type)

    @classmethod
    def precompile(cls, in_bits: int, group, prg, *, in_type=None,
                   frontier_depth: int = 16, threads_per_block: int = 256) -> None:
        """Prepare the compiled configuration without creating native state."""
        cls._precompile_config("dpf", in_bits=in_bits, group=group, prg=prg,
                               in_type=in_type,
                               frontier_depth=frontier_depth,
                               threads_per_block=threads_per_block)

    def gen(self, cws, s0s: torch.Tensor, a: int, b_buf: torch.Tensor) -> None:
        """Fill caller-owned correction words with one Core ``Gen`` call.

        ``s0s`` is preserved. ``cws`` includes the terminal correction word.
        GPU scalar payload conversion can synchronize the current stream.
        """
        self._native.gen(self._cws(cws), s0s, a, b_buf)

    def eval(self, b: bool, s0: torch.Tensor, cws, x: int) -> torch.Tensor:
        """Return one int4 share using the raw Core ``Eval`` buffers."""
        return self._native.eval(b, s0, self._cws(cws), x)

    def eval_range(self, b: bool, first: int, last: int, s0: torch.Tensor,
                   cws, ys: torch.Tensor, *, par_depth: int = 0,
                   frontier_depth: int = 16, threads_per_block: int = 256) -> None:
        """Fill caller-owned shares for the closed interval [first, last].

        first and last are Python integers in the configured input domain.
        ys contains last - first + 1 int4 blocks in input order.
        All tensors must be contiguous and use the same exact CPU/CUDA device.
        CPU uses par_depth and ignores the GPU launch knobs.
        CUDA ignores par_depth and uses frontier_depth/threads_per_block.
        Scalar seed fetch and backend launch-view preparation may synchronize.
        """
        self._execution_native(ys, frontier_depth, threads_per_block).eval_range(
            b, first, last, s0, self._cws(cws), ys,
            _parallel_depth(par_depth, ys))

    def eval_all(self, b: bool, s0: torch.Tensor, cws, ys: torch.Tensor, *, par_depth: int = 0,
                 frontier_depth: int = 16, threads_per_block: int = 256) -> None:
        """Fill caller-owned ``ys`` with one share for every domain input.

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
        outputs_per_key is 2**self.in_bits blocks.
        All keys share this instance's backends and the same party b.
        The first call on a device synchronizes to prepare its host launch view.
        Later launches are asynchronous on the current PyTorch stream.
        frontier_depth and threads_per_block select the Core GPU launch specialization.
        """
        self._execution_native(ys, frontier_depth, threads_per_block).eval_all_gpu_batch(b, s0s, cws, ys)


__all__ = ['Dpf']

"""Thin binding of ``fss::GrottoDcf`` and its parity-tree members."""

from dataclasses import dataclass, field

import torch

from fss_crypto._scheme import _Scheme, _parallel_depth


@dataclass(frozen=True)
class ParityTreeView:
    """Borrow a level-order bool tree of ``2**(in_bits + 1) - 1`` entries.

    For ``N = 2**in_bits``, p is a contiguous bool tensor of shape
    ``(2*N-1,)`` with ``2*N-1`` payload bytes per party.
    Root is p[0], leaf x is p[N-1+x], and each internal node is its
    two children's XOR. For in_bits=8, p has 511 entries, not 256.
    Both parties' trees use twice that payload. Key buffers, evaluation
    outputs, allocator overhead, and GPU temporary storage are separate.
    Counts and bytes must fit tensor address limits.
    Keeping this view alive keeps its tensor storage alive.
    Do not mutate the storage while evaluation is in flight.
    """

    p: torch.Tensor
    _native_views: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def _native_view(self, module):
        native_type = module.NativeParityTreeView
        if native_type not in self._native_views:
            self._native_views[native_type] = native_type(self.p)
        return self._native_views[native_type]


class OwnedParityTree:
    """Host storage allocated by the C++ scheme's ``PreprocessOwned`` member.

    The owned bool tensor has ``2**(in_bits + 1) - 1`` entries and payload
    bytes per party. The owner object and allocator overhead are separate.
    """

    def __init__(self, *args, **kwargs):
        raise TypeError("use scheme.preprocess_owned() to obtain an owned tree")

    @classmethod
    def _from_native(cls, p: torch.Tensor):
        result = object.__new__(cls)
        result._p = p
        return result

    def view(self) -> ParityTreeView:
        """Borrow this allocation, retaining its lifetime through the tensor."""
        return ParityTreeView(self._p)


class GrottoDcf(_Scheme):
    """Configure comparison shares with an explicit PRG constructor value."""

    def __init__(self, in_bits: int, prg, *, in_type=None):
        super().__init__("grotto_dcf", in_bits=in_bits, prg=prg,
                         in_type=in_type)

    def gen(self, cws, s0s: torch.Tensor, a: int) -> None:
        """Fill corrections with one Core ``Gen`` call, preserving ``s0s``."""
        self._native.gen(self._cws(cws), s0s, a)

    def eval(self, b: bool, s0: torch.Tensor, cws, x: int) -> bool:
        """Return one raw Core share. A CUDA bool result synchronizes its stream."""
        return self._native.eval(b, s0, self._cws(cws), x)

    def eval_all(self, b: bool, s0: torch.Tensor, cws, ys: torch.Tensor,
                 *, par_depth: int = 0, frontier_depth: int = 16,
                 threads_per_block: int = 256) -> None:
        """Fill bool outputs using Core's leaf-prefix full-domain route.

        This route can differ from raw scalar ``eval`` for each party.

        par_depth controls CPU partitioning; 0 is sequential.
        Positive values require OpenMP and a parallel-safe backend.
        CUDA calls ignore par_depth and use frontier_depth/threads_per_block.
        CPU calls ignore frontier_depth/threads_per_block.
        """
        self._execution_native(ys, frontier_depth, threads_per_block).eval_all(b, s0, self._cws(cws), ys, _parallel_depth(par_depth, ys))

    def preprocess(self, b: bool, s0: torch.Tensor, cws, p: torch.Tensor,
                   *, par_depth: int = 0, frontier_depth: int = 16,
                   threads_per_block: int = 256) -> None:
        """Fill caller-owned parity storage for ``ParityTreeView(p)``.

        With ``N = 2**in_bits``, allocate contiguous bool p with shape
        ``(2*N-1,)``. Its payload is ``2*N-1`` bytes per party.
        For in_bits=8, allocate 511 bool entries. Two parties require two
        such buffers. Keys and evaluation outputs are separate.
        Counts and bytes must fit tensor address limits.

        par_depth controls CPU partitioning; 0 is sequential.
        Positive values require OpenMP and a parallel-safe backend.
        CUDA calls ignore par_depth and use frontier_depth/threads_per_block.
        CPU calls ignore frontier_depth/threads_per_block.
        """
        self._execution_native(p, frontier_depth, threads_per_block).preprocess(b, s0, self._cws(cws), p, _parallel_depth(par_depth, p))

    def preprocess_owned(self, b: bool, s0: torch.Tensor, cws, *, par_depth: int = 0) -> OwnedParityTree:
        """Allocate a host tree through the C++ ``PreprocessOwned`` member.

        For ``N = 2**in_bits``, the tree contains ``2*N-1`` bool entries
        and uses ``2*N-1`` payload bytes per party.
        The owner object and allocator overhead are separate.

        par_depth controls CPU partitioning; 0 is sequential.
        Positive values require OpenMP and a parallel-safe backend.
        """
        return OwnedParityTree._from_native(self._native.preprocess_owned(b, s0, self._cws(cws), _parallel_depth(par_depth)))

    def eval_tree(self, pt: ParityTreeView, x: int) -> bool:
        """Evaluate a preprocessed view without a party or raw key."""
        return self._native.eval_tree(pt._native_view(self._ext), x)

    def eval_all_tree(self, pt: ParityTreeView, ys: torch.Tensor) -> None:
        """Fill outputs from a preprocessed view."""
        self._native.eval_all_tree(pt._native_view(self._ext), ys)


__all__ = ['GrottoDcf', 'ParityTreeView', 'OwnedParityTree']

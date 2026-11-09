"""Thin binding of ``fss::HalfTreeGrottoDcf`` and its parity-tree members."""

from dataclasses import dataclass, field

import torch

from fss_crypto._scheme import _Scheme, _parallel_depth


@dataclass(frozen=True)
class PackedParityTreeView:
    """Borrow packed int4 leaves and their level-order bool parity tree.

    For ``B = 2**kDepth``, blocks is a contiguous int32 tensor of shape
    ``(B, 4)`` and p is a contiguous bool tensor of shape ``(2*B-1,)``.
    Their combined payload is ``16*B + (2*B-1)`` bytes per party.
    Both parties' trees use twice that payload. Keys, evaluation outputs,
    allocator overhead, and GPU temporary storage are separate.
    For in_bits=8, kDepth=1: blocks has shape (2, 4) and p has shape (3,).
    Root is p[0]. Leaf p[B-1+j] is the parity of block j's 128 output bits.
    Counts and bytes must fit tensor address limits.
    Both tensors retain their storage. Do not mutate it during evaluation.
    """

    blocks: torch.Tensor
    p: torch.Tensor
    _native_views: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def _native_view(self, module):
        native_type = module.NativeParityTreeView
        if native_type not in self._native_views:
            self._native_views[native_type] = native_type(self.blocks, self.p)
        return self._native_views[native_type]


class OwnedPackedParityTree:
    """Host storage allocated by the C++ scheme's ``PreprocessOwned`` member.

    For ``B = 2**kDepth``, the owner holds B int4 entries and 2*B-1 bools.
    Their combined payload is ``16*B + (2*B-1)`` bytes per party.
    The owner object and allocator overhead are separate.
    """

    def __init__(self, *args, **kwargs):
        raise TypeError("use scheme.preprocess_owned() to obtain an owned tree")

    @classmethod
    def _from_native(cls, blocks: torch.Tensor, p: torch.Tensor):
        result = object.__new__(cls)
        result._blocks = blocks
        result._p = p
        return result

    def view(self) -> PackedParityTreeView:
        """Borrow both allocations, retaining their lifetimes through tensors."""
        return PackedParityTreeView(self._blocks, self._p)


class HalfTreeGrottoDcf(_Scheme):
    """Configure packed comparison shares with explicit CCR hash and seed."""

    def __init__(self, in_bits: int, hash, instance_seed, *, in_type=None):
        super().__init__("half_tree_grotto_dcf", in_bits=in_bits, hash=hash,
                         instance_seed=instance_seed, in_type=in_type)

    @classmethod
    def precompile(cls, in_bits: int, hash, *, in_type=None,
                   frontier_depth: int = 16, threads_per_block: int = 256) -> None:
        """Prepare the compiled configuration without creating native state."""
        cls._precompile_config("half_tree_grotto_dcf", in_bits=in_bits, hash=hash, in_type=in_type,
                               frontier_depth=frontier_depth,
                               threads_per_block=threads_per_block)

    @property
    def kDepth(self) -> int:
        return self._native.kDepth

    @property
    def kLanes(self) -> int:
        return self._native.kLanes

    def gen(self, cws, s0s: torch.Tensor, a: int) -> None:
        """Fill correction words with one Core ``Gen`` call, preserving ``s0s``."""
        self._native.gen(self._cws(cws), s0s, a)

    def eval(self, b: bool, s0: torch.Tensor, cws, x: int) -> bool:
        """Return one raw Core share. A CUDA bool result synchronizes its stream."""
        return self._native.eval(b, s0, self._cws(cws), x)

    def eval_all(self, b: bool, s0: torch.Tensor, cws, ys: torch.Tensor,
                 *, par_depth: int = 0, frontier_depth: int = 16,
                 threads_per_block: int = 256) -> None:
        """Fill bool outputs using the leaf-prefix route, matching Core EvalAll.

        Each party's shares can differ from raw scalar ``eval``.

        par_depth controls CPU partitioning; 0 is sequential.
        With OpenMP enabled, positive values use parallel evaluation and require
        a parallel-safe backend. Otherwise evaluation remains sequential.
        CUDA calls ignore par_depth and use frontier_depth/threads_per_block.
        CPU calls ignore frontier_depth/threads_per_block.
        """
        self._execution_native(ys, frontier_depth, threads_per_block).eval_all(b, s0, self._cws(cws), ys, _parallel_depth(par_depth, ys))

    def preprocess(self, b: bool, s0: torch.Tensor, cws, blocks: torch.Tensor,
                   p: torch.Tensor, *, par_depth: int = 0, frontier_depth: int = 16,
                   threads_per_block: int = 256) -> None:
        """Fill caller-owned packed leaves and their parity tree.

        For ``B = 2**kDepth``, allocate contiguous int32 blocks with shape
        ``(B, 4)`` and contiguous bool p with shape ``(2*B-1,)``.
        Their combined payload is ``16*B + (2*B-1)`` bytes per party.
        For in_bits=8, B=2: blocks has shape (2, 4) and p has shape (3,).
        Two parties require two sets of these buffers.
        Keys and evaluation outputs are separate.
        Counts and bytes must fit tensor address limits.

        par_depth controls CPU partitioning; 0 is sequential.
        With OpenMP enabled, positive values use parallel evaluation and require
        a parallel-safe backend. Otherwise evaluation remains sequential.
        CUDA calls ignore par_depth and use frontier_depth/threads_per_block.
        CPU calls ignore frontier_depth/threads_per_block.
        """
        self._execution_native(p, frontier_depth, threads_per_block).preprocess(b, s0, self._cws(cws), blocks, p, _parallel_depth(par_depth, p))

    def preprocess_owned(self, b: bool, s0: torch.Tensor, cws, *, par_depth: int = 0) -> OwnedPackedParityTree:
        """Allocate host leaves and parity storage through C++ ``PreprocessOwned``.

        For ``B = 2**kDepth``, the owned arrays have B int4 entries and
        2*B-1 bool entries, using ``16*B + (2*B-1)`` payload bytes per party.
        The owner object and allocator overhead are separate.

        par_depth controls CPU partitioning; 0 is sequential.
        With OpenMP enabled, positive values use parallel evaluation and require
        a parallel-safe backend. Otherwise evaluation remains sequential.
        """
        return OwnedPackedParityTree._from_native(*self._native.preprocess_owned(b, s0, self._cws(cws), _parallel_depth(par_depth)))

    def eval_tree(self, pt: PackedParityTreeView, x: int) -> bool:
        """Evaluate a preprocessed view without a party or raw key."""
        return self._native.eval_tree(pt._native_view(self._ext), x)

    def eval_all_tree(self, pt: PackedParityTreeView, ys: torch.Tensor, *,
                      threads_per_block: int = 256) -> None:
        """Fill outputs from a preprocessed view. CPU ignores threads_per_block."""
        module = self._execution_module(ys, threads_per_block=threads_per_block)
        state = self._execution_native(ys, threads_per_block=threads_per_block)
        state.eval_all_tree(pt._native_view(module), ys)


__all__ = ['HalfTreeGrottoDcf', 'PackedParityTreeView', 'OwnedPackedParityTree']

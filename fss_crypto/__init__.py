"""Thin PyTorch bindings for the public FSS C++ API.

Scheme operations select CPU or CUDA from their tensor arguments.
All tensor arguments must be on exactly the same device, including the CUDA
index. Operations never copy tensors between devices automatically.
Shared evaluation entry points ignore ``par_depth`` on CUDA and ignore
``frontier_depth`` and ``threads_per_block`` on CPU, where those options exist.
Chain-proof operations require CPU tensors and reject CUDA tensors.
"""

from fss_crypto import group, hash, jit, prg, prp, util
from fss_crypto.dcf import Dcf, DcfPred
from fss_crypto.dmpf import Dmpf
from fss_crypto.dpf import Dpf
from fss_crypto.grotto_dcf import GrottoDcf, OwnedParityTree, ParityTreeView
from fss_crypto.half_tree_dpf import HalfTreeDpf
from fss_crypto.half_tree_grotto_dcf import HalfTreeGrottoDcf, OwnedPackedParityTree, PackedParityTreeView
from fss_crypto.packed_half_tree_dpf import PackedHalfTreeDpf
from fss_crypto.util import input_dtype_and_shape
from fss_crypto.vdpf import Vdpf
from fss_crypto.vdmpf import Vdmpf

__all__ = [
    "Dcf", "DcfPred", "Dmpf", "Dpf", "GrottoDcf", "HalfTreeDpf", "HalfTreeGrottoDcf",
    "PackedHalfTreeDpf", "Vdpf", "Vdmpf", "ParityTreeView",
    "OwnedParityTree", "PackedParityTreeView", "OwnedPackedParityTree",
    "group", "hash", "jit", "prg", "prp", "util", "input_dtype_and_shape",
]

"""Shared construction and native buffer ownership for thin scheme wrappers."""

import torch

from fss_crypto import _jit
from fss_crypto._config import _words
from fss_crypto._primitive import native


class _Scheme:
    def __init__(self, scheme: str, *, in_bits: int, group=None, prg=None,
                 hash=None, xor_hash=None, prp=None, instance_seed=None,
                 in_type=None, out_bits=None,
                 max_points=None, bucket_bits=None, ch_lambda: int = 80,
                 pred: str = "lt"):
        half_tree = scheme in ("half_tree_dpf", "packed_half_tree_dpf", "half_tree_grotto_dcf")
        if half_tree:
            if instance_seed is None:
                raise ValueError("Half-Tree schemes require an explicit instance_seed")
            instance_seed = _words("instance_seed", instance_seed, 4)
        elif instance_seed is not None:
            raise ValueError("this scheme does not take an instance_seed")
        configurations = {"group": group, "prg": prg, "hash": hash,
                          "xor_hash": xor_hash, "prp": prp}
        self._load_config = dict(configurations, in_type=in_type, out_bits=out_bits,
                                 max_points=max_points, bucket_bits=bucket_bits,
                                 ch_lambda=ch_lambda, pred=pred)
        self._ext = _jit.load(scheme, in_bits, **self._load_config)
        backends = {name: native(value)._backend() for name, value in configurations.items()
                    if value is not None}
        self._backend_handles = backends
        self._instance_seed = instance_seed
        self._native = self._ext.NativeScheme(backends, instance_seed)
        self._gpu_specializations = {(16, 256): self._native}
        self._gpu_modules = {(16, 256): self._ext}
        self._scheme = scheme
        self._in_bits = in_bits
        self._in_type = in_type or next(name for name, bits in
            (("uint8", 8), ("uint16", 16), ("uint32", 32), ("uint64", 64), ("uint128", 128))
            if bits >= in_bits)

    def _execution_native(self, output, frontier_depth=16, threads_per_block=256):
        """Select launch code without reconstructing the configured backends."""
        if not output.is_cuda:
            return self._native
        _jit._check_gpu_launch(frontier_depth, threads_per_block)
        key = (frontier_depth, threads_per_block)
        if key not in self._gpu_specializations:
            module = _jit.load(self._scheme, self._in_bits, **self._load_config,
                               frontier_depth=frontier_depth,
                               threads_per_block=threads_per_block)
            self._gpu_modules[key] = module
            self._gpu_specializations[key] = module.NativeScheme(
                self._backend_handles, self._instance_seed)
        return self._gpu_specializations[key]

    def _execution_module(self, output, frontier_depth=16, threads_per_block=256):
        if not output.is_cuda:
            return self._ext
        self._execution_native(output, frontier_depth, threads_per_block)
        return self._gpu_modules[(frontier_depth, threads_per_block)]

    @property
    def in_bits(self) -> int:
        """Return the compiled input-domain bit width."""
        return self._in_bits

    @property
    def in_type(self) -> str:
        """Return the concrete unsigned C++ input transport type."""
        return self._in_type

    @property
    def cws_shape(self) -> tuple[int, ...]:
        """Shape for caller-allocated contiguous uint8 C++ correction records.

        Allocate with ``torch.empty(scheme.cws_shape, dtype=torch.uint8,
        device=device)``. Each final-axis row contains one complete Cw,
        including padding. Multi-point schemes include a leading bucket axis.
        Contents are uninitialized until a successful ``gen`` call.
        """
        shape = (self._native.cw_count, self._native.cw_size)
        if self._scheme in ("dmpf", "vdmpf"):
            shape = (self._native.cw_rows, *shape)
        return shape

    def _cws(self, value) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError("cws must be a tensor")
        if tuple(value.shape) != self.cws_shape or value.dtype != torch.uint8:
            raise ValueError("cws has the wrong native correction-word shape or dtype")
        if not value.is_contiguous():
            raise ValueError("cws must be contiguous")
        if value.data_ptr() % self._native.cw_alignment:
            raise ValueError("cws storage is not aligned for the native type")
        return value


def _parallel_depth(value: int, output=None) -> int:
    """Validate a per-call CPU partition depth without selecting it automatically."""
    if output is not None and output.is_cuda:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < (1 << 31):
        raise ValueError("par_depth must be a nonnegative C++ int")
    return value

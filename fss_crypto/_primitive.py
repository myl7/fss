"""Private ownership and validation shared by primitive wrappers."""

import threading

import torch

from . import _jit

_NATIVE_LOCK = threading.RLock()


def native(config, *, locations=None, expected_abi=None):
    """Retain one binding per current ABI while old owners remain valid."""
    with _NATIVE_LOCK:
        module = _jit.load_primitive(config, _locations=locations)
        abi = module._fss_native_abi
        if expected_abi is not None and abi != expected_abi:
            raise RuntimeError("native ABI changed during construction; create a new scheme")
        binding = getattr(config, "_primitive_binding", None)
        if binding is None or binding[0] != abi:
            dependencies = list(_jit._primitive_dependencies(config))
            values = ({"aes_mmo": native(dependencies[0], locations=locations,
                                         expected_abi=abi)._backend()}
                      if dependencies else config._runtime())
            handle = module.NativePrimitive(values)
            object.__setattr__(config, "_primitive_binding", (abi, handle))
        return config._primitive_binding[1]


class _PrecompileOperations:
    def precompile(self) -> None:
        """Prepare compiled plugins without constructing runtime material."""
        _jit.precompile_primitive(self)


def blocks(value, shape):
    if not isinstance(value, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    if value.dtype != torch.int32 or tuple(value.shape) != tuple(shape):
        raise ValueError(f"input must be an int32 tensor of shape {tuple(shape)}")
    if not value.is_contiguous():
        raise ValueError("input must be contiguous")
    return value


class GroupOperations(_PrecompileOperations):
    def zero(self, *, device="cpu"):
        """Return the additive identity on the requested device."""
        prototype = torch.empty(4, dtype=torch.int32, device=device)
        return Group._create(self, native(self).zero(prototype))

    def from_(self, buf):
        """Convert a clamped int4 block through the C++ Group::From operation."""
        return Group._create(self, native(self).from_(blocks(buf, (4,))))


class Group:
    """One group element; construction is through a configured group's methods."""

    def __init__(self):
        raise TypeError("construct group elements through zero() or from_()")

    @classmethod
    def _create(cls, group, value):
        result = object.__new__(cls)
        result._group = group
        result._native = value
        return result

    def into(self):
        """Return a new clamped int4 representation."""
        return self._native.into()

    def value(self):
        """Return the C++ convenience Value: an integer or a Bytes block."""
        return self._native.value()

    def __int__(self):
        """Return a Uint value as a Python int, synchronizing CUDA values."""
        if self._group._kind != "uint":
            raise TypeError("int conversion requires a Uint group element")
        return self._native.value()

    def __bytes__(self):
        """Return a Bytes value as 16 little-endian bytes, synchronizing CUDA values."""
        if self._group._kind != "bytes":
            raise TypeError("bytes conversion requires a Bytes group element")
        words = self._native.value().cpu().tolist()
        return b"".join((word & 0xffffffff).to_bytes(4, "little") for word in words)

    def __add__(self, rhs):
        if not isinstance(rhs, Group) or self._group != rhs._group:
            raise TypeError("group addition requires the same configured group")
        return self._create(self._group, self._native.add(rhs._native))

    def __neg__(self):
        return self._create(self._group, self._native.negate())


class PrgOperations(_PrecompileOperations):
    def gen(self, seed):
        """Expand one int4 seed into mul int4 blocks."""
        return native(self).gen(blocks(seed, (4,)))


class HashOperations(_PrecompileOperations):
    def hash(self, msg):
        """Call the C++ block, four-block, or two-block tuple overload."""
        if self._kind == "aes128_ccr":
            blocks(msg, (4,))
        elif isinstance(msg, torch.Tensor) and tuple(msg.shape) in ((2, 4), (4, 4)):
            blocks(msg, msg.shape)
        else:
            raise ValueError("hash input must have shape (2, 4) or (4, 4)")
        return native(self).hash(msg)


class PrpOperations(_PrecompileOperations):
    def permu(self, seed, x: int, domain_size: int) -> int:
        """Permute x within [0, domain_size), using the explicit int4 seed."""
        return native(self).permu(blocks(seed, (4,)), x, domain_size)

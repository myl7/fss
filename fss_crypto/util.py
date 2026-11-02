"""Tensor storage helpers for FSS inputs and int4 seed control bits."""

import torch as _torch


def input_dtype_and_shape(in_type: str) -> tuple[_torch.dtype, tuple[int, ...]]:
    """Return the tensor dtype and shape of one unsigned input value.

    ``in_type`` accepts the same names as scheme constructors: ``uint8``,
    ``uint16``, ``uint32``, ``uint64``, and ``uint128``.
    The first four use their matching unsigned dtype and scalar shape ``()``.
    ``uint128`` uses ``torch.uint64`` with item shape ``(2,)``, storing the
    low 64-bit limb first and the high 64-bit limb second.
    ``item_shape`` describes one value, excluding the batch count.

    Allocate caller-owned storage with::

        dtype, item_shape = input_dtype_and_shape(in_type)
        inputs = torch.empty((count, *item_shape), dtype=dtype, device=device)

    This function allocates no tensors and converts no values.
    Unknown names raise ``ValueError``.
    """
    formats = {
        "uint8": (_torch.uint8, ()),
        "uint16": (_torch.uint16, ()),
        "uint32": (_torch.uint32, ()),
        "uint64": (_torch.uint64, ()),
        "uint128": (_torch.uint64, (2,)),
    }
    try:
        return formats[in_type]
    except KeyError:
        raise ValueError("in_type must be uint8, uint16, uint32, uint64, or uint128") from None


def _check_seed(seed):
    if not isinstance(seed, _torch.Tensor):
        raise TypeError("seed must be a torch.Tensor")
    if seed.dtype != _torch.int32 or tuple(seed.shape) != (4,):
        raise ValueError("seed must be an int32 tensor of shape (4,)")
    if not seed.is_contiguous():
        raise ValueError("seed must be contiguous")


def set_seed_control_bit(seed, bit: bool):
    """Return a copy with bit 0 of the seed's fourth 32-bit word set to bit.

    seed must be a contiguous int32 tensor of shape (4,), representing one
    128-bit FSS seed. The result stays on the seed's device.
    The input is not modified.
    """
    _check_seed(seed)
    if not isinstance(bit, bool):
        raise TypeError("bit must be bool")
    result = seed.clone()
    result[3] = (result[3] & -2) | int(bit)
    return result


def get_seed_control_bit(seed) -> bool:
    """Return bit 0 of the seed's fourth 32-bit word as a Python bool.

    seed must be a contiguous int32 tensor of shape (4,), representing one
    128-bit FSS seed. Reading a CUDA result synchronizes its current stream.
    """
    _check_seed(seed)
    return bool((seed[3] & 1).item())


__all__ = ["input_dtype_and_shape", "set_seed_control_bit", "get_seed_control_bit"]

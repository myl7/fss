"""Tensor helpers for the control bit in an FSS int4 seed."""

import torch as _torch


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


__all__ = ["set_seed_control_bit", "get_seed_control_bit"]

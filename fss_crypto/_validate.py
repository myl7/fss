"""Shared validation for FSS binding parameters and tensors."""

import math
from numbers import Integral

import torch

_VALID_GROUPS = ("bytes", "uint")
_VALID_PRGS = ("chacha", "aes128_mmo")
_VALID_PRGS_BY_SCHEME = {
    "dpf": _VALID_PRGS,
    "dcf": _VALID_PRGS,
    "half_tree_dpf": _VALID_PRGS,
    "packed_half_tree_dpf": _VALID_PRGS,
    "grotto_dcf": _VALID_PRGS,
    "vdpf": _VALID_PRGS,
    "dmpf": _VALID_PRGS,
    "vdmpf": _VALID_PRGS,
}
_VALID_PREDS = ("lt", "gt")
_VALID_OUT_BITS = (1, 2, 4, 8, 16, 32, 64)
_VALID_HASHES = ("blake3", "sha256")
_VALID_PRPS = ("aes128_feistel",)


def validate_in_bits(in_bits: int) -> None:
    if not (1 <= in_bits <= 128):
        raise ValueError(f"in_bits must be between 1 and 128, got {in_bits}")


def validate_group(group: str) -> None:
    if group not in _VALID_GROUPS:
        raise ValueError(f"group must be one of {_VALID_GROUPS}, got {group!r}")


def validate_out_bits(out_bits: int) -> None:
    if out_bits not in _VALID_OUT_BITS:
        raise ValueError(f"out_bits must be one of {_VALID_OUT_BITS}, got {out_bits}")


def validate_hash(hash: str) -> None:
    if hash not in _VALID_HASHES:
        raise ValueError(f"hash must be one of {_VALID_HASHES}, got {hash!r}")


def validate_prp(prp: str) -> None:
    if prp not in _VALID_PRPS:
        raise ValueError(f"prp must be one of {_VALID_PRPS}, got {prp!r}")


def validate_retry(retry: bool) -> None:
    if not isinstance(retry, bool):
        raise TypeError(f"retry must be a bool, got {type(retry).__name__}")


def validate_max_points(max_points: int) -> None:
    if max_points < 30:
        raise ValueError(f"max_points must be at least 30, got {max_points}")


def validate_t(t: int, max_points: int) -> None:
    if not (30 <= t <= max_points):
        raise ValueError(f"t must be between 30 and max_points={max_points}, got {t}")


def _ch_bucket(t: int, ch_lambda: int = 80) -> int:
    """Mirror fss::cuckoo_hash::ChBucket."""
    e = (ch_lambda + 130 + math.log2(t)) / 123.5
    return math.ceil(t * e)


def validate_bucket_bits(in_bits: int, max_points: int, bucket_bits: int) -> None:
    """Check the bucket domain fits: with the smallest bucket count (t = 30)
    the runtime bucket size must fit 2 ** bucket_bits, as the C++ assert does."""
    if not (1 <= bucket_bits <= 64):
        raise ValueError(f"bucket_bits must be between 1 and 64, got {bucket_bits}")
    b_size = (2 ** in_bits * 3 + _ch_bucket(30) - 1) // _ch_bucket(30)
    if b_size > 2 ** bucket_bits:
        raise ValueError(
            f"bucket_bits={bucket_bits} is too small: the worst-case bucket "
            f"size {b_size} exceeds 2 ** bucket_bits"
        )


def validate_prg(prg: str, scheme: str) -> None:
    valid_prgs = _VALID_PRGS_BY_SCHEME.get(scheme)
    if valid_prgs is None:
        raise ValueError(
            f"scheme must be one of {tuple(_VALID_PRGS_BY_SCHEME)}, "
            f"got {scheme!r}"
        )
    if prg not in valid_prgs:
        raise ValueError(f"prg must be one of {valid_prgs}, got {prg!r}")


def validate_pred(pred: str) -> None:
    if pred not in _VALID_PREDS:
        raise ValueError(f"pred must be one of {_VALID_PREDS}, got {pred!r}")


def validate_party(party: int) -> None:
    if party not in (0, 1):
        raise ValueError(f"party must be 0 or 1, got {party}")


def validate_s0(s0: torch.Tensor) -> None:
    if s0.shape != (4,) or s0.dtype != torch.int32:
        raise TypeError(
            f"s0 must be a (4,) int32 tensor, "
            f"got shape {tuple(s0.shape)} dtype {s0.dtype}"
        )


def validate_s0s(s0s: torch.Tensor) -> None:
    if s0s.shape != (2, 4) or s0s.dtype != torch.int32:
        raise TypeError(
            f"s0s must be a (2, 4) int32 tensor, "
            f"got shape {tuple(s0s.shape)} dtype {s0s.dtype}"
        )


def validate_beta(beta: torch.Tensor) -> None:
    if beta.shape != (4,) or beta.dtype != torch.int32:
        raise TypeError(
            f"beta must be a (4,) int32 tensor, "
            f"got shape {tuple(beta.shape)} dtype {beta.dtype}"
        )


def validate_cws(cws: torch.Tensor, in_bits: int) -> None:
    expected = (in_bits + 1, 8)
    if cws.shape != expected or cws.dtype != torch.int32:
        raise TypeError(
            f"cws must be a {expected} int32 tensor, "
            f"got shape {tuple(cws.shape)} dtype {cws.dtype}"
        )


def validate_domain_value(name: str, value: int, in_bits: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 0 or value >= (1 << in_bits):
        raise ValueError(f"{name} must be in [0, 2^{in_bits}), got {value}")


def validate_alpha(alpha: int, in_bits: int) -> None:
    validate_domain_value("alpha", alpha, in_bits)


def validate_device_match(*tensors: torch.Tensor) -> None:
    devices = {t.device for t in tensors}
    if len(devices) > 1:
        dev_list = ", ".join(str(d) for d in sorted(devices, key=str))
        raise RuntimeError(
            f"expected all tensors to be on the same device, "
            f"but found at least two devices, {dev_list}!"
        )


def validate_cpu_only(*tensors: torch.Tensor, fn_name: str = "") -> None:
    for t in tensors:
        if t.device.type != "cpu":
            prefix = f"{fn_name} expects" if fn_name else "expected"
            raise RuntimeError(
                f"{prefix} all tensors to be on cpu, "
                f"but found tensor on {t.device}"
            )

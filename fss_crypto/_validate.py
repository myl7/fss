"""Shared validation for FSS binding parameters and tensors."""

from numbers import Integral

import torch

_VALID_GROUPS = ("bytes", "uint")
_VALID_PRGS = ("chacha", "aes128_mmo")
_VALID_PRGS_BY_SCHEME = {
    "dpf": _VALID_PRGS,
    "dcf": _VALID_PRGS,
}
_VALID_PREDS = ("lt", "gt")


def validate_in_bits(in_bits: int) -> None:
    if not (1 <= in_bits <= 128):
        raise ValueError(f"in_bits must be between 1 and 128, got {in_bits}")


def validate_group(group: str) -> None:
    if group not in _VALID_GROUPS:
        raise ValueError(f"group must be one of {_VALID_GROUPS}, got {group!r}")


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

"""Caller-owned input storage without native compilation."""

from types import SimpleNamespace

import pytest
import torch

import fss_crypto
from fss_crypto import _jit, _scheme, group, prg


@pytest.mark.parametrize("name,dtype,item_shape", [
    ("uint8", torch.uint8, ()),
    ("uint16", torch.uint16, ()),
    ("uint32", torch.uint32, ()),
    ("uint64", torch.uint64, ()),
    ("uint128", torch.uint64, (2,)),
])
def test_input_storage_allocation_and_constructor_names(monkeypatch, name, dtype, item_shape):
    # Replace native loading while preserving constructor configuration validation.
    extension = SimpleNamespace(_fss_native_abi="test", NativeScheme=lambda *args: object())
    monkeypatch.setattr(_jit, "_load_generated", lambda *args, **kwargs: extension)
    monkeypatch.setattr(_jit, "_assert_native_abi", lambda *args: None)
    monkeypatch.setattr(_scheme, "native", lambda *args, **kwargs:
                        SimpleNamespace(_backend=lambda: object()))
    scheme = fss_crypto.Dpf(8, group.Bytes(), prg.ChaCha(nonce=(1, 2), mul=2),
                            in_type=name)
    actual_dtype, actual_shape = fss_crypto.input_dtype_and_shape(scheme.in_type)
    assert (actual_dtype, actual_shape) == (dtype, item_shape)
    inputs = torch.empty((3, *actual_shape), dtype=actual_dtype, device="cpu")
    assert inputs.dtype == dtype
    assert tuple(inputs.shape) == (3, *item_shape)
    assert inputs.is_contiguous()


def test_uint128_low_high_roundtrip():
    values = [0, (1 << 64) - 1, 1 << 64, (1 << 127) + 17, (1 << 128) - 1]
    dtype, item_shape = fss_crypto.input_dtype_and_shape("uint128")
    inputs = torch.empty((len(values), *item_shape), dtype=dtype, device="cpu")
    limbs = [[value & ((1 << 64) - 1), value >> 64] for value in values]
    inputs.copy_(torch.tensor(limbs, dtype=dtype))
    assert [low | (high << 64) for low, high in inputs.tolist()] == values
    assert inputs.tolist()[3] == [17, 1 << 63]


@pytest.mark.parametrize("name", ["", "int32", "uint256", "uint32_t", "__uint128_t"])
def test_unknown_input_name(name):
    with pytest.raises(ValueError):
        fss_crypto.input_dtype_and_shape(name)

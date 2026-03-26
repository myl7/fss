"""JIT compilation manager for FSS C++ extensions."""

import os
import warnings
from pathlib import Path

import torch
from torch.utils.cpp_extension import load as _cpp_load

# Check for CUDA toolkit at import time.
try:
    from torch.utils.cpp_extension import _find_cuda_home

    _cuda_home = _find_cuda_home()
    if _cuda_home is None:
        warnings.warn(
            "CUDA toolkit not found. JIT compilation of FSS extensions "
            "will fail. Install the CUDA toolkit or set CUDA_HOME.",
            stacklevel=2,
        )
except Exception:
    warnings.warn(
        "Could not check for CUDA toolkit. JIT compilation may fail.",
        stacklevel=2,
    )

_EXT_CACHE: dict[str, object] = {}

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_PACKAGE_DIR = Path(__file__).resolve().parent
_CSRC_DIR = _PACKAGE_DIR / "_csrc"

# The include/ dir lives next to fss_crypto/ in the repo layout.
_INCLUDE_DIR = _PACKAGE_DIR.parent / "include"


def _find_include_dir() -> str:
    if _INCLUDE_DIR.is_dir():
        return str(_INCLUDE_DIR)
    raise RuntimeError(
        f"cannot find FSS include/ directory (tried {_INCLUDE_DIR})"
    )


# ---------------------------------------------------------------------------
# Type / include mappings
# ---------------------------------------------------------------------------

def _in_type(in_bits: int) -> str:
    if in_bits <= 32:
        return "uint32_t"
    if in_bits <= 64:
        return "uint64_t"
    return "__uint128_t"


_GROUP_MAP: dict[str, dict] = {
    "bytes": {
        "type": "fss::group::Bytes",
        "include": "fss/group/bytes.cuh",
    },
    "uint": {
        "include": "fss/group/uint.cuh",
    },
}


def _group_type(group: str, in_bits: int) -> str:
    if group == "bytes":
        return "fss::group::Bytes"
    # group == "uint"
    if in_bits <= 32:
        return "fss::group::Uint<uint32_t>"
    if in_bits <= 64:
        return "fss::group::Uint<uint64_t>"
    return (
        "fss::group::Uint<__uint128_t, "
        "static_cast<__uint128_t>(1) << 127>"
    )


_PRG_MAP: dict[str, dict] = {
    "chacha": {
        "include": "fss/prg/chacha.cuh",
    },
    "aes128_mmo": {
        "include": "fss/prg/aes128_mmo.cuh",
    },
}

# DPF uses mul=2, DCF uses mul=4 for their PRGs.
_SCHEME_PRG_MUL: dict[str, int] = {
    "dpf": 2,
    "dcf": 4,
}


def _prg_type(prg: str, scheme: str) -> str:
    mul = _SCHEME_PRG_MUL[scheme]
    if prg == "chacha":
        return f"fss::prg::ChaCha<{mul}>"
    return f"fss::prg::Aes128Mmo<{mul}>"


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

def _cache_key(scheme: str, in_bits: int, group: str, prg: str,
               pred: str | None = None) -> str:
    parts = [scheme, str(in_bits), group, prg, _in_type(in_bits)]
    if pred is not None:
        parts.append(pred)
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Source generation
# ---------------------------------------------------------------------------

_DPF_TEMPLATE = """\
#include <fss/dpf.cuh>
#include <{group_include}>
#include <{prg_include}>

constexpr int kInBits = {in_bits};
using InType = {in_type};
using GroupType = {group_type};
using PrgType = {prg_type};
using DpfInst = fss::Dpf<kInBits, GroupType, PrgType, InType>;

#include "dpf_binding_impl.cuh"
"""


def _generate_dpf_source(in_bits: int, group: str, prg: str) -> str:
    return _DPF_TEMPLATE.format(
        group_include=_GROUP_MAP[group]["include"],
        prg_include=_PRG_MAP[prg]["include"],
        in_bits=in_bits,
        in_type=_in_type(in_bits),
        group_type=_group_type(group, in_bits),
        prg_type=_prg_type(prg, "dpf"),
    )


_DCF_PRED_MAP: dict[str, str] = {
    "lt": "fss::DcfPred::kLt",
    "gt": "fss::DcfPred::kGt",
}

_DCF_TEMPLATE = """\
#include <fss/dcf.cuh>
#include <{group_include}>
#include <{prg_include}>

constexpr int kInBits = {in_bits};
using InType = {in_type};
using GroupType = {group_type};
using PrgType = {prg_type};
constexpr fss::DcfPred kPred = {pred};
using DcfInst = fss::Dcf<kInBits, GroupType, PrgType, InType, kPred>;

#include "dcf_binding_impl.cuh"
"""


def _generate_dcf_source(in_bits: int, group: str, prg: str,
                         pred: str) -> str:
    return _DCF_TEMPLATE.format(
        group_include=_GROUP_MAP[group]["include"],
        prg_include=_PRG_MAP[prg]["include"],
        in_bits=in_bits,
        in_type=_in_type(in_bits),
        group_type=_group_type(group, in_bits),
        prg_type=_prg_type(prg, "dcf"),
        pred=_DCF_PRED_MAP[pred],
    )


_SOURCE_GENERATORS: dict[str, object] = {
    "dpf": _generate_dpf_source,
    "dcf": _generate_dcf_source,
}

# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def load(scheme: str, in_bits: int, group: str, prg: str,
         pred: str | None = None) -> object:
    """Load (or return cached) JIT-compiled extension module."""
    key = _cache_key(scheme, in_bits, group, prg, pred)
    if key in _EXT_CACHE:
        return _EXT_CACHE[key]

    generator = _SOURCE_GENERATORS[scheme]
    if scheme == "dpf":
        source = generator(in_bits, group, prg)
    elif scheme == "dcf":
        source = generator(in_bits, group, prg, pred)
    else:
        raise ValueError(f"unsupported scheme {scheme!r}")

    include_dir = _find_include_dir()
    csrc_dir = str(_CSRC_DIR)

    build_dir = os.path.join(
        os.path.expanduser("~"), ".cache", "fss_crypto", key
    )
    os.makedirs(build_dir, exist_ok=True)

    src_path = os.path.join(build_dir, f"{key}.cu")
    with open(src_path, "w") as f:
        f.write(source)

    extra_cflags = ["-O3", "-std=c++20"]
    extra_cuda_cflags = ["-O3", "-std=c++20", "--extended-lambda"]
    extra_ldflags: list[str] = []

    if prg == "aes128_mmo":
        extra_cflags.append("-fopenmp")
        extra_ldflags.extend(["-lssl", "-lcrypto", "-lgomp"])

    # PyTorch's arch-flag detection crashes when no GPU is present and
    # TORCH_CUDA_ARCH_LIST is unset (it tries arch_list[-1] on an empty list).
    # Set a safe fallback so nvcc has a target even on headless machines.
    _arch_list_env = "TORCH_CUDA_ARCH_LIST"
    if not os.environ.get(_arch_list_env):
        import torch as _torch
        if _torch.cuda.device_count() == 0:
            os.environ[_arch_list_env] = "8.0"

    ext = _cpp_load(
        name=key,
        sources=[src_path],
        extra_include_paths=[include_dir, csrc_dir],
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=extra_ldflags if extra_ldflags else None,
        build_directory=build_dir,
    )

    _EXT_CACHE[key] = ext
    return ext

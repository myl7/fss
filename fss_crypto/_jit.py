"""JIT compilation manager for FSS C++ extensions."""

import hashlib
import importlib.util
import os
import sysconfig
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

# The include/ dir lives next to fss_crypto/ in a checkout. Wheels install the
# same header tree under the environment data prefix.
_CHECKOUT_INCLUDE_DIR = _PACKAGE_DIR.parent / "include"
_INSTALLED_INCLUDE_DIR = Path(sysconfig.get_paths()["data"]) / "include"


def _find_include_dir() -> str:
    for include_dir in (_CHECKOUT_INCLUDE_DIR, _INSTALLED_INCLUDE_DIR):
        if (include_dir / "fss").is_dir():
            return str(include_dir)
    raise RuntimeError(
        f"cannot find FSS include/ directory "
        f"(tried {_CHECKOUT_INCLUDE_DIR} and {_INSTALLED_INCLUDE_DIR})"
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

_HASH_MAP: dict[str, dict] = {
    "blake3": {
        "type": "fss::hash::Blake3",
        "include": "fss/hash/blake3.cuh",
    },
    "sha256": {
        "type": "fss::hash::Sha256",
        "include": "fss/hash/sha256.cuh",
    },
}

_PRP_MAP: dict[str, dict] = {
    "aes128_feistel": {
        "type": "fss::prp::Aes128Feistel",
        "include": "fss/prp/aes128_feistel.cuh",
    },
    "chacha_feistel": {
        "type": "fss::prp::ChaChaFeistel",
        "include": "fss/prp/chacha_feistel.cuh",
    },
}

_DCF_PRED_MAP: dict[str, str] = {
    "lt": "fss::DcfPred::kLt",
    "gt": "fss::DcfPred::kGt",
}

# ---------------------------------------------------------------------------
# Scheme registry
# ---------------------------------------------------------------------------
# One entry per scheme: the PRG multiplier its Prgable requirement fixes,
# whether it is templated on an output group, which extra knobs (beyond
# in_bits/group/prg) it takes, and how the generated extension names and
# constructs the scheme instance.

_SCHEMES: dict[str, dict] = {
    "dpf": {
        "prg_mul": 2,
        "has_group": True,
        "knobs": (),
        "header": "fss/dpf.cuh",
        "inst": "DpfInst",
        "inst_fmt": "fss::Dpf<kInBits, GroupType, PrgType, InType>",
        "impl": "dpf_binding_impl.cuh",
    },
    "dcf": {
        "prg_mul": 4,
        "has_group": True,
        "knobs": ("pred",),
        "header": "fss/dcf.cuh",
        "inst": "DcfInst",
        "inst_fmt": "fss::Dcf<kInBits, GroupType, PrgType, InType, kPred>",
        "impl": "dcf_binding_impl.cuh",
    },
    "half_tree_dpf": {
        "prg_mul": 1,
        "has_group": True,
        "knobs": (),
        "header": "fss/half_tree_dpf.cuh",
        "inst": "HtDpfInst",
        "inst_fmt": "fss::HalfTreeDpf<kInBits, GroupType, PrgType, InType>",
        "impl": "half_tree_dpf_binding_impl.cuh",
    },
    "packed_half_tree_dpf": {
        "prg_mul": 1,
        "has_group": False,
        "knobs": ("out_bits",),
        "header": "fss/packed_half_tree_dpf.cuh",
        "inst": "PackedInst",
        "inst_fmt": "fss::PackedHalfTreeDpf<kInBits, kOutBits, PrgType, InType>",
        "impl": "packed_half_tree_dpf_binding_impl.cuh",
    },
    "grotto_dcf": {
        "prg_mul": 2,
        "has_group": False,
        "knobs": (),
        "header": "fss/grotto_dcf.cuh",
        "inst": "GrottoInst",
        "inst_fmt": "fss::GrottoDcf<kInBits, PrgType, InType>",
        "impl": "grotto_dcf_binding_impl.cuh",
    },
    "vdpf": {
        "prg_mul": 2,
        "has_group": True,
        "knobs": ("xor_hash", "hash"),
        "header": "fss/vdpf.cuh",
        "inst": "VdpfInst",
        "inst_fmt": "fss::Vdpf<kInBits, GroupType, PrgType, XorHashType, HashType, InType>",
        "impl": "vdpf_binding_impl.cuh",
    },
    "dmpf": {
        "prg_mul": 2,
        "has_group": True,
        "knobs": ("max_points", "bucket_bits", "prp"),
        "header": "fss/dmpf.cuh",
        "inst": "DmpfInst",
        "inst_fmt": "fss::Dmpf<kInBits, kMaxPoints, kBucketBits, GroupType, PrgType, PrpType, InType>",
        "impl": "dmpf_binding_impl.cuh",
    },
    "vdmpf": {
        "prg_mul": 2,
        "has_group": True,
        "knobs": ("max_points", "bucket_bits", "xor_hash", "hash", "prp"),
        "header": "fss/vdmpf.cuh",
        "inst": "VdmpfInst",
        "inst_fmt": (
            "fss::Vdmpf<kInBits, kMaxPoints, kBucketBits, GroupType, PrgType, "
            "XorHashType, HashType, PrpType, InType>"
        ),
        "impl": "vdmpf_binding_impl.cuh",
    },
}


def _prg_type(prg: str, scheme: str) -> str:
    mul = _SCHEMES[scheme]["prg_mul"]
    if prg == "chacha":
        return f"fss::prg::ChaCha<{mul}>"
    return f"fss::prg::Aes128Mmo<{mul}>"


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

# Knobs that participate in the cache key, in a fixed order. All values must
# be str()able; None knobs of schemes that do not use them are skipped.
_KNOB_ORDER = ("group", "prg", "pred", "out_bits", "max_points", "bucket_bits", "xor_hash", "hash", "prp")


def _cache_key(scheme: str, in_bits: int, knobs: dict) -> str:
    parts = [scheme, str(in_bits)]
    for knob in _KNOB_ORDER:
        value = knobs.get(knob)
        if value is None:
            continue
        parts.append(f"{knob}{value}")
    parts.append(_in_type(in_bits).replace(" ", ""))
    return "_".join(parts)


_TREE_HASHES: dict[tuple[str, str], str] = {}


def _source_tree_hash(include_dir: str, csrc_dir: str) -> str:
    """Content hash of the header trees the generated sources include.

    Build dirs are keyed by this hash so checkouts with different header
    contents never share binaries: ninja's mtime check alone cannot tell a
    cached .so built from a divergent checkout (for example a parallel
    worktree writing the same cache root) from an up-to-date one.
    """
    cached = _TREE_HASHES.get((include_dir, csrc_dir))
    if cached is not None:
        return cached
    h = hashlib.sha1()
    for label, root in (("include", include_dir), ("csrc", csrc_dir)):
        h.update(label.encode())
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            for name in sorted(filenames):
                path = os.path.join(dirpath, name)
                h.update(os.path.relpath(path, root).encode())
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 16), b""):
                        h.update(chunk)
    digest = h.hexdigest()[:12]
    _TREE_HASHES[(include_dir, csrc_dir)] = digest
    return digest


def _import_built(name: str, path: str) -> object:
    # Mirrors torch's _import_module_from_library.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Source generation
# ---------------------------------------------------------------------------


def _generate_source(scheme: str, in_bits: int, knobs: dict) -> str:
    spec = _SCHEMES[scheme]
    includes = [spec["header"]]
    aliases = [
        f"constexpr int kInBits = {in_bits};",
        f"using InType = {_in_type(in_bits)};",
    ]
    inst = spec["inst_fmt"]

    if spec["has_group"]:
        includes.append(_GROUP_MAP[knobs["group"]]["include"])
        aliases.append(f"using GroupType = {_group_type(knobs['group'], in_bits)};")
    includes.append(_PRG_MAP[knobs["prg"]]["include"])
    aliases.append(f"using PrgType = {_prg_type(knobs['prg'], scheme)};")

    if "pred" in spec["knobs"]:
        aliases.append(f"constexpr fss::DcfPred kPred = {_DCF_PRED_MAP[knobs['pred']]};")
    if "out_bits" in spec["knobs"]:
        aliases.append(f"constexpr int kOutBits = {knobs['out_bits']};")
    if "xor_hash" in spec["knobs"]:
        includes.append(_HASH_MAP[knobs["xor_hash"]]["include"])
        aliases.append(f"using XorHashType = {_HASH_MAP[knobs['xor_hash']]['type']};")
    if "hash" in spec["knobs"]:
        if knobs["hash"] != knobs["xor_hash"]:
            includes.append(_HASH_MAP[knobs["hash"]]["include"])
        aliases.append(f"using HashType = {_HASH_MAP[knobs['hash']]['type']};")
    if "prp" in spec["knobs"]:
        includes.append(_PRP_MAP[knobs["prp"]]["include"])
        aliases.append(f"using PrpType = {_PRP_MAP[knobs['prp']]['type']};")
    if "max_points" in spec["knobs"]:
        aliases.append(f"constexpr int kMaxPoints = {knobs['max_points']};")
    if "bucket_bits" in spec["knobs"]:
        aliases.append(f"constexpr int kBucketBits = {knobs['bucket_bits']};")

    aliases.append(f"using {spec['inst']} = {inst};")

    lines = [f"#include <{path}>" for path in includes]
    lines.append("")
    lines.extend(aliases)
    lines.append("")
    lines.append(f'#include "{spec["impl"]}"')
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def load(scheme: str, in_bits: int, group: str | None = None, prg: str = "chacha",
         pred: str | None = None, *, out_bits: int | None = None,
         max_points: int | None = None, bucket_bits: int | None = None,
         xor_hash: str | None = None, hash: str | None = None,
         prp: str | None = None) -> object:
    """Load (or return cached) JIT-compiled extension module."""
    if scheme not in _SCHEMES:
        raise ValueError(f"unsupported scheme {scheme!r}")

    spec = _SCHEMES[scheme]
    if spec["has_group"] and group is None:
        raise ValueError(f"scheme {scheme!r} requires a group")
    knobs = {"group": group, "prg": prg, "pred": pred, "out_bits": out_bits,
             "max_points": max_points, "bucket_bits": bucket_bits,
             "xor_hash": xor_hash, "hash": hash, "prp": prp}
    for knob in spec["knobs"]:
        if knobs.get(knob) is None:
            if knob not in ("prp",):  # prp defaults below
                raise ValueError(f"scheme {scheme!r} requires {knob}")
    if "prp" in spec["knobs"] and prp is None:
        knobs["prp"] = "aes128_feistel"
    if "hash" in spec["knobs"] and xor_hash is not None and hash is None:
        knobs["hash"] = xor_hash
    # Drop knobs the scheme does not use so they cannot leak into the key.
    knobs = {knob: value for knob, value in knobs.items()
             if value is not None and (knob in ("group", "prg") or knob in spec["knobs"])}

    key = _cache_key(scheme, in_bits, knobs)
    if key in _EXT_CACHE:
        return _EXT_CACHE[key]

    source = _generate_source(scheme, in_bits, knobs)

    include_dir = _find_include_dir()
    csrc_dir = str(_CSRC_DIR)

    build_dir = os.path.join(
        os.path.expanduser("~"), ".cache", "fss_crypto",
        f"{key}-{torch.__version__}-{_source_tree_hash(include_dir, csrc_dir)}",
    )
    os.makedirs(build_dir, exist_ok=True)

    # The dir name carries the source tree hash and the torch version, so an
    # existing .so was built from identical inputs; import it directly.
    # Going through ninja would demand a pointless rebuild after every fresh
    # checkout: its deps log flags any dep mtime change, not only newer ones.
    so_path = os.path.join(build_dir, f"{key}.so")
    if os.path.exists(so_path):
        try:
            _EXT_CACHE[key] = _import_built(key, so_path)
            return _EXT_CACHE[key]
        except Exception:
            pass  # fall through and let ninja rebuild

    src_path = os.path.join(build_dir, f"{key}.cu")
    # Write only when the content changes: an unconditional rewrite bumps the
    # mtime and makes ninja recompile the extension in every fresh process.
    try:
        with open(src_path, "r") as f:
            existing = f.read()
    except FileNotFoundError:
        existing = None
    if existing != source:
        with open(src_path, "w") as f:
            f.write(source)

    extra_cflags = ["-O3", "-std=c++20"]
    extra_cuda_cflags = ["-O3", "-std=c++20", "--extended-lambda"]
    extra_ldflags: list[str] = []

    needs_crypto = (
        prg == "aes128_mmo"
        or knobs.get("prp") == "aes128_feistel"
        or knobs.get("xor_hash") == "sha256"
        or knobs.get("hash") == "sha256"
    )
    if prg == "aes128_mmo":
        extra_cflags.append("-fopenmp")
    if needs_crypto:
        extra_ldflags.extend(["-lssl", "-lcrypto"])
    if prg == "aes128_mmo":
        extra_ldflags.append("-lgomp")

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

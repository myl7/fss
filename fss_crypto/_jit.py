"""JIT compilation manager for FSS C++ extensions."""

import hashlib
import importlib.util
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import threading
import warnings
from functools import lru_cache
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
except (ImportError, OSError, RuntimeError):
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
        "inst_fmt": "fss::Dpf<kInBits, GroupType, PrgType, InType, kParDepth>",
        "impl": "dpf_binding_impl.cuh",
    },
    "dcf": {
        "prg_mul": 4,
        "has_group": True,
        "knobs": ("pred",),
        "header": "fss/dcf.cuh",
        "inst": "DcfInst",
        "inst_fmt": "fss::Dcf<kInBits, GroupType, PrgType, InType, kPred, kParDepth>",
        "impl": "dcf_binding_impl.cuh",
    },
    "half_tree_dpf": {
        "prg_mul": 1,
        "has_group": True,
        "knobs": (),
        "header": "fss/half_tree_dpf.cuh",
        "inst": "HtDpfInst",
        "inst_fmt": "fss::HalfTreeDpf<kInBits, GroupType, PrgType, InType, kParDepth>",
        "impl": "half_tree_dpf_binding_impl.cuh",
    },
    "packed_half_tree_dpf": {
        "prg_mul": 1,
        "has_group": False,
        "knobs": ("out_bits",),
        "header": "fss/packed_half_tree_dpf.cuh",
        "inst": "PackedInst",
        "inst_fmt": "fss::PackedHalfTreeDpf<kInBits, kOutBits, PrgType, InType, kParDepth>",
        "impl": "packed_half_tree_dpf_binding_impl.cuh",
    },
    "grotto_dcf": {
        "prg_mul": 2,
        "has_group": False,
        "knobs": (),
        "header": "fss/grotto_dcf.cuh",
        "inst": "GrottoInst",
        "inst_fmt": "fss::GrottoDcf<kInBits, PrgType, InType, kParDepth>",
        "impl": "grotto_dcf_binding_impl.cuh",
    },
    "vdpf": {
        "prg_mul": 2,
        "has_group": True,
        "knobs": ("xor_hash", "hash"),
        "header": "fss/vdpf.cuh",
        "inst": "VdpfInst",
        "inst_fmt": "fss::Vdpf<kInBits, GroupType, PrgType, XorHashType, HashType, InType, kParDepth>",
        "impl": "vdpf_binding_impl.cuh",
    },
    "dmpf": {
        "prg_mul": 2,
        "has_group": True,
        "knobs": ("max_points", "bucket_bits", "prp"),
        "header": "fss/dmpf.cuh",
        "inst": "DmpfInst",
        "inst_fmt": "fss::Dmpf<kInBits, kMaxPoints, kBucketBits, GroupType, PrgType, PrpType, InType, 3, 80, kParDepth>",
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
    return f"fss::python::ThreadLocalAes128Mmo<{mul}>"


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

# Knobs that participate in the cache key, in a fixed order. All values must
# be str()able; None knobs of schemes that do not use them are skipped.
_KNOB_ORDER = ("group", "prg", "pred", "out_bits", "max_points", "bucket_bits", "xor_hash", "hash", "prp", "par_depth")


def _cache_key(scheme: str, in_bits: int, knobs: dict) -> str:
    parts = [scheme, str(in_bits)]
    for knob in _KNOB_ORDER:
        value = knobs.get(knob)
        if value is None:
            continue
        parts.append(f"{knob}{str(value).replace('-', 'minus')}")
    parts.append(_in_type(in_bits).replace(" ", ""))
    return "_".join(parts)


def _source_tree_hash(include_dir: str, csrc_dir: str) -> str:
    """Hash included files on every request, including edits in this process."""
    h = hashlib.sha256()
    for label, root in (("include", include_dir), ("csrc", csrc_dir)):
        h.update(label.encode())
        for path in sorted(Path(root).rglob("*")):
            if not path.is_file() or path.suffix not in (".cuh", ".h", ".hpp"):
                continue
            h.update(path.relative_to(root).as_posix().encode())
            h.update(path.read_bytes())
    return h.hexdigest()


# Resolve these flags without assigning TORCH_CUDA_ARCH_LIST. Passing the
# explicit flags also prevents PyTorch from repeating its environment lookup.
_NAMED_ARCHES = {
    "Kepler+Tesla": "3.7", "Kepler": "3.5+PTX", "Maxwell+Tegra": "5.3",
    "Maxwell": "5.0;5.2+PTX", "Pascal": "6.0;6.1+PTX", "Volta+Tegra": "7.2",
    "Volta": "7.0+PTX", "Turing": "7.5+PTX", "Ampere+Tegra": "8.7",
    "Ampere": "8.0;8.6+PTX", "Ada": "8.9+PTX", "Hopper": "9.0+PTX",
    "Blackwell+Tegra": "11.0", "Blackwell": "10.0;12.0+PTX",
}


def _cuda_arch_flags() -> list[str]:
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    if arch_list:
        for name, arches in _NAMED_ARCHES.items():
            arch_list = arch_list.replace(name, arches)
        arches = arch_list.replace(" ", ";").split(";")
    else:
        arches = []
        supported = [int(a.split("_")[1]) for a in torch.cuda.get_arch_list()
                     if re.fullmatch(r"sm_[0-9]+", a)]
        for device in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(device)
            capability = min(10 * major + minor, max(supported)) if supported else 10 * major + minor
            arches.append(f"{capability // 10}.{capability % 10}")
        arches = sorted(set(arches), key=lambda a: tuple(map(int, a.split("."))))
        if arches:
            arches[-1] += "+PTX"
        else:
            arches = ["8.0"]
    flags = set()
    for arch in filter(None, arches):
        match = re.fullmatch(r"([0-9]+)\.([0-9]+[a-z]?)(\+PTX)?", arch)
        if not match:
            raise ValueError(f"unsupported CUDA architecture {arch!r}")
        target = match[1] + match[2]
        flags.add(f"-gencode=arch=compute_{target},code=sm_{target}")
        if match[3]:
            flags.add(f"-gencode=arch=compute_{target},code=compute_{target}")
    if not flags:
        raise ValueError("TORCH_CUDA_ARCH_LIST must contain a CUDA architecture")
    return sorted(flags)


@lru_cache(maxsize=32)
def _compiler_version(command: tuple[str, ...], stamp: tuple | None) -> str:
    try:
        result = subprocess.run([*command, "--version"], capture_output=True,
                                text=True, timeout=10, check=False)
        return result.stdout.strip() + result.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {type(exc).__name__}"


def _compiler_identity(command: str) -> dict:
    argv = shlex.split(command)
    resolved = shutil.which(argv[0]) or argv[0]
    try:
        stat = Path(resolved).stat()
        stamp = (stat.st_size, stat.st_mtime_ns)
    except OSError:
        stamp = None
    return {"command": [resolved, *argv[1:]], "stamp": stamp,
            "version": _compiler_version((resolved, *argv[1:]), stamp)}


@lru_cache(maxsize=32)
def _require_openmp(command: tuple[str, ...], version: str) -> None:
    try:
        probe = subprocess.run(
            [*command, "-fopenmp", "-x", "c++", "-o", os.devnull, "-"],
            input="#include <omp.h>\nint main() { return omp_get_max_threads() == 0; }\n",
            text=True, capture_output=True, timeout=20, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("par_depth requires an available OpenMP host compiler") from exc
    if probe.returncode:
        raise RuntimeError(
            "par_depth requires OpenMP compiler and linker support: " + probe.stderr.strip()[-500:]
        )


def _build_options(knobs: dict) -> tuple[list[str], list[str], list[str]]:
    cflags = ["-O3", "-std=c++20"]
    cuda_flags = ["-O3", "-std=c++20", "--extended-lambda", *_cuda_arch_flags()]
    ldflags = []
    if knobs.get("par_depth", 0) != 0:
        compiler = _compiler_identity(os.environ.get("CXX", "c++"))
        _require_openmp(tuple(compiler["command"]), compiler["version"])
        if os.environ.get("CC"):
            host_compiler = _compiler_identity(os.environ["CC"])
            _require_openmp(tuple(host_compiler["command"]), host_compiler["version"])
        cflags.append("-fopenmp")
        cuda_flags.append("-Xcompiler=-fopenmp")
        ldflags.append("-fopenmp")
    if (knobs["prg"] == "aes128_mmo" or knobs.get("prp") == "aes128_feistel"
            or "sha256" in (knobs.get("xor_hash"), knobs.get("hash"))):
        ldflags.extend(["-lssl", "-lcrypto"])
    return cflags, cuda_flags, ldflags


def _build_manifest(source: str, include_dir: str, csrc_dir: str,
                    options: tuple[list[str], list[str], list[str]]) -> dict:
    from torch.utils import cpp_extension

    cuda_home = cpp_extension.CUDA_HOME or os.environ.get("CUDA_HOME")
    nvcc = os.environ.get("PYTORCH_NVCC") or (str(Path(cuda_home) / "bin/nvcc") if cuda_home else "nvcc")
    return {
        "schema": 1, "source": source,
        "headers": _source_tree_hash(include_dir, csrc_dir),
        "python": {"version": sys.version, "soabi": sysconfig.get_config_var("SOABI"),
                   "platform": platform.platform(), "executable": sys.executable},
        "torch": {"version": str(torch.__version__), "cuda": torch.version.cuda,
                  "cxx11_abi": getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", None),
                  "pybind_abi": [getattr(torch._C, name, None) for name in
                                 ("_PYBIND11_COMPILER_TYPE", "_PYBIND11_STDLIB", "_PYBIND11_BUILD_ABI")]},
        "compilers": {"cxx": _compiler_identity(os.environ.get("CXX", "c++")),
                      "nvcc": _compiler_identity(nvcc),
                      "cc": _compiler_identity(os.environ.get("CC", "cc"))},
        "environment": {key: os.environ.get(key) for key in
                        ("CUDAHOSTCXX", "NVCC_CCBIN", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS")},
        "options": options,
    }


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

    aliases.append(f"constexpr int kParDepth = {knobs.get('par_depth', 0)};")
    aliases.append(f"constexpr bool kGpuPrg = {str(knobs['prg'] == 'chacha').lower()};")
    aliases.append(f"constexpr bool kGpuXorHash = {str(knobs.get('xor_hash', 'blake3') == 'blake3').lower()};")
    aliases.append(f"constexpr bool kGpuHash = {str(knobs.get('hash', 'blake3') == 'blake3').lower()};")
    aliases.append(f"using {spec['inst']} = {inst};")

    lines = [f"#include <{path}>" for path in includes]
    if knobs["prg"] == "aes128_mmo":
        lines.append('#include "binding_prg.cuh"')
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
         prp: str | None = None, par_depth: int = 0) -> object:
    """Load (or return cached) JIT-compiled extension module."""
    if scheme not in _SCHEMES:
        raise ValueError(f"unsupported scheme {scheme!r}")

    if isinstance(par_depth, bool) or not isinstance(par_depth, int) or par_depth < -1:
        raise ValueError("par_depth must be an integer >= -1")
    # Larger depths resolve to the same clamped partition count in C++.
    par_depth = min(par_depth, (1 << 31) - 1)
    if scheme == "vdmpf" and par_depth != 0:
        raise ValueError("vdmpf does not expose par_depth")
    spec = _SCHEMES[scheme]
    if spec["has_group"] and group is None:
        raise ValueError(f"scheme {scheme!r} requires a group")
    knobs = {"group": group, "prg": prg, "pred": pred, "out_bits": out_bits,
             "max_points": max_points, "bucket_bits": bucket_bits,
             "xor_hash": xor_hash, "hash": hash, "prp": prp,
             "par_depth": par_depth}
    if "hash" in spec["knobs"] and xor_hash is not None and hash is None:
        knobs["hash"] = xor_hash
    for knob in spec["knobs"]:
        if knobs.get(knob) is None and knob != "prp":
            raise ValueError(f"scheme {scheme!r} requires {knob}")
    if "prp" in spec["knobs"] and prp is None:
        knobs["prp"] = "aes128_feistel"
    # Drop knobs the scheme does not use so they cannot leak into the key.
    knobs = {knob: value for knob, value in knobs.items()
             if value is not None and (knob in ("group", "prg", "par_depth") or knob in spec["knobs"])}

    source = _generate_source(scheme, in_bits, knobs)
    include_dir = _find_include_dir()
    csrc_dir = str(_CSRC_DIR)
    options = _build_options(knobs)
    manifest = _build_manifest(source, include_dir, csrc_dir, options)
    manifest_text = json.dumps(manifest, sort_keys=True, indent=2)
    fingerprint = hashlib.sha256(manifest_text.encode()).hexdigest()[:20]
    key = f"{_cache_key(scheme, in_bits, knobs)}_{fingerprint}"
    if key in _EXT_CACHE:
        return _EXT_CACHE[key]

    build_dir = Path.home() / ".cache" / "fss_crypto" / key
    build_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = build_dir / "build_inputs.json"
    so_path = build_dir / f"{key}.so"
    if so_path.exists() and manifest_path.exists() and manifest_path.read_text() == manifest_text:
        try:
            ext = _import_built(key, str(so_path))
            _EXT_CACHE[key] = ext
            return ext
        except (ImportError, OSError):
            pass

    src_path = build_dir / f"{key}.cu"
    if not src_path.exists() or src_path.read_text() != source:
        src_path.write_text(source)
    cflags, cuda_flags, ldflags = options
    ext = _cpp_load(
        name=key,
        sources=[str(src_path)],
        extra_include_paths=[include_dir, csrc_dir],
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_flags,
        extra_ldflags=ldflags or None,
        build_directory=str(build_dir),
    )
    # Publish the manifest only after a successful build. PyTorch retains its
    # build lock and performs source dependency checks on the compile path.
    temporary = manifest_path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(manifest_text)
    temporary.replace(manifest_path)
    _EXT_CACHE[key] = ext
    return ext

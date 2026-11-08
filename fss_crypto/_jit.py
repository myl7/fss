"""JIT compilation manager for FSS C++ extensions."""

import hashlib
import builtins
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
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from filelock import FileLock

from .jit import _Locations, _current_locations
from ._jit_worker import _torch_identity

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
_ARTIFACT_PATHS: dict[str, Path] = {}
_KEY_LOCKS: dict[str, threading.RLock] = {}
_KEY_LOCKS_LOCK = threading.Lock()

# Keeps worker environment snapshots synchronized across local builds.
_BUILD_ENV_LOCK = threading.Lock()

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


# Backend instances are supplied to NativeScheme; only types enter generated source.

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


# Resolve these flags without assigning TORCH_CUDA_ARCH_LIST: the build
# identity must record the effective architectures even when the caller
# leaves the environment unset. Torch releases before 2.7 still resolve
# their own arch list inside load() from the environment or the visible
# devices and never inspect extra_cuda_cflags for arch flags. The isolated
# worker supplies the resolved value without changing the parent environment.
_NAMED_ARCHES = {
    "Kepler+Tesla": "3.7", "Kepler": "3.5+PTX", "Maxwell+Tegra": "5.3",
    "Maxwell": "5.0;5.2+PTX", "Pascal": "6.0;6.1+PTX", "Volta+Tegra": "7.2",
    "Volta": "7.0+PTX", "Turing": "7.5+PTX", "Ampere+Tegra": "8.7",
    "Ampere": "8.0;8.6+PTX", "Ada": "8.9+PTX", "Hopper": "9.0+PTX",
    "Blackwell+Tegra": "11.0", "Blackwell": "10.0;12.0+PTX",
}


def _resolved_arches() -> list[str]:
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    if arch_list:
        for name, arches in _NAMED_ARCHES.items():
            arch_list = arch_list.replace(name, arches)
        return [arch for arch in arch_list.replace(" ", ";").split(";") if arch]
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
    return arches


def _cuda_arch_flags() -> list[str]:
    flags = set()
    for arch in _resolved_arches():
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
def _supports_openmp(command: tuple[str, ...], version: str) -> bool:
    try:
        probe = subprocess.run(
            [*command, "-fopenmp", "-x", "c++", "-o", os.devnull, "-"],
            input="#include <omp.h>\nint main() { return omp_get_max_threads() == 0; }\n",
            text=True, capture_output=True, timeout=20, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


def _build_options(knobs: dict) -> tuple[list[str], list[str], list[str]]:
    cflags = ["-O3", "-std=c++20"]
    cuda_flags = ["-O3", "-std=c++20", "--extended-lambda", *_cuda_arch_flags()]
    ldflags = []
    if "primitive" not in knobs:
        compiler = _compiler_identity(os.environ.get("CXX", "c++"))
        openmp = _supports_openmp(tuple(compiler["command"]), compiler["version"])
        if os.environ.get("CC"):
            host_compiler = _compiler_identity(os.environ["CC"])
            openmp = openmp and _supports_openmp(tuple(host_compiler["command"]), host_compiler["version"])
        if openmp:
            cflags.append("-fopenmp")
            cuda_flags.append("-Xcompiler=-fopenmp")
            ldflags.append("-fopenmp")
    material = json.dumps(knobs, sort_keys=True)
    if any(name in material for name in ('"aes128_mmo"', '"aes128_feistel"', '"sha256"')):
        ldflags.extend(["-lssl", "-lcrypto"])
    if any(name in material for name in ('"aes128_mmo_raw"', '"aes128_feistel_raw"')):
        cflags.extend(["-maes", "-msse4.2"])
        cuda_flags.append("-Xcompiler=-maes,-msse4.2")
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
        "torch": _torch_identity(),
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


_SCHEMES = {
    "dpf": (1, "fss/dpf.cuh", "fss::Dpf<kInBits, GroupType, PrgType, InType>", 2),
    "dcf": (2, "fss/dcf.cuh", "fss::Dcf<kInBits, GroupType, PrgType, InType, kPred>", 4),
    "half_tree_dpf": (3, "fss/half_tree_dpf.cuh", "fss::HalfTreeDpf<kInBits, GroupType, HashType, InType>", 0),
    "packed_half_tree_dpf": (4, "fss/packed_half_tree_dpf.cuh", "fss::PackedHalfTreeDpf<kInBits, kOutBits, HashType, InType>", 0),
    "grotto_dcf": (5, "fss/grotto_dcf.cuh", "fss::GrottoDcf<kInBits, PrgType, InType>", 2),
    "half_tree_grotto_dcf": (6, "fss/half_tree_grotto_dcf.cuh", "fss::HalfTreeGrottoDcf<kInBits, HashType, InType>", 0),
    "vdpf": (7, "fss/vdpf.cuh", "fss::Vdpf<kInBits, GroupType, PrgType, XorHashType, HashType, InType>", 2),
    "dmpf": (8, "fss/dmpf.cuh", "fss::Dmpf<kInBits, kMaxPoints, kBucketBits, GroupType, PrgType, PrpType, InType, kChLambda>", 2),
    "vdmpf": (9, "fss/vdmpf.cuh", "fss::Vdmpf<kInBits, kMaxPoints, kBucketBits, GroupType, PrgType, XorHashType, HashType, PrpType, InType, kChLambda>", 2),
}


def _backend_cpp(spec: dict | None) -> tuple[str, list[str], bool]:
    if spec is None:
        return "fss::python::NoBackend", [], True
    kind = spec["kind"]
    if kind == "bytes":
        return "fss::group::Bytes", ["fss/group/bytes.cuh"], True
    if kind == "uint":
        if spec["bits"] == 128:
            mod = spec["mod"]
            low, high = mod & ((1 << 64) - 1), mod >> 64
            return (f"fss::group::Uint128<fss::group::Uint128Modulus{{.low = {low}ULL, .high = {high}ULL}}>",
                ["fss/group/uint128.cuh"], True)
        return f"fss::group::Uint<uint{spec['bits']}_t, {spec['mod']}>", ["fss/group/uint.cuh"], True
    if kind == "chacha":
        return f"fss::prg::ChaCha<{spec['mul']}, {spec['rounds']}>", ["fss/prg/chacha.cuh"], True
    if kind in ("aes128_mmo", "aes128_mmo_raw", "aes128_soft"):
        names = {"aes128_mmo": "Aes128Mmo", "aes128_mmo_raw": "Aes128MmoRaw", "aes128_soft": "Aes128Soft"}
        file = "aes128_mmo_soft" if kind == "aes128_soft" else kind
        return f"fss::prg::{names[kind]}<{spec['mul']}>", [f"fss/prg/{file}.cuh"], kind == "aes128_soft"
    if kind in ("blake3", "sha256"):
        return f"fss::hash::{ {'blake3': 'Blake3', 'sha256': 'Sha256'}[kind]}", [f"fss/hash/{kind}.cuh"], kind == "blake3"
    if kind == "aes128_ccr":
        aes, includes, gpu = _backend_cpp(spec["aes_mmo"])
        return f"fss::hash::Aes128Ccr<{aes}>", [*includes, "fss/hash/aes128_ccr.cuh"], gpu
    names = {"chacha_feistel": "ChaChaFeistel", "aes128_feistel": "Aes128Feistel", "aes128_feistel_raw": "Aes128FeistelRaw"}
    if kind in names:
        return f"fss::prp::{names[kind]}", [f"fss/prp/{kind}.cuh"], kind == "chacha_feistel"
    raise ValueError(f"unsupported backend kind {kind!r}")


def _input_type(in_bits: int, in_type: str | None) -> str:
    widths = {"uint8": 8, "uint16": 16, "uint32": 32, "uint64": 64, "uint128": 128}
    if in_type is None:
        width = next(width for width in widths.values() if width >= in_bits)
    else:
        if in_type not in widths:
            raise ValueError("in_type must be uint8, uint16, uint32, uint64, or uint128")
        width = widths[in_type]
    if width < in_bits:
        raise ValueError("in_type cannot represent the input domain")
    return "__uint128_t" if width == 128 else f"uint{width}_t"


def _normalize_backend(value):
    if value is None or isinstance(value, dict):
        return value
    from fss_crypto._config import _backend_spec
    return _backend_spec(value)


def _generate_source(scheme: str, in_bits: int, knobs: dict) -> str:
    kind, header, core, _ = _SCHEMES[scheme]
    includes = [header]
    aliases = [f"constexpr int kInBits = {in_bits};", f"using InType = {_input_type(in_bits, knobs.get('in_type'))};"]
    gpu = True
    for role, name in (("group", "GroupType"), ("prg", "PrgType"), ("xor_hash", "XorHashType"), ("hash", "HashType"), ("prp", "PrpType")):
        native, dependencies, supported = _backend_cpp(knobs.get(role))
        includes.extend(dependencies)
        aliases.append(f"using {name} = {native};")
        gpu = gpu and supported
    aliases += [
        f"constexpr int kSchemeKind = {kind};",
        f"constexpr int kOutBits = {knobs.get('out_bits') or 1};",
        f"constexpr int kMaxPoints = {knobs.get('max_points') or 30};",
        f"constexpr int kBucketBits = {knobs.get('bucket_bits') or 1};",
        f"constexpr int kChLambda = {knobs['ch_lambda']};",
        f"constexpr int kFrontierDepth = {knobs.get('frontier_depth', 16)};",
        f"constexpr int kThreadsPerBlock = {knobs.get('threads_per_block', 256)};",
        f"constexpr fss::DcfPred kPred = fss::DcfPred::{'kLt' if knobs.get('pred', 'lt') == 'lt' else 'kGt'};",
        f"constexpr bool kGpuSupported = {str(gpu).lower()};",
        f"using SchemeType = {core};",
    ]
    # DcfPred and NoBackend are needed before generated aliases, including in non-DCF modules.
    if "fss/dcf.cuh" not in includes:
        includes.append("fss/dcf.cuh")
    lines = [f"#include <{path}>" for path in dict.fromkeys(includes)]
    lines += ['#include "binding_types.cuh"', "", *aliases, "", '#include "binding_common.cuh"', f'#include "{scheme}_binding_impl.cuh"', ""]
    return "\n".join(lines)


def _generate_primitive_source(spec: dict) -> str:
    kind = spec["kind"]
    native, includes, gpu = _backend_cpp(spec)
    category = 1 if kind in ("bytes", "uint") else 2 if kind in ("chacha", "aes128_mmo", "aes128_mmo_raw", "aes128_soft") else 3 if kind in ("blake3", "sha256", "aes128_ccr") else 4
    mul = spec.get("mul", 1)
    lines = [f"#include <{path}>" for path in dict.fromkeys(includes)]
    lines += ['#include "binding_types.cuh"', f"using PrimitiveType = {native};", f"constexpr int kPrimitiveKind = {category};", f"constexpr int kPrgMul = {mul};", f"constexpr bool kGpuSupported = {str(gpu).lower()};", '#include "primitive_binding.cuh"', ""]
    return "\n".join(lines)


def _scheme_request(scheme: str, in_bits: int, *, group=None, prg=None, pred: str = "lt", out_bits: int | None = None,
         max_points: int | None = None, bucket_bits: int | None = None, xor_hash=None, hash=None,
         prp=None, in_type: str | None = None,
         ch_lambda: int = 80, frontier_depth: int = 16,
         threads_per_block: int = 256) -> object:
    """Compile a concrete Core type, without including runtime key material in its identity."""
    if scheme not in _SCHEMES:
        raise ValueError(f"unsupported scheme {scheme!r}")
    if isinstance(in_bits, bool) or not isinstance(in_bits, int) or not 1 <= in_bits <= (126 if scheme in ("dmpf", "vdmpf") else 128):
        raise ValueError("in_bits is outside the Core domain range")
    _input_type(in_bits, in_type)
    _check_gpu_launch(frontier_depth, threads_per_block)
    knobs = dict(group=_normalize_backend(group), prg=_normalize_backend(prg), pred=pred,
                 out_bits=out_bits, max_points=max_points, bucket_bits=bucket_bits,
                 xor_hash=_normalize_backend(xor_hash), hash=_normalize_backend(hash),
                 prp=_normalize_backend(prp), in_type=in_type,
                 ch_lambda=ch_lambda, frontier_depth=frontier_depth,
                 threads_per_block=threads_per_block)
    _check_scheme_config(scheme, knobs)
    if scheme == "packed_half_tree_dpf" and in_bits <= (128 // out_bits).bit_length() - 1:
        raise ValueError("in_bits must exceed log2(128 / out_bits)")
    if scheme == "half_tree_grotto_dcf" and in_bits <= 7:
        raise ValueError("HalfTreeGrottoDcf requires in_bits > 7")
    return scheme, _generate_source(scheme, in_bits, knobs), knobs


def _check_scheme_config(scheme: str, knobs: dict) -> None:
    roles = {"dpf": ("group", "prg"), "dcf": ("group", "prg"), "half_tree_dpf": ("group", "hash"),
             "packed_half_tree_dpf": ("hash",), "grotto_dcf": ("prg",), "half_tree_grotto_dcf": ("hash",),
             "vdpf": ("group", "prg", "xor_hash", "hash"), "dmpf": ("group", "prg", "prp"),
             "vdmpf": ("group", "prg", "xor_hash", "hash", "prp")}
    required = roles[scheme]
    for role in required:
        if knobs.get(role) is None:
            raise ValueError(f"{scheme} requires explicit {role} configuration")
    for role in ("group", "prg", "xor_hash", "hash", "prp"):
        if role not in required and knobs.get(role) is not None:
            raise ValueError(f"{scheme} does not take a {role} backend")
    allowed_kinds = {"group": ("bytes", "uint"),
                     "prg": ("chacha", "aes128_mmo", "aes128_mmo_raw", "aes128_soft"),
                     "prp": ("chacha_feistel", "aes128_feistel", "aes128_feistel_raw")}
    for role, kinds in allowed_kinds.items():
        if role in required and knobs[role]["kind"] not in kinds:
            raise ValueError(f"unsupported {role} backend for {scheme}")
    mul = _SCHEMES[scheme][3]
    if mul and knobs["prg"].get("mul") != mul:
        raise ValueError(f"{scheme} requires a PRG with mul={mul}")
    for role in ("xor_hash",):
        if role in required and knobs[role]["kind"] not in ("blake3", "sha256"):
            raise ValueError("xor_hash must implement XorHashable")
    if "hash" in required:
        allowed = ("aes128_ccr",) if scheme in ("half_tree_dpf", "packed_half_tree_dpf", "half_tree_grotto_dcf") else ("blake3", "sha256")
        if knobs["hash"]["kind"] not in allowed:
            raise ValueError(f"{scheme} requires an appropriate explicit hash backend")
    if scheme == "dcf" and knobs["pred"] not in ("lt", "gt"):
        raise ValueError("pred must be lt or gt")
    if scheme == "packed_half_tree_dpf":
        width = knobs["out_bits"]
        if isinstance(width, bool) or not isinstance(width, int) or width not in (1, 2, 4, 8, 16, 32, 64):
            raise ValueError("out_bits must be one of 1, 2, 4, 8, 16, 32, 64")
    if scheme in ("dmpf", "vdmpf"):
        for name, minimum, maximum in (("max_points", 30, (1 << 31)-1), ("bucket_bits", 1, 127), ("ch_lambda", 0, (1 << 31)-1)):
            value = knobs[name]
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise ValueError(f"{name} is outside the C++ parameter range")


def _check_gpu_launch(frontier_depth, threads_per_block):
    if isinstance(frontier_depth, bool) or not isinstance(frontier_depth, int) or not 0 <= frontier_depth < (1 << 31):
        raise ValueError("frontier_depth must be a nonnegative C++ int")
    if isinstance(threads_per_block, bool) or not isinstance(threads_per_block, int) or not 1 <= threads_per_block <= 1024 or threads_per_block & (threads_per_block - 1):
        raise ValueError("threads_per_block must be a power of two in [1, 1024]")


def load_primitive(config, *, _locations=None) -> object:
    """Compile the selected public Group, PRG, Hash or PRP implementation."""
    spec = _normalize_backend(config)
    if spec is None:
        raise ValueError("primitive configuration is required")
    knobs = {"primitive": spec}
    return _load_generated("primitive", _generate_primitive_source(spec), knobs, locations=_locations)


@dataclass(frozen=True)
class _Prepared:
    label: str
    raw_source: str
    source: str
    knobs: dict
    key: str
    native_abi: str
    manifest_text: str
    options: tuple[list[str], list[str], list[str]]
    include_dir: str
    csrc_dir: str


def _prepare_generated(label: str, source: str, knobs: dict) -> _Prepared:
    include_dir = _find_include_dir()
    csrc_dir = str(_CSRC_DIR)
    options = _build_options(knobs)
    manifest = _build_manifest(source, include_dir, csrc_dir, options)
    native_identity = {name: manifest[name] for name in
                       ("headers", "python", "torch", "compilers", "environment")}
    native_identity["bridge_schema"] = 1
    native_abi = hashlib.sha256(json.dumps(native_identity, sort_keys=True).encode()).hexdigest()[:20]
    raw_source = source
    source = f"#define FSS_NATIVE_ABI abi_{native_abi}\n" + source
    manifest["source"] = source
    manifest["native_abi"] = native_abi
    manifest_text = json.dumps(manifest, sort_keys=True, indent=2)
    fingerprint = hashlib.sha256(manifest_text.encode()).hexdigest()[:20]
    return _Prepared(label, raw_source, source, knobs, f"fss_{label}_{fingerprint}",
                     native_abi, manifest_text, options, include_dir, csrc_dir)


def _assert_native_abi(expected: str) -> None:
    spec = {"kind": "bytes"}
    current = _prepare_generated("primitive", _generate_primitive_source(spec), {"primitive": spec})
    if current.native_abi != expected:
        raise RuntimeError("native ABI changed during construction; create a new scheme")


def _revalidate(prepared: _Prepared) -> None:
    current = _prepare_generated(prepared.label, prepared.raw_source, prepared.knobs)
    if current.manifest_text != prepared.manifest_text:
        raise RuntimeError("jit build inputs changed during preparation; retry the operation")


def _primitive_dependencies(config):
    """Share recursive dependency ordering with native owner construction."""
    spec = _normalize_backend(config)
    if spec is None:
        raise ValueError("primitive configuration is required")
    if spec["kind"] == "aes128_ccr":
        yield config["aes_mmo"] if isinstance(config, dict) else config.aes_mmo


def _primitive_requests(config):
    for dependency in _primitive_dependencies(config):
        yield from _primitive_requests(dependency)
    spec = _normalize_backend(config)
    if spec is None:
        raise ValueError("primitive configuration is required")
    yield "primitive", _generate_primitive_source(spec), {"primitive": spec}


def load(scheme: str, in_bits: int, *, _locations=None, **kwargs) -> object:
    """Load the selected scheme specialization using full build identity."""
    return _load_generated(*_scheme_request(scheme, in_bits, **kwargs), locations=_locations)


def _precompile_requests(requests) -> None:
    locations = _current_locations()
    prepared = list({item.key: item for item in
                     (_prepare_generated(*request) for request in requests)}.values())
    if len({item.native_abi for item in prepared}) != 1:
        raise RuntimeError("native ABI changed during preparation; retry the operation")
    for item in prepared:
        _load_prepared(item, locations, materialize=True)
    for item in prepared:
        _revalidate(item)


def precompile(scheme: str, in_bits: int, **kwargs) -> None:
    """Prepare the material-free dependency closure and selected launches."""
    selected = _scheme_request(scheme, in_bits, **kwargs)
    requests = []
    for role in ("group", "prg", "xor_hash", "hash", "prp"):
        config = selected[2][role]
        if config is not None:
            requests.extend(_primitive_requests(config))
    default_kwargs = dict(kwargs, frontier_depth=16, threads_per_block=256)
    requests.append(_scheme_request(scheme, in_bits, **default_kwargs))
    requests.append(selected)
    _precompile_requests(requests)


def precompile_primitive(config) -> None:
    """Prepare a primitive and every separately loaded dependency."""
    _precompile_requests(list(_primitive_requests(config)))


def _load_generated(label: str, source: str, knobs: dict, *, locations=None) -> object:
    return _load_prepared(_prepare_generated(label, source, knobs), locations)


def _matching_artifact(root: Path, prepared: _Prepared) -> Path | None:
    directory = root / prepared.key
    try:
        if not (directory / f"{prepared.key}.so").stat().st_size:
            return None
        if (directory / "build_inputs.json").read_text() != prepared.manifest_text:
            return None
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError, UnicodeError):
        return None
    return directory


def _remember(prepared: _Prepared, module, directory: Path):
    module._fss_native_abi = prepared.native_abi
    _ARTIFACT_PATHS[prepared.key] = directory
    _EXT_CACHE[prepared.key] = module
    return module


def _import_artifact(prepared: _Prepared, directory: Path):
    return _import_built(prepared.key, str(directory / f"{prepared.key}.so"))


def _validate_artifact(prepared: _Prepared, directory: Path) -> None:
    # CPython can reuse a loaded extension after its on-disk file was replaced.
    # Only a fresh process establishes that explicit preparation remains reusable.
    with _BUILD_ENV_LOCK:
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    try:
        result = subprocess.run(
            [sys.executable, str(_PACKAGE_DIR / "_jit_worker.py"), "--validate",
             prepared.key, str(directory / f"{prepared.key}.so"),
             json.dumps(json.loads(prepared.manifest_text)["torch"])], env=environment,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except OSError as error:
        raise RuntimeError(f"cannot start artifact validation worker for {prepared.key}") from error
    prefix = "FSS_ARTIFACT_VALIDATION="
    try:
        status = json.loads(result.stdout.rsplit(prefix, 1)[1])
    except (IndexError, ValueError) as error:
        raise RuntimeError(f"artifact validation worker failed for {prepared.key}:\n{result.stdout}") from error
    kind = status.get("kind")
    if kind not in ("ok", "miss", "error") or result.returncode != (0 if kind == "ok" else 1):
        raise RuntimeError(f"artifact validation worker failed for {prepared.key}:\n{result.stdout}")
    if kind == "ok":
        return
    message = f"artifact validation failed for {prepared.key}:\n{result.stdout}"
    if status["kind"] == "miss":
        raise ImportError(message)
    error_type = getattr(builtins, status.get("type", ""), None)
    error = RuntimeError(message)
    if status.get("module") == "builtins" and isinstance(error_type, type) and issubclass(error_type, Exception):
        try:
            error = error_type(*status["args"])
        except TypeError:
            pass
    raise error from RuntimeError(message)


def _build_worker(prepared: _Prepared, staging: Path) -> None:
    source = staging / f"{prepared.key}.cu"
    source.write_text(prepared.source)
    cflags, cuda_flags, ldflags = prepared.options
    request = {"name": prepared.key, "sources": [str(source)],
               "extra_include_paths": [prepared.include_dir, prepared.csrc_dir],
               "extra_cflags": cflags, "extra_cuda_cflags": cuda_flags,
               "extra_ldflags": ldflags or None, "build_directory": str(staging),
               "arch_list": ";".join(_resolved_arches()),
               "torch_identity": json.loads(prepared.manifest_text)["torch"]}
    request_path = staging / "build_request.json"
    request_path.write_text(json.dumps(request))
    # Capture environment consistently with other FSS builds in this process.
    with _BUILD_ENV_LOCK:
        environment = os.environ.copy()
    result = subprocess.run([sys.executable, str(_PACKAGE_DIR / "_jit_worker.py"),
                             str(request_path)], env=environment, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (staging / "build.log").write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"jit build failed for {prepared.key}:\n{result.stdout}")


def _publish(prepared: _Prepared, locations: _Locations, source: Path | None):
    root = locations.write_root
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{prepared.key}.", dir=root))
    try:
        if source is None:
            _revalidate(prepared)
            _build_worker(prepared, staging)
        else:
            shutil.copyfile(source / f"{prepared.key}.so", staging / f"{prepared.key}.so")
        _revalidate(prepared)
        # The dynamic loader keys a DSO by inode: importing staging before its
        # rename makes a later import of the final path re-run PyInit on the
        # same DSO and corrupt the static PyModuleDef refcount freed at exit.
        _validate_artifact(prepared, staging)
        destination = root / prepared.key
        # Compilers run in separate staging directories. Only publication is locked.
        with FileLock(str(root / f".{prepared.key}.publish.lock")):
            existing = _matching_artifact(root, prepared)
            if existing is not None:
                try:
                    _validate_artifact(prepared, existing)
                except (ImportError, OSError):
                    existing = None
            _revalidate(prepared)
            if existing is not None:
                module = _import_artifact(prepared, destination)
                _revalidate(prepared)
            else:
                stale = None
                if destination.exists() or destination.is_symlink():
                    stale = root / f".{prepared.key}.stale.{os.getpid()}.{threading.get_ident()}"
                    destination.rename(stale)
                committed = False
                try:
                    # Without build_inputs.json, readers skip the destination
                    # until the stable import and drift checks have passed.
                    staging.rename(destination)
                    module = _import_artifact(prepared, destination)
                    _revalidate(prepared)
                    pending = destination / f".{prepared.key}.inputs.{os.getpid()}.{threading.get_ident()}"
                    pending.write_text(prepared.manifest_text)
                    os.replace(pending, destination / "build_inputs.json")
                    committed = True
                finally:
                    if not committed:
                        if destination.is_dir() and not destination.is_symlink():
                            shutil.rmtree(destination, ignore_errors=True)
                        else:
                            destination.unlink(missing_ok=True)
                        if stale is not None and not destination.exists():
                            stale.rename(destination)
                if stale is not None:
                    if stale.is_dir() and not stale.is_symlink():
                        shutil.rmtree(stale)
                    else:
                        stale.unlink()
        return _remember(prepared, module, destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _load_prepared(prepared: _Prepared, locations=None, *, materialize=False):
    locations = locations or _current_locations()
    with _KEY_LOCKS_LOCK:
        lock = _KEY_LOCKS.setdefault(prepared.key, threading.RLock())
    with lock:
        _revalidate(prepared)
        cached = _EXT_CACHE.get(prepared.key)
        if cached is not None and not materialize:
            return cached
        roots = list(locations.read_roots)
        if locations.write_root is not None:
            roots.append(locations.write_root)
        source = None
        last_error = None
        for root in roots:
            directory = _matching_artifact(root, prepared)
            if directory is None:
                continue
            try:
                if materialize:
                    _validate_artifact(prepared, directory)
                module = _import_artifact(prepared, directory)
            except (ImportError, OSError) as exc:
                last_error = exc
                continue
            source = directory
            if not materialize or locations.write_root is None or root == locations.write_root:
                return _remember(prepared, module, directory)
            break
        if locations.write_root is None:
            raise RuntimeError(f"no compatible compiled plugin for {prepared.key} in read-only cache") from last_error
        # A warm module alone cannot satisfy explicit preparation at a new root.
        if source is None and cached is not None:
            origin = _ARTIFACT_PATHS.get(prepared.key)
            if origin is not None:
                candidate = _matching_artifact(origin.parent, prepared)
                if candidate is not None:
                    try:
                        _validate_artifact(prepared, candidate)
                        source = candidate
                    except (ImportError, OSError):
                        pass
        return _publish(prepared, locations, source)

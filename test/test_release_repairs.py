"""JIT cache, compilation-option and source-selection contracts."""

import os
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fss_crypto import _jit, group, prg


def _load_dpf():
    return _jit.load("dpf", 8, group=group.Bytes(),
                     prg=prg.ChaCha(nonce=(1, 2), mul=2))


@pytest.fixture
def mock_jit(monkeypatch, tmp_path):
    calls = {"build": [], "import": []}
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    include = tmp_path / "include"
    (include / "fss").mkdir(parents=True)
    (include / "fss" / "test.cuh").write_text("// original\n")
    monkeypatch.setattr(_jit, "_find_include_dir", lambda: str(include))
    monkeypatch.setattr(_jit, "_compiler_identity", lambda command: {
        "command": [command], "stamp": None, "version": "GCC 12"})
    monkeypatch.setattr(_jit, "_supports_openmp", lambda *a: True)
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.0")
    monkeypatch.setattr(_jit.torch.cuda, "get_arch_list", list)
    monkeypatch.setattr(_jit.torch.cuda, "device_count", lambda: 0)
    _jit._EXT_CACHE.clear()

    def build(prepared, staging):
        cflags, cuda_flags, ldflags = prepared.options
        calls["build"].append({
            "name": prepared.key,
            "build_directory": str(staging.parent / prepared.key),
            "extra_cflags": cflags,
            "extra_cuda_cflags": cuda_flags,
            "extra_ldflags": ldflags,
            "torch_arch_env": ";".join(_jit._resolved_arches()),
        })
        (staging / (prepared.key + ".so")).write_bytes(b"native")

    def import_built(name, path):
        calls["import"].append((name, path))
        return SimpleNamespace()

    monkeypatch.setattr(_jit, "_build_worker", build)
    monkeypatch.setattr(_jit, "_validate_artifact", lambda *args: None)
    monkeypatch.setattr(_jit, "_import_built", import_built)
    yield calls
    _jit._EXT_CACHE.clear()


def test_jit_memory_and_manifest_cache(mock_jit):
    ext = _load_dpf()
    assert _load_dpf() is ext
    assert len(mock_jit["build"]) == 1
    manifest = Path(mock_jit["build"][0]["build_directory"]) / "build_inputs.json"
    assert manifest.is_file()
    _jit._EXT_CACHE.clear()
    _load_dpf()
    assert len(mock_jit["import"]) == 2 and len(mock_jit["build"]) == 1
    manifest.write_text("{}")
    _jit._EXT_CACHE.clear()
    _load_dpf()
    assert len(mock_jit["build"]) == 2


def test_jit_architecture_compiler_options_and_abi_identity(mock_jit, monkeypatch):
    first = _load_dpf()
    first_build = mock_jit["build"][0]
    assert "-fopenmp" in first_build["extra_cflags"]
    assert "-Xcompiler=-fopenmp" in first_build["extra_cuda_cflags"]
    assert "-fopenmp" in first_build["extra_ldflags"]
    for arch in ("9.0", "9.0+PTX", "8.0+PTX"):
        monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", arch)
        assert _load_dpf() is not first
    assert len(mock_jit["build"]) == 4
    monkeypatch.setattr(_jit, "_supports_openmp", lambda *a: False)
    _load_dpf()
    sequential_build = mock_jit["build"][-1]
    assert "-fopenmp" not in sequential_build["extra_cflags"]
    assert "-Xcompiler=-fopenmp" not in sequential_build["extra_cuda_cflags"]
    assert not sequential_build["extra_ldflags"]
    monkeypatch.setattr(_jit.torch._C, "_GLIBCXX_USE_CXX11_ABI", "changed")
    _load_dpf()
    assert len(mock_jit["build"]) == 6


def test_jit_headless_architecture_does_not_change_environment(mock_jit, monkeypatch):
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST")
    _load_dpf()
    assert "TORCH_CUDA_ARCH_LIST" not in _jit.os.environ
    assert "-gencode=arch=compute_80,code=sm_80" in mock_jit["build"][0]["extra_cuda_cflags"]
    assert mock_jit["build"][0]["torch_arch_env"] == "8.0"


def test_jit_header_changes_invalidate_same_process(mock_jit):
    _load_dpf()
    header = Path(_jit._find_include_dir()) / "fss" / "test.cuh"
    header.write_text("// changed\n")
    _load_dpf()
    assert len(mock_jit["build"]) == 2


def test_jit_source_uses_selected_aes_backend():
    for scheme, spec in _jit._SCHEMES.items():
        half_tree = scheme in ("half_tree_dpf", "packed_half_tree_dpf", "half_tree_grotto_dcf")
        knobs = {"group": None, "prg": None, "hash": None, "xor_hash": None,
                 "prp": None, "pred": "gt", "out_bits": 64,
                 "max_points": 30, "bucket_bits": 4, "ch_lambda": 80}
        if scheme in ("dpf", "dcf", "half_tree_dpf", "vdpf", "dmpf", "vdmpf"):
            knobs["group"] = {"kind": "bytes"}
        if spec[3]:
            knobs["prg"] = {"kind": "aes128_mmo", "mul": spec[3]}
        if half_tree:
            knobs["hash"] = {"kind": "aes128_ccr", "aes_mmo": {"kind": "aes128_mmo", "mul": 1}}
        if scheme in ("vdpf", "vdmpf"):
            knobs["xor_hash"] = {"kind": "blake3"}
            knobs["hash"] = {"kind": "sha256"}
        if scheme in ("dmpf", "vdmpf"):
            knobs["prp"] = {"kind": "chacha_feistel"}
        source = _jit._generate_source(scheme, 8, knobs)
        assert f'#include "{scheme}_binding_impl.cuh"' in source
        if half_tree:
            assert "using HashType = fss::hash::Aes128Ccr<fss::prg::Aes128Mmo<1>>;" in source
        else:
            assert f"using PrgType = fss::prg::Aes128Mmo<{spec[3]}>;" in source
        assert "constexpr bool kGpuSupported = false;" in source


def test_openmp_probe_failure_disables_parallel_build_flags(monkeypatch):
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.0")
    _jit._supports_openmp.cache_clear()
    monkeypatch.setattr(_jit.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1))
    monkeypatch.setattr(_jit, "_compiler_identity", lambda command: {
        "command": [command], "stamp": None, "version": "test"})
    assert not _jit._supports_openmp(("missing-compiler",), "test")
    cflags, cuda_flags, ldflags = _jit._build_options({"group": {"kind": "bytes"}})
    assert "-fopenmp" not in cflags
    assert "-Xcompiler=-fopenmp" not in cuda_flags
    assert "-fopenmp" not in ldflags
    _jit._supports_openmp.cache_clear()


def test_native_bridge_identity_matches_primitive_and_scheme(mock_jit):
    backend = prg.ChaCha(nonce=(1, 2), mul=2)
    _jit.load_primitive(backend)
    _load_dpf()
    manifests = [json.loads((Path(call["build_directory"]) / "build_inputs.json").read_text())
                 for call in mock_jit["build"]]
    assert len(manifests) == 2
    assert manifests[0]["native_abi"] == manifests[1]["native_abi"]
    for manifest in manifests:
        assert manifest["source"].startswith(f"#define FSS_NATIVE_ABI abi_{manifest['native_abi']}\n")


def test_native_bridge_identity_changes_with_source(mock_jit):
    _load_dpf()
    first = json.loads((Path(mock_jit["build"][0]["build_directory"]) / "build_inputs.json").read_text())
    header = Path(_jit._find_include_dir()) / "fss" / "test.cuh"
    header.write_text("// changed native layout\n")
    _load_dpf()
    second = json.loads((Path(mock_jit["build"][-1]["build_directory"]) / "build_inputs.json").read_text())
    assert first["native_abi"] != second["native_abi"]

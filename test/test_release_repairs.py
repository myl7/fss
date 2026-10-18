"""Fast API and cache-contract tests. Native integration is tested separately."""

import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fss_crypto import Dcf, Dmpf, Dpf, GrottoDcf, HalfTreeDpf, PackedHalfTreeDpf, Vdmpf, Vdpf, _jit
from fss_crypto._validate import (
    validate_bucket_bits,
    validate_cuda_backends,
    validate_full_domain_size,
    validate_query_domain,
)


@pytest.mark.parametrize("cls,args", [
    (Dpf, (8,)), (Dcf, (8,)), (HalfTreeDpf, (8,)),
    (PackedHalfTreeDpf, (8, 64)), (GrottoDcf, (8,)), (Vdpf, (8,)),
    (Dmpf, (8, 30, 4)),
])
def test_depth_is_keyword_only_and_reaches_jit(monkeypatch, cls, args):
    calls = []
    monkeypatch.setattr(importlib.import_module(cls.__module__), "load",
                        lambda *a, **kw: calls.append((a, kw)) or SimpleNamespace())
    assert inspect.signature(cls).parameters["par_depth"].kind == inspect.Parameter.KEYWORD_ONLY
    assert cls(*args).par_depth == 0
    for depth in (-1, 1, 2, 1000):
        assert cls(*args, par_depth=depth).par_depth == depth
        assert calls[-1][1]["par_depth"] == depth
    for depth in (-2, True, 1.5):
        with pytest.raises((ValueError, TypeError), match="par_depth"):
            cls(*args, par_depth=depth)


def test_released_python_dpf_dcf_positional_calls(monkeypatch):
    for cls, args in ((Dpf, (8, "bytes", "chacha")),
                      (Dcf, (8, "uint", "chacha", "gt"))):
        monkeypatch.setattr(importlib.import_module(cls.__module__), "load", lambda *a, **kw: object())
        assert cls(*args).in_bits == 8


def test_grotto_party_first(monkeypatch):
    calls = []
    monkeypatch.setattr(importlib.import_module(GrottoDcf.__module__), "load",
                        lambda *a, **kw: SimpleNamespace(eval=lambda *a: calls.append(a) or True))
    tree = torch.zeros(15, dtype=torch.bool)
    assert GrottoDcf(3).eval(1, tree, 2)
    assert calls[0] == (1, tree, 2, 0)


def test_static_verify_requires_no_jit(monkeypatch):
    def fail_load(*a, **kw):
        raise AssertionError("verification must not build an extension")
    for cls in (Vdpf, Vdmpf):
        monkeypatch.setattr(importlib.import_module(cls.__module__), "load", fail_load)
        proof = torch.arange(16, dtype=torch.int32).reshape(4, 4)
        assert cls.verify(proof, proof.clone())
        assert object.__new__(cls).verify(proof, proof.clone())
        assert not cls.verify(proof, proof + 1)
        with pytest.raises(TypeError, match="pi0"):
            cls.verify(proof.long(), proof)
        with pytest.raises(TypeError, match="pi1"):
            cls.verify(proof, proof[:, :2])


@pytest.mark.parametrize("bits", [1, 8, 62, 63, 64, 128])
def test_query_domains_and_empty_inputs(bits):
    upper = min((1 << bits) - 1, (1 << 63) - 1)
    validate_query_domain(torch.tensor([0, upper, upper]), bits)
    validate_query_domain(torch.tensor([], dtype=torch.int64), bits)
    validate_query_domain(torch.tensor([upper, -1, upper])[::2], bits)
    with pytest.raises(ValueError, match="xs must be"):
        validate_query_domain(torch.tensor([-1]), bits)
    if bits < 63:
        with pytest.raises(ValueError, match="xs must be"):
            validate_query_domain(torch.tensor([1 << bits]), bits)


def test_bucket_bounds_and_capacity():
    validate_bucket_bits(1, 30, 1)
    validate_bucket_bits(34, 30, 30)
    with pytest.raises(ValueError, match="too small"):
        validate_bucket_bits(35, 30, 30)
    for bits in (0, 31, 64):
        with pytest.raises(ValueError, match="between 1 and 30"):
            validate_bucket_bits(35, 30, bits)


def test_vdpf_size_helper_boundaries():
    assert validate_full_domain_size(3) == 8
    assert validate_full_domain_size(60) == 1 << 60
    assert validate_full_domain_size(22, cuda=True, cuda_limit=22) == 1 << 22
    for bits in (61, 64, 128):
        with pytest.raises(RuntimeError, match="cannot represent"):
            validate_full_domain_size(bits)
    with pytest.raises(RuntimeError, match="supports in_bits <= 22"):
        validate_full_domain_size(23, cuda=True, cuda_limit=22)


def test_host_only_backends_rejected_without_cuda_runtime():
    fake_cuda = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    for kwargs, message in [
        ({"prg": "aes128_mmo"}, "aes128_mmo"),
        ({"prg": "chacha", "prp": "aes128_feistel"}, "prp="),
        ({"prg": "chacha", "xor_hash": "sha256"}, "xor_hash="),
        ({"prg": "chacha", "hash": "sha256"}, "hash="),
    ]:
        with pytest.raises(RuntimeError, match=message):
            validate_cuda_backends(fake_cuda, **kwargs)
    validate_cuda_backends(fake_cuda, prg="chacha", prp="chacha_feistel",
                           xor_hash="blake3", hash="blake3")


@pytest.mark.parametrize("cls", [Dmpf, Vdmpf])
def test_multipoint_query_validation_precedes_extension(monkeypatch, cls):
    scheme = object.__new__(cls)
    scheme.in_bits = 8
    scheme.prg, scheme.prp = "chacha", "chacha_feistel"
    scheme.xor_hash = scheme.hash = "blake3"
    scheme._check_key = lambda key: None
    scheme._ext = SimpleNamespace(batch_eval=lambda *a: a[-1], batch_eval_merkle=lambda *a: a[-1])
    key = SimpleNamespace(cws=torch.zeros(1), cs=torch.zeros(1), ocw=torch.zeros(1),
                          s0s=torch.zeros(2), sigma=torch.zeros(1), meta=torch.zeros(1))
    for method in ([scheme.batch_eval, scheme.batch_eval_merkle] if cls is Vdmpf else [scheme.batch_eval]):
        for xs in (torch.tensor([-1]), torch.tensor([256])):
            with pytest.raises(ValueError, match="xs must be"):
                method(0, key, xs)
        xs = torch.tensor([2, -1, 2, -1])[::2]
        result = method(0, key, xs)
        assert result.is_contiguous() and result.tolist() == [2, 2]
        assert method(0, key, torch.empty(0, dtype=torch.int64)).numel() == 0


@pytest.mark.parametrize("beta", [0, (1 << 63) - 1, 1 << 63, (1 << 64) - 1])
def test_packed_beta_64_transport(monkeypatch, beta):
    calls = []
    monkeypatch.setattr(importlib.import_module(PackedHalfTreeDpf.__module__), "load",
                        lambda *a, **kw: SimpleNamespace(gen=lambda *a: calls.append(a)))
    PackedHalfTreeDpf(3, 64).gen(torch.zeros((2, 4), dtype=torch.int32), 1, beta)
    assert calls[-1][-1] & ((1 << 64) - 1) == beta


@pytest.fixture
def mock_jit(monkeypatch, tmp_path):
    calls = {"build": [], "import": []}
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    include = tmp_path / "include"
    (include / "fss").mkdir(parents=True)
    (include / "fss" / "test.cuh").write_text("// original\n")
    monkeypatch.setattr(_jit, "_find_include_dir", lambda: str(include))
    monkeypatch.setattr(_jit, "_compiler_identity", lambda command: {"command": command, "version": "GCC 12"})
    monkeypatch.setattr(_jit.sys, "platform", "linux")
    monkeypatch.setattr(_jit, "_require_openmp", lambda *a: None)
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.0")
    monkeypatch.setattr(_jit.torch.cuda, "get_arch_list", list)
    monkeypatch.setattr(_jit.torch.cuda, "device_count", lambda: 0)
    _jit._EXT_CACHE.clear()

    def build(**kwargs):
        calls["build"].append(kwargs)
        (Path(kwargs["build_directory"]) / (kwargs["name"] + ".so")).touch()
        return object()

    def import_built(name, path):
        calls["import"].append((name, path))
        return object()

    monkeypatch.setattr(_jit, "_cpp_load", build)
    monkeypatch.setattr(_jit, "_import_built", import_built)
    yield calls
    _jit._EXT_CACHE.clear()


def test_jit_memory_and_manifest_cache(mock_jit):
    ext = _jit.load("dpf", 8, "bytes")
    assert _jit.load("dpf", 8, "bytes") is ext
    assert len(mock_jit["build"]) == 1
    manifest = Path(mock_jit["build"][0]["build_directory"]) / "build_inputs.json"
    assert manifest.is_file()
    _jit._EXT_CACHE.clear()
    _jit.load("dpf", 8, "bytes")
    assert len(mock_jit["import"]) == 1 and len(mock_jit["build"]) == 1
    manifest.write_text("{}")
    _jit._EXT_CACHE.clear()
    _jit.load("dpf", 8, "bytes")
    assert len(mock_jit["build"]) == 2


def test_jit_architecture_depth_and_abi_identity(mock_jit, monkeypatch):
    first = _jit.load("dpf", 8, "bytes")
    for arch in ("9.0", "9.0+PTX", "8.0+PTX"):
        monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", arch)
        assert _jit.load("dpf", 8, "bytes") is not first
    assert len(mock_jit["build"]) == 4
    _jit.load("dpf", 8, "bytes", par_depth=2)
    parallel = mock_jit["build"][-1]
    assert "-fopenmp" in parallel["extra_cflags"]
    assert "-Xcompiler=-fopenmp" in parallel["extra_cuda_cflags"]
    assert "-fopenmp" in parallel["extra_ldflags"]
    monkeypatch.setattr(_jit.torch._C, "_GLIBCXX_USE_CXX11_ABI", "changed")
    _jit.load("dpf", 8, "bytes", par_depth=2)
    assert len(mock_jit["build"]) == 6


def test_jit_headless_architecture_does_not_change_environment(mock_jit, monkeypatch):
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST")
    _jit.load("dpf", 8, "bytes")
    assert "TORCH_CUDA_ARCH_LIST" not in _jit.os.environ
    assert "-gencode=arch=compute_80,code=sm_80" in mock_jit["build"][0]["extra_cuda_cflags"]


def test_jit_header_changes_invalidate_same_process(mock_jit):
    _jit.load("dpf", 8, "bytes")
    header = Path(_jit._find_include_dir()) / "fss" / "test.cuh"
    header.write_text("// changed\n")
    _jit.load("dpf", 8, "bytes")
    assert len(mock_jit["build"]) == 2


def test_jit_source_uses_configured_depth_and_aes_adapter():
    for scheme, spec in _jit._SCHEMES.items():
        knobs = {"group": "bytes", "prg": "aes128_mmo", "pred": "gt", "out_bits": 64,
                 "max_points": 30, "bucket_bits": 4, "xor_hash": "blake3", "hash": "sha256",
                 "prp": "chacha_feistel", "par_depth": 0 if scheme == "vdmpf" else 2}
        source = _jit._generate_source(scheme, 8, knobs)
        assert '"binding_prg.cuh"' in source
        assert f"ThreadLocalAes128Mmo<{spec['prg_mul']}>" in source
        assert "constexpr bool kGpuPrg = false" in source
        if scheme != "vdmpf":
            assert "kParDepth>" in source


def test_openmp_probe_failure_precedes_extension_compile(monkeypatch):
    _jit._require_openmp.cache_clear()
    monkeypatch.setattr(_jit.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1, stderr="omp.h missing"))
    with pytest.raises(RuntimeError, match="OpenMP compiler and linker support"):
        _jit._require_openmp(("missing-compiler",), "test")


def test_automatic_depth_has_valid_extension_name(mock_jit):
    _jit.load("dpf", 8, "bytes", par_depth=-1)
    assert mock_jit["build"][0]["name"].isidentifier()


def test_oversized_python_depth_is_clamped_to_cpp_int(mock_jit):
    _jit.load("dpf", 8, "bytes", par_depth=1 << 100)
    source = Path(mock_jit["build"][0]["sources"][0]).read_text()
    assert "constexpr int kParDepth = 2147483647;" in source


@pytest.mark.parametrize("value", [0, (1 << 63) - 1, 1 << 63, (1 << 64) - 1,
                                   1 << 127, (1 << 127) + (1 << 63) + 1,
                                   (1 << 128) - 1])
def test_uint128_transport_uses_signed_int64_halves(value):
    from fss_crypto._int import split_uint128
    lo, hi = split_uint128(value)
    assert -(1 << 63) <= lo < 1 << 63
    assert -(1 << 63) <= hi < 1 << 63
    assert ((hi & ((1 << 64) - 1)) << 64) | (lo & ((1 << 64) - 1)) == value

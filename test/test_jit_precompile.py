"""Precompile closure and scoped compiled-plugin cache contracts."""

from concurrent.futures import ThreadPoolExecutor
import inspect
import json
from pathlib import Path
import shutil
import tempfile
import threading
import time
from types import SimpleNamespace

import pytest

from fss_crypto import Dcf, DcfPred, Dpf, HalfTreeDpf, Vdmpf
from fss_crypto import _jit, group, hash, jit, prg


@pytest.fixture
def cache_workspace():
    build = Path(__file__).resolve().parents[1] / "build"
    build.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="jit-precompile-test-", dir=build) as root:
        yield Path(root)


@pytest.fixture
def fake_compiler(monkeypatch, cache_workspace):
    calls = {"build": [], "import": []}
    include = cache_workspace / "include"
    (include / "fss").mkdir(parents=True)
    (include / "fss" / "probe.cuh").write_text("// original\n")
    monkeypatch.setattr(_jit, "_find_include_dir", lambda: str(include))
    monkeypatch.setattr(_jit, "_compiler_identity", lambda command: {
        "command": [command], "stamp": None, "version": "test compiler"})
    monkeypatch.setattr(_jit, "_supports_openmp", lambda *args: True)
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.0")
    monkeypatch.setattr(_jit.torch.cuda, "get_arch_list", list)
    monkeypatch.setattr(_jit.torch.cuda, "device_count", lambda: 0)
    _jit._EXT_CACHE.clear()

    def build(prepared, staging):
        calls["build"].append({"name": prepared.key, "build_directory": str(staging)})
        assert staging.is_relative_to(cache_workspace)
        (staging / (prepared.key + ".so")).write_bytes(b"fake extension")

    def import_built(name, path):
        assert Path(path).is_file()
        calls["import"].append((name, Path(path)))
        return SimpleNamespace()

    monkeypatch.setattr(_jit, "_build_worker", build)
    monkeypatch.setattr(_jit, "_validate_artifact", lambda *args: None)
    monkeypatch.setattr(_jit, "_import_built", import_built)
    yield calls
    _jit._EXT_CACHE.clear()


def _load_dpf():
    return _jit.load("dpf", 8, group=group.Bytes(),
                     prg=prg.ChaCha(nonce=(1, 2), mul=2))


def _snapshot(root):
    return {str(path.relative_to(root)): (path.stat().st_mtime_ns,
            path.read_bytes() if path.is_file() else None)
            for path in root.rglob("*")}


def test_cache_requires_explicit_write_policy():
    with pytest.raises(TypeError):
        jit.cache()


def test_cache_nested_restore_and_relative_paths(cache_workspace, monkeypatch):
    original = jit._current_locations()
    monkeypatch.chdir(cache_workspace)
    with jit.cache(read_roots=("reader",), write_root="outer"):
        outer = jit._current_locations()
        assert tuple(map(Path, outer.read_roots)) == (cache_workspace / "reader",)
        assert Path(outer.write_root) == cache_workspace / "outer"
        with pytest.raises(RuntimeError, match="restore"):
            with jit.cache(write_root=None):
                assert jit._current_locations().write_root is None
                raise RuntimeError("restore")
        assert jit._current_locations() == outer
        monkeypatch.chdir(cache_workspace.parent)
        assert Path(jit._current_locations().write_root) == cache_workspace / "outer"
    assert jit._current_locations() == original


@pytest.mark.parametrize("reader,writer", [("same", "same"), ("parent", "parent/child"),
                                          ("parent/child", "parent")])
def test_cache_rejects_overlapping_roots(cache_workspace, reader, writer):
    with pytest.raises(ValueError):
        with jit.cache(read_roots=(cache_workspace / reader,),
                       write_root=cache_workspace / writer):
            pytest.fail("overlapping roots were accepted")


def test_cache_rejects_symlink_alias(cache_workspace):
    target = cache_workspace / "target"
    target.mkdir()
    alias = cache_workspace / "alias"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError):
        with jit.cache(read_roots=(alias,), write_root=target):
            pytest.fail("symlink alias was accepted")


def test_cache_new_thread_does_not_inherit_context(cache_workspace):
    original = jit._current_locations()
    with jit.cache(write_root=cache_workspace / "local"):
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(jit._current_locations).result() == original


def test_read_only_miss_creates_nothing(fake_compiler, cache_workspace):
    reader = cache_workspace / "missing"
    before = _snapshot(cache_workspace)
    with jit.cache(read_roots=(reader,), write_root=None):
        with pytest.raises(RuntimeError):
            _load_dpf()
    assert _snapshot(cache_workspace) == before
    assert not fake_compiler["build"]


def test_read_root_is_unchanged_and_manifest_must_match(fake_compiler, cache_workspace):
    reader = cache_workspace / "reader"
    with jit.cache(write_root=reader):
        _load_dpf()
    _jit._EXT_CACHE.clear()
    before = _snapshot(reader)
    with jit.cache(read_roots=(reader,), write_root=None):
        _load_dpf()
    assert _snapshot(reader) == before
    assert len(fake_compiler["build"]) == 1
    manifests = list(reader.glob("*/build_inputs.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    manifest["unexpected_identity_field"] = True
    manifests[0].write_text(json.dumps(manifest))
    _jit._EXT_CACHE.clear()
    before = _snapshot(reader)
    with jit.cache(read_roots=(reader,), write_root=None):
        with pytest.raises(RuntimeError):
            _load_dpf()
    assert _snapshot(reader) == before
    assert len(fake_compiler["build"]) == 1


def test_precompile_exports_warm_memory_complete_closure(fake_compiler, cache_workspace):
    source, destination = cache_workspace / "source", cache_workspace / "destination"
    config = prg.ChaCha(nonce=(11, 23), mul=2)
    with jit.cache(write_root=source):
        Dpf.precompile(8, group.Bytes(), config)
    expected = {path.name for path in source.iterdir() if (path / "build_inputs.json").is_file()}
    assert len(expected) == 3
    count = len(fake_compiler["build"])
    with jit.cache(read_roots=(source,), write_root=destination):
        assert Dpf.precompile(8, group.Bytes(), config) is None
    assert len(fake_compiler["build"]) == count
    assert {path.name for path in destination.iterdir()
            if (path / "build_inputs.json").is_file()} == expected
    _jit._EXT_CACHE.clear()
    with jit.cache(read_roots=(destination,), write_root=None):
        Dpf.precompile(8, group.Bytes(), config)
    assert len(fake_compiler["build"]) == count


def test_read_only_precompile_requires_disk_even_when_warm(fake_compiler, cache_workspace):
    config = prg.ChaCha(nonce=(1, 2), mul=2)
    with jit.cache(write_root=cache_workspace / "source"):
        Dpf.precompile(8, group.Bytes(), config)
    with jit.cache(read_roots=(cache_workspace / "absent",), write_root=None):
        with pytest.raises(RuntimeError):
            Dpf.precompile(8, group.Bytes(), config)


def test_same_key_threads_build_once(fake_compiler, cache_workspace, monkeypatch):
    original = _jit._build_worker
    barrier = threading.Barrier(2)

    def slow_build(*args):
        time.sleep(0.02)
        return original(*args)

    monkeypatch.setattr(_jit, "_build_worker", slow_build)

    def worker():
        with jit.cache(write_root=cache_workspace / "shared"):
            barrier.wait(timeout=5)
            return _load_dpf()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker) for _ in range(2)]
        results = [future.result(timeout=10) for future in futures]
    assert results[0] is results[1]
    assert len(fake_compiler["build"]) == 1


def test_half_tree_precompile_is_material_free_and_prepares_nested_closure(
        fake_compiler, cache_workspace, monkeypatch):
    aes = prg.Aes128Soft(mul=1, keys=(bytes(range(16)),))
    ccr = hash.Aes128Ccr(aes)
    configs = [group.Bytes(), aes, ccr]

    def forbidden(*args, **kwargs):
        pytest.fail("precompile accessed runtime material or constructed a native owner")

    for config in configs:
        monkeypatch.setattr(type(config), "_runtime", forbidden)
    monkeypatch.setattr(HalfTreeDpf, "__init__", forbidden)
    monkeypatch.setattr(_jit.torch, "empty", forbidden)
    with jit.cache(write_root=cache_workspace / "closure"):
        assert HalfTreeDpf.precompile(16, configs[0], ccr,
                                      frontier_depth=17, threads_per_block=128) is None
    manifests = [json.loads(path.read_text()) for path in
                 (cache_workspace / "closure").glob("*/build_inputs.json")]
    assert len(manifests) == 5
    assert len({manifest["native_abi"] for manifest in manifests}) == 1
    sources = [manifest["source"] for manifest in manifests]
    assert sum("half_tree_dpf_binding_impl.cuh" in source for source in sources) == 2
    assert not any(hasattr(config, "_primitive_binding") for config in configs)


def test_public_precompile_signatures_exclude_runtime_arguments():
    assert "instance_seed" not in inspect.signature(HalfTreeDpf.precompile).parameters
    assert "par_depth" not in inspect.signature(Dpf.precompile).parameters
    assert "frontier_depth" not in inspect.signature(Vdmpf.precompile).parameters
    assert inspect.signature(Dcf.precompile).parameters["pred"].default == DcfPred.LT
    assert inspect.signature(Dpf.precompile).parameters["threads_per_block"].default == 256


def test_primitive_precompile_includes_nested_aes(fake_compiler, cache_workspace):
    ccr = hash.Aes128Ccr(prg.Aes128Soft(mul=1, keys=(bytes(16),)))
    with jit.cache(write_root=cache_workspace / "primitive"):
        assert ccr.precompile() is None
    assert len(fake_compiler["build"]) == 2
    assert not hasattr(ccr, "_primitive_binding")
    assert not hasattr(ccr.aes_mmo, "_primitive_binding")


def test_precompile_rejects_mixed_abi_before_loading(fake_compiler, monkeypatch):
    from dataclasses import replace

    prepare = _jit._prepare_generated
    calls = []

    def changing_abi(*args):
        item = prepare(*args)
        return replace(item, native_abi="changed" if item.label == "dpf" else item.native_abi)

    monkeypatch.setattr(_jit, "_prepare_generated", changing_abi)
    monkeypatch.setattr(_jit, "_load_prepared", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(RuntimeError, match="ABI"):
        Dpf.precompile(8, group.Bytes(), prg.ChaCha(nonce=(1, 2), mul=2))
    assert not calls


def test_primitive_rebind_preserves_previous_owner(monkeypatch):
    from fss_crypto import _primitive

    config = group.Bytes()
    current_abi = ["first"]
    owners = []

    def load(config, **kwargs):
        def construct(values):
            owner = SimpleNamespace(abi=current_abi[0], values=values)
            owners.append(owner)
            return owner
        return SimpleNamespace(_fss_native_abi=current_abi[0], NativePrimitive=construct)

    monkeypatch.setattr(_jit, "load_primitive", load)
    first = _primitive.native(config)
    assert _primitive.native(config) is first
    current_abi[0] = "second"
    second = _primitive.native(config)
    assert second is not first
    assert first.abi == "first" and second.abi == "second"
    assert len(owners) == 2
    with pytest.raises(RuntimeError, match="ABI"):
        _primitive.native(config, expected_abi="first")
    assert len(owners) == 2


def test_delayed_launch_rejects_new_abi_before_native_construction(monkeypatch):
    from fss_crypto._scheme import _Scheme

    owner = object.__new__(_Scheme)
    owner._native_abi = "retained"
    owner._gpu_specializations = {}
    owner._gpu_modules = {}
    owner._load_config = {}
    owner._scheme = "dpf"
    owner._in_bits = 8
    owner._locations = jit._current_locations()
    requests = []

    def construct(*args):
        pytest.fail("new ABI launch attempted to consume retained backend owners")

    def load(*args, **kwargs):
        requests.append(kwargs)
        return SimpleNamespace(_fss_native_abi="changed", NativeScheme=construct)

    monkeypatch.setattr(_jit, "load", load)
    with pytest.raises(RuntimeError, match="ABI"):
        owner._execution_native(SimpleNamespace(is_cuda=True), 17, 128)
    assert requests[0]["_locations"] is owner._locations
    assert not owner._gpu_specializations and not owner._gpu_modules


def test_read_roots_try_next_import_candidate(fake_compiler, cache_workspace, monkeypatch):
    first, second = cache_workspace / "first", cache_workspace / "second"
    config = prg.ChaCha(nonce=(1, 2), mul=2)
    with jit.cache(write_root=first):
        Dpf.precompile(8, group.Bytes(), config)
    with jit.cache(read_roots=(first,), write_root=second):
        Dpf.precompile(8, group.Bytes(), config)
    _jit._EXT_CACHE.clear()
    original = _jit._import_built
    attempts = []

    def import_built(name, path):
        attempts.append(Path(path))
        if Path(path).is_relative_to(first):
            raise ImportError("incompatible candidate")
        return original(name, path)

    monkeypatch.setattr(_jit, "_import_built", import_built)
    before = (_snapshot(first), _snapshot(second))
    with jit.cache(read_roots=(first, second), write_root=None):
        _load_dpf()
    assert len(attempts) == 2
    assert (_snapshot(first), _snapshot(second)) == before


def test_native_initialization_error_is_not_treated_as_cache_miss(
        fake_compiler, cache_workspace, monkeypatch):
    source = cache_workspace / "source"
    with jit.cache(write_root=source):
        _load_dpf()
    _jit._EXT_CACHE.clear()

    def broken_import(*args):
        raise ValueError("native initialization failed")

    monkeypatch.setattr(_jit, "_import_built", broken_import)
    with jit.cache(read_roots=(source,), write_root=None):
        with pytest.raises(ValueError, match="native initialization failed"):
            _load_dpf()
    assert not _jit._EXT_CACHE
    assert len(fake_compiler["build"]) == 1


@pytest.mark.parametrize("damage", ["invalid-utf8", "not-directory"])
def test_corrupt_read_candidate_falls_through(fake_compiler, cache_workspace, damage):
    source = cache_workspace / "valid"
    with jit.cache(write_root=source):
        module = _load_dpf()
    key = next(iter(_jit._EXT_CACHE))
    broken = cache_workspace / "broken"
    broken.mkdir()
    if damage == "not-directory":
        (broken / key).write_bytes(b"invalid cache directory")
    else:
        entry = broken / key
        entry.mkdir()
        (entry / (key + ".so")).write_bytes(b"fake extension")
        (entry / "build_inputs.json").write_bytes(b"\xff")
    _jit._EXT_CACHE.clear()
    before = _snapshot(broken)
    with jit.cache(read_roots=(broken, source), write_root=None):
        assert _load_dpf() is not None
    assert _snapshot(broken) == before
    assert len(fake_compiler["build"]) == 1


def test_build_input_drift_is_not_published(fake_compiler, cache_workspace, monkeypatch):
    build = _jit._build_worker

    def changing_build(prepared, staging):
        build(prepared, staging)
        (Path(_jit._find_include_dir()) / "fss" / "probe.cuh").write_text("// changed during build\n")

    monkeypatch.setattr(_jit, "_build_worker", changing_build)
    output = cache_workspace / "changed-inputs"
    with jit.cache(write_root=output):
        with pytest.raises(RuntimeError, match="inputs changed"):
            group.Bytes().precompile()
    assert not list(output.rglob("build_inputs.json"))
    assert not _jit._EXT_CACHE



def test_import_input_drift_is_not_published(fake_compiler, cache_workspace, monkeypatch):
    import_built = _jit._import_built

    def changing_import(*args):
        module = import_built(*args)
        (Path(_jit._find_include_dir()) / "fss" / "probe.cuh").write_text("// changed during import\n")
        return module

    monkeypatch.setattr(_jit, "_import_built", changing_import)
    output = cache_workspace / "changed-import-inputs"
    with jit.cache(write_root=output):
        with pytest.raises(RuntimeError, match="inputs changed"):
            group.Bytes().precompile()
    assert not list(output.rglob("build_inputs.json"))
    assert not _jit._EXT_CACHE


def test_publish_imports_only_stable_destination(fake_compiler, cache_workspace):
    output = cache_workspace / "stable-destination"
    with jit.cache(write_root=output):
        _load_dpf()
    key = next(iter(_jit._EXT_CACHE))
    assert fake_compiler["import"] == [(key, output / key / f"{key}.so")]


def test_ready_marker_lands_only_after_stable_import(fake_compiler, cache_workspace, monkeypatch):
    import_built = _jit._import_built
    observed = []

    def probe(name, path):
        directory = Path(path).parent
        observed.append((Path(path), (directory / "build_inputs.json").is_file()))
        return import_built(name, path)

    monkeypatch.setattr(_jit, "_import_built", probe)
    output = cache_workspace / "marker-timing"
    with jit.cache(write_root=output):
        _load_dpf()
    key = next(iter(_jit._EXT_CACHE))
    directory = output / key
    assert observed == [(directory / f"{key}.so", False)]
    assert (directory / "build_inputs.json").is_file()
    assert json.loads((directory / "build_inputs.json").read_text())["source"]


def test_failed_stable_import_publishes_nothing(fake_compiler, cache_workspace, monkeypatch):
    import_built = _jit._import_built

    def broken_import(name, path):
        import_built(name, path)
        raise ImportError("stable import failed")

    monkeypatch.setattr(_jit, "_import_built", broken_import)
    output = cache_workspace / "failed-import"
    with jit.cache(write_root=output):
        with pytest.raises(ImportError, match="stable import failed"):
            _load_dpf()
    assert not list(output.rglob("build_inputs.json"))
    assert not list(output.rglob("*.so"))
    assert not _jit._EXT_CACHE


def test_valid_winner_is_loaded_from_stable_path(fake_compiler, cache_workspace):
    source = cache_workspace / "winner-source"
    destination = cache_workspace / "winner-destination"
    with jit.cache(write_root=source):
        group.Bytes().precompile()
    key = next(iter(_jit._EXT_CACHE))
    # A committed artifact at the write root models a concurrent publisher
    # that won the publish lock while this process was still preparing.
    shutil.copytree(source / key, destination / key)
    before = (_snapshot(source / key), _snapshot(destination / key))
    _jit._EXT_CACHE.clear()
    fake_compiler["import"].clear()
    with jit.cache(read_roots=(source,), write_root=destination):
        group.Bytes().precompile()
    assert fake_compiler["import"] == [(key, source / key / f"{key}.so"),
                                       (key, destination / key / f"{key}.so")]
    assert (_snapshot(source / key), _snapshot(destination / key)) == before
    leftovers = [path for path in destination.glob(f".{key}.*")
                 if not path.name.endswith(".publish.lock")]
    assert not leftovers
    assert len(fake_compiler["build"]) == 1


def test_winner_import_input_drift_is_not_remembered(fake_compiler, cache_workspace, monkeypatch):
    source = cache_workspace / "winner-drift-source"
    destination = cache_workspace / "winner-drift-destination"
    with jit.cache(write_root=source):
        group.Bytes().precompile()
    key = next(iter(_jit._EXT_CACHE))
    shutil.copytree(source / key, destination / key)
    before = _snapshot(destination / key)
    _jit._EXT_CACHE.clear()
    import_built = _jit._import_built

    def changing_import(name, path):
        module = import_built(name, path)
        if Path(path) == destination / key / f"{key}.so":
            (Path(_jit._find_include_dir()) / "fss" / "probe.cuh").write_text(
                "// changed during winner import\n")
        return module

    monkeypatch.setattr(_jit, "_import_built", changing_import)
    with jit.cache(read_roots=(source,), write_root=destination):
        with pytest.raises(RuntimeError, match="inputs changed"):
            group.Bytes().precompile()
    assert fake_compiler["import"][-1] == (key, destination / key / f"{key}.so")
    assert _snapshot(destination / key) == before
    leftovers = [path for path in destination.glob(f".{key}.*")
                 if not path.name.endswith(".publish.lock")]
    assert not leftovers
    assert not _jit._EXT_CACHE


def test_replacement_failure_preserves_published_artifact(fake_compiler, cache_workspace, monkeypatch):
    output = cache_workspace / "replacement-failure"
    with jit.cache(write_root=output):
        _load_dpf()
    key = next(iter(_jit._EXT_CACHE))
    before = _snapshot(output / key)
    _jit._EXT_CACHE.clear()

    def rejecting_validation(prepared, directory):
        if directory == output / key:
            raise ImportError("winner is corrupt")

    def broken_import(name, path):
        assert Path(path).is_file()
        raise ImportError("replacement import failed")

    monkeypatch.setattr(_jit, "_validate_artifact", rejecting_validation)
    monkeypatch.setattr(_jit, "_import_built", broken_import)
    with jit.cache(write_root=output):
        with pytest.raises(ImportError, match="replacement import failed"):
            _load_dpf()
    assert _snapshot(output / key) == before
    leftovers = [path for path in output.glob(f".{key}.*")
                 if not path.name.endswith(".publish.lock")]
    assert not leftovers
    assert not _jit._EXT_CACHE



def test_explicit_precompile_revalidates_warm_binary(fake_compiler, cache_workspace, monkeypatch):
    root = cache_workspace / "warm-corruption"
    config = group.Bytes()
    with jit.cache(write_root=root):
        config.precompile()
    binary = next(root.glob("*/*.so"))
    replacement = root / "replacement.so"
    replacement.write_bytes(b"invalid binary")
    replacement.replace(binary)

    def validate(prepared, directory):
        if (directory / (prepared.key + ".so")).read_bytes() == b"invalid binary":
            raise ImportError("corrupt binary")

    monkeypatch.setattr(_jit, "_validate_artifact", validate)
    with jit.cache(read_roots=(root,), write_root=None):
        with pytest.raises(RuntimeError, match="read-only cache"):
            config.precompile()
    with jit.cache(write_root=root):
        config.precompile()
    assert binary.read_bytes() == b"fake extension"
    assert len(fake_compiler["build"]) == 2


def test_validation_worker_failure_is_not_a_cache_miss(monkeypatch):
    prepared = SimpleNamespace(key="test_key", manifest_text=json.dumps({"torch": {}}))
    monkeypatch.setattr(_jit.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=9, stdout="worker crashed"))
    with pytest.raises(RuntimeError, match="worker failed"):
        _jit._validate_artifact(prepared, Path("unused"))


def test_validation_worker_preserves_native_initialization_error(monkeypatch):
    status = {"kind": "error", "type": "ValueError", "module": "builtins",
              "args": ["native initialization failed"]}
    monkeypatch.setattr(_jit.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=1, stdout="FSS_ARTIFACT_VALIDATION=" + json.dumps(status)))
    with pytest.raises(ValueError, match="native initialization failed") as error:
        _jit._validate_artifact(SimpleNamespace(key="test_key", manifest_text=json.dumps({"torch": {}})), Path("unused"))
    assert error.value.__cause__ is not None


def test_ordinary_warm_hit_has_no_worker_or_cache_write(fake_compiler, cache_workspace, monkeypatch):
    with jit.cache(write_root=cache_workspace / "ordinary"):
        first = _load_dpf()

        def forbidden(*args, **kwargs):
            pytest.fail("ordinary warm hit invoked a worker or cache write")

        monkeypatch.setattr(_jit, "_validate_artifact", forbidden)
        monkeypatch.setattr(_jit, "_build_worker", forbidden)
        monkeypatch.setattr(_jit.subprocess, "run", forbidden)
        monkeypatch.setattr(Path, "mkdir", forbidden)
        monkeypatch.setattr(Path, "write_text", forbidden)
        monkeypatch.setattr(Path, "write_bytes", forbidden)
        assert _load_dpf() is first



def test_validation_worker_crash_after_result_is_not_a_miss(monkeypatch):
    status = {"kind": "miss", "type": "ImportError", "module": "builtins", "args": ["bad file"]}
    monkeypatch.setattr(_jit.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=-11, stdout="FSS_ARTIFACT_VALIDATION=" + json.dumps(status)))
    prepared = SimpleNamespace(key="test_key", manifest_text=json.dumps({"torch": {}}))
    with pytest.raises(RuntimeError, match="worker failed"):
        _jit._validate_artifact(prepared, Path("unused"))


def test_worker_rejects_different_torch_identity():
    from fss_crypto import _jit_worker

    with pytest.raises(RuntimeError, match="Torch identity differs"):
        _jit_worker._check_torch_identity({"version": "different"})



@pytest.mark.parametrize("future", [False, True])
def test_cache_rejects_inode_aliases(cache_workspace, monkeypatch, future):
    actual = cache_workspace / "actual"
    actual.mkdir()
    alias = cache_workspace / "inode-alias"
    alias.symlink_to(actual, target_is_directory=True)
    # Preserve distinct spellings to model aliases such as bind mounts.
    monkeypatch.setattr(Path, "resolve", lambda path: path.absolute())
    reader = alias / "future" if future else alias
    writer = actual / "future" / "plugins" if future else actual
    with pytest.raises(ValueError, match="overlap"):
        with jit.cache(read_roots=(reader,), write_root=writer):
            pass


def test_cache_allows_separate_missing_siblings(cache_workspace):
    with jit.cache(read_roots=(cache_workspace / "future" / "reader",),
                   write_root=cache_workspace / "future" / "writer"):
        assert jit._current_locations().write_root.name == "writer"
    assert not (cache_workspace / "future").exists()



def test_cache_rejects_missing_case_aliases(cache_workspace, monkeypatch):
    actual = cache_workspace / "actual"
    actual.mkdir()
    alias = cache_workspace / "inode-alias"
    alias.symlink_to(actual, target_is_directory=True)
    monkeypatch.setattr(Path, "resolve", lambda path: path.absolute())
    monkeypatch.setattr(jit, "_case_sensitive", lambda path: False)
    with pytest.raises(ValueError, match="overlap"):
        with jit.cache(read_roots=(alias / "Future",),
                       write_root=actual / "future" / "plugins"):
            pass


def test_cache_preserves_case_sensitive_siblings(cache_workspace, monkeypatch):
    monkeypatch.setattr(jit, "_case_sensitive", lambda path: True)
    with jit.cache(read_roots=(cache_workspace / "Future",),
                   write_root=cache_workspace / "future"):
        assert jit._current_locations().write_root.name == "future"



def test_cache_rejects_missing_darwin_unicode_aliases(cache_workspace, monkeypatch):
    actual = cache_workspace / "actual"
    actual.mkdir()
    alias = cache_workspace / "inode-alias"
    alias.symlink_to(actual, target_is_directory=True)
    monkeypatch.setattr(Path, "resolve", lambda path: path.absolute())
    monkeypatch.setattr(jit, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(jit, "_case_sensitive", lambda path: False)
    with pytest.raises(ValueError, match="overlap"):
        with jit.cache(read_roots=(alias / "Café",),
                       write_root=actual / "Café" / "plugins"):
            pass


def test_cache_preserves_linux_unicode_siblings(cache_workspace, monkeypatch):
    monkeypatch.setattr(jit, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(jit, "_case_sensitive", lambda path: True)
    with jit.cache(read_roots=(cache_workspace / "Café",),
                   write_root=cache_workspace / "Café"):
        assert jit._current_locations().write_root.name == "Café"

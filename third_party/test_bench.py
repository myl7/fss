"""Tests for benchmark data and host-state safety. No benchmark tools required."""

import sys

sys.dont_write_bytecode = True

import importlib.util
import json
from pathlib import Path
import argparse
import tempfile
import signal
import subprocess
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("bench", Path(__file__).with_name("bench.py"))
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class BenchmarkResultsTest(unittest.TestCase):
    def setUp(self):
        (bench.ROOT / "build").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=bench.ROOT / "build")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def google(self, values, library="fss", device="cpu", status="ok"):
        data = self.directory / f"{library}-{device}"
        data.mkdir(exist_ok=True)
        (data / "raw.json").write_text(json.dumps({"benchmarks": values}))
        entry = {"library": library, "device": device, "framework": "google",
                 "command": ["--benchmark_out=/old/host/raw.json"],
                 "status": status, "log": "raw.log"}
        if status != "ok":
            entry["error"] = "process exited with status 1"
        return entry, data

    def value(self, unit="ns", **extra):
        return dict(name="fss/CPU/DPF/Gen", real_time=2, cpu_time=9,
                    time_unit=unit, **extra)

    def test_units_and_cuda_wall_time(self):
        for unit, multiplier in (("ns", 1), ("us", 1000), ("ms", 10**6), ("s", 10**9)):
            entry, data = self.google([self.value(unit)], device="gpu")
            rows, errors = bench.result_rows(entry, data)
            self.assertEqual(errors, [])
            self.assertEqual(rows[0][4], 2 * multiplier)
            self.assertEqual(rows[0][5], 2 * multiplier / (1 << 20))

    def test_native_throughput_does_not_change_exact_domain_work(self):
        value = dict(name="fss/CPU/DPF/EvalAll", real_time=1000, cpu_time=900,
                     time_unit="ns", items_per_second=(1 << 20) * 1e9 / 900)
        entry, data = self.google([value])
        rows, errors = bench.result_rows(entry, data)
        self.assertEqual(errors, [])
        self.assertEqual(rows[0][6], 1 << 20)
        self.assertEqual(rows[0][7], value["items_per_second"])
        self.assertEqual(rows[0][14:], [900, "cpu_time", value["items_per_second"]])

    def test_median_aggregate_is_used_once(self):
        values = [self.value(), self.value(aggregate_name="mean"),
                  self.value(aggregate_name="median")]
        entry, data = self.google(values)
        rows, errors = bench.result_rows(entry, data)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][8], "median")
        self.assertEqual(errors, [])

    def test_empty_filter_and_skip_with_error_fail(self):
        for values in ([], [self.value(error_occurred=True, error_message="cuda allocation failed")]):
            entry, data = self.google(values)
            rows, errors = bench.result_rows(entry, data)
            self.assertEqual(rows, [])
            self.assertTrue(errors)

    def test_failed_process_preserves_partial_results_and_failure(self):
        entry, _ = self.google([self.value()], status="failed")
        (self.directory / "run.json").write_text(json.dumps({"commands": [entry]}))
        self.assertEqual(bench.summarize(self.directory), 1)
        summary = json.loads((self.directory / "summary.json").read_text())
        manifest = json.loads((self.directory / "run.json").read_text())
        self.assertEqual(len(summary), 1)
        self.assertEqual(manifest["commands"][0]["status"], "failed")

    def test_zero_exit_without_results_is_failure(self):
        entry, _ = self.google([])
        (self.directory / "run.json").write_text(json.dumps({"commands": [entry]}))
        self.assertEqual(bench.summarize(self.directory), 1)
        manifest = json.loads((self.directory / "run.json").read_text())
        self.assertEqual(manifest["commands"][0]["status"], "failed")

    def test_invalid_throughput_does_not_produce_successful_rows(self):
        for throughput in (0, -1, float("nan"), float("inf")):
            entry, data = self.google([self.value(items_per_second=throughput)])
            with self.assertRaises(ValueError):
                bench.result_rows(entry, data)

    def test_criterion_median_and_domain_units(self):
        data = self.directory / "rust"
        raw = data / "criterion" / "bench" / "new"
        raw.mkdir(parents=True)
        (raw / "benchmark.json").write_text(json.dumps({"full_id": "libdpf/CPU/DPF/EvalAll"}))
        (raw / "estimates.json").write_text(json.dumps({"median": {"point_estimate": 1000}}))
        rows, errors = bench.result_rows({"library": "libdpf", "device": "cpu", "framework": "criterion"}, data)
        self.assertEqual(errors, [])
        self.assertEqual(rows[0][6], 1 << 20)
        self.assertEqual(rows[0][4], 1000)

    def test_python_batch_and_time(self):
        (self.directory / "python.json").write_text(json.dumps({"benchmarks": [
            {"name": "GPU-DPF/GPU/Eval", "time_ns": 512000, "batch": 512}]}))
        rows, errors = bench.result_rows({"library": "gpu_dpf", "device": "gpu", "framework": "python"}, self.directory)
        self.assertEqual(errors, [])
        self.assertEqual(rows[0][5], 1000)

    def test_malformed_and_zero_time_are_failures(self):
        entry, _ = self.google([self.value()])
        data = self.directory / "fss-cpu"
        for content in ('invalid json', json.dumps({"benchmarks": [dict(name="zero", real_time=0, time_unit="ns")]})):
            (data / "raw.json").write_text(content)
            (self.directory / "run.json").write_text(json.dumps({"commands": [entry]}))
            self.assertEqual(bench.summarize(self.directory), 1)


class RunnerSelectionTest(unittest.TestCase):
    def test_cli_flags_override_environment_and_keep_safe_defaults(self):
        with mock.patch.dict(bench.os.environ, {"CPU_ID": "2", "GPU_ID": "3", "JOBS": "2", "CUDA_ARCH": "86"}), \
             mock.patch.object(bench.sys, "argv", ["bench.py", "run", "--libraries", "main", "--platform", "gpu", "--gpu", "1"]), \
             mock.patch.object(bench, "run", return_value=0) as run:
            self.assertEqual(bench.main(), 0)
        args, selected = run.call_args.args
        self.assertEqual((args.cpu, args.gpu, args.jobs, args.cuda_arch), (2, "1", 2, "86"))
        self.assertEqual((args.governor, args.repetitions), ("performance", 5))
        self.assertEqual(selected, [("main", "gpu")])

    def test_environment_limits_threads_and_pins_rust_build_output(self):
        args = argparse.Namespace(gpu="1", jobs=4, cuda_arch="86")
        with mock.patch.dict(bench.os.environ, {"OMP_NUM_THREADS": "8", "RAYON_NUM_THREADS": "16"}):
            env = bench.environment(args, "libdpf")
        self.assertEqual((env["OMP_NUM_THREADS"], env["RAYON_NUM_THREADS"]), ("1", "1"))
        self.assertEqual(env["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertEqual(env["CARGO_TARGET_DIR"], str(bench.BUILD / "libdpf" / "target"))
        self.assertEqual(env["TORCH_CUDA_ARCH_LIST"], "8.6")
        self.assertEqual(bench.RUST, {"libdpf": "nightly-2025-09-01", "fss_v060": "nightly-2025-09-01"})

    def test_bazel_probe_and_build_share_pinned_isolated_environment(self):
        args = argparse.Namespace(libraries="google_dpf", platform="cpu", gpu="0", jobs=4,
                                  cuda_arch=None, cpu=0, smoke=True, repetitions=1, min_time=.01)
        with mock.patch.object(bench, "capture", return_value="test probe") as capture:
            bench.metadata(args, "performance")
        call = next(call for call in capture.call_args_list if call.args[0] == ["bazel", "--version"])
        self.assertEqual(call.kwargs["cwd"], bench.source("google_dpf"))
        env = call.kwargs["env"]
        self.assertEqual(env["USE_BAZEL_VERSION"], "7.6.1")
        self.assertEqual(Path(env["BAZELISK_HOME"]), bench.BUILD / "google_dpf" / "bazelisk")
        with mock.patch.dict(bench.os.environ, {"CPLUS_INCLUDE_PATH": "/openssl", "C_INCLUDE_PATH": "/openssl", "CPATH": "/openssl"}):
            bazel_env = bench.environment(args, "google_dpf")
            main_env = bench.environment(args, "main")
        for key in ("CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH", "CPATH"):
            self.assertNotIn(key, bazel_env)
            self.assertEqual(main_env[key], "/openssl")
        if "USE_BAZEL_VERSION" not in bench.os.environ:
            self.assertNotIn("USE_BAZEL_VERSION", bench.environment(args, "main"))

    def test_source_hashes_include_harness_configuration_and_patches(self):
        (bench.ROOT / "build").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=bench.ROOT / "build") as temp:
            root = Path(temp)
            relative = ["third_party/fss/bench.cu", "third_party/fss/CMakeLists.txt",
                        "third_party/fss/patches/fix.patch", "third_party/fss/check.h",
                        "third_party/torchcsprng/torchcsprng/aes128_mmo_soft.cuh", "include/fss/dpf.cuh"]
            for name in relative:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name)
            args = argparse.Namespace(libraries="fss,torchcsprng", platform="cpu")
            with mock.patch.object(bench, "ROOT", root):
                hashes = bench.benchmark_source_hashes(args)
                original = hashes[relative[0]]
                (root / relative[0]).write_text("changed harness")
                changed = bench.benchmark_source_hashes(args)
            self.assertEqual(set(hashes), set(relative))
            self.assertNotEqual(original, changed[relative[0]])

    def test_cargo_does_not_receive_duplicate_criterion_bench_flag(self):
        (bench.ROOT / "build").mkdir(exist_ok=True)
        args = argparse.Namespace(gpu="0", jobs=2, cuda_arch=None, cpu=16,
                                  filter=None, smoke=True, min_time=.01,
                                  repetitions=1, native_args=[])
        for library in ("libdpf", "fss_v060"):
            commands = []
            with tempfile.TemporaryDirectory(dir=bench.ROOT / "build") as temp, \
                 mock.patch.object(bench, "BUILD", Path(temp) / "build"), \
                 mock.patch.object(bench, "execute"), \
                 mock.patch.object(bench, "result_rows", return_value=([[]], [])):
                bench.record(args, library, "cpu", Path(temp), commands)
            command = commands[0]["command"]
            self.assertEqual(command.count("--bench"), 1)
            self.assertNotIn("--bench", command[command.index("--") + 1:])

    def test_no_build_metadata_records_actual_configured_architecture(self):
        (bench.ROOT / "build").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=bench.ROOT / "build") as temp:
            root = Path(temp)
            directory = root / "build" / "main" / "cmake"
            directory.mkdir(parents=True)
            (directory / "CMakeCache.txt").write_text(
                "CMAKE_CUDA_ARCHITECTURES:STRING=120\nCMAKE_BUILD_TYPE:STRING=Release\n"
                "CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/g++\nUNRELATED:STRING=ignored\n")
            with mock.patch.object(bench, "ROOT", root), mock.patch.object(bench, "BUILD", root / "build"):
                config = bench.build_configuration([("main", "gpu")])
            recorded = config["build/main/cmake/CMakeCache.txt"]
            self.assertEqual(recorded["CMAKE_CUDA_ARCHITECTURES"], "120")
            self.assertEqual(recorded["CMAKE_BUILD_TYPE"], "Release")
            self.assertNotIn("UNRELATED", recorded)

    def test_original_library_is_opt_in(self):
        selected = bench.selection(argparse.Namespace(libraries="all", platform="all"))
        self.assertNotIn(("main", "cpu"), selected)
        self.assertIn(("torchcsprng", "gpu"), selected)
        with self.assertRaises(RuntimeError):
            bench.selection(argparse.Namespace(libraries="libdpf", platform="gpu"))

    def test_smoke_matches_raw_aes_and_excludes_eval_all(self):
        (bench.ROOT / "build").mkdir(exist_ok=True)
        args = argparse.Namespace(gpu="0", jobs=1, cuda_arch=None, cpu=0,
                                  filter=None, smoke=True, min_time=.01,
                                  repetitions=1, native_args=[])
        commands = []
        with tempfile.TemporaryDirectory(dir=bench.ROOT / "build") as temp, \
             mock.patch.object(bench, "execute"), \
             mock.patch.object(bench, "result_rows", return_value=([[]], [])):
            bench.record(args, "torchcsprng", "cpu", Path(temp), commands)
            pattern = next(value.split("=", 1)[1] for value in commands[0]["command"]
                           if value.startswith("--benchmark_filter="))
            self.assertIsNotNone(bench.re.search(pattern, "torchcsprng/CPU/AesSoft"))
            commands.clear()
            bench.record(args, "main", "cpu", Path(temp), commands)
            pattern = next(value.split("=", 1)[1] for value in commands[0]["command"]
                           if value.startswith("--benchmark_filter="))
            self.assertIsNone(bench.re.search(pattern, "BM_DpfEvalAll_Uint_Aes/20"))
            self.assertIsNotNone(bench.re.search(pattern, "BM_DpfEval_Uint_Aes/20"))

    def test_historic_targets_are_split_by_group_and_scheme(self):
        args = argparse.Namespace(gpu="0", jobs=2, cuda_arch="120")
        with mock.patch.object(bench, "cmake") as cmake:
            bench.build(args, [("fss_v070", "cpu")])
        targets = cmake.call_args.args[3]
        self.assertEqual(set(targets), {f"bench_cpu_{group}_{scheme}"
                                       for group in ("uint", "bytes") for scheme in ("dpf", "dcf")})
        self.assertEqual(cmake.call_args.args[4], ["-DFSS070_BUILD_GPU=OFF"])
        self.assertEqual({path.name for path, _ in bench.binaries("fss_v070", "gpu", args)},
                         {f"bench_gpu_{group}_{scheme}"
                          for group in ("uint", "bytes") for scheme in ("dpf", "dcf")})

    def test_historic_cpu_scheme_filter_skips_unrelated_binaries(self):
        (bench.ROOT / "build").mkdir(exist_ok=True)
        args = argparse.Namespace(gpu="0", jobs=2, cuda_arch=None, cpu=16,
                                  filter="/DPF/", smoke=False, min_time=.01,
                                  repetitions=1, native_args=[])
        commands = []
        with tempfile.TemporaryDirectory(dir=bench.ROOT / "build") as temp, \
             mock.patch.object(bench, "execute"), \
             mock.patch.object(bench, "result_rows", return_value=([[]], [])):
            bench.record(args, "fss_v070", "cpu", Path(temp), commands)
        self.assertEqual(len(commands), 2)
        self.assertTrue(all(Path(entry["command"][3]).name.endswith("_dpf") for entry in commands))

    def test_gpu_failure_does_not_block_bytes_variant(self):
        (bench.ROOT / "build").mkdir(exist_ok=True)
        args = argparse.Namespace(gpu="0", jobs=1, cuda_arch=None, cpu=0,
                                  filter=None, smoke=True, min_time=.01,
                                  repetitions=1, native_args=[])
        commands = []
        def execute(command, **kwargs):
            if "bench_gpu_uint" in " ".join(map(str, command)):
                raise RuntimeError("uint process failed")
        with tempfile.TemporaryDirectory(dir=bench.ROOT / "build") as temp, \
             mock.patch.object(bench, "execute", side_effect=execute), \
             mock.patch.object(bench, "result_rows", return_value=([[]], [])):
            bench.record(args, "fss_v070", "gpu", Path(temp), commands)
        self.assertEqual(len(commands), 8)
        self.assertEqual([c["status"] for c in commands], ["failed"] * 4 + ["ok"] * 4)
        for entry in commands:
            pattern = next(value.split("=", 1)[1] for value in entry["command"]
                           if value.startswith("--benchmark_filter="))
            binary = Path(entry["command"][3]).name
            group, scheme = binary.split("_")[-2:]
            self.assertIn(f"/{group}/{scheme.upper()}/", pattern)
            if "/DPF/Gen" in pattern:
                self.assertIsNotNone(bench.re.search(pattern, "fss-v0.7.0/GPU/" +
                                     ("uint" if entry["status"] == "failed" else "bytes") +
                                     "/DPF/Gen/manual_time"))


class GovernorTest(unittest.TestCase):
    def lifecycle(self, interruption=None):
        state = {"value": "powersave"}
        writes = []
        def write(_, text):
            state["value"] = text.strip()
            writes.append(state["value"])
        with mock.patch.object(Path, "exists", return_value=True), \
             mock.patch.object(Path, "read_text", side_effect=lambda: state["value"]), \
             mock.patch.object(Path, "write_text", autospec=True, side_effect=write), \
             mock.patch.object(bench.os, "access", return_value=True):
            try:
                with bench.governor(0, "performance") as active:
                    self.assertEqual(active, "performance")
                    if interruption:
                        raise interruption
            except BaseException as error:
                if interruption is None or type(error) != type(interruption):
                    raise
        self.assertEqual(state["value"], "powersave")
        self.assertEqual(writes, ["performance", "powersave"])

    def test_restore_after_success_error_and_signals(self):
        for interruption in (None, RuntimeError("failed benchmark"), KeyboardInterrupt(), SystemExit(143)):
            with self.subTest(interruption=interruption):
                self.lifecycle(interruption)

    def test_missing_governor_requires_explicit_keep(self):
        with mock.patch.object(Path, "exists", return_value=False):
            with self.assertRaises(RuntimeError):
                with bench.governor(0, "performance"):
                    self.fail("must not time benchmarks")
            with bench.governor(0, "keep") as active:
                self.assertEqual(active, "unavailable")

    def test_failed_governor_change_restores_original(self):
        state = {"value": "powersave"}
        def write(_, text):
            state["value"] = text.strip()
            if state["value"] == "performance":
                raise OSError("write completed before error")
        with mock.patch.object(Path, "exists", return_value=True), \
             mock.patch.object(Path, "read_text", side_effect=lambda: state["value"]), \
             mock.patch.object(Path, "write_text", autospec=True, side_effect=write), \
             mock.patch.object(bench.os, "access", return_value=True):
            with self.assertRaises(OSError):
                with bench.governor(0, "performance"):
                    self.fail("must not time benchmarks")
        self.assertEqual(state["value"], "powersave")

    def test_actual_sigterm_restores_governor(self):
        (bench.ROOT / "build").mkdir(exist_ok=True)
        script = """
import importlib.util
from pathlib import Path
import signal
import sys
spec = importlib.util.spec_from_file_location("bench", sys.argv[1])
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
bench.Path = lambda _: Path(sys.argv[2])
signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
with bench.governor(0, "performance"):
    print("ready", flush=True)
    signal.pause()
"""
        with tempfile.TemporaryDirectory(dir=bench.ROOT / "build") as temp:
            path = Path(temp) / "governor"
            path.write_text("powersave\n")
            process = subprocess.Popen([sys.executable, "-B", "-c", script, str(Path(bench.__file__)), str(path)],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                       env={**bench.os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
            try:
                self.assertEqual(process.stdout.readline().strip(), "ready")
                self.assertEqual(path.read_text().strip(), "performance")
                process.send_signal(signal.SIGTERM)
                process.communicate(timeout=5)
                self.assertEqual(process.returncode, 143)
                self.assertEqual(path.read_text().strip(), "powersave")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()

    def test_process_failure_cannot_succeed(self):
        (bench.ROOT / "build").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=bench.ROOT / "build") as temp:
            with self.assertRaises(RuntimeError):
                bench.execute(["python3", "-c", "raise SystemExit(7)"], log=Path(temp) / "process.log")


if __name__ == "__main__":
    unittest.main()

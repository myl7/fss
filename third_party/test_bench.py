"""Tests for benchmark data and host-state safety. No benchmark tools required."""

import sys

sys.dont_write_bytecode = True

import importlib.util
import csv
import json
from pathlib import Path
import argparse
import contextlib
import io
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

    def v1(self, **extra):
        value = dict(name="BM_DpfEvalAllGpu_Uint_ChaCha/20", real_time=2, cpu_time=9,
                     time_unit="ns", work_schema_version=1, batch_keys=1024,
                     calls_per_iteration=1)
        value.update(extra)
        return value

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
        self.assertEqual(rows[0][14:17], [900, "cpu_time", value["items_per_second"]])
        self.assertEqual(rows[0][17:], ["", "", "", "", "", "", "legacy"])

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

    def test_schema_v1_full_domain_row_normalizes_from_per_iteration_counters(self):
        for launches, keys in ((1, 1024), (1024, 1024 << 10)):
            value = self.v1(real_time=1_000_000, keys_per_iteration=keys, queries_per_iteration=0,
                            logical_outputs_per_iteration=keys << 20, output_bytes_per_iteration=1 << 24,
                            batch_keys=1024, calls_per_iteration=launches,
                            host_wall_ns_per_iteration=900_000,
                            items_per_second=1e9 * (keys << 20) / 1_000_000)
            entry, data = self.google([value], library="main", device="gpu")
            entry["full_launches"] = launches
            rows, errors = bench.result_rows(entry, data)
            self.assertEqual(errors, [])
            self.assertEqual(rows[0][3], 1024)
            self.assertEqual(rows[0][5], 1_000_000 / keys)
            self.assertEqual(rows[0][6], keys << 20)
            self.assertEqual(rows[0][13], "domain_outputs")
            self.assertEqual(rows[0][7], value["items_per_second"])
            self.assertEqual(rows[0][17:], [keys << 20, 1 << 24, "", "", "", 900_000, "v1"])

    def test_schema_v1_executed_counters_must_match_the_plan(self):
        entry, data = self.google([self.v1(keys_per_iteration=1024 << 10,
                                           logical_outputs_per_iteration=(1024 << 10) << 20,
                                           batch_keys=1024, calls_per_iteration=1024)],
                                  library="main", device="gpu")
        entry["full_launches"] = 1
        with self.assertRaisesRegex(ValueError, "calls_per_iteration"):
            bench.result_rows(entry, data)
        entry, data = self.google([self.v1(keys_per_iteration=1024,
                                           logical_outputs_per_iteration=1024 << 20,
                                           batch_keys=512, calls_per_iteration=1)],
                                  library="main", device="gpu")
        entry["full_launches"] = 1
        with self.assertRaisesRegex(ValueError, "batch_keys"):
            bench.result_rows(entry, data)

    def test_schema_v1_median_is_not_rescaled_by_aggregate_iterations(self):
        counters = dict(keys_per_iteration=1024, logical_outputs_per_iteration=1024 << 20)
        values = [self.v1(real_time=2000, aggregate_name="median", iterations=4096, **counters),
                  self.v1(real_time=1900, **counters), self.v1(real_time=2100, **counters)]
        entry, data = self.google(values, library="main", device="gpu")
        entry["full_launches"] = 1
        rows, errors = bench.result_rows(entry, data)
        self.assertEqual(errors, [])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][8], "median")
        self.assertEqual(rows[0][5], 2000 / 1024)

    def test_schema_v1_median_requires_matching_repetition_work(self):
        values = [self.v1(real_time=2000, aggregate_name="median", keys_per_iteration=1024,
                          logical_outputs_per_iteration=1024 << 20),
                  self.v1(real_time=1900, keys_per_iteration=1024,
                          logical_outputs_per_iteration=1024 << 20),
                  self.v1(real_time=2100, keys_per_iteration=2048,
                          logical_outputs_per_iteration=2048 << 20)]
        entry, data = self.google(values, library="main", device="gpu")
        entry["full_launches"] = 1
        with self.assertRaisesRegex(ValueError, "repetitions"):
            bench.result_rows(entry, data)

    def test_schema_v1_packed_row_keeps_logical_outputs_and_packed_bytes(self):
        value = self.v1(name="BM_PackedHalfTreeDpfEvalAllGpu_OwnedAesCcr/20", real_time=500,
                        keys_per_iteration=1, logical_outputs_per_iteration=1 << 20,
                        output_bytes_per_iteration=1 << 17,
                        items_per_second=1e9 * (1 << 20) / 500)
        entry, data = self.google([value], library="main", device="gpu")
        rows, errors = bench.result_rows(entry, data)
        self.assertEqual(errors, [])
        self.assertEqual(rows[0][6], 1 << 20)
        self.assertEqual(rows[0][17:], [1 << 20, 1 << 17, "", "", "", "", "v1"])
        self.assertEqual(rows[0][7], value["items_per_second"])

    def test_schema_v1_rejects_missing_fields_bad_versions_and_conflicting_rates(self):
        entry, data = self.google([self.v1(work_schema_version=2)], library="main", device="gpu")
        with self.assertRaisesRegex(ValueError, "work schema version"):
            bench.result_rows(entry, data)
        entry, data = self.google([self.v1()], library="main", device="gpu")
        with self.assertRaisesRegex(ValueError, "work counters"):
            bench.result_rows(entry, data)
        missing = self.v1(keys_per_iteration=8, logical_outputs_per_iteration=8 << 20)
        del missing["real_time"]
        entry, data = self.google([missing], library="main", device="gpu")
        with self.assertRaisesRegex(ValueError, "real_time"):
            bench.result_rows(entry, data)
        entry, data = self.google([self.v1(real_time=100, keys_per_iteration=8,
                                           logical_outputs_per_iteration=8 << 20,
                                           items_per_second=1.0)], library="main", device="gpu")
        with self.assertRaisesRegex(ValueError, "conflicts"):
            bench.result_rows(entry, data)

    def test_receipts_without_schema_marker_keep_legacy_reading(self):
        entry, data = self.google([self.value()], library="main", device="cpu")
        rows, errors = bench.result_rows(entry, data)
        self.assertEqual(errors, [])
        self.assertEqual(rows[0][13], "keys_or_prg_calls")
        self.assertEqual(rows[0][17:], ["", "", "", "", "", "", "legacy"])

    def test_proof_row_counts_proofs_and_keeps_separate_proof_bytes(self):
        value = dict(name="BM_VdpfProve_Uint_ChaCha_Blake3/20", real_time=194.94876043818238,
                     cpu_time=194.95026761503186, time_unit="ns", work_schema_version=1.0,
                     api_calls_per_iteration=1.0, keys_per_iteration=0.0,
                     queries_per_iteration=0.0, logical_outputs_per_iteration=0.0,
                     output_bytes_per_iteration=0.0, proofs_per_iteration=1.0,
                     proof_bytes_per_iteration=64.0)
        entry, data = self.google([value], library="main", device="cpu")
        entry["full_launches"] = 1
        entry["required_work_schema"] = 1
        rows, errors = bench.result_rows(entry, data)
        self.assertEqual(errors, [])
        self.assertEqual((rows[0][6], rows[0][13]), (1, "proofs"))
        self.assertEqual(rows[0][5], "")
        self.assertEqual(rows[0][17:], [0, 0, 1, 64, "", "", "v1"])
        (self.directory / "run.json").write_text(json.dumps({"commands": [entry]}))
        self.assertEqual(bench.summarize(self.directory), 0)
        summary = json.loads((self.directory / "summary.json").read_text())[0]
        self.assertEqual((summary["logical_outputs_per_iteration"],
                          summary["output_bytes_per_iteration"],
                          summary["proofs_per_iteration"], summary["proof_bytes_per_iteration"]),
                         (0, 0, 1, 64))
        with (self.directory / "summary.csv").open() as stream:
            row = next(row for row in csv.DictReader(stream) if "VdpfProve" in row["benchmark"])
        self.assertEqual((row["proofs_per_iteration"], row["proof_bytes_per_iteration"],
                          row["output_bytes_per_iteration"]), ("1", "64", "0"))
        line = next(l for l in (self.directory / "summary.md").read_text().splitlines()
                    if "VdpfProve" in l)
        self.assertIn("|  | 1 | proofs | ", line)

    def test_preprocess_row_counts_prepared_leaves_per_key(self):
        value = dict(name="BM_GrottoDcfPreprocess_Aes/20", real_time=73431249.94076788,
                     cpu_time=73432564.0, time_unit="ns", work_schema_version=1.0,
                     api_calls_per_iteration=1.0, keys_per_iteration=1.0,
                     queries_per_iteration=0.0, logical_outputs_per_iteration=0.0,
                     output_bytes_per_iteration=0.0, prepared_leaves_per_iteration=1048576.0)
        entry, data = self.google([value], library="main", device="cpu")
        rows, errors = bench.result_rows(entry, data)
        self.assertEqual(errors, [])
        self.assertEqual((rows[0][6], rows[0][13]), (1048576, "prepared_leaves"))
        self.assertEqual(rows[0][3], 1)
        self.assertAlmostEqual(rows[0][5], 73431249.94076788)
        self.assertAlmostEqual(rows[0][7], 1e9 * 1048576 / 73431249.94076788, places=3)
        (self.directory / "run.json").write_text(json.dumps({"commands": [entry]}))
        self.assertEqual(bench.summarize(self.directory), 0)
        line = next(l for l in (self.directory / "summary.md").read_text().splitlines()
                    if "Preprocess_Aes" in l)
        self.assertIn("| 1 | 1048576 | prepared_leaves | ", line)

    def test_vdpf_point_row_preserves_point_hash_bytes_separately(self):
        point = dict(name="BM_VdpfEval_Uint_ChaCha_Blake3/20/manual_time", real_time=1000,
                     cpu_time=2000, time_unit="ns", work_schema_version=1,
                     keys_per_iteration=1024, queries_per_iteration=1024,
                     logical_outputs_per_iteration=1024, output_bytes_per_iteration=1024 * 16,
                     point_hash_bytes_per_iteration=1024 * 64)
        entry, data = self.google([point], library="main", device="gpu")
        entry["full_launches"] = 1
        entry["required_work_schema"] = 1
        rows, errors = bench.result_rows(entry, data)
        self.assertEqual(errors, [])
        self.assertEqual(rows[0][13], "domain_outputs")
        self.assertEqual(rows[0][17:], [1024, 1024 * 16, "", "", 1024 * 64, "", "v1"])
        for bad in (-1, 1.5, float("nan")):
            invalid = dict(point, point_hash_bytes_per_iteration=bad)
            entry, data = self.google([invalid], library="main", device="gpu")
            with self.assertRaisesRegex(ValueError, "invalid per-iteration counter"):
                bench.result_rows(entry, data)
        values = [point, dict(point, point_hash_bytes_per_iteration=1),
                  dict(point, aggregate_name="median")]
        entry, data = self.google(values, library="main", device="gpu")
        with self.assertRaisesRegex(ValueError, "repetitions"):
            bench.result_rows(entry, data)

    def test_new_gpu_vdpf_point_collections_require_point_hash_bytes(self):
        point = dict(name="BM_VdpfEval_Uint_ChaCha_Blake3/20/manual_time", real_time=1000,
                     cpu_time=2000, time_unit="ns", work_schema_version=1,
                     keys_per_iteration=1024, queries_per_iteration=1024,
                     logical_outputs_per_iteration=1024, output_bytes_per_iteration=1024 * 16)
        entry, data = self.google([point], library="main", device="gpu")
        entry["full_launches"] = 1
        entry["required_work_schema"] = 1
        with self.assertRaisesRegex(ValueError, "point_hash_bytes_per_iteration"):
            bench.result_rows(entry, data)
        del entry["required_work_schema"]
        rows, errors = bench.result_rows(entry, data)
        self.assertEqual(errors, [])
        self.assertEqual(rows[0][21], "")
        merkle = self.v1(name="BM_VdpfEvalAllMerkleGpu_Uint_ChaCha_Blake3/20", real_time=1000,
                         keys_per_iteration=1, queries_per_iteration=0,
                         logical_outputs_per_iteration=1 << 20)
        entry, data = self.google([merkle], library="main", device="gpu")
        entry["required_work_schema"] = 1
        rows, errors = bench.result_rows(entry, data)
        self.assertEqual(errors, [])

    def test_cpu_native_rate_is_validated_on_cpu_time(self):
        value = dict(name="fss/CPU/DCF-bytes/EvalAll", real_time=127316862.11191118,
                     cpu_time=125900684.0, time_unit="ns", work_schema_version=1.0,
                     keys_per_iteration=1.0, queries_per_iteration=1048576.0,
                     logical_outputs_per_iteration=1048576.0,
                     output_bytes_per_iteration=16777216.0,
                     items_per_second=8328596.530897323)
        entry, data = self.google([value], library="fss", device="cpu")
        rows, errors = bench.result_rows(entry, data)
        self.assertEqual(errors, [])
        self.assertAlmostEqual(rows[0][7], 1e9 * 1048576 / 127316862.11191118, places=3)
        self.assertAlmostEqual(rows[0][16], 8328596.530897323, places=3)
        self.assertEqual(rows[0][15], "cpu_time")

    def test_median_native_rate_is_kept_without_inverting_median_time(self):
        counters = dict(keys_per_iteration=1024, logical_outputs_per_iteration=1024 << 20)
        values = [self.v1(real_time=2000, aggregate_name="median",
                          items_per_second=1.5 * (1024 << 20) * 1e9 / 2000, **counters),
                  self.v1(real_time=1900, items_per_second=(1024 << 20) * 1e9 / 1900, **counters),
                  self.v1(real_time=2100, items_per_second=(1024 << 20) * 1e9 / 2100, **counters)]
        entry, data = self.google(values, library="main", device="gpu")
        rows, errors = bench.result_rows(entry, data)
        self.assertEqual(errors, [])
        self.assertAlmostEqual(rows[0][7], (1024 << 20) * 1e9 / 2000, places=3)
        self.assertAlmostEqual(rows[0][16], 1.5 * (1024 << 20) * 1e9 / 2000, places=3)

    def test_v1_counters_must_be_finite_nonnegative_and_integral(self):
        for counters in (dict(queries_per_iteration=-2, logical_outputs_per_iteration=-5,
                              output_bytes_per_iteration=-7),
                         dict(keys_per_iteration=1.5), dict(keys_per_iteration=float("nan")),
                         dict(proofs_per_iteration=-1)):
            entry, data = self.google([self.v1(**counters)], library="main", device="gpu")
            with self.assertRaisesRegex(ValueError, "invalid per-iteration counter"):
                bench.result_rows(entry, data)

    def test_new_collections_require_work_schema_v1(self):
        legacy = dict(name="fss/CPU/DPF-bytes/Gen", real_time=2241.064285643592,
                      cpu_time=2241.1702026599114, time_unit="ns",
                      keys_per_iteration=1.0, work_schema_version=1.0)
        del legacy["work_schema_version"]
        entry, data = self.google([legacy], library="fss", device="cpu")
        entry["required_work_schema"] = 1
        with self.assertRaisesRegex(ValueError, "work_schema_version"):
            bench.result_rows(entry, data)
        del entry["required_work_schema"]
        rows, errors = bench.result_rows(entry, data)
        self.assertEqual(errors, [])
        self.assertEqual(rows[0][23], "legacy")

    def test_batch_full_domain_rows_require_planned_calls_and_batch(self):
        stripped = self.v1(keys_per_iteration=1024 << 10,
                           logical_outputs_per_iteration=(1024 << 10) << 20)
        del stripped["batch_keys"]
        del stripped["calls_per_iteration"]
        entry, data = self.google([stripped], library="main", device="gpu")
        entry["full_launches"] = 7
        with self.assertRaisesRegex(ValueError, "missing"):
            bench.result_rows(entry, data)
        for name in ("BM_GrottoDcfEvalAllGpu_ChaCha/20",
                     "BM_VdpfEvalAllMerkleGpu_Uint_ChaCha_Blake3/20"):
            single_key = self.v1(name=name, keys_per_iteration=1,
                                 logical_outputs_per_iteration=1 << 20)
            del single_key["batch_keys"]
            del single_key["calls_per_iteration"]
            entry, data = self.google([single_key], library="main", device="gpu")
            rows, errors = bench.result_rows(entry, data)
            self.assertEqual(errors, [])


class RunnerSelectionTest(unittest.TestCase):
    def test_cli_flags_override_environment_and_keep_safe_defaults(self):
        with mock.patch.dict(bench.os.environ, {"CPU_ID": "2", "GPU_ID": "3", "JOBS": "2", "CUDA_ARCH": "86"}), \
             mock.patch.object(bench.sys, "argv", ["bench.py", "run", "--libraries", "main", "--platform", "gpu", "--gpu", "1"]), \
             mock.patch.object(bench, "run", return_value=0) as run:
            self.assertEqual(bench.main(), 0)
        args, selected = run.call_args.args
        self.assertEqual((args.cpu, args.gpu, args.jobs, args.cuda_arch), (2, "1", 2, "86"))
        self.assertEqual((args.governor, args.repetitions, args.min_time, args.profile),
                         ("performance", 1, .3, "decision"))
        self.assertEqual(selected, [("main", "gpu")])

    def test_environment_limits_threads_and_pins_rust_build_output(self):
        args = argparse.Namespace(gpu="1", jobs=4, cuda_arch="86")
        with mock.patch.dict(bench.os.environ, {"OMP_NUM_THREADS": "8", "RAYON_NUM_THREADS": "16"}):
            env = bench.environment(args, "libdpf")
        self.assertEqual((env["OMP_NUM_THREADS"], env["RAYON_NUM_THREADS"]), ("1", "1"))
        self.assertEqual(env["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertEqual(env["CARGO_TARGET_DIR"], str(bench.BUILD / "libdpf" / "target"))
        self.assertEqual(env["TORCH_CUDA_ARCH_LIST"], "8.6")
        self.assertEqual(bench.RUST, {"libdpf": "nightly-2025-09-01", "fss_v060": "nightly-2025-09-01",
                                  "mastic": "stable", "plasma": "stable", "sycret": "stable"})

    def test_bazel_probe_and_build_share_pinned_isolated_environment(self):
        args = argparse.Namespace(libraries="google_dpf", platform="cpu", gpu="0", jobs=4,
                                  cuda_arch=None, cpu=0, smoke=True, repetitions=1, min_time=.01)
        with mock.patch.object(bench, "capture", return_value="test probe") as capture:
            metadata = bench.metadata(args, "performance")
        call = next(call for call in capture.call_args_list if call.args[0] == ["bazel", "--version"])
        self.assertEqual(call.kwargs["cwd"], bench.source("google_dpf"))
        env = call.kwargs["env"]
        self.assertEqual(env["USE_BAZEL_VERSION"], "7.6.1")
        self.assertEqual(Path(env["BAZELISK_HOME"]), bench.BUILD / "google_dpf" / "bazelisk")
        self.assertEqual(metadata["library_versions"], {"google_dpf": None})
        for field in ("source_sha256", "runner_sha256", "lockfile_sha256"):
            self.assertNotIn(field, metadata)
        self.assertNotIn("git_commit", metadata["probes"])
        with mock.patch.dict(bench.os.environ, {"CPLUS_INCLUDE_PATH": "/openssl", "C_INCLUDE_PATH": "/openssl", "CPATH": "/openssl"}):
            bazel_env = bench.environment(args, "google_dpf")
            main_env = bench.environment(args, "main")
        for key in ("CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH", "CPATH"):
            self.assertNotIn(key, bazel_env)
            self.assertEqual(main_env[key], "/openssl")
        if "USE_BAZEL_VERSION" not in bench.os.environ:
            self.assertNotIn("USE_BAZEL_VERSION", bench.environment(args, "main"))

    def test_declared_version_labels_unreleased_working_tree(self):
        declared = bench.declared_version("fss")
        declared_in_cmake = bench.re.search(
            r"project\(\s*fss\b[^)]*VERSION\s+(\d+\.\d+\.\d+)",
            (bench.ROOT / "CMakeLists.txt").read_text()).group(1)
        self.assertEqual(declared, declared_in_cmake + "-unreleased")
        self.assertEqual(bench.declared_version("main"), declared)
        self.assertEqual(bench.declared_version("fss_v060"), "0.6.0")
        self.assertEqual(bench.declared_version("fss_v070"), "0.7.0")

    def test_third_party_versions_stay_unknown_instead_of_guessed(self):
        for name in ("libdpf", "libfss", "google_dpf", "gpu_dpf", "ezpc",
                     "mastic", "plasma", "sycret", "dpf-cpp", "torchcsprng"):
            self.assertIsNone(bench.declared_version(name))

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

    def test_smoke_matches_raw_aes_and_includes_full_domain(self):
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
            self.assertIsNotNone(bench.re.search(pattern, "BM_DpfEvalAll_Uint_Aes/20"))
            self.assertIsNotNone(bench.re.search(pattern, "BM_DpfEval_Uint_Aes/20"))

    def test_plan_resolves_selection_and_budget_without_admission(self):
        output = io.StringIO()
        with mock.patch.object(bench.sys, "argv", ["bench.py", "run", "--plan"]), \
             mock.patch.object(bench, "build") as build, mock.patch.object(bench, "run") as run, \
             mock.patch.object(bench, "metadata") as metadata, \
             contextlib.redirect_stdout(output):
            self.assertEqual(bench.main(), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual((plan["profile"], plan["min_time_s"], plan["repetitions"],
                          plan["full_launches"], plan["full_launches_env"]),
                         ("decision", .3, 1, 1, "FSS_BENCH_FULL_LAUNCHES"))
        self.assertEqual(plan["libraries"], "main")
        self.assertEqual([(c["library"], c["device"]) for c in plan["commands"]],
                         [("main", "cpu"), ("main", "gpu")])
        for command in plan["commands"]:
            self.assertTrue(command["result_row_count"].startswith("unresolved"))
            self.assertIsNone(bench.re.search(command["filter"], "BM_DpfEvalPoint_Uint_ChaCha/20"))
            self.assertIsNotNone(bench.re.search(command["filter"], "BM_DpfEvalAllGpu_Uint_ChaCha/20"))
        self.assertFalse(build.called or run.called or metadata.called)

    def test_decision_default_excludes_internal_point_rows(self):
        (bench.ROOT / "build").mkdir(exist_ok=True)
        commands = []
        for profile, minimum, repetitions in (("decision", .3, 1), ("publication", 1, 5)):
            args = argparse.Namespace(gpu="0", jobs=1, cuda_arch=None, cpu=0, filter=None,
                                      smoke=False, min_time=minimum, repetitions=repetitions,
                                      native_args=[], profile=profile, full_launches=1, workload="daily")
            with tempfile.TemporaryDirectory(dir=bench.ROOT / "build") as temp, \
                 mock.patch.object(bench, "execute"), \
                 mock.patch.object(bench, "result_rows", return_value=([[]], [])):
                bench.record(args, "main", "cpu", Path(temp), commands)
            entry = commands[-1]
            pattern = next(value.split("=", 1)[1] for value in entry["command"]
                           if value.startswith("--benchmark_filter="))
            point = bench.re.search(pattern, "BM_DpfEvalPoint_Uint_AesRaw/20")
            if profile == "decision":
                self.assertIsNone(point)
            else:
                self.assertIsNotNone(point)
            self.assertIsNotNone(bench.re.search(pattern, "BM_DpfEvalAllGpu_Uint_ChaCha/20"))
            self.assertIn(f"--benchmark_min_time={minimum}s", entry["command"])
            self.assertIn(f"--benchmark_repetitions={repetitions}", entry["command"])
            self.assertEqual(entry["full_launches"], 1)

    def test_environment_forwards_resolved_full_launches_for_main(self):
        args = argparse.Namespace(gpu="0", jobs=4, cuda_arch=None, full_launches=7)
        self.assertEqual(bench.environment(args, "main")["FSS_BENCH_FULL_LAUNCHES"], "7")
        args.full_launches = None
        self.assertNotIn("FSS_BENCH_FULL_LAUNCHES", bench.environment(args, "main"))
        args.full_launches = 7
        self.assertNotIn("FSS_BENCH_FULL_LAUNCHES", bench.environment(args, "fss"))

    def test_main_filters_use_posix_alternations_without_lookaround(self):
        registered = ["BM_DpfEvalAllGpu_Uint_ChaCha/20", "BM_DcfEvalAllGpu_Uint_ChaCha/20",
                      "BM_HalfTreeDpfEvalAllGpu_Uint_OwnedAesCcr/20",
                      "BM_PackedHalfTreeDpfEvalAllGpu_OwnedAesCcr/20",
                      "BM_GrottoDcfEvalAllGpu_ChaCha/20", "BM_VdpfEvalAllMerkleGpu_Uint_ChaCha_Blake3/20",
                      "BM_VdpfEvalAllMerkle_Uint_ChaCha_Blake3/20", "BM_GrottoDcfPreprocess_Aes/20",
                      "BM_VdpfProve_Uint_ChaCha_Blake3/20", "BM_GrottoDcfPreprocessEvalAll_ChaCha/20",
                      "BM_DpfEval_Uint_ChaCha/20", "BM_DpfGen_Bytes_AesRaw/20"]
        points = ["BM_DpfEvalPoint_Uint_ChaCha/20", "BM_HalfTreeDpfEvalPoint_Bytes_AesRaw/20",
                  "BM_GrottoDcfEvalPointGpu_ChaCha/20", "BM_VdpfEvalPointGpu_Uint_ChaCha_Blake3/20",
                  "BM_PackedHalfTreeDpfEvalPointGpu_OwnedAesCcr/20"]
        decision = bench.default_filter("main", "gpu", exclude_internal_points=True)
        smoke = bench.resolved_filter(argparse.Namespace(filter=None, smoke=True), "main", "gpu")
        for pattern in (decision, smoke, bench.default_filter("main", "cpu")):
            self.assertNotIn("(?!", pattern)
        for name in registered:
            if name not in points:
                self.assertIsNotNone(bench.re.search(decision, name), name)
        for name in points:
            self.assertIsNone(bench.re.search(decision, name), name)
            self.assertIsNone(bench.re.search(smoke, name), name)
        for name in ("BM_DpfEvalAllGpu_Uint_ChaCha/20", "BM_VdpfEvalAllMerkleGpu_Uint_ChaCha_Blake3/20",
                     "BM_VdpfEvalAllMerkle_Uint_ChaCha_Blake3/20", "BM_DpfEvalAll_Uint_Aes/20",
                     "BM_DpfEval_Uint_Aes/20", "BM_DpfGen_Uint_Aes/20"):
            self.assertIsNotNone(bench.re.search(smoke, name), name)
        self.assertIsNone(bench.re.search(smoke, "BM_GrottoDcfPreprocess_Aes/20"))

    def test_google_commands_preserve_raw_repetitions_and_resolved_timing(self):
        (bench.ROOT / "build").mkdir(exist_ok=True)
        args = argparse.Namespace(gpu="0", jobs=1, cuda_arch=None, cpu=0,
                                  filter=None, smoke=True, min_time=.7, repetitions=3,
                                  native_args=[], profile="smoke", full_launches=1)
        commands = []
        with tempfile.TemporaryDirectory(dir=bench.ROOT / "build") as temp, \
             mock.patch.object(bench, "execute"), \
             mock.patch.object(bench, "result_rows", return_value=([], [])):
            bench.record(args, "main", "cpu", Path(temp), commands)
        command = commands[0]["command"]
        self.assertIn("--benchmark_min_time=0.7s", command)
        self.assertIn("--benchmark_repetitions=3", command)
        self.assertIn("--benchmark_display_aggregates_only=true", command)
        self.assertNotIn("--benchmark_report_aggregates_only=true", command)
        self.assertEqual(commands[0]["required_work_schema"], 1)

    def test_repeated_current_library_commands_run_interleaved(self):
        (bench.ROOT / "build").mkdir(exist_ok=True)
        commands = []
        for library, repetitions, expected, flag in (
                ("main", 5, True, "--benchmark_enable_random_interleaving=true"),
                ("main", 1, False, None),
                ("fss", 5, True, "--benchmark_enable_random_interleaving=true"),
                ("fss", 1, False, None)):
            args = argparse.Namespace(gpu="0", jobs=1, cuda_arch=None, cpu=0, filter=None,
                                      smoke=False, min_time=.3, repetitions=repetitions,
                                      native_args=[], profile="decision", full_launches=1)
            with tempfile.TemporaryDirectory(dir=bench.ROOT / "build") as temp, \
                 mock.patch.object(bench, "execute"), \
                 mock.patch.object(bench, "result_rows", return_value=([], [])):
                bench.record(args, library, "cpu", Path(temp), commands)
            entry = commands[-1]
            self.assertEqual([value for value in entry["command"]
                              if value.startswith(bench.benchmark_policy.INTERLEAVING_FLAG)],
                             [flag] if flag else [], library)
            self.assertEqual(entry["interleaving"], expected)

    def test_explicit_native_interleaving_leaves_exactly_one_flag(self):
        (bench.ROOT / "build").mkdir(exist_ok=True)
        args = argparse.Namespace(gpu="0", jobs=1, cuda_arch=None, cpu=0, filter=None,
                                  smoke=False, min_time=1, repetitions=5, native_args=[],
                                  profile="publication", full_launches=1,
                                  explicit_interleaving=False)
        commands = []
        with tempfile.TemporaryDirectory(dir=bench.ROOT / "build") as temp, \
             mock.patch.object(bench, "execute"), \
             mock.patch.object(bench, "result_rows", return_value=([], [])):
            bench.record(args, "main", "cpu", Path(temp), commands)
        self.assertEqual([value for value in commands[0]["command"]
                          if value.startswith(bench.benchmark_policy.INTERLEAVING_FLAG)],
                         ["--benchmark_enable_random_interleaving=false"])
        self.assertFalse(commands[0]["interleaving"])

    def test_historical_and_third_party_commands_do_not_interleave(self):
        (bench.ROOT / "build").mkdir(exist_ok=True)
        for library in ("fss_v070", "torchcsprng", "libdpf"):
            args = argparse.Namespace(gpu="0", jobs=1, cuda_arch=None, cpu=16,
                                      filter=None, smoke=False, min_time=.01, repetitions=5,
                                      native_args=[], profile="publication", full_launches=1)
            commands = []
            with tempfile.TemporaryDirectory(dir=bench.ROOT / "build") as temp, \
                 mock.patch.object(bench, "execute"), \
                 mock.patch.object(bench, "result_rows", return_value=([[]], [])):
                bench.record(args, library, "cpu", Path(temp), commands)
            self.assertTrue(commands, library)
            for entry in commands:
                self.assertNotIn("interleaving", entry)
                self.assertTrue(all(not value.startswith(bench.benchmark_policy.INTERLEAVING_FLAG)
                                    for value in entry["command"]), library)

    def test_decision_repetition_override_interleaves(self):
        output = io.StringIO()
        with mock.patch.object(bench.sys, "argv",
                               ["bench.py", "run", "--plan", "--profile", "decision",
                                "--repetitions", "3"]), \
             mock.patch.object(bench, "build"), mock.patch.object(bench, "run"), \
             mock.patch.object(bench, "metadata"), \
             contextlib.redirect_stdout(output):
            self.assertEqual(bench.main(), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["repetitions"], 3)
        self.assertTrue(all(command["interleaving"] for command in plan["commands"]))

    def test_native_interleaving_passthrough_resolves_before_plan(self):
        output = io.StringIO()
        with mock.patch.object(bench.sys, "argv",
                               ["bench.py", "run", "--plan", "--repetitions", "5", "--",
                                "--benchmark_enable_random_interleaving=false"]), \
             mock.patch.object(bench, "build"), mock.patch.object(bench, "run"), \
             contextlib.redirect_stdout(output):
            self.assertEqual(bench.main(), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["repetitions"], 5)
        self.assertTrue(all(command["interleaving"] is False for command in plan["commands"]))
        with mock.patch.object(bench.sys, "argv",
                               ["bench.py", "run", "--plan", "--",
                                "--benchmark_enable_random_interleaving=maybe"]), \
             mock.patch.object(bench, "build"), mock.patch.object(bench, "run"):
            with self.assertRaisesRegex(RuntimeError, "only true or false"):
                bench.main()

    def test_plan_honors_a_native_passthrough_filter(self):
        output = io.StringIO()
        with mock.patch.object(bench.sys, "argv",
                               ["bench.py", "run", "--plan", "--", "--benchmark_filter=only-this"]), \
             mock.patch.object(bench, "build"), mock.patch.object(bench, "run"), \
             contextlib.redirect_stdout(output):
            self.assertEqual(bench.main(), 0)
        plan = json.loads(output.getvalue())
        self.assertTrue(all(command["filter"] == "only-this" for command in plan["commands"]))

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

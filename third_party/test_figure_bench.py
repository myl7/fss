"""Sweep planning and native-result normalization tests without GPU/toolchains."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
import figure_bench as sweep


class FigureSweepTest(unittest.TestCase):
    def setUp(self):
        (sweep.ROOT / "build").mkdir(exist_ok=True)
        temp = tempfile.TemporaryDirectory(dir=sweep.ROOT / "build")
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)

    def args(self, **extra):
        defaults = dict(domain_bits=None, libraries="fss,libfss,gpu_dpf", platform="all",
                        sweep="domain", threads_per_block=None, num_keys=262144, cpu_prg="aes-ni")
        return argparse.Namespace(**(defaults | extra))

    def case(self, **extra):
        return sweep.cases(self.args(libraries="fss", platform="gpu", domain_bits=[12]))[0] | extra

    def test_pilot_endpoint_and_slow_full_subset(self):
        cases = sweep.cases(self.args())
        self.assertEqual(cases[0]["domain_bits"], 20)
        for library in ("libfss", "gpu_dpf"):
            domains = {c["domain_bits"] for c in cases if c["library"] == library and c["platform"] == "cpu" and c["operation"] == "EvalAll"}
            self.assertEqual(domains, set(sweep.SLOW_DOMAINS))
        self.assertEqual({c["domain_bits"] for c in cases if c["library"] == "fss" and c["operation"] == "Gen"}, set(sweep.DOMAINS))

    def test_full_keys_and_threads_are_independent(self):
        cases = sweep.cases(self.args(libraries="fss", platform="gpu", domain_bits=[20]))
        point = next(c for c in cases if c["operation"] == "Eval")
        full = next(c for c in cases if c["operation"] == "EvalAll")
        self.assertEqual((point["num_keys"], point["threads_per_block"]), (262144, 256))
        self.assertEqual((full["num_keys"], full["threads_per_block"]), (1, 128))
        self.assertEqual(full["configured_num_keys"], 262144)

    def test_ezpc_filter_selects_exact_key_count(self):
        case = self.case(library="ezpc", operation="Eval")
        pattern = sweep.pattern(case)
        import re
        self.assertIsNotNone(re.search(pattern, "EzPC/GPU/DPF/Eval/262144/manual_time"))
        self.assertIsNone(re.search(pattern, "EzPC/GPU/DPF/Eval/1024/manual_time"))

    def test_all_block_cases_and_software_aes_filter(self):
        cases = sweep.cases(self.args(libraries="fss", platform="gpu", sweep="block"))
        self.assertEqual(len(cases), 14)
        for operation in ("Eval", "EvalAll"):
            self.assertEqual({c["threads_per_block"] for c in cases if c["operation"] == operation}, set(sweep.BLOCKS))
        self.assertIn("AesSoft", sweep.pattern(next(c for c in cases if c["operation"] == "Eval")))

    def test_google_median_prefers_aggregate_over_repetitions(self):
        raw = self.directory / "raw.json"
        raw.write_text(json.dumps({"benchmarks": [
            dict(name="x", real_time=1, time_unit="us"),
            dict(name="x", real_time=3, time_unit="us"),
            dict(name="x_median", run_name="x", real_time=7, time_unit="us", aggregate_name="median"),
            dict(name="x_mean", run_name="x", real_time=4, time_unit="us", aggregate_name="mean"),
            dict(name="bad", error_occurred=True, error_message="unsupported block size")]}))
        values, failures = sweep.parse_google(raw)
        self.assertEqual(values, [("x", 7000)])
        self.assertEqual(failures, [("bad", "unsupported block size")])

    def test_google_repeated_samples_and_invalid_time(self):
        raw = self.directory / "raw.json"
        raw.write_text(json.dumps({"benchmarks": [dict(name="x", real_time=x, time_unit="ns") for x in (2, 9, 4)]}))
        self.assertEqual(sweep.parse_google(raw)[0], [("x", 4)])
        raw.write_text(json.dumps({"benchmarks": [dict(name="x", real_time=0, time_unit="ns")]}))
        with self.assertRaisesRegex(ValueError, "invalid benchmark time"):
            sweep.parse_google(raw)

    def test_normalization_preserves_total_time_and_distinct_half_tree(self):
        point = sweep.normalize(self.case(), "fss/GPU/DPF-bytes/AesSoft/Gen", 2621440, "raw", {})
        self.assertEqual((point["median_ns"], point["ns_per_key"], point["num_keys"]), (2621440, 10, 262144))
        full = sweep.normalize(self.case(operation="EvalAll", num_keys=1), "fss/GPU/HalfTreeDPF-bytes/EvalAll", 1000, "raw", {})
        self.assertEqual(full["ns_per_key"], 1000)
        self.assertEqual(full["variant"], "HalfTreeDPF-bytes-ChaCha20-x1")
        self.assertEqual(full["logical_output_bits"], 127)

    def test_cpu_pattern_covers_specialty_schemes(self):
        case = self.case(platform="cpu", operation="EvalAll")
        for name in ("fss/CPU/DPF-bytes/EvalAll", "fss/CPU/HalfTreeDPF-bytes/EvalAll",
                     "fss/CPU/VDPF-bytes/EvalAll", "fss/CPU/PackedHalfTreeDPF-bits1/EvalAll",
                     "fss/CPU/GrottoDCF/EvalAll", "fss/CPU/DMPF-bytes/EvalAll",
                     "fss/CPU/VDMPF-bytes/EvalAll"):
            self.assertRegex(name, sweep.pattern(case))

    def test_specialty_scheme_normalize_metadata(self):
        packed = sweep.normalize(self.case(platform="cpu", operation="EvalAll"),
                                 "fss/CPU/PackedHalfTreeDPF-bits1/EvalAll", 1000, "raw", {})
        self.assertEqual((packed["scheme"], packed["group"], packed["logical_output_bits"], packed["lanes"]),
                         ("PackedHalfTreeDPF", "bits1", 1, 128))
        self.assertEqual(packed["output_storage"], "packed_lanes")
        self.assertTrue(packed["label"].endswith("(experimental)"))
        grotto = sweep.normalize(self.case(platform="cpu", operation="EvalAll"),
                                 "fss/CPU/GrottoDCF/EvalAll", 1000, "raw", {})
        self.assertEqual((grotto["scheme"], grotto["logical_output_bits"], grotto["output_storage"]),
                         ("GrottoDCF", 1, "bool_scalar"))
        for name, scheme in (("fss/CPU/DMPF-bytes/EvalAll", "DMPF"), ("fss/CPU/VDMPF-bytes/Eval", "VDMPF")):
            row = sweep.normalize(self.case(platform="cpu", operation="Eval"), name, 1000, "raw", {})
            self.assertEqual(row["scheme"], scheme)
            self.assertEqual(row["num_points"], 64)
        servan = sweep.normalize(self.case(library="servan_vdpf", platform="cpu", operation="EvalAll"),
                                 "servan_vdpf/CPU/VDPF/EvalAll", 1000, "raw", {})
        self.assertEqual((servan["scheme"], servan["num_points"], servan["output_storage"], servan["prg"]),
                         ("VDPF", 1, "uint128_scalar", "AES128/OpenSSL"))
        ours = sweep.normalize(self.case(platform="cpu", operation="EvalAll"),
                               "fss/CPU/VDPF-bytes/EvalAll", 1000, "raw", {})
        self.assertEqual((ours["scheme"], ours["group"], ours["logical_output_bits"]), ("VDPF", "bytes", 127))
        self.assertNotIn("num_points", ours)
        self.assertEqual(ours["output_storage"], "materialized")

    def test_kernel_resource_failure_preserves_incomplete_raw_as_unsupported(self):
        case = self.case(operation="Eval", sweep="block", threads_per_block=1024)
        args = argparse.Namespace(cpu=16, repetitions=5, min_time=.1, warmup=.1)

        def execute(command, **kwargs):
            raw = Path(next(argument.split("=", 1)[1] for argument in command if argument.startswith("--benchmark_out=")))
            raw.write_text('{"benchmarks": [')
            kwargs["log"].write_text("cuda error: too many resources requested for launch\n")
            raise RuntimeError("command exited with status -6")

        with mock.patch.object(sweep, "environment", return_value={}), \
             mock.patch.object(sweep, "executables", return_value=[Path("bench")]), \
             mock.patch.object(sweep.bench, "execute", side_effect=execute):
            rows = sweep.collect(args, case, self.directory / "case", {})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "unsupported")
        self.assertEqual(rows[0]["failure_kind"], "kernel_resources_exhausted")
        self.assertIsNone(rows[0]["median_ns"])
        self.assertEqual(Path(rows[0]["raw_result"]).read_text(), '{"benchmarks": [')
        self.assertEqual(len(rows[0]["raw_result_sha256"]), 64)

    def test_criterion_variant_failure_preserves_other_medians(self):
        target = self.directory / "target"
        case = self.case(library="fss_v060", platform="cpu", num_keys=1, operation="Eval", include_uint=True)
        args = argparse.Namespace(cpu=24, repetitions=2, min_time=.1, warmup=.1)
        counts = {}

        def execute(command, **unused):
            group = command[-1].split("/")[1]
            if group == "DCF-uint":
                raise RuntimeError("correctness check failed")
            counts[group] = counts.get(group, 0) + 1
            directory = target / "criterion" / group / "new"
            directory.mkdir(parents=True)
            (directory / "benchmark.json").write_text(json.dumps({"full_id": f"fss-v0.6.0/CPU/{group}/Eval"}))
            (directory / "estimates.json").write_text(json.dumps({"median": {"point_estimate": 100 * counts[group]}}))

        with mock.patch.object(sweep, "environment", return_value={"CARGO_TARGET_DIR": str(target)}), \
             mock.patch.object(sweep.bench, "execute", side_effect=execute):
            rows = sweep.collect(args, case, self.directory / "case", {})
        self.assertEqual([r["status"] for r in rows], ["ok", "ok", "ok", "known_failure"])
        self.assertEqual([r["median_ns"] for r in rows[:3]], [150, 150, 150])
        self.assertEqual(rows[0]["process_medians_ns"], [100, 200])
        self.assertEqual(rows[-1]["scheme"], "DCF")

    def test_stale_native_configuration_is_rejected(self):
        raw = self.directory / "raw.json"
        raw.write_text(json.dumps({"benchmarks": [dict(name="x", domain_bits=20, keys=262144)]}))
        with self.assertRaisesRegex(ValueError, "domain_bits"):
            sweep.validate_native_configuration(self.case(), raw)
        raw.write_text(json.dumps({"benchmarks": [dict(name="x", domain_bits=12, keys=262144, threads_per_block=256)]}))
        sweep.validate_native_configuration(self.case(), raw)

    def test_export_keeps_failures_and_merges_cases(self):
        (self.directory / "run.json").write_text(json.dumps({"source_sha256": {"bench.cu": "abc"}}))
        for index, status in enumerate(("ok", "unsupported", "known_failure")):
            directory = self.directory / str(index)
            directory.mkdir()
            row = sweep.normalize(self.case(), "case", 10 if status == "ok" else None, "raw", {}, status)
            (directory / "case.json").write_text(json.dumps({"rows": [row]}))
        self.assertEqual(sweep.export(self.directory), 1)
        result = json.loads((self.directory / "chart-data.json").read_text())
        self.assertEqual([r["status"] for r in result["records"]], ["ok", "unsupported", "known_failure"])
        self.assertEqual(result["metadata"]["source_sha256"], {"bench.cu": "abc"})
        self.assertIn("source_sha256", (self.directory / "chart-data.csv").read_text())

    def test_gpu_dpf_storage_refresh_preserves_measurements(self):
        for device, storage in (("cpu", "uint128_scalar"), ("gpu", "uint128_scalar_bit_reversed")):
            row = sweep.normalize(self.case(library="gpu_dpf", platform=device, operation="EvalAll", num_keys=1),
                                  f"GPU-DPF/{device.upper()}/DPF/EvalAll", 456, "observed-raw", {"source": "verified"})
            row["storage_output_bits"] = None
            refreshed = sweep.polish_metadata(row)
            self.assertEqual((refreshed["logical_output_bits"], refreshed["storage_output_bits"]), (128, 128))
            self.assertEqual(refreshed["output_storage"], storage)
            if device == "gpu":
                self.assertEqual(refreshed["output_order"], "bit_reversed")
            for field in ("median_ns", "time_ns", "ns_per_key", "raw_result", "source_sha256", "status"):
                self.assertEqual(refreshed[field], row[field])

    def test_native_cpu_prg_refresh_preserves_values_and_explicit_variants(self):
        for library, scheme, label, blocks in (("libdpf", "DPF", "AES128-MMO/RustCrypto", None),
                                               ("libfss", "DPF", "AES128-MMO/OpenSSL", 3),
                                               ("libfss", "DCF", "AES128-MMO/OpenSSL", 4)):
            row = sweep.normalize(self.case(library=library, platform="cpu"), f"{library}/CPU/{scheme}/Gen", 123, "observed-raw", {"source": "pinned"})
            row["prg"] = "native"
            refreshed = sweep.polish_metadata(row)
            self.assertEqual(refreshed["prg"], label)
            if blocks is not None:
                self.assertEqual(refreshed["prg_output_blocks"], blocks)
            for field in ("median_ns", "time_ns", "ns_per_key", "status", "raw_result", "source_sha256"):
                self.assertEqual(refreshed[field], row[field])
            row.update(prg="explicit-alternative", prg_output_blocks=9)
            preserved = sweep.polish_metadata(row)
            self.assertEqual(preserved["prg"], "explicit-alternative")
            self.assertEqual(preserved["prg_output_blocks"], 9)

    def test_current_gpu_prg_block_labels_and_existing_dcf_refresh(self):
        for scheme, blocks in (("DPF", 2), ("DCF", 4), ("HalfTreeDPF", 1)):
            row = sweep.normalize(self.case(operation="Eval"), f"fss/GPU/{scheme}-bytes/Eval", 500, "raw", {})
            self.assertEqual(row["prg"], f"ChaCha20-x{blocks}")
            self.assertEqual(row["prg_output_blocks"], blocks)
        row = sweep.normalize(self.case(operation="Eval"), "fss/GPU/DCF-bytes/Eval", 500, "observed-raw", {})
        row.update(prg="ChaCha20-x2", status="ok")
        refreshed = sweep.polish_metadata(row)
        self.assertEqual(refreshed["prg"], "ChaCha20-x4")
        for field in ("median_ns", "time_ns", "ns_per_key", "raw_result", "status"):
            self.assertEqual(refreshed[field], row[field])
        soft = sweep.normalize(self.case(operation="Eval"), "fss/GPU/DPF-bytes/AesSoft/Eval", 500, "raw", {})
        self.assertEqual(sweep.polish_metadata(soft)["prg"], "AES-software")

    def test_historical_google_xor_metadata_is_not_reinterpreted(self):
        row = sweep.normalize(self.case(library="google_dpf", platform="cpu"), "google_dpf/CPU/DCF/Gen", 100, "historical-raw", {})
        row.update(group="xor_128", config={}, source_sha256={"third_party/distributed_point_functions/bench.cc": "historical-xor-hash"}, status="failed")
        refreshed = sweep.polish_metadata(row)
        self.assertEqual(refreshed["group"], "xor_128")
        for field in ("median_ns", "time_ns", "ns_per_key", "raw_result", "source_sha256", "status"):
            self.assertEqual(refreshed[field], row[field])
        row["group"] = "native"
        self.assertEqual(sweep.polish_metadata(row)["group"], "native")
        row["source_sha256"]["third_party/distributed_point_functions/bench.cc"] = sweep.hashlib.sha256((sweep.ROOT / "third_party/distributed_point_functions/bench.cc").read_bytes()).hexdigest()
        self.assertEqual(sweep.polish_metadata(row)["group"], "additive_mod_2^128")
        row["group"] = "xor_128"
        self.assertEqual(sweep.polish_metadata(row)["group"], "xor_128")

    def test_export_refreshes_metadata_without_changing_measurements(self):
        row = sweep.normalize(self.case(library="fss_v070"), "fss-v0.7.0/GPU/bytes/DPF/Eval", 500, "raw", {"source": "hash"})
        row.update(prg="Salsa20", timing_boundary="cuda_event_kernel", median_ns=500, ns_per_key=500 / 262144)
        sweep.write_export(self.directory, {"runner_sha256": "original"}, [row])
        result = json.loads((self.directory / "chart-data.json").read_text())
        refreshed = result["records"][0]
        self.assertEqual(refreshed["prg"], "Salsa12")
        self.assertEqual(refreshed["median_ns"], 500)
        self.assertEqual(refreshed["ns_per_key"], 500 / 262144)
        self.assertEqual(refreshed["raw_result"], "raw")
        self.assertEqual(refreshed["source_sha256"], {"source": "hash"})
        self.assertEqual(result["metadata"]["runner_sha256"], "original")

    def test_group_and_native_api_timing_metadata(self):
        for scheme, group in (("DPF", "xor_128"), ("DCF", "additive_mod_2^128")):
            row = sweep.normalize(self.case(library="google_dpf", platform="cpu"), f"google_dpf/CPU/{scheme}/Gen", 100, "raw", {})
            self.assertEqual(row["group"], group)
            self.assertEqual(row["logical_output_bits"], 128)
        row = sweep.normalize(self.case(library="libfss", platform="cpu"), "libfss/CPU/DCF/Gen", 100, "raw", {})
        self.assertEqual((row["group"], row["logical_output_bits"]), ("additive_mod_2^64", 64))
        row = sweep.normalize(self.case(library="ezpc", operation="Gen"), "EzPC/GPU/DPF/Gen/262144", 100, "raw", {})
        self.assertEqual(row["timing_boundary"], "cuda_event_native_key_generation_api")
        self.assertIn("transfers", row["timing_boundary_detail"])

    def test_merge_preserves_run_provenance(self):
        inputs = []
        for index in range(2):
            directory = self.directory / f"run-{index}"
            sweep.write_export(directory, {"host": f"host-{index}"}, [sweep.normalize(self.case(domain_bits=8 + index), "x", 10, "raw", {})])
            inputs.append(directory)
        self.assertEqual(sweep.merge(self.directory / "merged", inputs), 0)
        result = json.loads((self.directory / "merged" / "chart-data.json").read_text())
        self.assertEqual(len(result["records"]), 2)
        self.assertEqual(result["metadata"]["runs"][1]["metadata"]["host"], "host-1")
        self.assertEqual(len(result["metadata"]["runs"][0]["sha256"]), 64)

    def test_explicit_retry_supersedes_case_but_preserves_failed_attempt(self):
        original = self.directory / "original"
        retry = self.directory / "retry"
        failed = sweep.normalize(self.case(operation="Eval"), "", None, "failed-raw", {}, "failed", "incomplete JSON")
        recovered = sweep.normalize(self.case(operation="Eval"), "fss/GPU/DPF-bytes/AesSoft/Eval", None, "retry-raw", {}, "unsupported", "kernel resources")
        sweep.write_export(original, {}, [failed])
        sweep.write_export(retry, {}, [recovered])
        self.assertEqual(sweep.merge(self.directory / "merged", [original], [retry]), 0)
        result = json.loads((self.directory / "merged" / "chart-data.json").read_text())
        self.assertEqual([row["status"] for row in result["records"]], ["unsupported"])
        self.assertEqual(result["metadata"]["superseded_records"][0]["record"]["raw_result"], "failed-raw")
        self.assertEqual(result["metadata"]["runs"][1]["retry"], True)

    def test_old_bench_parser_uses_configured_domain_and_key_count(self):
        raw = self.directory / "raw.json"
        raw.write_text(json.dumps({"benchmarks": [dict(name="fss/GPU/DPF/EvalAll", real_time=10, time_unit="ns")]}))
        entry = dict(library="fss", device="gpu", framework="google", domain_bits=12,
                     num_keys=262144, command=[f"--benchmark_out={raw}"])
        rows, errors = sweep.bench.result_rows(entry, self.directory)
        self.assertFalse(errors)
        self.assertEqual((rows[0][3], rows[0][6]), (1, 4096))
        entry.update(device="cpu", framework="criterion")
        new = self.directory / "criterion" / "case" / "new"
        new.mkdir(parents=True)
        (new / "benchmark.json").write_text(json.dumps({"full_id": "libdpf/CPU/DPF/EvalAll"}))
        (new / "estimates.json").write_text(json.dumps({"median": {"point_estimate": 20}}))
        self.assertEqual(sweep.bench.result_rows(entry, self.directory)[0][0][6], 4096)


if __name__ == "__main__":
    unittest.main()

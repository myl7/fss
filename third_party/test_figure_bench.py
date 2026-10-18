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

    def test_prepare_collected_with_eval_and_not_counted_as_query(self):
        case = self.case(platform="cpu", num_keys=1, operation="Eval")
        name = "fss/CPU/DPF-bytes/PackedKey/Prepare"
        self.assertRegex(name, sweep.pattern(case))
        row = sweep.normalize(case, name, 20, "absent", {})
        self.assertEqual((row["operation"], row["timed_operation"], row["time_denominator"]), ("Prepare", "Prepare", "key"))
        self.assertEqual((row["query_count"], row["key_count"], row["ns_per_unit"], row["setup_included"]), (0, 1, 20, True))
        self.assertTrue(row["variant"].endswith("-packedkey-setup"))
        self.assertIn("all point queries are outside timing", row["timing_boundary_detail"])

    def test_vdpf_point_times_the_single_element_chain_proof(self):
        case = self.case(platform="cpu", operation="Eval", num_keys=1)
        row = sweep.normalize(case, "fss/CPU/VDPF-bytes/Eval", 500, "absent", {})
        self.assertEqual((row["variant"], row["proof_accumulator"], row["proof_generation_included"]), ("VDPF-bytes-AES-NI-chains", "chain", True))
        self.assertIn("one-element chain proof", row["timing_boundary_detail"])
        generation = sweep.normalize(dict(case, operation="Gen"), "fss/CPU/VDPF-bytes/Gen", 500, "absent", {})
        self.assertEqual((generation["proof_accumulator"], generation["proof_generation_included"]), ("none", False))
        legacy = dict(row, variant="VDPF-bytes-AES-NI")
        self.assertEqual(sweep.polish_metadata(legacy)["variant"], row["variant"])
        self.assertEqual(sweep.polish_metadata(legacy)["median_ns"], 500)

    def test_packed_gpu_prg_metadata_matches_one_block_wrapper(self):
        row = sweep.normalize(self.case(operation="EvalAll", num_keys=1), "fss/GPU/PackedHalfTreeDPF-bits1/EvalAll", 1000, "absent", {})
        self.assertEqual((row["prg"], row["prg_output_blocks"]), ("ChaCha20-x1", 1))
        legacy = dict(row, prg="ChaCha20-x2", prg_output_blocks=2)
        repaired = sweep.polish_metadata(legacy)
        self.assertEqual((repaired["prg"], repaired["variant"]), ("ChaCha20-x1", "PackedHalfTreeDPF-bits1-ChaCha20-x1"))
        self.assertEqual(repaired["median_ns"], legacy["median_ns"])

    def test_specialty_scheme_selection_uses_native_names_and_supported_ops(self):
        for platform in ("cpu", "gpu"):
            selected = sweep.cases(self.args(platform=platform, libraries="fss", scheme="PackedHalfTreeDPF", domain_bits=[12]))
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]["operation"], "EvalAll")
            self.assertRegex(f"fss/{platform.upper()}/PackedHalfTreeDPF-bits1/EvalAll", sweep.pattern(selected[0]))
            grotto = sweep.cases(self.args(platform=platform, libraries="fss", scheme="GrottoDCF", domain_bits=[12]))
            for case in grotto:
                self.assertRegex(f"fss/{platform.upper()}/GrottoDCF/{case['operation']}", sweep.pattern(case))

    def test_explicit_quantity_denominators_and_receipt_launch(self):
        raw = self.directory / "quantities.json"
        name = "fss/GPU/VDMPF-bytes/Eval/manual_time"
        raw.write_text(json.dumps({"benchmarks": [dict(name=name, key_count=1, query_count=4,
                                                       threads_per_block=256)]}))
        row = sweep.normalize(self.case(operation="Eval", num_keys=4, configured_num_keys=4,
                                        threads_per_block=128), name, 2000, raw, {})
        self.assertEqual((row["key_count"], row["query_count"], row["time_denominator"], row["ns_per_unit"]),
                         (1, 4, "query", 500))
        self.assertEqual((row["requested_threads_per_block"], row["threads_per_block"]), (128, 256))
        self.assertEqual(row["workload"], "single_key_batch")
        self.assertNotIn("metadata_status", row)
        self.assertFalse(row["block_size_tunable"])
        full = sweep.normalize(self.case(operation="EvalAll", num_keys=1), "fss/GPU/DPF-bytes/EvalAll", 1000, "absent", {})
        self.assertEqual((full["key_count"], full["domain_output_count"], full["ns_per_unit"]), (1, 4096, 1000))
        cpu = sweep.normalize(self.case(platform="cpu", num_keys=1, operation="Eval"), "fss/CPU/DPF-bytes/Eval", 100, "absent", {})
        self.assertEqual((cpu["query_count"], cpu["time_denominator"], cpu["ns_per_unit"]), (1, "query", 100))
        independent = sweep.normalize(self.case(operation="Eval", num_keys=4), "fss/GPU/DPF-bytes/Eval", 2000, "absent", {})
        self.assertEqual((independent["key_count"], independent["query_count"], independent["ns_per_unit"]), (4, 4, 500))

    def test_full_multi_point_receipt_keys_per_launch_is_authoritative(self):
        raw = self.directory / "full-quantities.json"
        name = "fss/GPU/DMPF-bytes/EvalAll/manual_time"
        raw.write_text(json.dumps({"benchmarks": [dict(name=name, keys_per_launch=1, domain_outputs=4096, threads_per_block=256)]}))
        row = sweep.normalize(self.case(operation="EvalAll", num_keys=1, threads_per_block=128), name, 4096, raw, {})
        self.assertNotIn("metadata_status", row)
        self.assertEqual((row["key_count"], row["domain_output_count"], row["ns_per_domain_output"]), (1, 4096, 1))
        self.assertEqual((row["quantity_source"], row["threads_per_block"]), ("raw_counter", 256))

    def test_unknown_batch_metadata_requires_recollection(self):
        row = sweep.normalize(self.case(operation="Eval"), "fss/GPU/DMPF-bytes/Eval", 1000, "absent", {})
        self.assertEqual(row["metadata_status"], "requires_recollection")
        known = sweep.normalize(self.case(operation="Eval", threads_per_block=128), "fss/GPU/DMPF-bytes/Eval", 1000,
                                "absent", {"third_party/fss/bench.cu": next(iter(sweep.LEGACY_BATCH_WRAPPERS))})
        self.assertEqual((known["key_count"], known["query_count"], known["threads_per_block"]), (1, 262144, 256))

    def test_default_gpu_point_collection_includes_all_supported_schemes(self):
        case = self.case(operation="Eval")
        args = argparse.Namespace(cpu=100, repetitions=5, min_time=.1, warmup=.1)
        filters = []
        def execute(command, **unused):
            selected = next(arg.split("=", 1)[1] for arg in command if arg.startswith("--benchmark_filter="))
            filters.append(selected)
            raw = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--benchmark_out=")))
            name = selected.removeprefix("^").removesuffix("(/|$)")
            raw.write_text(json.dumps({"benchmarks": [dict(name=name, real_time=100, time_unit="ns", key_count=1 if "DMPF" in name else 262144, query_count=262144, threads_per_block=256)]}))
        with mock.patch.object(sweep, "environment", return_value={}), mock.patch.object(sweep, "executables", return_value=[Path("bench")]), mock.patch.object(sweep.bench, "execute", side_effect=execute):
            rows = sweep.collect(args, case, self.directory / "case", {})
        self.assertEqual(len(rows), 5)
        self.assertEqual(len(set(filters)), 5)
        self.assertTrue(any("DMPF-bytes" in pattern for pattern in filters))
        self.assertTrue(any("VDMPF-bytes" in pattern for pattern in filters))
        self.assertEqual(sweep.cases(self.args(platform="gpu", sweep="block", scheme="DMPF")), [])
        self.assertNotRegex("fss/GPU/VDMPF-bytes/EvalAll", sweep.pattern(self.case(operation="EvalAll", sweep="block")))

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

    def test_block_scheme_selection_preserves_isolation(self):
        for scheme, operations in sweep.FSS_GPU_OPERATIONS.items():
            selected = sweep.cases(self.args(libraries="fss", platform="gpu", sweep="block", scheme=scheme))
            expected = set(operations) - {"Gen"}
            if scheme in ("DMPF", "VDMPF"):
                expected = set()
            elif scheme != "DPF":
                expected.discard("Eval")
            self.assertEqual({case["operation"] for case in selected}, expected, scheme)
            for case in selected:
                native = "PackedHalfTreeDPF-bits1" if scheme == "PackedHalfTreeDPF" else "GrottoDCF" if scheme == "GrottoDCF" else scheme + "-bytes"
                middle = "AesSoft/" if case["operation"] == "Eval" else ""
                self.assertRegex(f"fss/GPU/{native}/{middle}{case['operation']}", sweep.pattern(case))
                if scheme != "DPF":
                    self.assertNotRegex("fss/GPU/DPF-bytes/AesSoft/Eval", sweep.pattern(case))

    def test_default_filters_follow_supported_operation_registry(self):
        for operation in ("Gen", "Eval", "EvalAll"):
            pattern = sweep.pattern(self.case(operation=operation))
            for scheme, operations in sweep.FSS_GPU_OPERATIONS.items():
                native = "PackedHalfTreeDPF-bits1" if scheme == "PackedHalfTreeDPF" else "GrottoDCF" if scheme == "GrottoDCF" else scheme + "-bytes"
                assertion = self.assertRegex if operation in operations else self.assertNotRegex
                assertion(f"fss/GPU/{native}/{operation}", pattern)

    def test_native_work_counters_reject_stale_configuration(self):
        raw = self.directory / "counts.json"
        def validate(case, name, **counters):
            raw.write_text(json.dumps({"benchmarks": [dict(name=name, **counters)]}))
            sweep.validate_native_configuration(case, raw)
        full = self.case(operation="EvalAll", num_keys=1)
        validate(full, "fss/GPU/VDMPF-bytes/EvalAll", key_count=1, query_count=4096, domain_outputs=4096, keys_per_launch=1)
        for field, value in (("domain_outputs", 1024), ("keys_per_launch", 4), ("query_count", 262144)):
            with self.assertRaisesRegex(ValueError, field):
                validate(full, "fss/GPU/VDMPF-bytes/EvalAll", **{field: value})
        point = self.case(operation="Eval")
        validate(point, "fss/GPU/DMPF-bytes/Eval", key_count=1, query_count=262144)
        validate(point, "fss/GPU/DPF-bytes/Eval", key_count=262144, query_count=262144)
        for field, value in (("key_count", 1), ("query_count", 1024)):
            with self.assertRaisesRegex(ValueError, field):
                validate(point, "fss/GPU/DPF-bytes/Eval", **{field: value})
        validate(self.case(platform="cpu", operation="Eval", num_keys=1), "fss/CPU/DPF-bytes/PackedKey/Prepare", key_count=1, query_count=0)

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

    def test_cpu_pattern_collects_packed_key_and_merkle_rows(self):
        case = self.case(platform="cpu", operation="Eval")
        for name in ("fss/CPU/DPF-bytes/Eval", "fss/CPU/DPF-bytes/PackedKey/Eval",
                     "fss/CPU/DPF-bytes/PackedKeyOneShot/Eval", "fss/CPU/HalfTreeDPF-bytes/PackedKey/Eval",
                     "fss/CPU/HalfTreeDPF-bytes/PackedKeyOneShot/Eval"):
            self.assertRegex(name, sweep.pattern(case))
        # The CPU AesSoft variant, EvalAll names, and the Merkle row stay
        # outside a plain Eval sweep.
        self.assertNotRegex("fss/CPU/DPF-bytes/AesSoft/Eval", sweep.pattern(case))
        self.assertNotRegex("fss/CPU/DPF-bytes/EvalAll", sweep.pattern(case))
        self.assertNotRegex("fss/CPU/VDPF-bytes/Merkle/EvalAll", sweep.pattern(case))
        eval_all = self.case(platform="cpu", operation="EvalAll")
        for name in ("fss/CPU/DPF-bytes/EvalAll", "fss/CPU/VDPF-bytes/EvalAll", "fss/CPU/VDPF-bytes/Merkle/EvalAll"):
            self.assertRegex(name, sweep.pattern(eval_all))
        scheme_case = self.case(platform="cpu", operation="EvalAll", scheme="VDPF")
        self.assertRegex("fss/CPU/VDPF-bytes/Merkle/EvalAll", sweep.pattern(scheme_case))
        self.assertRegex("fss/CPU/VDPF-bytes/EvalAll", sweep.pattern(scheme_case))
        self.assertNotRegex("fss/CPU/VDMPF-bytes/EvalAll", sweep.pattern(scheme_case))

    def test_packed_key_and_vdpf_merkle_variant_metadata(self):
        prepared = sweep.normalize(self.case(platform="cpu", operation="Eval"),
                                   "fss/CPU/DPF-bytes/PackedKey/Eval", 1000, "raw", {})
        self.assertEqual((prepared["variant"], prepared["label"]),
                         ("DPF-bytes-AES-NI-packedkey", "fss/DPF-bytes-AES-NI-packedkey"))
        self.assertEqual(prepared["timing_boundary"], "wall_time_operation")
        self.assertIn("outside timing", prepared["timing_boundary_detail"])
        oneshot = sweep.normalize(self.case(platform="cpu", operation="Eval"),
                                  "fss/CPU/DPF-bytes/PackedKeyOneShot/Eval", 1000, "raw", {})
        self.assertEqual((oneshot["variant"], oneshot["label"]),
                         ("DPF-bytes-AES-NI-packedkey-oneshot", "fss/DPF-bytes-AES-NI-packedkey-oneshot"))
        self.assertIn("inside timing", oneshot["timing_boundary_detail"])
        half = sweep.normalize(self.case(platform="cpu", operation="Eval"),
                               "fss/CPU/HalfTreeDPF-bytes/PackedKey/Eval", 1000, "raw", {})
        self.assertEqual(half["variant"], "HalfTreeDPF-bytes-AES-NI-packedkey")
        plain = sweep.normalize(self.case(platform="cpu", operation="Eval"), "fss/CPU/DPF-bytes/Eval", 1000, "raw", {})
        self.assertEqual(len({plain["variant"], prepared["variant"], oneshot["variant"], half["variant"]}), 4)
        merkle = sweep.normalize(self.case(platform="cpu", operation="EvalAll"),
                                 "fss/CPU/VDPF-bytes/Merkle/EvalAll", 1000, "raw", {})
        self.assertEqual((merkle["variant"], merkle["implementation"]),
                         ("VDPF-bytes-AES-NI-merkle", "merkle_fold_proof"))
        chain = sweep.normalize(self.case(platform="cpu", operation="EvalAll"),
                                "fss/CPU/VDPF-bytes/EvalAll", 1000, "raw", {})
        self.assertEqual((chain["variant"], chain["implementation"]),
                         ("VDPF-bytes-AES-NI-chains", "native"))
        # Point VDPF operations keep their base variant.
        self.assertEqual(sweep.normalize(self.case(platform="cpu", operation="Eval"), "fss/CPU/VDPF-bytes/Eval", 1000,
                                         "raw", {})["variant"], "VDPF-bytes-AES-NI-chains")

    def test_accumulator_variants_and_old_row_refresh(self):
        cpu_vdmpf = sweep.normalize(self.case(platform="cpu", operation="EvalAll"),
                                    "fss/CPU/VDMPF-bytes/EvalAll", 1000, "raw", {})
        self.assertEqual(cpu_vdmpf["variant"], "VDMPF-bytes-AES-NI-chains")
        gpu_vdmpf = sweep.normalize(self.case(operation="EvalAll", num_keys=1),
                                    "fss/GPU/VDMPF-bytes/EvalAll", 1000, "raw", {})
        self.assertEqual(gpu_vdmpf["variant"], "VDMPF-bytes-ChaCha20-x2-merkle-fold")
        gpu_vdpf = sweep.normalize(self.case(operation="EvalAll", num_keys=1),
                                   "fss/GPU/VDPF-bytes/EvalAll", 1000, "raw", {})
        self.assertEqual((gpu_vdpf["variant"], gpu_vdpf["implementation"]),
                         ("VDPF-bytes-ChaCha20-x2-merkle", "merkle_fold_proof"))
        # Rows exported before the qualifiers existed refresh at merge time.
        legacy = sweep.normalize(self.case(platform="cpu", operation="EvalAll"),
                                 "fss/CPU/VDPF-bytes/EvalAll", 1000, "raw", {})
        legacy.update(variant="VDPF-bytes-AES-NI", label="fss/VDPF-bytes-AES-NI")
        refreshed = sweep.polish_metadata(legacy)
        self.assertEqual((refreshed["variant"], refreshed["label"]),
                         ("VDPF-bytes-AES-NI-chains", "fss/VDPF-bytes-AES-NI-chains"))

    def test_rawprp_pattern_and_variant(self):
        # The AES-NI raw-PRP DMPF/VDMPF twins join the plain rows in one CPU sweep.
        case = self.case(platform="cpu", operation="EvalAll")
        for name in ("fss/CPU/DMPF-bytes/EvalAll", "fss/CPU/DMPF-bytes-AES-NI-rawprp/EvalAll",
                     "fss/CPU/VDMPF-bytes/EvalAll", "fss/CPU/VDMPF-bytes-AES-NI-rawprp/EvalAll"):
            self.assertRegex(name, sweep.pattern(case))
        # The suffix stays scoped to the two schemes that register raw-PRP rows.
        self.assertNotRegex("fss/CPU/DPF-bytes-AES-NI-rawprp/EvalAll", sweep.pattern(case))
        self.assertNotRegex("fss/CPU/VDMPF-bytes/Merkle-AES-NI-rawprp/EvalAll", sweep.pattern(case))
        dmpf = sweep.normalize(self.case(platform="cpu", operation="Eval"), "fss/CPU/DMPF-bytes-AES-NI-rawprp/Eval", 1000, "raw", {})
        self.assertEqual((dmpf["scheme"], dmpf["group"], dmpf["prg"], dmpf["num_points"]), ("DMPF", "bytes", "AES-NI", 64))
        self.assertEqual((dmpf["variant"], dmpf["label"]), ("DMPF-bytes-AES-NI-rawprp", "fss/DMPF-bytes-AES-NI-rawprp"))
        vdmpf = sweep.normalize(self.case(platform="cpu", operation="EvalAll"),
                                "fss/CPU/VDMPF-bytes-AES-NI-rawprp/EvalAll", 1000, "raw", {})
        self.assertEqual((vdmpf["variant"], vdmpf["implementation"]), ("VDMPF-bytes-AES-NI-chains-rawprp", "native"))
        plain = sweep.normalize(self.case(platform="cpu", operation="EvalAll"), "fss/CPU/VDMPF-bytes/EvalAll", 1000, "raw", {})
        self.assertEqual(plain["variant"], "VDMPF-bytes-AES-NI-chains")
        # A --scheme DMPF sweep collects the plain and raw-PRP rows together.
        scheme_case = self.case(platform="cpu", operation="Gen", scheme="DMPF")
        self.assertRegex("fss/CPU/DMPF-bytes/Gen", sweep.pattern(scheme_case))
        self.assertRegex("fss/CPU/DMPF-bytes-AES-NI-rawprp/Gen", sweep.pattern(scheme_case))
        # Other schemes never match the suffix.
        dpf_case = self.case(platform="cpu", operation="Gen", scheme="DPF")
        self.assertNotRegex("fss/CPU/DPF-bytes-AES-NI-rawprp/Gen", sweep.pattern(dpf_case))
        # Rows exported before the qualifier existed refresh at merge time.
        legacy = sweep.normalize(self.case(platform="cpu", operation="Eval"), "fss/CPU/DMPF-bytes-AES-NI-rawprp/Eval", 10, "raw", {})
        legacy.update(variant="DMPF-bytes-AES-NI", label="fss/DMPF-bytes-AES-NI")
        refreshed = sweep.polish_metadata(legacy)
        self.assertEqual((refreshed["variant"], refreshed["label"]), ("DMPF-bytes-AES-NI-rawprp", "fss/DMPF-bytes-AES-NI-rawprp"))

    def test_new_library_case_coverage_matches_registered_operations(self):
        cases = sweep.cases(self.args(libraries="dpf-cpp,dpfplusplus,mastic,plasma", platform="cpu"))
        seen = {(c["library"], c["operation"], c["domain_bits"]) for c in cases}
        for library, operations in sweep.NEW_LIBRARY_DOMAINS.items():
            for operation, domains in operations.items():
                self.assertEqual({n for lib, op, n in seen if lib == library and op == operation}, set(domains),
                                 f"{library}/{operation}")

    def test_new_library_patterns_and_native_id_validation(self):
        import re
        scalar = sweep.pattern(self.case(library="dpf-cpp", platform="cpu", operation="EvalAll", domain_bits=18))
        simd = sweep.pattern(self.case(library="dpf-cpp", platform="cpu", operation="EvalAll8", domain_bits=18))
        self.assertTrue(re.search(scalar, "dpf-cpp/CPU/EvalAll/18"))
        self.assertFalse(re.search(scalar, "dpf-cpp/CPU/EvalAll8/18"))
        self.assertTrue(re.search(simd, "dpf-cpp/CPU/EvalAll8/18"))
        self.assertFalse(re.search(simd, "dpf-cpp/CPU/EvalAll/18"))
        # Domain-pinned: one raw file never mixes the other registered sizes.
        self.assertFalse(re.search(scalar, "dpf-cpp/CPU/EvalAll/16"))
        self.assertFalse(re.search(scalar, "dpf-cpp/CPU/EvalAll/20"))
        self.assertFalse(re.search(scalar, "dpf-cpp/CPU/EvalAll/24"))
        raw = self.directory / "raw.json"
        raw.write_text(json.dumps({"benchmarks": [{"name": "dpf-cpp/CPU/EvalAll/16", "aggregate_name": None}]}))
        sweep.validate_native_configuration(
            self.case(library="dpf-cpp", operation="EvalAll", domain_bits=16), raw)
        with self.assertRaises(ValueError):
            sweep.validate_native_configuration(
                self.case(library="dpf-cpp", operation="EvalAll", domain_bits=20), raw)

    def test_vidpf_and_simd8_normalization(self):
        mastic = sweep.normalize(self.case(library="mastic", platform="cpu", operation="EvalAll", num_keys=1),
                                 "mastic/CPU/EvalAll/16", 225000, "raw", {})
        self.assertEqual(mastic["scheme"], "VIDPF")
        self.assertEqual(mastic["evaluation_parties"], "both")
        self.assertIn("digest", mastic["timing_includes"])
        self.assertTrue(mastic["variant"].startswith("VIDPF-"))
        plasma = sweep.normalize(self.case(library="plasma", platform="cpu", operation="Eval", num_keys=1),
                                 "plasma/CPU/Eval/16", 4600, "raw", {})
        self.assertEqual((plasma["scheme"], plasma["evaluation_parties"]), ("VIDPF", "one"))
        scalar = sweep.normalize(self.case(library="dpf-cpp", platform="cpu", operation="EvalAll", num_keys=1),
                                 "dpf-cpp/CPU/EvalAll/18", 1000, "raw", {})
        simd = sweep.normalize(self.case(library="dpf-cpp", platform="cpu", operation="EvalAll8", num_keys=1),
                               "dpf-cpp/CPU/EvalAll8/18", 100, "raw", {})
        self.assertEqual(simd["operation"], "EvalAll")
        self.assertNotEqual(simd["variant"], scalar["variant"])
        self.assertEqual(simd["implementation"], "simd8_frontier")

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

    def test_full_case_resource_failure_does_not_invent_point_variant(self):
        case = self.case(operation="EvalAll", sweep="block", threads_per_block=1024)
        args = argparse.Namespace(cpu=100, repetitions=5, min_time=.1, warmup=.1)
        def execute(command, **kwargs):
            raw = Path(next(argument.split("=", 1)[1] for argument in command if argument.startswith("--benchmark_out=")))
            raw.write_text('{"benchmarks": [')
            kwargs["log"].write_text("cuda error: too many resources requested for launch\n")
            raise RuntimeError("command exited with status -6")
        with mock.patch.object(sweep, "environment", return_value={}), \
             mock.patch.object(sweep, "executables", return_value=[Path("bench")]), \
             mock.patch.object(sweep.bench, "execute", side_effect=execute):
            row, = sweep.collect(args, case, self.directory / "full-case", {})
        self.assertEqual((row["failure_scope"], row["operation"], row["timed_operation"]), ("case", "EvalAll", "EvalAll"))
        self.assertEqual(row["original_benchmark"], sweep.pattern(case))
        self.assertEqual(row["failure_config_id"], sweep.case_id(case))
        for field in ("benchmark", "scheme", "variant", "prg", "key_count", "query_count", "domain_output_count", "ns_per_unit"):
            self.assertIsNone(row[field], field)
        # Apply the same explicit correction when loading an old fallback row.
        legacy = dict(row, benchmark="fss/GPU/DPF-bytes/AesSoft/Eval", scheme="DPF", failure_scope="benchmark")
        legacy.pop("original_benchmark")
        repaired = sweep.polish_metadata(legacy)
        self.assertEqual(repaired["original_benchmark"], legacy["benchmark"])
        self.assertIsNone(repaired["benchmark"])
        for field in ("raw_result", "raw_result_sha256", "source_sha256", "command", "error", "median_ns"):
            self.assertEqual(repaired[field], legacy[field])
        self.assertEqual(sweep.polish_metadata(repaired), repaired)
        point_case = dict(case, operation="Eval")
        repeated = [sweep.normalize(point_case, "fss/GPU/DPF-bytes/AesSoft/Eval/manual_time", None,
                                   "native-error-receipt", {}, "unsupported",
                                   "unsupported: insufficient software AES kernel resources for the block size")
                    for _ in range(5)]
        self.assertEqual({item["failure_config_id"] for item in repeated}, {sweep.case_id(point_case)})
        self.assertTrue(all(item["failure_scope"] == "benchmark" for item in repeated))

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

    def test_dmpf_gpu_reports_its_fixed_block_size(self):
        raw = self.directory / "raw.json"
        raw.write_text(json.dumps({"benchmarks": [
            dict(name="fss/GPU/DMPF-bytes/EvalAll", domain_bits=12, keys=262144, threads_per_block=256)]}))
        # The DMPF GPU kernels launch fixed 256-thread blocks, so the reported
        # size is accepted against a smaller configured EvalAll default.
        sweep.validate_native_configuration(self.case(operation="EvalAll", threads_per_block=128), raw)
        raw.write_text(json.dumps({"benchmarks": [
            dict(name="fss/GPU/DPF-bytes/EvalAll", domain_bits=12, keys=262144, threads_per_block=256)]}))
        with self.assertRaisesRegex(ValueError, "threads_per_block"):
            sweep.validate_native_configuration(self.case(operation="EvalAll", threads_per_block=128), raw)

    def test_scheme_filter_restricts_native_collections(self):
        cases = sweep.cases(self.args(libraries="fss", platform="gpu", domain_bits=[12], scheme="DMPF"))
        self.assertTrue(cases)
        for case in cases:
            self.assertEqual(case["scheme"], "DMPF")
            selected = sweep.pattern(case)
            self.assertRegex(f"fss/GPU/DMPF-bytes/{case['operation']}", selected)
            self.assertNotRegex("fss/GPU/DPF-bytes/EvalAll", selected)
            self.assertNotRegex("fss/GPU/GrottoDCF/EvalAll", selected)
        with self.assertRaisesRegex(ValueError, "unknown sweep scheme"):
            sweep.cases(self.args(libraries="fss", scheme="Nope"))

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

    def test_merge_drop_removes_library_platform_rows(self):
        source = self.directory / "source"
        rows = [sweep.normalize(self.case(), "fss/GPU/DPF-bytes/EvalAll/manual_time", 10, "raw", {}),
                sweep.normalize(self.case() | {"library": "fss_v070"}, "fss-v0.7.0/GPU/uint/DPF/Gen/manual_time", 10,
                                "raw-v070", {})]
        sweep.write_export(source, {}, rows)
        self.assertEqual(sweep.merge(self.directory / "merged", [source], drops=("fss_v070:gpu:upstream crash",)), 0)
        result = json.loads((self.directory / "merged" / "chart-data.json").read_text())
        self.assertEqual([row["benchmark"] for row in result["records"]], ["fss/GPU/DPF-bytes/EvalAll/manual_time"])
        self.assertEqual(result["metadata"]["dropped"], [dict(library="fss_v070", platform="gpu", reason="upstream crash", rows=1)])

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

    def test_retry_supersedes_only_the_same_scheme_and_variant(self):
        original = self.directory / "original"
        retry = self.directory / "retry"
        dpf = sweep.normalize(self.case(operation="EvalAll"), "fss/CPU/DPF-bytes/EvalAll/8", 10, "old-dpf", {})
        vdpf = sweep.normalize(self.case(operation="EvalAll"), "fss/CPU/VDPF-bytes/EvalAll/8", 20, "old-vdpf", {})
        sweep.write_export(original, {}, [dpf, vdpf])
        # The retry re-measures the DPF series at the same domain and operation; the
        # retry key must keep it from superseding the VDPF row as well.
        refreshed = sweep.normalize(self.case(operation="EvalAll"), "fss/CPU/DPF-bytes/EvalAll/8", 5, "new-dpf", {})
        sweep.write_export(retry, {}, [refreshed])
        self.assertEqual(sweep.merge(self.directory / "merged", [original], [retry]), 0)
        result = json.loads((self.directory / "merged" / "chart-data.json").read_text())
        by_raw = {row["raw_result"]: row for row in result["records"]}
        self.assertEqual(set(by_raw), {"old-vdpf", "new-dpf"})
        self.assertEqual(result["metadata"]["superseded_records"][0]["record"]["raw_result"], "old-dpf")

    def test_retry_supersedes_rows_exported_before_variant_qualifiers(self):
        # An input chart-data written before a qualifier existed keeps its old
        # variant; polishing on load must let a qualified retry supersede it.
        legacy = self.directory / "legacy"
        legacy.mkdir()
        cpu_case = self.case(operation="EvalAll") | {"platform": "cpu"}
        row = sweep.polish_metadata(sweep.normalize(cpu_case, "fss/CPU/VDPF-bytes/EvalAll/8", 10, "legacy-raw", {}))
        row["variant"] = row["variant"].replace("-chains", "")
        row["label"] = "fss/" + row["variant"]
        (legacy / "chart-data.json").write_text(json.dumps(dict(schema_version=1, metadata={}, records=[row])))
        retry = self.directory / "retry"
        sweep.write_export(retry, {}, [sweep.normalize(cpu_case, "fss/CPU/VDPF-bytes/EvalAll/8", 5, "new-raw", {})])
        self.assertEqual(sweep.merge(self.directory / "merged", [legacy], [retry]), 0)
        result = json.loads((self.directory / "merged" / "chart-data.json").read_text())
        self.assertEqual([row["raw_result"] for row in result["records"]], ["new-raw"])
        self.assertTrue(result["records"][0]["variant"].endswith("-chains"))
        self.assertEqual(result["metadata"]["superseded_records"][0]["record"]["raw_result"], "legacy-raw")

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

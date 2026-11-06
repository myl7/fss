import argparse
import importlib.util
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True

SPEC = importlib.util.spec_from_file_location("benchmark_policy",
                                              Path(__file__).with_name("benchmark_policy.py"))
benchmark_policy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark_policy)


def arguments(**extra):
    defaults = dict(profile=None, smoke=False, min_time=None, repetitions=None, warmup=None,
                    workload=None, full_launches=None, libraries=None, domain_bits=None)
    return argparse.Namespace(**(defaults | extra))


class ResolvePolicyTest(unittest.TestCase):
    def test_decision_is_the_default_profile_for_both_runners(self):
        for runner in benchmark_policy.RUNNERS:
            policy = benchmark_policy.resolve_policy(arguments(), runner)
            self.assertEqual((policy["profile"], policy["workload"], policy["full_launches"]),
                             ("decision", "daily", 1))
            self.assertEqual((policy["min_time_s"], policy["repetitions"], policy["warmup_s"]),
                             (0.3, 1, 0.0))
            self.assertEqual(policy["sources"]["profile"], "default")

    def test_explicit_timing_overrides_the_profile(self):
        policy = benchmark_policy.resolve_policy(arguments(min_time=2, repetitions=3, warmup=.5),
                                                "figure")
        self.assertEqual((policy["min_time_s"], policy["repetitions"], policy["warmup_s"]),
                         (2, 3, .5))
        self.assertEqual([policy["sources"][key] for key in ("min_time_s", "repetitions", "warmup_s")],
                         ["explicit", "explicit", "explicit"])

    def test_publication_restores_each_runner_historical_timing(self):
        policy = benchmark_policy.resolve_policy(arguments(profile="publication"), "bench")
        self.assertEqual((policy["min_time_s"], policy["repetitions"]), (1.0, 5))
        policy = benchmark_policy.resolve_policy(arguments(profile="publication"), "figure")
        self.assertEqual((policy["min_time_s"], policy["repetitions"], policy["warmup_s"]),
                         (0.1, 5, 0.1))

    def test_smoke_alias_and_conflicting_explicit_profiles_fail(self):
        policy = benchmark_policy.resolve_policy(arguments(smoke=True), "bench")
        self.assertEqual(policy["profile"], "smoke")
        self.assertEqual((policy["min_time_s"], policy["repetitions"]), (0.01, 1))
        self.assertEqual(benchmark_policy.resolve_policy(arguments(smoke=True, profile="smoke"),
                                                        "bench")["profile"], "smoke")
        with self.assertRaisesRegex(ValueError, "conflict"):
            benchmark_policy.resolve_policy(arguments(smoke=True, profile="decision"), "bench")

    def test_workload_legacy_restores_1024_full_launches(self):
        policy = benchmark_policy.resolve_policy(arguments(workload="legacy"), "bench")
        self.assertEqual(policy["full_launches"], 1024)
        self.assertEqual(policy["sources"]["full_launches"], "legacy workload")
        policy = benchmark_policy.resolve_policy(arguments(workload="legacy", full_launches=1024),
                                                "bench")
        self.assertEqual(policy["full_launches"], 1024)

    def test_full_launches_range_and_legacy_conflict(self):
        policy = benchmark_policy.resolve_policy(arguments(full_launches=512), "bench")
        self.assertEqual((policy["full_launches"], policy["sources"]["full_launches"]), (512, "explicit"))
        for value in (0, 1025, -1):
            with self.assertRaisesRegex(ValueError, "1 to 1024"):
                benchmark_policy.resolve_policy(arguments(full_launches=value), "bench")
        with self.assertRaisesRegex(ValueError, "legacy"):
            benchmark_policy.resolve_policy(arguments(workload="legacy", full_launches=512), "bench")

    def test_unknown_runner_rejected(self):
        with self.assertRaisesRegex(ValueError, "runner"):
            benchmark_policy.resolve_policy(arguments(), "native")

    def test_decision_and_publication_selection_defaults(self):
        for profile, libraries in (("decision", "main"), ("smoke", "main"), ("publication", "all")):
            args = arguments(libraries=None, profile=profile)
            policy = benchmark_policy.resolve_policy(args, "bench")
            benchmark_policy.apply_default_selection(args, "bench", policy["profile"], policy)
            self.assertEqual(args.libraries, libraries)
            self.assertEqual(policy["sources"]["libraries"], "profile default")
        args = arguments(libraries="fss,torchcsprng")
        policy = benchmark_policy.resolve_policy(args, "bench")
        benchmark_policy.apply_default_selection(args, "bench", policy["profile"], policy)
        self.assertEqual(args.libraries, "fss,torchcsprng")
        self.assertNotIn("libraries", policy["sources"])

    def test_figure_decision_selection_defaults_and_explicit_override(self):
        args = arguments()
        policy = benchmark_policy.resolve_policy(args, "figure")
        benchmark_policy.apply_default_selection(args, "figure", policy["profile"], policy)
        self.assertEqual((args.libraries, args.domain_bits), ("fss", [8, 20]))
        args = arguments(libraries="libdpf", domain_bits=[16])
        policy = benchmark_policy.resolve_policy(args, "figure")
        benchmark_policy.apply_default_selection(args, "figure", policy["profile"], policy)
        self.assertEqual((args.libraries, args.domain_bits), ("libdpf", [16]))
        args = arguments(profile="publication")
        policy = benchmark_policy.resolve_policy(args, "figure")
        benchmark_policy.apply_default_selection(args, "figure", policy["profile"], policy)
        self.assertIsNone(args.libraries)
        self.assertIsNone(args.domain_bits)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Reproducible domain and CUDA block-size sweeps for the README figures.

Raw logs, native results, build caches, and normalized exports stay under build.
Run `list` for exact cases. The domain endpoint is run first as a cost pilot.
"""
import argparse
import csv
import datetime
import hashlib
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import statistics
import sys

sys.dont_write_bytecode = True

import bench

ROOT = bench.ROOT
BUILD = bench.BUILD / "sweep"
DOMAINS = (8, 10, 12, 14, 16, 18, 20)
SLOW_DOMAINS = (8, 12, 14, 18, 20)
BLOCKS = (16, 32, 64, 128, 256, 512, 1024)
# GPU Eval cases use the same registry for default and per-scheme collections.
FSS_GPU_OPERATIONS = {
    "DPF": ("Gen", "Eval", "EvalAll"), "DCF": ("Gen", "Eval", "EvalAll"),
    "HalfTreeDPF": ("EvalAll",), "PackedHalfTreeDPF": ("EvalAll",),
    "GrottoDCF": ("EvalAll",), "VDPF": ("EvalAll",),
    "DMPF": ("Eval", "EvalAll"), "VDMPF": ("Eval", "EvalAll"),
}
FSS_CPU_OPERATIONS = {scheme: ("EvalAll",) if scheme == "PackedHalfTreeDPF" else ("Gen", "Eval", "EvalAll") for scheme in FSS_GPU_OPERATIONS}
GPU_POINT_SCHEMES = tuple(scheme for scheme, ops in FSS_GPU_OPERATIONS.items() if "Eval" in ops)
# These preserved wrappers use one key and kN queries, with fixed 256-thread
# kernels. Their raw receipts predate explicit batch counters. The hashes were
# verified against e25a42c and 84b93e5, respectively.
LEGACY_BATCH_WRAPPERS = {
    "2e9a30754964d05df3522d72a30c3174d6b57a6f8c683375c3facd638b52c534",
    "e09d80c6f73f093847b3d6b6d154df5aa2920b60bb02318e177790dd3fdf9049",
}
CPU_LIBRARIES = ("fss", "libdpf", "libfss", "google_dpf", "gpu_dpf", "fss_v060", "fss_v070", "servan_vdpf",
                 "mastic", "plasma", "dpf-cpp", "dpfplusplus")
TARGETS = {"fss": "bench_fss", "libfss": "bench_dpf_libfss", "gpu_dpf": "bench_dpf_gpu_dpf",
           "ezpc": "bench_dpf_ezpc", "torchcsprng": "bench_aes128_soft", "servan_vdpf": "bench_vdpf_servan",
           "dpf-cpp": "bench_dpf_cpp", "dpfplusplus": "bench_dpfplusplus"}
# Operations and domains the new third-party entries register; sweeps skip the
# rest. sycret stays bench.py-only because its EQ/CMP batch operations do not
# follow the domain-sweep model.
NEW_LIBRARY_DOMAINS = {
    "mastic": {"Gen": (12, 16, 20), "Eval": (16,), "EvalAll": (12, 16, 20)},
    "plasma": {"Gen": (16, 20), "Eval": (16,)},
    "dpf-cpp": {"Gen": (16, 20), "Eval": (16, 20), "EvalAll": (16, 18, 20), "EvalAll8": (16, 18, 20)},
    "dpfplusplus": {"Gen": (16, 20), "Eval": (16, 20), "EvalAll": (16, 18, 20)},
}


def cases(args):
    result = []
    domains = tuple(args.domain_bits) if args.domain_bits else DOMAINS
    libraries = args.libraries.split(",") if args.libraries else None
    for device in (("cpu", "gpu") if args.platform == "all" else (args.platform,)):
        names = CPU_LIBRARIES if device == "cpu" else ("fss", "gpu_dpf", "ezpc", "fss_v070")
        for name in names:
            if libraries and name not in libraries:
                continue
            for n in sorted(domains, reverse=True):
                requested = getattr(args, "operations", None) or ("Gen", "Eval", "EvalAll")
                registered = NEW_LIBRARY_DOMAINS.get(name)
                if registered:
                    # Registered libraries sweep the operations their wrappers
                    # expose. EvalAll8 rides along with an EvalAll request; the
                    # CLI operation choices cannot name it directly.
                    selected = [op for op in requested if op in registered]
                    if "EvalAll" in requested and "EvalAll8" in registered and "EvalAll8" not in selected:
                        selected.append("EvalAll8")
                    operations = sorted(selected)
                else:
                    operations = requested or ("Gen", "Eval", "EvalAll")
                for operation in operations:
                    scheme = getattr(args, "scheme", None)
                    registry = FSS_GPU_OPERATIONS if device == "gpu" else FSS_CPU_OPERATIONS
                    if name == "fss" and scheme and operation not in registry.get(scheme, ()):
                        continue
                    if device == "gpu" and name == "gpu_dpf" and operation != "EvalAll":
                        continue
                    if operation == "EvalAll" and device == "cpu" and name in ("libfss", "gpu_dpf") and n not in SLOW_DOMAINS:
                        continue
                    if registered and n not in registered[operation]:
                        continue
                    if args.sweep == "block" and (device != "gpu" or name != "fss" or n != 20):
                        continue
                    if args.sweep == "block" and (operation == "Gen" or scheme in ("DMPF", "VDMPF")
                                                  or operation == "Eval" and scheme not in (None, "DPF")):
                        continue
                    blocks = args.threads_per_block or (BLOCKS if args.sweep == "block" else (128 if operation == "EvalAll" else 256,))
                    for t in blocks:
                        if device == "cpu":
                            t = None
                        result.append(dict(library=name, platform=device, operation=operation,
                                           domain_bits=n, domain_size=1 << n,
                                           configured_num_keys=args.num_keys if device == "gpu" else 1,
                                           num_keys=1 if device == "cpu" or operation == "EvalAll" else args.num_keys,
                                           threads_per_block=(256 if name == "ezpc" else t), cpu_prg=args.cpu_prg,
                                           sweep=args.sweep, include_uint=getattr(args, "include_uint", False),
                                           scheme=getattr(args, "scheme", None)))
                        if device == "cpu":
                            break
    if libraries and set(libraries) - set(CPU_LIBRARIES) - {"ezpc"}:
        raise ValueError("unknown sweep library")
    scheme = getattr(args, "scheme", None)
    if scheme and scheme not in FSS_GPU_OPERATIONS:
        raise ValueError("unknown sweep scheme")
    return result


def case_id(case):
    return "-".join(str(case[k]) for k in ("library", "platform", "operation", "domain_bits", "threads_per_block"))


def environment(args, case):
    env = bench.environment(args, case["library"])
    cache = BUILD / case["library"] / case["platform"]
    env.update(CARGO_TARGET_DIR=str(cache / "target"), FSS_BENCH_DOMAIN_BITS=str(case["domain_bits"]))
    return env


def pattern(case):
    operation = case["operation"]
    if case["library"] == "fss_v060" and operation == "EvalAll":
        operation = "FullEval"
    if case["platform"] == "gpu" and case["library"] in ("gpu_dpf", "ezpc") and operation == "EvalAll":
        operation = "EvalAllFull"
    if case["library"] == "fss" and case["platform"] == "cpu":
        groups = "(bytes|uint)" if case["include_uint"] else "bytes"
        # Preparation is collected alongside Eval without changing CLI defaults.
        operation = "(Eval|Prepare)" if operation == "Eval" else operation
        # The packed-key point evaluations and the CPU VDPF Merkle fold carry a
        # middle name segment, mirroring the GPU AesSoft variant; the optional
        # segment keeps every plain name matching exactly as before. The DMPF and
        # VDMPF AES-NI raw-PRP twins append the backend to the scheme-group name.
        segment = "(PackedKeyOneShot|PackedKey|Merkle)/"
        rawprp = "" if case.get("scheme") and case["scheme"] not in ("DMPF", "VDMPF") else "(-AES-NI-rawprp)?"
        if case.get("scheme"):
            if case["scheme"] in ("PackedHalfTreeDPF", "GrottoDCF"):
                native = "PackedHalfTreeDPF-bits1" if case["scheme"] == "PackedHalfTreeDPF" else "GrottoDCF"
                return f"^fss/CPU/{native}/{operation}(/|$)"
            return f"^fss/CPU/{case['scheme']}-{groups}{rawprp}/({segment})?{operation}(/|$)"
        return (f"^fss/CPU/((DPF|DCF|HalfTreeDPF|VDPF)-{groups}|(DMPF|VDMPF)-{groups}{rawprp}"
                f"|PackedHalfTreeDPF-bits1|GrottoDCF)/({segment})?{operation}(/|$)")
    if case["library"] == "ezpc" and case["operation"] != "EvalAll":
        return f"^EzPC/GPU/(DPF|DCF)/{operation}/{case['num_keys']}(/|$)"
    prefix = "fss" if case["library"] == "fss" else ".*"
    if case["sweep"] == "block" and operation == "Eval":
        return "^fss/GPU/DPF-bytes/AesSoft/Eval(/|$)"
    if case["library"] == "fss":
        groups = "(bytes|uint)" if case["include_uint"] else "bytes"
        if case.get("scheme"):
            if case["scheme"] in ("PackedHalfTreeDPF", "GrottoDCF"):
                native = "PackedHalfTreeDPF-bits1" if case["scheme"] == "PackedHalfTreeDPF" else "GrottoDCF"
                return f"^fss/GPU/{native}/{operation}(/|$)"
            return f"^fss/GPU/{case['scheme']}-{groups}/(AesSoft/)?{operation}(/|$)"
        selected = [scheme for scheme, operations in FSS_GPU_OPERATIONS.items() if operation in operations
                    and not (case["sweep"] == "block" and scheme in ("DMPF", "VDMPF"))]
        native = ["PackedHalfTreeDPF-bits1" if scheme == "PackedHalfTreeDPF" else "GrottoDCF"
                  if scheme == "GrottoDCF" else f"{scheme}-{groups}" for scheme in selected]
        return f"^fss/GPU/({'|'.join(native)})/(AesSoft/)?{operation}(/|$)"
    if case["library"] in ("dpf-cpp", "dpfplusplus"):
        # Their benchmarks register several domain sizes under one operation,
        # so the filter must pin the requested case; otherwise one raw file
        # mixes 16/18/20/24 and the native-id consistency check rejects it.
        return f"^{case['library']}/{case['platform'].upper()}/{operation}/{case['domain_bits']}(/|$)"
    return f"^{prefix}/{case['platform'].upper()}/.*{operation}(/|$)"


def configuration(case):
    return [f"-DFSS_BENCH_DOMAIN_BITS={case['domain_bits']}",
            f"-DFSS_BENCH_THREADS_PER_BLOCK={case['threads_per_block'] or 256}",
            f"-DFSS_BENCH_NUM_KEYS={case['configured_num_keys']}",
            f"-DFSS_BENCH_CPU_PRG={case['cpu_prg']}"]


def build_execute(args, command, **kwargs):
    if platform.system() == "Linux":
        available = sorted(os.sched_getaffinity(0) - {args.cpu, 16, 24})
        if not available:
            raise RuntimeError("no CPU remains for builders after reserving benchmark cores")
        command = ["taskset", "-c", ",".join(map(str, available)), *command]
    bench.execute(command, **kwargs)


def build_case(args, case):
    name, device = case["library"], case["platform"]
    cache = BUILD / name / device
    cache.mkdir(parents=True, exist_ok=True)
    env = environment(args, case)
    if name in bench.RUST:
        target = "bench" if name == "libdpf" else "bench_dpf"
        build_execute(args, ["cargo", f"+{bench.RUST[name]}", "bench", "--locked", "--no-run", "--bench", target,
                       "--jobs", str(args.jobs)], cwd=bench.source(name), env=env)
    elif name == "google_dpf":
        build_execute(args, ["bazel", f"--output_user_root={cache / 'bazel'}", "build", ":bench_dpf_google",
                       f"--define=fss_bench_domain_bits={case['domain_bits']}", f"--jobs={args.jobs}",
                       "--symlink_prefix=/"], cwd=bench.source(name), env=env)
    else:
        target = TARGETS.get(name)
        extra = configuration(case)
        if name == "fss":
            pass
        if name == "dpfplusplus":
            fetched = bench.BUILD / "dpfplusplus" / "src" / "include"
            if not fetched.exists():
                bench.execute(["bash", str(bench.source(name) / "fetch.sh"),
                               str(bench.BUILD / "dpfplusplus")], env=env)
            extra.append(f"-DDPFPP_INCLUDE_DIR={fetched}")
        if name == "gpu_dpf" and device == "gpu":
            extra.append("-DFSS_BENCH_BUILD_GPU_FULL=ON")
            target = "bench_dpf_gpu_dpf_full"
        if name == "fss_v070":
            extra.append(f"-DFSS070_BUILD_GPU={'ON' if device == 'gpu' else 'OFF'}")
        if name == "ezpc":
            dependency = bench.BUILD / name / "sytorch"
            if not (dependency / "CMakeCache.txt").exists():
                build_execute(args, ["cmake", "-S", bench.source(name) / "sytorch", "-B", dependency,
                               "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_CUDA_ARCHITECTURES={args.cuda_arch or 'native'}"], env=env)
                build_execute(args, ["cmake", "--build", dependency, "--target", "sytorch", "--parallel", str(args.jobs)], env=env)
            extra.append(f"-DEZPC_SYTORCH_BUILD={dependency}")
        # Share downloaded Benchmark sources, while keeping its compiled cache per platform.
        downloaded = sorted(bench.BUILD.glob("*/cmake/_deps/benchmark-src"))
        if downloaded:
            extra.append(f"-DFETCHCONTENT_SOURCE_DIR_BENCHMARK={downloaded[0]}")
        command = ["cmake", "-S", bench.source(name), "-B", cache / "cmake", "-DCMAKE_BUILD_TYPE=Release",
                   "-DBUILD_TESTING=OFF", *extra]
        if args.cuda_arch:
            command.append(f"-DCMAKE_CUDA_ARCHITECTURES={args.cuda_arch}")
        build_execute(args, command, env=env)
        targets = [target] if target else [f"bench_{device}_{g}_{s}" for g in (("uint", "bytes") if case["include_uint"] else ("bytes",)) for s in ("dpf", "dcf")]
        build_execute(args, ["cmake", "--build", cache / "cmake", "--parallel", str(args.jobs), "--target", *targets], env=env)


def executables(args, case):
    name, device = case["library"], case["platform"]
    cache = BUILD / name / device
    if name == "google_dpf":
        output = bench.capture(["bazel", f"--output_user_root={cache / 'bazel'}", "info", "bazel-bin"],
                               cwd=bench.source(name), env=environment(args, case))
        return [Path(output) / "bench_dpf_google"]
    if name == "fss_v070":
        return [cache / "cmake" / f"bench_{device}_{g}_{s}" for g in (("uint", "bytes") if case["include_uint"] else ("bytes",)) for s in ("dpf", "dcf")]
    target = "bench_dpf_gpu_dpf_full" if name == "gpu_dpf" and device == "gpu" else TARGETS[name]
    return [cache / "cmake" / ("bench_dpf_gpu_dpf_full" if name == "gpu_dpf" and device == "gpu" else "bench")]


def parse_google(path):
    values = json.loads(path.read_text())["benchmarks"]
    grouped, failures = {}, []
    for value in values:
        name = value.get("run_name", value["name"])
        if value.get("error_occurred"):
            failures.append((name, value.get("error_message", "benchmark failed")))
            continue
        aggregate = value.get("aggregate_name")
        if aggregate and aggregate != "median":
            continue
        scale = {"ns": 1, "us": 1e3, "ms": 1e6, "s": 1e9}[value["time_unit"]]
        time = value["real_time"] * scale
        if not math.isfinite(time) or time <= 0:
            raise ValueError("invalid benchmark time")
        group = grouped.setdefault(name, {"samples": [], "median": None})
        if aggregate:
            group["median"] = time
        else:
            group["samples"].append(time)
    return [(name, group["median"] if group["median"] is not None else statistics.median(group["samples"]))
            for name, group in grouped.items()], failures


def normalize(case, name, time, raw, sources, status="ok", error=None):
    if "/Prepare" in name:
        case = dict(case, requested_operation=case["operation"], operation="Prepare")
    if case["library"] == "google_dpf":
        case = dict(case, google_dcf_group="additive_mod_2^128")
    row = dict(case, benchmark=name, status=status, median_ns=time, time_ns=time,
               ns_per_key=time / case["num_keys"] if time is not None else None,
               raw_result=str(raw), source_sha256=sources, error=error)
    scheme = ("PackedHalfTreeDPF" if "PackedHalfTree" in name else "GrottoDCF" if "GrottoDCF" in name else
              "VDPF" if "VDPF" in name else "VDMPF" if "VDMPF" in name else "DMPF" if "DMPF" in name else
              "HalfTreeDPF" if "HalfTree" in name else "DCF" if "DCF" in name else "DPF")
    if case["library"] in ("mastic", "plasma"):
        # VIDPF prototypes: verifiable-incremental DPFs, not plain DPF rows.
        scheme = "VIDPF"
    group = "bytes" if "bytes" in name else "uint" if "uint" in name else "native"
    prg = "AES-software" if "AesSoft" in name else ("ChaCha12" if case["library"] == "gpu_dpf" and case["platform"] == "gpu" else
          "AES-NI" if case["platform"] == "cpu" and case["library"] == "fss" and case["cpu_prg"] == "aes-ni" else
          "OpenSSL-AES" if case["platform"] == "cpu" and case["library"] == "fss" else
          ("ChaCha20-x1" if scheme in ("HalfTreeDPF", "PackedHalfTreeDPF") else "ChaCha20-x4" if scheme == "DCF" else "ChaCha20-x2") if case["library"] == "fss" else "AES128" if case["library"] in ("gpu_dpf", "ezpc", "google_dpf") else "native")
    if case["library"] == "fss_v070":
        prg = "AES128-MMO-NI" if case["platform"] == "cpu" else "Salsa12"
    if case["library"] in ("dpf-cpp", "mastic"):
        prg = "AES128-MMO"
    elif case["library"] == "dpfplusplus":
        prg = "AES128"
    elif case["library"] == "plasma":
        prg = "AES128-CTR"
    bits = 127 if group in ("bytes", "uint") and case["library"] in ("fss", "fss_v070") else 128 if group in ("bytes", "uint") else None
    row.update(scheme=scheme, group=group, prg=prg, logical_output_bits=bits,
               scan="threads" if case["sweep"] == "block" else "domain",
               variant=scheme + "-" + group + "-" + prg,
               label=case["library"] + "/" + scheme + "-" + group + "/" + prg,
               implementation="materialized_point_loop" if case["library"] in ("libfss", "gpu_dpf") and case["platform"] == "cpu" and case["operation"] == "EvalAll" else "native",
               output_storage="packed_bits" if case["library"] == "ezpc" and case["operation"] == "EvalAll" else "materialized",
               timing_boundary="cuda_event_kernel" if case["platform"] == "gpu" else "wall_time_operation",
               statistic="median", config=dict(case), storage_output_bits=128 if group in ("bytes", "uint") else bits)
    if scheme == "PackedHalfTreeDPF":
        width = int(next(iter(re.findall(r"-bits(\d+)", name)), 1))
        row.update(group=f"bits{width}", logical_output_bits=width, storage_output_bits=128,
                   output_storage="packed_lanes", lanes=128 // width)
    elif scheme == "GrottoDCF":
        row.update(logical_output_bits=1, storage_output_bits=8, output_storage="bool_scalar")
    elif scheme == "VDPF" and case["platform"] == "gpu" and case["operation"] == "EvalAll":
        # Merkle-fold proof accumulation computed on device; the paper-faithful
        # chain stays on the CPU curves (doc/vdpf_merkle_fold_security.md).
        row.update(implementation="merkle_fold_proof")
    elif scheme == "VDPF" and case["platform"] == "cpu" and "Merkle" in name:
        # CPU Merkle-fold row: the counterpart of the GPU row above, kept apart
        # from the paper-faithful chain row by its variant qualifier.
        row.update(implementation="merkle_fold_proof")
    elif scheme == "VDPF" and case["library"] == "servan_vdpf":
        # Single-point VDPF reference implementation (sachaservan/vdpf).
        row.update(logical_output_bits=1, storage_output_bits=128, output_storage="uint128_scalar", num_points=1)
    elif scheme in ("DMPF", "VDMPF"):
        # Number of (alpha, beta) point pairs fixed by the fss CPU bench configuration.
        row["num_points"] = 64
    if case["library"] == "libdpf":
        row.update(logical_output_bits=1, output_storage="packed_128_binary_leaf", storage_output_bits=128)
    elif case["library"] in ("gpu_dpf", "google_dpf", "fss_v060"):
        row["logical_output_bits"] = 128
        if case["library"] == "fss_v060":
            row["prg"] = "AES128"
    elif case["library"] == "ezpc":
        row["logical_output_bits"] = 1
    elif case["library"] == "libfss":
        row.update(group="uint64" if scheme == "DCF" else "prime_field",
                   logical_output_bits=64 if scheme == "DCF" else 33,
                   output_storage="uint64_scalar" if scheme == "DCF" else "gmp_scalar")
    elif case["library"] == "mastic":
        # VIDPF prototype. EvalAll times eval_tree over the full domain for
        # BOTH parties and includes the proof digest comparison; each leaf
        # carries two Field128 payload words plus the appended counter.
        row.update(logical_output_bits=128, output_storage="field128x2_plus_counter",
                   implementation=("vidpf_eval_tree_proof_both_parties" if case["operation"] == "EvalAll"
                                   else "vidpf_chained_proof"),
                   evaluation_parties="both",
                   timing_includes=("proof digest comparison" if case["operation"] == "EvalAll"
                                    else "chained proof"))
    elif case["library"] == "plasma":
        # VIDPF prototype over u64 shares mod 2^63-25; point eval carries the
        # running proof. No full-domain API exists upstream.
        row.update(logical_output_bits=64, output_storage="u64_mod_p_scalar",
                   implementation="vidpf_chained_proof", evaluation_parties="one",
                   timing_includes="chained proof")
    elif case["library"] == "dpf-cpp":
        # Bit-valued point function; EvalFull returns a packed-bit truth table.
        row.update(logical_output_bits=1, output_storage="packed_bits", storage_output_bits=8)
        if "EvalAll8" in name:
            # The 8-wide SIMD frontier variant is its own series in the EvalAll
            # panels, never merged into the scalar EvalFull curve.
            row.update(operation="EvalAll", variant=row["variant"] + "-simd8",
                       implementation="simd8_frontier", label=row["label"] + " (SIMD8)")
    elif case["library"] == "dpfplusplus":
        # bool leaves packed 128 per __m128i node; Gen draws real randomness.
        row.update(logical_output_bits=1, output_storage="packed_128_binary_leaf", storage_output_bits=128)
    if case["library"] in ("fss", "fss_v070"):
        row["prg_output_blocks"] = ((2 if scheme == "DCF" else 1) if case["library"] == "fss_v070" and case["platform"] == "gpu" else
                                    4 if scheme == "DCF" else 1 if scheme in ("HalfTreeDPF", "PackedHalfTreeDPF") else 2)
    variant_suffix = "-simd8" if row.get("implementation") == "simd8_frontier" else ""
    row["variant"] = scheme + "-" + row["group"] + "-" + prg + variant_suffix
    row["label"] = case["library"] + "/" + row["variant"]
    return polish_metadata(row)


@lru_cache(maxsize=1024)
def native_receipts(raw, expected_hash):
    path = Path(raw)
    if not path.is_file() and "/build/" in raw:
        path = ROOT / "build" / raw.split("/build/", 1)[1]
    if not path.is_file():
        return {}
    content = path.read_bytes()
    if expected_hash and hashlib.sha256(content).hexdigest() != expected_hash:
        raise ValueError(f"raw result hash mismatch: {path}")
    try:
        values = json.loads(content).get("benchmarks", [])
    except (ValueError, UnicodeDecodeError):
        return {}
    return {value.get("run_name", value["name"]): value for value in values
            if not value.get("error_occurred") and value.get("aggregate_name") in (None, "median")}


def measurement_metadata(row):
    """Recover work quantities from receipts and documented wrapper contracts."""
    row = dict(row)
    receipt = native_receipts(str(row.get("raw_result", "")), row.get("raw_result_sha256")).get(row.get("benchmark"), {})
    fields = ("key_count", "query_count", "domain_outputs", "keys_per_launch", "threads_per_block")
    counters = {field: int(receipt[field]) for field in fields if field in receipt}
    row["native_counters"] = counters or row.get("native_counters", {})
    counters = row["native_counters"]
    row.setdefault("requested_threads_per_block", row.get("config", row).get("threads_per_block"))
    if "threads_per_block" in counters:
        row["threads_per_block"] = counters["threads_per_block"]
        row["launch_metadata_source"] = "raw_counter"
    batch = row["library"] == "fss" and row["platform"] == "gpu" and row.get("scheme") in ("DMPF", "VDMPF")
    legacy_batch = row.get("source_sha256", {}).get("third_party/fss/bench.cu") in LEGACY_BATCH_WRAPPERS
    if batch and legacy_batch and not counters.get("threads_per_block"):
        row.update(threads_per_block=256, launch_metadata_source="verified_wrapper_source")
    if batch:
        row["block_size_tunable"] = False
    else:
        row.setdefault("block_size_tunable", True)
    keys = counters.get("key_count", counters.get("keys_per_launch"))
    if keys is None:
        keys = 1 if row["operation"] == "EvalAll" or row["platform"] == "cpu" else row.get("num_keys", 1)
    if batch:
        if counters.get("key_count", counters.get("keys_per_launch")) == 1 or legacy_batch:
            keys = 1
            row["quantity_source"] = "raw_counter" if counters.get("key_count", counters.get("keys_per_launch")) else "verified_wrapper_source"
            if row.get("metadata_status") == "requires_recollection":
                row.pop("metadata_status")
        else:
            # Single-key batch semantics cannot be established from an old
            # num_keys field alone. Keep the receipt but omit it from plots.
            row["metadata_status"] = "requires_recollection"
    row["key_count"] = keys
    op = row["operation"]
    query_count = counters.get("query_count")
    if query_count is None:
        query_count = (row.get("configured_num_keys", row.get("num_keys", 1)) if batch else keys) if op == "Eval" else row["domain_size"] if batch and row.get("scheme") == "VDMPF" and op == "EvalAll" and legacy_batch else 0
    row["query_count"] = query_count
    row["domain_output_count"] = counters.get("domain_outputs", keys * row["domain_size"] if op == "EvalAll" else 0)
    row["ns_per_domain_output"] = row.get("median_ns", row.get("time_ns")) / row["domain_output_count"] if row.get("median_ns", row.get("time_ns")) is not None and row["domain_output_count"] else None
    row["legacy_quantity_fields"] = {field: row[field] for field in ("num_keys", "ns_per_key") if field in row}
    row["timed_operation"] = op
    row["time_denominator"] = "query" if op == "Eval" else "key"
    count = query_count if op == "Eval" else keys
    time = row.get("median_ns", row.get("time_ns"))
    row["ns_per_unit"] = time / count if time is not None and count else None
    row["ns_per_query"] = time / query_count if time is not None and query_count else None
    # Keep num_keys and historical ns_per_key for provenance. New consumers
    # use explicit counts and ns_per_unit.
    row["setup_included"] = op == "Prepare" or "PackedKeyOneShot" in row.get("benchmark", "") or row.get("scheme") == "GrottoDCF" and op == "EvalAll"
    row["workload"] = "single_key_batch" if batch and op == "Eval" else "independent_keys" if op == "Eval" and keys > 1 else "single_key"
    return row


def failure_metadata(row):
    """Describe a failed collection without inventing a successful variant."""
    resource_failure = row.get("failure_kind") == "kernel_resources_exhausted" or \
        "too many resources requested for launch" in str(row.get("error", "")) or \
        "kernel resources" in str(row.get("error", ""))
    if row.get("status") != "unsupported" or not resource_failure:
        return row
    row = dict(row)
    case = row.get("config", row)
    selected_filter = next((arg.split("=", 1)[1] for arg in row.get("command", ())
                            if arg.startswith("--benchmark_filter=")), "")
    expression = selected_filter.removesuffix("(/|$)")
    case_level = row.get("failure_scope") == "case" or "(" in expression or \
        str(row.get("benchmark", "")).startswith("^")
    row.update(operation=case["operation"], timed_operation=case["operation"],
               failure_scope="case" if case_level else "benchmark", failure_config_id=case_id(case))
    if not case_level:
        return row
    row.setdefault("original_benchmark", row.get("benchmark"))
    row.update(benchmark=None, scheme=None, variant=None, group=None, prg=None,
               logical_output_bits=None, storage_output_bits=None, output_storage=None,
               prg_output_blocks=None, implementation=None, key_count=None, query_count=None,
               domain_output_count=None, ns_per_unit=None, ns_per_query=None, ns_per_domain_output=None,
               setup_included=None, workload=None,
               time_denominator="query" if case["operation"] == "Eval" else "key",
               label=f"{row['library']}/{row['platform']}/{case['operation']} case",
               timing_boundary_detail="collection failed before a variant-level result was available")
    row["metadata_correction"] = "case-level resource failure has no identified benchmark variant"
    return row


def polish_metadata(row):
    """Refresh descriptive fields without changing preserved measurements."""
    row = failure_metadata(dict(row))
    if row.get("failure_scope") == "case":
        return row
    library, device, operation = row["library"], row["platform"], row["operation"]
    scheme = row.get("scheme", "DPF")
    benchmark = row.get("benchmark", "")
    if library == "fss_v070" and device == "gpu":
        row["prg"] = "Salsa12"
    if library == "gpu_dpf":
        row.update(logical_output_bits=128, storage_output_bits=128,
                   output_storage="uint128_scalar_bit_reversed" if device == "gpu" else "uint128_scalar")
        if device == "gpu" and operation == "EvalAll":
            row["output_order"] = "bit_reversed"
    if library in ("libdpf", "libfss") and row.get("prg", "native") == "native":
        row["prg"] = "AES128-MMO/RustCrypto" if library == "libdpf" else "AES128-MMO/OpenSSL"
        row["prg_backend_detail"] = ("RustCrypto aes 0.8.4 with runtime AES-NI dispatch" if library == "libdpf" else
                                     "OpenSSL low-level AES_encrypt; no AES-NI implementation claim")
        if library == "libfss":
            row["prg_output_blocks"] = 4 if scheme == "DCF" else 3
    if library == "servan_vdpf" and row.get("prg", "native") == "native":
        row.update(prg="AES128/OpenSSL",
                   prg_backend_detail="OpenSSL EVP AES-128: ECB for the PRG expansion and CTR for the MMO hash; "
                                      "research-purposes-only upstream code")
    if library == "fss" and device == "gpu" and "AesSoft" not in row.get("benchmark", ""):
        blocks = 4 if scheme == "DCF" else 1 if scheme in ("HalfTreeDPF", "PackedHalfTreeDPF") else 2
        row.update(prg=f"ChaCha20-x{blocks}", prg_output_blocks=blocks)
    if library == "google_dpf" and scheme == "DPF":
        row.update(group="xor_128", logical_output_bits=128, storage_output_bits=128)
    if library == "google_dpf" and scheme == "DCF":
        # Never reinterpret an explicitly historical XOR measurement. A matching
        # measured source hash proves that a legacy generic label used this adapter.
        group = str(row.get("group", "native"))
        relative = "third_party/distributed_point_functions/bench.cc"
        source = ROOT / relative
        recorded_hash = row.get("source_sha256", {}).get(relative)
        corrected_source = source.exists() and "DCF uses additive uint128 output" in source.read_text()
        same_source = corrected_source and recorded_hash == hashlib.sha256(source.read_bytes()).hexdigest()
        configured = row.get("config", {}).get("google_dcf_group") == "additive_mod_2^128"
        if "xor" not in group.lower() and (group == "additive_mod_2^128" or same_source or configured):
            row.update(group="additive_mod_2^128", logical_output_bits=128, storage_output_bits=128)
    if library == "libfss" and scheme == "DCF":
        row.update(group="additive_mod_2^64", logical_output_bits=64, storage_output_bits=64,
                   output_storage="uint64_scalar")
    if device == "gpu":
        if library == "ezpc" and operation != "EvalAll":
            row["timing_boundary"] = "cuda_event_native_key_generation_api" if operation == "Gen" else "cuda_event_native_point_evaluation_api"
            row["timing_boundary_detail"] = "CUDA events around native GPU API calls, including their GPU work and transfers within the event interval"
        elif library == "ezpc":
            row["timing_boundary"] = "cuda_event_packed_eval_all_kernel"
            row["timing_boundary_detail"] = "CUDA events around the packed full-domain adapter kernel for one party; setup and copies are outside timing"
        elif library == "gpu_dpf":
            row["timing_boundary"] = "cuda_event_native_dpf_hybrid_api"
            row["timing_boundary_detail"] = "CUDA events around the native dpf_hybrid call for one party; input key upload and result download are outside timing"
        else:
            row["timing_boundary"] = "cuda_event_operation_kernel_launches"
            row["timing_boundary_detail"] = "CUDA events around operation kernel launches; allocation and input copies are outside timing"
    else:
        row["timing_boundary"] = "wall_time_operation"
        row["timing_boundary_detail"] = "Native framework wall time per operation; setup outside the benchmark iteration is excluded"
        if library == "fss":
            if "PackedKeyOneShot" in benchmark:
                row["timing_boundary_detail"] = "Native framework wall time per point evaluation; the one-time key pack is inside timing"
            elif "/Prepare" in benchmark:
                row["timing_boundary_detail"] = "Native framework wall time per key preparation; key generation and all point queries are outside timing"
            elif "PackedKey" in benchmark:
                row["timing_boundary_detail"] = "Native framework wall time per prepared-key point evaluation; the one-time key pack is outside timing"
    # Variant qualifiers keep the packed-key rows and the two proof accumulators
    # apart in every legend and on the site. Polish recomputes them at export
    # and merge time, so rows collected before a qualifier existed align with
    # the current naming. The raw-PRP qualifier stacks onto the base one so the
    # CPU VDMPF chain row keeps its accumulator marker.
    qualifier = ""
    if library == "fss":
        if "PackedKeyOneShot" in benchmark:
            qualifier = "-packedkey-oneshot"
        elif "/Prepare" in benchmark:
            qualifier = "-packedkey-setup"
        elif "PackedKey" in benchmark:
            qualifier = "-packedkey"
        elif scheme == "VDPF" and operation in ("Eval", "EvalAll"):
            qualifier = "-merkle" if row.get("implementation") == "merkle_fold_proof" else "-chains"
        elif scheme == "VDMPF":
            qualifier = "-merkle-fold" if device == "gpu" else "-chains"
        if "AES-NI-rawprp" in benchmark:
            qualifier += "-rawprp"
    if library == "fss" and scheme in ("VDPF", "VDMPF"):
        row["proof_accumulator"] = "none" if operation == "Gen" else "merkle" if "-merkle" in qualifier else "chain"
        row["proof_generation_included"] = operation in ("Eval", "EvalAll")
        if scheme == "VDPF" and device == "cpu" and operation == "Eval":
            row["timing_boundary_detail"] = "Native framework wall time per point evaluation plus a one-element chain proof; key generation and validation are outside timing"
    variant_suffix = ("-simd8" if row.get("implementation") == "simd8_frontier" else "") + qualifier
    row["variant"] = scheme + "-" + row.get("group", "native") + "-" + row.get("prg", "native") + variant_suffix
    row["label"] = library + "/" + row["variant"]
    if row.get("scheme") == "PackedHalfTreeDPF":
        row["label"] += " (experimental)"
    return measurement_metadata(row)


def validate_native_configuration(case, raw):
    for value in json.loads(raw.read_text())["benchmarks"]:
        if value.get("error_occurred") or value.get("aggregate_name") not in (None, "median"):
            continue
        if case["library"] in ("dpf-cpp", "dpfplusplus"):
            # The new C++ wrappers carry the domain size in the benchmark ID
            # instead of custom counters; the ID must match the requested case.
            match = re.search(r"/(Gen|Eval|EvalAll8?)/(\d+)$", value.get("name", ""))
            if match and int(match.group(2)) != case["domain_bits"]:
                raise ValueError(f"benchmark id {value['name']} does not match the configured domain {case['domain_bits']}")
        if case["library"] == "fss":
            name = value.get("run_name", value.get("name", ""))
            match = re.search(r"/(Gen|Prepare|Eval|EvalAll)(/|$)", name)
            operation = match.group(1) if match else case["operation"]
            batch = case["platform"] == "gpu" and "DMPF" in name
            keys = 1 if case["platform"] == "cpu" or operation in ("Prepare", "EvalAll") or batch else case["num_keys"]
            queries = (case["num_keys"] if case["platform"] == "gpu" else 1) if operation == "Eval" else 0
            if batch and "VDMPF" in name and operation == "EvalAll":
                queries = case["domain_size"]
            expected_counts = {"key_count": keys, "query_count": queries,
                               "domain_outputs": case["domain_size"] if operation == "EvalAll" else 0,
                               "keys_per_launch": keys}
            for field, expected in expected_counts.items():
                if field in value and value[field] != expected:
                    raise ValueError(f"benchmark {field} {value[field]} does not match configured {expected}")
        for field, expected in (("domain_bits", case["domain_bits"]), ("keys", case["num_keys"]),
                                ("threads_per_block", case["threads_per_block"])):
            if expected is None or field not in value or value[field] == expected:
                continue
            # The packed GPU EvalAll kernel shrinks its block to 2^z when the
            # frontier is smaller than the configured block, so its reported
            # size is a power of two at or below the configured one.
            if field == "threads_per_block" and case["platform"] == "gpu" \
                    and "PackedHalfTreeDPF" in value.get("name", ""):
                reported = int(value[field])
                if 0 < reported <= expected and reported & (reported - 1) == 0:
                    continue
            # The DMPF GPU kernels launch fixed 256-thread blocks; the
            # configured block size is not a tiling parameter for them.
            if field == "threads_per_block" and case["platform"] == "gpu" \
                    and "DMPF" in value.get("name", "") and int(value[field]) == 256:
                continue
            raise ValueError(f"benchmark {field} {value[field]} does not match configured {expected}")


def collect(args, case, directory, sources):
    directory.mkdir(parents=True, exist_ok=True)
    env = environment(args, case)
    pin = ["taskset", "-c", str(args.cpu)]
    rows = []
    if case["library"] in bench.RUST:
        target = Path(env["CARGO_TARGET_DIR"])
        crate_bench = "bench" if case["library"] == "libdpf" else "bench_dpf"
        operation = "FullEval" if case["library"] == "fss_v060" and case["operation"] == "EvalAll" else case["operation"]
        groups = (("DPF-bytes", "DPF-uint", "DCF-bytes", "DCF-uint") if case["include_uint"] else ("DPF-bytes", "DCF-bytes")) if case["library"] == "fss_v060" else ("DPF",)
        # The mastic and plasma criterion IDs carry the domain size, so the
        # filter must pin the exact case; /DPF/Gen$ would match nothing.
        exact_id = (f"{case['library']}/CPU/{case['operation']}/{case['domain_bits']}"
                    if case["library"] in ("mastic", "plasma") else None)
        for group in groups:
            all_values = {}
            failure = None
            for repetition in range(args.repetitions):
                criterion = target / "criterion"
                if criterion.exists():
                    shutil.rmtree(criterion)
                command = pin + ["cargo", f"+{bench.RUST[case['library']]}", "bench", "--locked", "--bench", crate_bench,
                                 "--", "--sample-size", "20", "--warm-up-time", "0.2", "--measurement-time", "0.2",
                                 (f"^{exact_id}$" if exact_id else f"/{group}/{operation}$")]
                (directory / f"command-{group}-{repetition}.json").write_text(json.dumps(command) + "\n")
                try:
                    bench.execute(command, cwd=bench.source(case["library"]), env=env,
                                  log=directory / f"run-{group}-{repetition}.log")
                except RuntimeError as error:
                    failure = str(error)
                    break
                raw = directory / f"criterion-{group}-{repetition}"
                shutil.copytree(criterion, raw)
                for path in raw.glob("**/new/estimates.json"):
                    name = json.loads((path.parent / "benchmark.json").read_text())["full_id"]
                    value = json.loads(path.read_text())["median"]["point_estimate"]
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError("invalid Criterion median")
                    if exact_id and name != exact_id:
                        raise ValueError(f"criterion id {name} does not match the requested case {exact_id}")
                    all_values.setdefault(name, []).append(value)
            if failure:
                rows.append(normalize(case, exact_id or f"{case['library']}/CPU/{group}/{operation}", None,
                                      directory, sources, "known_failure", failure))
                continue
            for name, values in all_values.items():
                row = normalize(case, name, statistics.median(values), directory, sources)
                row.update(statistic="median_of_process_medians", process_medians_ns=values,
                           criterion_sample_size=20, criterion_warmup_s=.2, criterion_measurement_s=.2)
                rows.append(row)
    else:
        units = [(binary, pattern(case)) for binary in executables(args, case)]
        if case["library"] == "fss" and case["platform"] == "gpu" and case["operation"] == "Eval" and case["sweep"] != "block" \
                and not case.get("scheme"):
            groups = ("bytes", "uint") if case["include_uint"] else ("bytes",)
            filters = [f"^fss/GPU/{scheme}-{group}/Eval(/|$)" for group in groups for scheme in GPU_POINT_SCHEMES if group == "bytes" or scheme not in ("DMPF", "VDMPF")]
            filters.append("^fss/GPU/DPF-bytes/AesSoft/Eval(/|$)")
            units = [(binary, selected_filter) for binary in executables(args, case) for selected_filter in filters]
        for index, (binary, selected_filter) in enumerate(units):
            raw = directory / f"raw-{index}.json"
            command = pin + [str(binary), f"--benchmark_filter={selected_filter}", f"--benchmark_min_time={args.min_time}s",
                             f"--benchmark_min_warmup_time={args.warmup}", f"--benchmark_repetitions={args.repetitions}",
                             "--benchmark_report_aggregates_only=true", "--benchmark_out_format=json", f"--benchmark_out={raw}"]
            (directory / f"command-{index}.json").write_text(json.dumps(command) + "\n")
            process_error = None
            try:
                bench.execute(command, cwd=bench.source(case["library"]), env=env, log=directory / f"run-{index}.log")
            except RuntimeError as error:
                process_error = str(error)
                log_text = (directory / f"run-{index}.log").read_text()
                if "too many resources requested for launch" in log_text:
                    exact_point = "^fss/GPU/DPF-bytes/AesSoft/Eval(/|$)"
                    name = "fss/GPU/DPF-bytes/AesSoft/Eval" if selected_filter == exact_point else selected_filter
                    row = normalize(case, name, None, raw, sources, "unsupported", "too many resources requested for launch")
                    row["failure_kind"] = "kernel_resources_exhausted"
                    row["command"] = command
                    row["log"] = str(directory / f"run-{index}.log")
                    rows.append(failure_metadata(row))
                    continue
                if not raw.exists():
                    raise
            validate_native_configuration(case, raw)
            values, failures = parse_google(raw)
            rows.extend(normalize(case, name, value, raw, sources, "failed" if process_error else "ok", process_error) for name, value in values)
            rows.extend(normalize(case, name, None, raw, sources,
                                  "unsupported" if "kernel resources" in error or "too many resources requested for launch" in error else "known_failure", error)
                        for name, error in failures)
    if not rows:
        raise RuntimeError("no benchmarks matched this case")
    for row in rows:
        row.update(repetitions=args.repetitions, min_time_s=args.min_time, warmup_s=args.warmup,
                   framework="criterion" if case["library"] in bench.RUST else "google")
        path = Path(row["raw_result"])
        if path.is_file():
            row["raw_result_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return rows


def export(directory):
    manifest = json.loads((directory / "run.json").read_text())
    rows = []
    for path in sorted(directory.glob("*/case.json")):
        rows.extend(json.loads(path.read_text())["rows"])
    return write_export(directory, manifest, rows)


def write_export(directory, metadata, rows):
    directory.mkdir(parents=True, exist_ok=True)
    rows = [polish_metadata(row) for row in rows]
    output = dict(schema_version=1, metadata=metadata, records=rows)
    (directory / "chart-data.json").write_text(json.dumps(output, indent=2) + "\n")
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else ["status"]
    with (directory / "chart-data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()} for row in rows)
    return 1 if not rows or any(row["status"] in ("failed", "known_failure") for row in rows) else 0


def merge(directory, inputs, retry_inputs=(), drops=()):
    drop_specs = {}
    for item in drops:
        parts = item.split(":", 2)
        drop_specs[(parts[0], parts[1])] = parts[2] if len(parts) > 2 else None
    dropped = []
    runs, rows, superseded = [], [], []
    for path, is_retry in [(path, False) for path in inputs] + [(path, True) for path in retry_inputs]:
        source = path / "chart-data.json" if path.is_dir() else path
        data = json.loads(source.read_text())
        if data.get("schema_version") != 1 or "records" not in data:
            raise ValueError("unsupported chart-data schema")
        runs.append(dict(source=str(source.resolve()), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                         metadata=data["metadata"], retry=is_retry))
        records = data["records"]
        if drop_specs:
            for row in records:
                spec = (row.get("library"), row.get("platform"))
                if spec in drop_specs:
                    entry = next((d for d in dropped if (d["library"], d["platform"]) == spec), None)
                    if entry is None:
                        dropped.append(dict(library=spec[0], platform=spec[1], reason=drop_specs[spec], rows=1))
                    else:
                        entry["rows"] += 1
            records = [row for row in records if (row.get("library"), row.get("platform")) not in drop_specs]
        # Polish on load so the retry matching below compares both sides under the
        # current naming: inputs exported before a variant qualifier existed carry
        # the old variant, which would otherwise dodge a retry from a newer run.
        records = [polish_metadata(row) for row in records]
        if is_retry:
            # The scheme and variant keep a retry scoped to the series it re-measured:
            # without them a retried CPU sweep row would supersede every scheme's row
            # at the same domain and operation. Rows without a benchmark name (a
            # crashed attempt) have no meaningful variant, so they supersede by the
            # case fields alone, as before.
            case_fields = ("library", "platform", "operation", "domain_bits", "threads_per_block", "num_keys",
                "cpu_prg", "scan")
            name_fields = ("scheme", "variant")
            def case_key(row):
                return tuple(row.get(field) for field in case_fields)
            def name_key(row):
                return tuple(row.get(field) for field in name_fields)
            replaced_cases = {case_key(row) for row in records}
            replaced_names = {name_key(row) for row in records}
            def superseded_by_retry(row):
                return case_key(row) in replaced_cases and (not row.get("benchmark") or name_key(row) in replaced_names)
            superseded.extend(dict(record=row, replaced_by=str(source.resolve())) for row in rows
                if superseded_by_retry(row))
            rows = [row for row in rows if not superseded_by_retry(row)]
        rows.extend(records)
    return write_export(directory, dict(runs=runs, superseded_records=superseded, dropped=dropped), rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "build", "run", "export", "merge"))
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--inputs", type=Path, nargs="+", help="chart JSON files or result directories for merge")
    parser.add_argument("--retry-inputs", type=Path, nargs="+", default=(), help="explicit retry runs superseding matching cases while preserving attempt provenance")
    parser.add_argument("--drop", action="append",
                        help="drop all rows of 'library:platform[:reason]' from every input, e.g. fss_v070:gpu:'upstream crashes on the reference GPU'")
    parser.add_argument("--libraries")
    parser.add_argument("--scheme", help="restrict fss native collections to one scheme, e.g. DMPF")
    parser.add_argument("--platform", choices=("cpu", "gpu", "all"), default="all")
    parser.add_argument("--sweep", choices=("domain", "block"), default="domain")
    parser.add_argument("--operations", choices=("Gen", "Eval", "EvalAll"), nargs="+", default=("Gen", "Eval", "EvalAll"))
    parser.add_argument("--domain-bits", type=int, nargs="+")
    parser.add_argument("--threads-per-block", type=int, nargs="+")
    parser.add_argument("--num-keys", type=int, default=262144)
    parser.add_argument("--include-uint", action="store_true", help="include optional FSS integer output groups")
    parser.add_argument("--cpu-prg", choices=("aes-ni", "openssl"), default="aes-ni")
    parser.add_argument("--cpu", type=int, default=int(os.environ.get("CPU_ID", "24")))
    parser.add_argument("--gpu", default=os.environ.get("GPU_ID", "1"))
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--cuda-arch", default=os.environ.get("CUDA_ARCH"))
    parser.add_argument("--governor", choices=("performance", "keep"), default="performance")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--min-time", type=float, default=.1)
    parser.add_argument("--warmup", type=float, default=.1)
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.domain_bits and any(n < 8 or n > 20 for n in args.domain_bits):
        raise ValueError("domain bits must be between 8 and 20")
    if args.threads_per_block and any(t not in BLOCKS for t in args.threads_per_block):
        raise ValueError("threads per block must be a power of two from 16 to 1024")
    if args.jobs <= 0 or args.repetitions <= 0 or args.num_keys <= 0 or not math.isfinite(args.min_time) or args.min_time <= 0 or not math.isfinite(args.warmup) or args.warmup < 0:
        raise ValueError("jobs, repetitions, keys, and minimum time must be positive")
    selected = cases(args)
    if args.action == "list" or args.dry_run:
        print(json.dumps(selected, indent=2))
        return 0
    if args.action in ("build", "run") and not selected:
        raise ValueError("no benchmark cases selected for the requested libraries, operations, and domains")
    directory = args.directory or BUILD / "results" / datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = directory.resolve()
    if not directory.is_relative_to(ROOT / "build"):
        raise ValueError("result directory must be under build")
    if args.action == "export":
        return export(directory)
    if args.action == "merge":
        if not args.inputs:
            raise ValueError("merge requires --inputs")
        return merge(directory, args.inputs, args.retry_inputs, args.drop or ())
    if args.action == "build":
        for case in selected:
            build_case(args, case)
        return 0
    if platform.system() != "Linux" or args.cpu not in os.sched_getaffinity(0):
        raise RuntimeError("runs require Linux and an allowed CPU affinity")
    directory.mkdir(parents=True, exist_ok=False)
    hashes_args = argparse.Namespace(libraries=",".join(dict.fromkeys(c["library"] for c in selected)), platform=args.platform)
    manifest = dict(schema_version=1, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    arguments=vars(args) | {"directory": str(directory)}, cases=selected,
                    source_sha256=bench.benchmark_source_hashes(hashes_args),
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    excluded_variants=[] if args.include_uint else [dict(library="fss_v060", variant="DCF-uint",
                        reason="original algorithm fails equal-alpha reconstruction correctness check")],
                    host=platform.node(), cpu_model=bench.capture(["lscpu"]), gpu_model=bench.capture(["nvidia-smi"]),
                    git_revision=bench.capture(["git", "rev-parse", "HEAD"]),
                    git_status=bench.capture(["git", "status", "--porcelain"]),
                    toolchain_versions={name: bench.capture(command) for name, command in
                                        {"cmake": ["cmake", "--version"], "nvcc": ["nvcc", "--version"],
                                         "cxx": ["c++", "--version"], "cargo": ["cargo", "--version"]}.items()},
                    lockfile_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                     for name in bench.RUST for path in [bench.source(name) / "Cargo.lock"] if path.exists()})
    try:
        with bench.governor(args.cpu, args.governor) as active:
            manifest["governor"] = active
            for case in selected:
                data = directory / case_id(case)
                data.mkdir()
                if case["library"] == "fss_v070" and case["operation"] == "EvalAll":
                    rows = [normalize(case, "", None, data, manifest["source_sha256"], "unsupported", "upstream has no full-domain API")]
                    (data / "case.json").write_text(json.dumps(dict(case=case, rows=rows), indent=2) + "\n")
                    continue
                try:
                    if not args.no_build:
                        build_case(args, case)
                    rows = collect(args, case, data, manifest["source_sha256"])
                except (RuntimeError, OSError, ValueError) as error:
                    rows = [normalize(case, "", None, data, manifest["source_sha256"], "failed", str(error))]
                (data / "case.json").write_text(json.dumps(dict(case=case, rows=rows), indent=2) + "\n")
    finally:
        (directory / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
        result = export(directory)
    print(directory / "chart-data.json")
    return result


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        sys.exit(main())
    except (RuntimeError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)

#!/usr/bin/env python3
"""Reproducible domain and CUDA block-size sweeps for the README figures.

Raw logs, native results, build caches, and normalized exports stay under build.
Run `list` for exact cases. The domain endpoint is run first as a cost pilot.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
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
CPU_LIBRARIES = ("fss", "libdpf", "libfss", "google_dpf", "gpu_dpf", "fss_v060", "fss_v070")
TARGETS = {"fss": "bench_fss", "libfss": "bench_dpf_libfss", "gpu_dpf": "bench_dpf_gpu_dpf",
           "ezpc": "bench_dpf_ezpc", "torchcsprng": "bench_aes128_soft"}


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
                for operation in getattr(args, "operations", ("Gen", "Eval", "EvalAll")):
                    if device == "gpu" and name == "gpu_dpf" and operation != "EvalAll":
                        continue
                    if operation == "EvalAll" and device == "cpu" and name in ("libfss", "gpu_dpf") and n not in SLOW_DOMAINS:
                        continue
                    if args.sweep == "block" and (device != "gpu" or name != "fss" or n != 20):
                        continue
                    if args.sweep == "block" and operation == "Gen":
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
                                           sweep=args.sweep, include_uint=getattr(args, "include_uint", False)))
                        if device == "cpu":
                            break
    if libraries and set(libraries) - set(CPU_LIBRARIES) - {"ezpc"}:
        raise ValueError("unknown sweep library")
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
        return f"^fss/CPU/(DPF|DCF)-{groups}/{operation}(/|$)"
    if case["library"] == "ezpc" and case["operation"] != "EvalAll":
        return f"^EzPC/GPU/(DPF|DCF)/{operation}/{case['num_keys']}(/|$)"
    prefix = "fss" if case["library"] == "fss" else ".*"
    if case["sweep"] == "block" and operation == "Eval":
        return "^fss/GPU/DPF-bytes/AesSoft/Eval(/|$)"
    if case["library"] == "fss":
        groups = "(bytes|uint)" if case["include_uint"] else "bytes"
        return f"^fss/GPU/(DPF|DCF|HalfTreeDPF)-{groups}/(AesSoft/)?{operation}(/|$)"
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
    if case["library"] == "google_dpf":
        case = dict(case, google_dcf_group="additive_mod_2^128")
    row = dict(case, benchmark=name, status=status, median_ns=time, time_ns=time,
               ns_per_key=time / case["num_keys"] if time is not None else None,
               raw_result=str(raw), source_sha256=sources, error=error)
    scheme = "HalfTreeDPF" if "HalfTree" in name else "DCF" if "DCF" in name else "DPF"
    group = "bytes" if "bytes" in name else "uint" if "uint" in name else "native"
    prg = "AES-software" if "AesSoft" in name else ("ChaCha12" if case["library"] == "gpu_dpf" and case["platform"] == "gpu" else
          "AES-NI" if case["platform"] == "cpu" and case["library"] == "fss" and case["cpu_prg"] == "aes-ni" else
          "OpenSSL-AES" if case["platform"] == "cpu" and case["library"] == "fss" else
          ("ChaCha20-x1" if scheme == "HalfTreeDPF" else "ChaCha20-x4" if scheme == "DCF" else "ChaCha20-x2") if case["library"] == "fss" else "AES128" if case["library"] in ("gpu_dpf", "ezpc", "google_dpf") else "native")
    if case["library"] == "fss_v070":
        prg = "AES128-MMO-NI" if case["platform"] == "cpu" else "Salsa12"
    bits = 127 if group in ("bytes", "uint") and case["library"] in ("fss", "fss_v070") else 128 if group in ("bytes", "uint") else None
    row.update(scheme=scheme, group=group, prg=prg, logical_output_bits=bits,
               scan="threads" if case["sweep"] == "block" else "domain",
               variant=scheme + "-" + group + "-" + prg,
               label=case["library"] + "/" + scheme + "-" + group + "/" + prg,
               implementation="materialized_point_loop" if case["library"] in ("libfss", "gpu_dpf") and case["platform"] == "cpu" and case["operation"] == "EvalAll" else "native",
               output_storage="packed_bits" if case["library"] == "ezpc" and case["operation"] == "EvalAll" else "materialized",
               timing_boundary="cuda_event_kernel" if case["platform"] == "gpu" else "wall_time_operation",
               statistic="median", config=dict(case), storage_output_bits=128 if group in ("bytes", "uint") else bits)
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
    if case["library"] in ("fss", "fss_v070"):
        row["prg_output_blocks"] = ((2 if scheme == "DCF" else 1) if case["library"] == "fss_v070" and case["platform"] == "gpu" else
                                    4 if scheme == "DCF" else 1 if scheme == "HalfTreeDPF" else 2)
    row["variant"] = scheme + "-" + row["group"] + "-" + row["prg"]
    row["label"] = case["library"] + "/" + row["variant"]
    return polish_metadata(row)


def polish_metadata(row):
    """Refresh descriptive fields without changing preserved measurements."""
    row = dict(row)
    library, device, operation = row["library"], row["platform"], row["operation"]
    scheme = row.get("scheme", "DPF")
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
    if library == "fss" and device == "gpu" and "AesSoft" not in row.get("benchmark", ""):
        blocks = 4 if scheme == "DCF" else 1 if scheme == "HalfTreeDPF" else 2
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
    row["variant"] = scheme + "-" + row.get("group", "native") + "-" + row.get("prg", "native")
    row["label"] = library + "/" + row["variant"]
    return row


def validate_native_configuration(case, raw):
    for value in json.loads(raw.read_text())["benchmarks"]:
        if value.get("error_occurred") or value.get("aggregate_name") not in (None, "median"):
            continue
        for field, expected in (("domain_bits", case["domain_bits"]), ("keys", case["num_keys"]),
                                ("threads_per_block", case["threads_per_block"])):
            if expected is not None and field in value and value[field] != expected:
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
        for group in groups:
            all_values = {}
            failure = None
            for repetition in range(args.repetitions):
                criterion = target / "criterion"
                if criterion.exists():
                    shutil.rmtree(criterion)
                command = pin + ["cargo", f"+{bench.RUST[case['library']]}", "bench", "--locked", "--bench", crate_bench,
                                 "--", "--sample-size", "20", "--warm-up-time", "0.2", "--measurement-time", "0.2",
                                 f"/{group}/{operation}$"]
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
                    all_values.setdefault(name, []).append(value)
            if failure:
                rows.append(normalize(case, f"{case['library']}/CPU/{group}/{operation}", None,
                                      directory, sources, "known_failure", failure))
                continue
            for name, values in all_values.items():
                row = normalize(case, name, statistics.median(values), directory, sources)
                row.update(statistic="median_of_process_medians", process_medians_ns=values,
                           criterion_sample_size=20, criterion_warmup_s=.2, criterion_measurement_s=.2)
                rows.append(row)
    else:
        units = [(binary, pattern(case)) for binary in executables(args, case)]
        if case["library"] == "fss" and case["platform"] == "gpu" and case["operation"] == "Eval" and case["sweep"] != "block":
            groups = ("bytes", "uint") if case["include_uint"] else ("bytes",)
            filters = [f"^fss/GPU/{scheme}-{group}/Eval(/|$)" for group in groups for scheme in ("DPF", "DCF")]
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
                    name = "fss/GPU/DPF-bytes/AesSoft/Eval" if "AesSoft" in selected_filter else selected_filter
                    row = normalize(case, name, None, raw, sources, "unsupported", "too many resources requested for launch")
                    row["failure_kind"] = "kernel_resources_exhausted"
                    row["command"] = command
                    row["log"] = str(directory / f"run-{index}.log")
                    rows.append(row)
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


def merge(directory, inputs, retry_inputs=()):
    runs, rows, superseded = [], [], []
    for path, is_retry in [(path, False) for path in inputs] + [(path, True) for path in retry_inputs]:
        source = path / "chart-data.json" if path.is_dir() else path
        data = json.loads(source.read_text())
        if data.get("schema_version") != 1 or "records" not in data:
            raise ValueError("unsupported chart-data schema")
        runs.append(dict(source=str(source.resolve()), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                         metadata=data["metadata"], retry=is_retry))
        if is_retry:
            fields = ("library", "platform", "operation", "domain_bits", "threads_per_block", "num_keys", "cpu_prg", "scan")
            def key(row):
                return tuple(row.get(field) for field in fields)
            replaced = {key(row) for row in data["records"]}
            superseded.extend(dict(record=row, replaced_by=str(source.resolve())) for row in rows if key(row) in replaced)
            rows = [row for row in rows if key(row) not in replaced]
        rows.extend(data["records"])
    return write_export(directory, dict(runs=runs, superseded_records=superseded), rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "build", "run", "export", "merge"))
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--inputs", type=Path, nargs="+", help="chart JSON files or result directories for merge")
    parser.add_argument("--retry-inputs", type=Path, nargs="+", default=(), help="explicit retry runs superseding matching cases while preserving attempt provenance")
    parser.add_argument("--libraries")
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
    directory = args.directory or BUILD / "results" / datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = directory.resolve()
    if not directory.is_relative_to(ROOT / "build"):
        raise ValueError("result directory must be under build")
    if args.action == "export":
        return export(directory)
    if args.action == "merge":
        if not args.inputs:
            raise ValueError("merge requires --inputs")
        return merge(directory, args.inputs, args.retry_inputs)
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

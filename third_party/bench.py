#!/usr/bin/env python3
"""Build and reproduce the comparison benchmarks using their native tools."""

import argparse
import contextlib
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark_policy

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build" / "third_party"
LIBRARIES = {
    "libdpf": ("libdpf-bench", "cargo", ("cpu",), "third_party/libdpf"),
    "libfss": ("libfss", "cmake", ("cpu",), "third_party/libfss/libfss"),
    "servan_vdpf": ("vdpf", "cmake", ("cpu",), "third_party/vdpf/vdpf"),
    "mastic": ("mastic-bench", "cargo", ("cpu",), "third_party/mastic"),
    "plasma": ("plasma-bench", "cargo", ("cpu",), "third_party/plasma"),
    "sycret": ("sycret-bench", "cargo", ("cpu",), "third_party/sycret"),
    "dpf-cpp": ("dpf-cpp", "cmake", ("cpu",), "third_party/dpf-cpp/dpf-cpp"),
    "dpfplusplus": ("dpfplusplus-bench", "cmake", ("cpu",), None),
    "google_dpf": ("distributed_point_functions", "bazel", ("cpu",),
                   "third_party/distributed_point_functions/distributed_point_functions"),
    "gpu_dpf": ("GPU-DPF", "cmake", ("cpu", "gpu"), "third_party/GPU-DPF/GPU-DPF"),
    "ezpc": ("EzPC", "cmake", ("gpu",), "third_party/EzPC/EzPC"),
    "fss_v060": ("fss-v0.6.0", "cargo", ("cpu",), None),
    "fss_v070": ("fss-v0.7.0", "cmake", ("cpu", "gpu"),
                 "third_party/fss-v0.7.0/fss-v0.7.0"),
    "fss": ("fss", "cmake", ("cpu", "gpu"), None),
    "torchcsprng": ("torchcsprng", "cmake", ("cpu", "gpu"), None),
    "main": ("..", "cmake", ("cpu", "gpu"), None),
}
RUST = {"libdpf": "nightly-2025-09-01", "fss_v060": "nightly-2025-09-01",
        "mastic": "stable", "plasma": "stable", "sycret": "stable"}
TIME_UNITS = {"ns": 1, "us": 1000, "ms": 1_000_000, "s": 1_000_000_000}
V1_WORK_FIELDS = ("api_calls_per_iteration", "keys_per_iteration", "queries_per_iteration",
                  "logical_outputs_per_iteration", "output_bytes_per_iteration",
                  "proofs_per_iteration", "proof_bytes_per_iteration",
                  "prepared_leaves_per_iteration", "point_hash_bytes_per_iteration",
                  "batch_keys", "calls_per_iteration")
V1_COUNTABLE_FIELDS = ("work_schema_version",) + V1_WORK_FIELDS + ("fixture_keys",)


def v1_counter(value, field):
    if not isinstance(value, (int, float)) or isinstance(value, bool) \
            or not math.isfinite(value) or value < 0 or value != int(value):
        raise ValueError(f"invalid per-iteration counter {field}={value!r}")
    return int(value)


def source(library):
    return (ROOT / "third_party" / LIBRARIES[library][0]).resolve()


def capture(command, cwd=ROOT, env=None):
    try:
        return subprocess.check_output(command, cwd=cwd, env=env, stderr=subprocess.STDOUT,
                                       text=True).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return str(error)


def execute(command, *, cwd=ROOT, env=None, log=None):
    print("+ " + " ".join(map(str, command)), flush=True)
    if log is None:
        logs = BUILD / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        log = logs / (datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".log")
    with contextlib.ExitStack() as stack:
        output = stack.enter_context(log.open("w")) if log else None
        if output:
            output.write("command: " + json.dumps(list(map(str, command))) + "\n")
            output.flush()
        process = subprocess.Popen(list(map(str, command)), cwd=cwd, env=env,
                                   stdout=output, stderr=subprocess.STDOUT if output else None,
                                   start_new_session=True)
        try:
            code = process.wait()
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        if code:
            raise RuntimeError(f"command exited with status {code}" +
                               (f"; see {log}" if log else ""))


def selection(args):
    names = list(LIBRARIES)[:-1] if args.libraries == "all" else args.libraries.split(",")
    unknown = set(names) - LIBRARIES.keys()
    if unknown:
        raise RuntimeError("unknown libraries: " + ", ".join(sorted(unknown)))
    platforms = ("cpu", "gpu") if args.platform == "all" else (args.platform,)
    result = [(name, p) for name in dict.fromkeys(names) for p in platforms
              if p in LIBRARIES[name][2]]
    if not result:
        raise RuntimeError("no selected library supports the requested platform")
    return result


def environment(args, library):
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", RAYON_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               CUDA_VISIBLE_DEVICES=args.gpu, MAX_JOBS=str(args.jobs),
               CARGO_TARGET_DIR=str(BUILD / library / "target"),
               UV_PROJECT_ENVIRONMENT=str(BUILD / "gpu_dpf" / "venv"),
               UV_CACHE_DIR=str(BUILD / "uv-cache"))
    if library == "google_dpf":
        for key in ("CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH", "CPATH"):
            env.pop(key, None)
        env.update(BAZELISK_HOME=str(BUILD / "google_dpf" / "bazelisk"),
                   USE_BAZEL_VERSION="7.6.1")
    env.setdefault("FSS_EZPC_POOL_MIB", "512")
    if library == "main" and getattr(args, "full_launches", None) is not None:
        env["FSS_BENCH_FULL_LAUNCHES"] = str(args.full_launches)
    if args.cuda_arch:
        arch = args.cuda_arch
        if not re.fullmatch(r"[0-9]+", arch):
            raise RuntimeError("cuda architecture must be a number such as 80 or 120")
        env["TORCH_CUDA_ARCH_LIST"] = f"{int(arch) // 10}.{int(arch) % 10}"
    return env


def gpu_uv(args):
    """Install the project-required uv in an isolated build directory if required."""
    version = "0.12.24"
    isolated = BUILD / "gpu_dpf" / "uv-tool"
    binary = isolated / "bin" / "uv"
    if binary.exists() and capture([str(binary), "--version"]).split()[:2] == ["uv", version]:
        return str(binary)
    if capture(["uv", "--version"]).split()[:2] == ["uv", version]:
        return "uv"
    env = environment(args, "gpu_dpf")
    execute(["uv", "venv", str(isolated), "--python", sys.executable], env=env)
    execute(["uv", "pip", "install", "--python", str(isolated / "bin" / "python"),
             f"uv=={version}"], env=env)
    return str(binary)


def prepare(args, selected):
    names = list(dict.fromkeys(name for name, _ in selected))
    paths = [LIBRARIES[name][3] for name in names if LIBRARIES[name][3]]
    if paths:
        execute(["git", "submodule", "update", "--init", "--recursive", "--", *paths])
    if "dpfplusplus" in names:
        # The upstream headers forbid redistribution: clone and patch on demand
        # under build/, never commit them (doc/bench-dpfplusplus.md).
        execute(["bash", str(ROOT / "third_party" / "dpfplusplus-bench" / "fetch.sh"),
                 str(BUILD / "dpfplusplus")])
    for name in names:
        if name in RUST:
            execute(["rustup", "toolchain", "install", RUST[name], "--profile", "minimal"])
            execute(["cargo", f"+{RUST[name]}", "fetch", "--locked"], cwd=source(name))
    if ("gpu_dpf", "gpu") in selected:
        execute([gpu_uv(args), "sync", "--project", str(source("gpu_dpf")), "--frozen"],
                env=environment(args, "gpu_dpf"))


def cmake(args, name, env, targets=None, extra=()):
    directory = BUILD / name / "cmake"
    command = ["cmake", "-S", source(name), "-B", directory,
               "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TESTING=OFF",
               "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", *extra]
    if args.cuda_arch:
        command.append(f"-DCMAKE_CUDA_ARCHITECTURES={args.cuda_arch}")
    execute(command, env=env)
    command = ["cmake", "--build", directory, "--parallel", str(args.jobs)]
    if targets:
        command += ["--target", *targets]
    execute(command, env=env)


def build(args, selected):
    for name in dict.fromkeys(n for n, _ in selected):
        platforms = {p for n, p in selected if n == name}
        env = environment(args, name)
        directory = BUILD / name
        directory.mkdir(parents=True, exist_ok=True)
        if name in RUST:
            bench = "bench" if name == "libdpf" else "bench_dpf"
            execute(["cargo", f"+{RUST[name]}", "bench", "--locked", "--no-run",
                     "--bench", bench, "--jobs", str(args.jobs)], cwd=source(name), env=env)
        elif name == "google_dpf":
            execute(["bazel", f"--output_user_root={directory / 'bazel'}", "build",
                     ":bench_dpf_google", f"--jobs={args.jobs}", "--symlink_prefix=/"],
                    cwd=source(name), env=env)
        elif name == "gpu_dpf":
            if "cpu" in platforms:
                cmake(args, name, env)
            if "gpu" in platforms:
                upstream = source(name) / "GPU-DPF"
                copied = directory / "extension-source"
                if copied.exists():
                    shutil.rmtree(copied)
                shutil.copytree(upstream, copied, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns(".git", "build", "*.egg-info"))
                patches = source(name) / "patches"
                for patch in sorted(patches.glob("*.patch")):
                    execute(["git", "apply", str(patch)], cwd=copied)
                execute([directory / "venv/bin/python", "setup.py", "build_ext",
                         "--build-temp", str(directory / "extension-temp"),
                         "--build-lib", str(directory / "extension")], cwd=copied, env=env)
        elif name == "ezpc":
            sytorch = source(name) / "sytorch"
            dependency = directory / "sytorch"
            command = ["cmake", "-S", sytorch, "-B", dependency,
                       "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"]
            if args.cuda_arch:
                command.append(f"-DCMAKE_CUDA_ARCHITECTURES={args.cuda_arch}")
            execute(command, env=env)
            execute(["cmake", "--build", dependency, "--target", "sytorch",
                     "--parallel", str(args.jobs)], env=env)
            cmake(args, name, env, extra=[f"-DEZPC_SYTORCH_BUILD={dependency}"])
        elif name == "fss_v070":
            targets = [f"bench_{p}_{group}_{scheme}" for p in sorted(platforms)
                       for group in ("uint", "bytes") for scheme in ("dpf", "dcf")]
            cmake(args, name, env, targets,
                  [f"-DFSS070_BUILD_GPU={'ON' if 'gpu' in platforms else 'OFF'}"])
        elif name == "main":
            cmake(args, name, env, [f"bench_{p}" for p in sorted(platforms)],
                  ["-DBUILD_BENCH=ON"])
        else:
            extra = ()
            if name == "dpfplusplus":
                extra = (f"-DDPFPP_INCLUDE_DIR={BUILD / 'dpfplusplus' / 'src' / 'include'}",)
            cmake(args, name, env, extra=extra)


@contextlib.contextmanager
def governor(cpu, mode):
    path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor")
    previous = path.read_text().strip() if path.exists() else None
    if mode == "performance" and previous is None:
        raise RuntimeError(f"cpu {cpu} has no scaling governor interface; use --governor keep explicitly")

    def write(value):
        if os.access(path, os.W_OK):
            path.write_text(value + "\n")
        else:
            subprocess.run(["sudo", "-n", "tee", str(path)], input=value + "\n",
                           text=True, stdout=subprocess.DEVNULL, check=True)
        if path.read_text().strip() != value:
            raise RuntimeError(f"could not verify CPU {cpu} governor {value}")

    try:
        if mode == "performance" and previous != "performance":
            write("performance")
        yield path.read_text().strip() if path.exists() else "unavailable"
    finally:
        if mode == "performance" and previous and path.read_text().strip() != previous:
            write(previous)


def benchmark_source_hashes(args):
    """Identify local benchmark inputs independently of Git dirty-state summaries."""
    paths = {ROOT / "CMakeLists.txt", ROOT / "third_party" / "CMakeLists.txt",
             ROOT / "third_party" / "bench.py", ROOT / "third_party" / "figure_bench.py",
             ROOT / "third_party" / "benchmark_policy.py"}
    for name in dict.fromkeys(n for n, _ in selection(args)):
        directory = source(name)
        for pattern in ("*.cu", "*.cc", "*.cpp", "*.cuh", "*.h", "*.py", "CMakeLists.txt",
                        "Cargo.toml", "BUILD", "BUILD.bazel", "WORKSPACE", "WORKSPACE.bazel",
                        ".bazelversion", "pyproject.toml", "benches/*.rs", "patches/**/*.patch",
                        "sytorch/CMakeLists.txt", "torchcsprng/*.cuh", "*.sh"):
            paths.update(directory.glob(pattern))
        if name == "main":
            paths.update((ROOT / "src").rglob("*.cu"))
            paths.update((ROOT / "src").rglob("*.h"))
    paths.update((ROOT / "include").rglob("*.cuh"))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths) if path.is_file()}


def build_configuration(selected):
    """Record configured toolchains and architectures, including no-build runs."""
    configurations = {}
    for name in dict.fromkeys(name for name, _ in selected):
        for subdirectory in ("cmake", "sytorch"):
            cache = BUILD / name / subdirectory / "CMakeCache.txt"
            if not cache.exists():
                continue
            values = {}
            for line in cache.read_text().splitlines():
                if line.startswith(("#", "//")) or "=" not in line or ":" not in line:
                    continue
                definition, value = line.split("=", 1)
                key = definition.split(":", 1)[0]
                if key.startswith(("CMAKE_CXX_", "CMAKE_CUDA_", "CMAKE_C_COMPILER", "CMAKE_C_FLAGS",
                                   "CMAKE_BUILD_TYPE", "CMAKE_EXE_LINKER_FLAGS")) or key in (
                        "EZPC_SYTORCH_BUILD", "FSS070_BUILD_GPU", "BUILD_BENCH", "CMAKE_EXPORT_COMPILE_COMMANDS",
                        "FSS_BENCH_CPU_PRG", "FSS_BENCH_DOMAIN_BITS", "FSS_BENCH_NUM_KEYS"):
                    values[key] = value
            configurations[str(cache.relative_to(ROOT))] = values
    return configurations


def metadata(args, active_governor):
    probes = {
        "git_commit": ["git", "rev-parse", "HEAD"],
        "git_changes": ["git", "status", "--porcelain", "--untracked-files=no"],
        "submodules": ["git", "submodule", "status", "--recursive"],
        "cpu": ["lscpu"], "gpu": ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,"
                                "memory.used,utilization.gpu,driver_version", "--format=csv"],
        "nvcc": ["nvcc", "--version"], "cxx": ["c++", "--version"],
        "cmake": ["cmake", "--version"], "bazel": ["bazel", "--version"],
        "uv": ["uv", "--version"], "rustup": ["rustup", "show"],
    }
    return {"timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "host": platform.node(), "platform": platform.platform(),
            "cpu_id": args.cpu, "gpu_id": args.gpu, "governor": active_governor,
            "omp_num_threads": 1, "rayon_num_threads": 1,
            "cuda_arch": args.cuda_arch, "smoke": args.smoke,
            "profile": getattr(args, "profile", None), "workload": getattr(args, "workload", None),
            "full_launches": getattr(args, "full_launches", None),
            "policy": getattr(args, "policy", None),
            "repetitions": args.repetitions, "min_time_seconds": args.min_time,
            "selected": selection(args),
            "rust_toolchains": RUST,
            "environment": {key: environment(args, "gpu_dpf").get(key)
                            for key in ("OMP_NUM_THREADS", "RAYON_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                        "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "CUDA_VISIBLE_DEVICES",
                                        "MAX_JOBS", "CMAKE_PREFIX_PATH", "LIBRARY_PATH", "CPATH",
                                        "CPLUS_INCLUDE_PATH", "CUDA_HOME", "TORCH_CUDA_ARCH_LIST",
                                        "FSS_EZPC_POOL_MIB")},
            "build_configuration": build_configuration(selection(args)),
            "source_sha256": benchmark_source_hashes(args),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "lockfile_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in [source("libdpf") / "Cargo.lock",
                                             source("fss_v060") / "Cargo.lock",
                                             source("mastic") / "Cargo.lock",
                                             source("plasma") / "Cargo.lock",
                                             source("sycret") / "Cargo.lock",
                                             source("gpu_dpf") / "uv.lock"] if path.exists()},
            "probes": {key: capture(command,
                                    cwd=source("google_dpf") if key == "bazel" else ROOT,
                                    env=environment(args, "google_dpf" if key == "bazel" else "main"))
                       for key, command in probes.items()}}


def binaries(name, device, args=None):
    directory = BUILD / name
    if name == "google_dpf":
        output = capture(["bazel", f"--output_user_root={directory / 'bazel'}", "info",
                          "bazel-bin"], cwd=source(name), env=environment(args, name))
        return [(Path(output) / "bench_dpf_google", "")]
    if name == "fss_v070":
        return [(directory / "cmake" / f"bench_{device}_{group}_{scheme}", f"{group}-{scheme}")
                for group in ("uint", "bytes") for scheme in ("dpf", "dcf")]
    binary = "bench"
    if name == "main":
        binary = f"bench_{device}"
    return [(directory / "cmake" / binary, "")]


# The native binaries compile filters with the POSIX extended regex backend,
# which rejects lookaround. Both patterns are plain alternations over the
# operation tokens the main binaries register (src/bench_cpu.cu and
# src/bench_gpu.cu), so EvalPoint rows never match.
MAIN_OPERATION_TOKENS = r"(Gen|Eval|EvalAll(Merkle)?(Gpu)?|Prove|Preprocess(EvalAll)?(Gpu)?)_"
SMOKE_MAIN_OPERATION_TOKENS = r"(Gen|Eval|EvalAll(Merkle)?(Gpu)?)_"


def default_filter(name, device, exclude_internal_points=False):
    if name == "main":
        return "^BM_.*" + MAIN_OPERATION_TOKENS if exclude_internal_points else ".*"
    prefixes = {"fss": "fss", "torchcsprng": "(torchcsprng|fss-prg)"}
    return f"^{prefixes.get(name, '.*')}/{device.upper()}/"


def resolved_filter(args, name, device):
    if args.filter:
        return args.filter
    if args.smoke:
        return (default_filter(name, device) + r"(.*/)?(Gen|Eval|EvalAll|FullEval|AesSoft)(/|$)"
                if name != "main" else "^BM_.*" + SMOKE_MAIN_OPERATION_TOKENS + ".*")
    return default_filter(name, device,
                          exclude_internal_points=getattr(args, "profile", None) == "decision")


def resolved_interleaving(args, name):
    if name not in benchmark_policy.INTERLEAVED_LIBRARIES:
        return None
    return benchmark_policy.interleaving_enabled(name, args.repetitions,
                                                 getattr(args, "explicit_interleaving", None))


def record(args, name, device, directory, commands):
    env = environment(args, name)
    pin = ["taskset", "-c", str(args.cpu)]
    interleaving = resolved_interleaving(args, name)
    units = []
    if name in RUST:
        target = BUILD / name / "target"
        criterion = target / "criterion"
        if criterion.exists():
            shutil.rmtree(criterion)
        bench = "bench" if name == "libdpf" else "bench_dpf"
        command = pin + ["cargo", f"+{RUST[name]}", "bench", "--locked", "--bench", bench,
                         "--", "--sample-size", "10" if args.smoke else "100",
                         "--warm-up-time", "0.1" if args.smoke else "3",
                         "--measurement-time", str(args.min_time)]
        if args.filter:
            command.append(args.filter)
        elif args.smoke:
            # Keygen covers sycret's EqKeygen/LeKeygen, which end lowercase.
            command.append(r"/(Gen|Eval|Keygen)(/|$)")
        units.append((command, "criterion", "", target))
    elif name == "gpu_dpf" and device == "gpu":
        directory_env = BUILD / name
        env["PYTHONPATH"] = os.pathsep.join([str(directory_env / "extension"),
                                             str(directory_env / "extension-source")])
        command = pin + [str(directory_env / "venv/bin/python"), str(source(name) / "bench_gpu.py"),
                         "--output", str(directory / "python.json"),
                         "--repetitions", str(args.repetitions),
                         "--warmup", "1" if args.smoke else "2"]
        if args.filter:
            command += ["--filter", args.filter]
        units.append((command, "python", "", None))
    else:
        native_interleaving = benchmark_policy.interleaving_flag(
            name, args.repetitions, getattr(args, "explicit_interleaving", None))
        for binary, variant in binaries(name, device, args):
            pattern = resolved_filter(args, name, device)
            filters = [pattern]
            if name == "fss_v070":
                group, scheme = variant.split("-")
                prefix = f"fss-v0.7.0/{device.upper()}/{group}/{scheme.upper()}"
                operations = [operation for operation in ("Gen", "Eval")
                              if not args.filter or re.search(args.filter, f"{prefix}/{operation}")
                              or (device == "gpu" and re.search(args.filter, f"{prefix}/{operation}/manual_time"))]
                if not operations:
                    continue
                if device == "gpu":
                    filters = [f"^{prefix}/{operation}(/|$)" for operation in operations]
            for pattern in filters:
                suffix = hashlib.sha256(pattern.encode()).hexdigest()[:8]
                output = directory / f"google-{variant}-{suffix}.json"
                command = pin + [str(binary), f"--benchmark_filter={pattern}",
                                 f"--benchmark_min_time={args.min_time}s",
                                 f"--benchmark_repetitions={args.repetitions}",
                                 "--benchmark_display_aggregates_only=true",
                                 "--benchmark_out_format=json", f"--benchmark_out={output}"]
                if native_interleaving:
                    command.append(native_interleaving)
                units.append((command, "google", variant + "-" + suffix, None))
    if not units:
        entry = {"library": name, "device": device, "framework": "selection",
                 "command": [], "log": "", "status": "failed", "error": "no benchmarks matched"}
        commands.append(entry)
        return [entry]
    records = []
    for command, framework, variant, target in units:
        command += getattr(args, "native_args", [])
        log = directory / (variant + ".log" if variant else framework + ".log")
        entry = {"library": name, "device": device, "framework": framework,
                 "command": command, "log": str(log.relative_to(directory.parent)),
                 "status": "ok"}
        if name == "main":
            entry["full_launches"] = getattr(args, "full_launches", None)
        if name in ("main", "fss"):
            entry["required_work_schema"] = 1
        if interleaving is not None:
            entry["interleaving"] = interleaving
        commands.append(entry)
        started = time.monotonic()
        try:
            execute(command, cwd=source(name), env=env, log=log)
        except (RuntimeError, OSError) as error:
            entry.update(status="failed", error=str(error))
            # Preserve known failures as failures, and still measure other processes.
            print(f"error: {name}/{device}: {error}", file=sys.stderr)
        except BaseException as error:
            entry.update(status="failed", error=str(error) or type(error).__name__)
            raise
        finally:
            entry["process_wall_seconds"] = round(time.monotonic() - started, 3)
        if target and (target / "criterion").exists():
            shutil.copytree(target / "criterion", directory / "criterion", dirs_exist_ok=True)
        try:
            _, errors = result_rows(entry, directory)
            if errors:
                raise RuntimeError("; ".join(errors))
        except (RuntimeError, OSError, ValueError, KeyError, TypeError) as error:
            if entry["status"] == "ok":
                entry.update(status="failed", error=str(error))
        records.append(entry)
    return records


def plan_document(args, selected):
    commands = []
    for name, device in selected:
        command = {"library": name, "device": device,
                   "filter": resolved_filter(args, name, device),
                   "result_row_count": "unresolved until the native run"}
        interleaving = resolved_interleaving(args, name)
        if interleaving is not None:
            command["interleaving"] = interleaving
        commands.append(command)
    return {"profile": args.profile, "workload": args.workload,
            "min_time_s": args.min_time, "repetitions": args.repetitions,
            "warmup_s": args.warmup_s, "full_launches": args.full_launches,
            "full_launches_env": benchmark_policy.FULL_LAUNCHES_ENV,
            "full_launches_scope": "main GPU batch full-domain rows; batch keys stay "
                                   f"{benchmark_policy.FULL_EVAL_BATCH_KEYS}",
            "libraries": args.libraries, "sources": args.policy["sources"],
            "commands": commands}


def result_rows(entry, data):
    """Parse preserved output, returning usable rows and explicit data failures."""
    name, device = entry["library"], entry["device"]
    rows, errors = [], []

    def row(benchmark, total, batch=1, work=None, statistic="median", item_unit=None):
        if not math.isfinite(total) or total <= 0 or batch <= 0:
            raise ValueError("invalid benchmark time or batch")
        work = batch if work is None else work
        if not math.isfinite(work) or work <= 0:
            raise ValueError("invalid benchmark work count")
        unit = item_unit or ("domain_outputs" if "EvalAll" in benchmark and name != "ezpc"
                             else "keys_or_prg_calls")
        rows.append([name, device, benchmark, batch, total, total / batch,
                     work, 1e9 * work / total, statistic,
                     "cuda_event" if device == "gpu" else "wall_time", "", "", "",
                     unit, "", "real_time", "", "", "", "", "", "", "", "legacy"])

    def schema_v1_row(benchmark, value, aggregate, repetition_work):
        """Normalize one schema v1 receipt row from per-iteration counters.

        Google Benchmark v1.9.5 reports time fields per iteration, so the
        aggregate ``iterations`` count never rescales the normalization.
        """
        for field in V1_COUNTABLE_FIELDS:
            if field in value:
                v1_counter(value[field], field)
        if value.get("work_schema_version") != 1:
            raise ValueError(f"{benchmark}: unsupported work schema version "
                             f"{value.get('work_schema_version')}")
        for field in ("real_time", "time_unit"):
            if field not in value:
                raise ValueError(f"{benchmark}: schema v1 receipt is missing {field}")
        if repetition_work and len(repetition_work) > 1:
            raise ValueError(f"{benchmark}: repetitions report different per-iteration work counters")
        total = value["real_time"] * TIME_UNITS[value["time_unit"]]
        keys = v1_counter(value.get("keys_per_iteration", 0), "keys_per_iteration")
        queries = v1_counter(value.get("queries_per_iteration", 0), "queries_per_iteration")
        outputs = v1_counter(value.get("logical_outputs_per_iteration", 0),
                             "logical_outputs_per_iteration")
        proofs = v1_counter(value.get("proofs_per_iteration", 0), "proofs_per_iteration")
        prepared = v1_counter(value.get("prepared_leaves_per_iteration", 0),
                              "prepared_leaves_per_iteration")
        work, unit = ((outputs, "domain_outputs") if outputs > 0 else
                      (queries, "keys_or_prg_calls") if queries > 0 else
                      (prepared, "prepared_leaves") if prepared > 0 else
                      (proofs, "proofs") if proofs > 0 else
                      (keys, "keys_or_prg_calls") if keys > 0 else (0, "keys_or_prg_calls"))
        if work <= 0:
            raise ValueError(f"{benchmark}: schema v1 receipt has no per-iteration work counters")
        batch_keys = v1_counter(value.get("batch_keys", 0), "batch_keys")
        if name == "main":
            batch_full_domain = (device == "gpu" and "EvalAll" in benchmark
                                 and "Grotto" not in benchmark and "VdpfEvalAllMerkle" not in benchmark)
            if batch_full_domain and ("calls_per_iteration" not in value or "batch_keys" not in value):
                raise ValueError(f"{benchmark}: batch full-domain row is missing "
                                 "calls_per_iteration or batch_keys")
            planned = entry.get("full_launches") or benchmark_policy.DAILY_FULL_LAUNCHES
            calls = value.get("calls_per_iteration")
            if calls is not None and calls != planned:
                raise ValueError(f"{benchmark}: executed calls_per_iteration {calls} does not "
                                 f"match the planned {planned} full launches")
            if batch_keys and batch_keys != benchmark_policy.FULL_EVAL_BATCH_KEYS:
                raise ValueError(f"{benchmark}: executed batch_keys {batch_keys} does not match "
                                 f"the planned {benchmark_policy.FULL_EVAL_BATCH_KEYS} batch keys")
            if (entry.get("required_work_schema") and "VdpfEval" in benchmark
                    and "EvalAll" not in benchmark and "EvalPoint" not in benchmark
                    and "point_hash_bytes_per_iteration" not in value):
                raise ValueError(f"{benchmark}: new main GPU VDPF point receipt is missing "
                                 "point_hash_bytes_per_iteration")
        batch = batch_keys or keys or queries or work
        row(benchmark, total, batch, work, "median" if aggregate else "single", unit)
        rows[-1][5] = total / keys if keys > 0 else ""
        if "cpu_time" in value:
            rows[-1][14] = value["cpu_time"] * TIME_UNITS[value["time_unit"]]
        native_rate = value.get("items_per_second")
        if native_rate is not None:
            if not math.isfinite(native_rate) or native_rate <= 0:
                raise ValueError("invalid native benchmark throughput")
            basis = ("manual_time" if device == "gpu" or "/manual_time" in benchmark
                     else "real_time" if "/real_time" in benchmark else "cpu_time")
            if not aggregate:
                if basis == "cpu_time" and "cpu_time" in value:
                    basis_time = value["cpu_time"] * TIME_UNITS[value["time_unit"]]
                else:
                    basis = "manual_time" if basis == "manual_time" else "real_time"
                    basis_time = total
                computed = 1e9 * work / basis_time
                if abs(native_rate - computed) > 1e-3 * computed:
                    raise ValueError(f"{benchmark}: native items_per_second {native_rate:.6g} "
                                     f"conflicts with the per-iteration counters on {basis} "
                                     f"({computed:.6g})")
            rows[-1][16] = native_rate
            rows[-1][15] = basis
        def receipt_counter(field):
            return v1_counter(value[field], field) if field in value else ""

        rows[-1][17:] = [receipt_counter("logical_outputs_per_iteration"),
                         receipt_counter("output_bytes_per_iteration"),
                         receipt_counter("proofs_per_iteration"),
                         receipt_counter("proof_bytes_per_iteration"),
                         receipt_counter("point_hash_bytes_per_iteration"),
                         value.get("host_wall_ns_per_iteration", ""), "v1"]

    framework = entry["framework"]
    if framework == "google":
        filename = next(x.split("=", 1)[1] for x in entry["command"]
                        if x.startswith("--benchmark_out="))
        values = json.loads((data / Path(filename).name).read_text())["benchmarks"]
        units = TIME_UNITS
        aggregate_names = {v.get("run_name", v["name"]) for v in values
                           if v.get("aggregate_name") == "median"}
        repetition_work = {}
        for value in values:
            if (value.get("work_schema_version") is not None and not value.get("aggregate_name")
                    and not value.get("error_occurred")):
                repetition_work.setdefault(value.get("run_name", value["name"]), set()).add(
                    tuple(value.get(field) for field in V1_WORK_FIELDS))
        for value in values:
            benchmark = value.get("run_name", value["name"])
            if value.get("error_occurred"):
                errors.append(f"{benchmark}: {value.get('error_message', 'benchmark failed')}")
                continue
            aggregate = value.get("aggregate_name")
            if aggregate and aggregate != "median":
                continue
            if not aggregate and benchmark in aggregate_names:
                continue
            if value.get("work_schema_version") is not None:
                schema_v1_row(benchmark, value, aggregate, repetition_work.get(benchmark))
                continue
            if entry.get("required_work_schema"):
                raise ValueError(f"{benchmark}: new {name} receipt is missing work_schema_version; "
                                 "historical legacy rows are only valid for re-summarized runs")
            batch = 1
            if device == "gpu":
                batch = entry.get("num_keys", 1 << 20)
                if "EvalAllFull" in benchmark or (name == "fss" and "EvalAll" in benchmark):
                    batch = 1
                if name == "main" and "EvalAll" in benchmark and ("Grotto" in benchmark or "VdpfEvalAllMerkle" in benchmark):
                    # Single-key-per-iteration GPU EvalAll benchmarks in bench_gpu.
                    batch = 1
                if name == "ezpc" and "EvalAllFull" not in benchmark:
                    match = re.search(r"/(\d+)(?:/|$)", benchmark)
                    if not match:
                        raise ValueError("missing EzPC key count")
                    batch = int(match.group(1))
            total = value["real_time"] * units[value["time_unit"]]
            domain = 1 << entry.get("domain_bits", 20)
            if name in ("main", "dpf-cpp", "dpfplusplus") and "EvalAll" in benchmark:
                match = re.search(r"/(\d+)(?:/|$)", benchmark)
                if match:
                    domain = 1 << int(match.group(1))
            work = batch * domain if "EvalAll" in benchmark and (name != "ezpc" or "EvalAllFull" in benchmark) else batch
            row(benchmark, total, batch, work, "median" if aggregate else "single")
            rows[-1][14] = (value["cpu_time"] * units[value["time_unit"]]
                            if "cpu_time" in value else "")
            if "items_per_second" in value:
                native_rate = value["items_per_second"]
                if not math.isfinite(native_rate) or native_rate <= 0:
                    raise ValueError("invalid native benchmark throughput")
                rows[-1][7] = native_rate
                rows[-1][16] = native_rate
                rows[-1][15] = ("manual_time" if device == "gpu" or "/manual_time" in benchmark
                                else "real_time" if "/real_time" in benchmark else "cpu_time")
    elif framework == "criterion":
        for path in sorted((data / "criterion").glob("**/new/estimates.json")):
            definition = json.loads((path.parent / "benchmark.json").read_text())
            estimates = json.loads(path.read_text())
            benchmark = definition["full_id"]
            batch = 1
            work = 1
            match = re.search(r"/(\d+)$", benchmark)
            if "EvalAll" in benchmark or "FullEval" in benchmark:
                work = 1 << (int(match.group(1)) if match else entry.get("domain_bits", 20))
            elif name == "sycret" and match:
                # sycret processes N keys or points per batched call; the suffix is N.
                batch = int(match.group(1))
                work = batch
            row(benchmark, estimates["median"]["point_estimate"], batch, work)
    elif framework == "python":
        for value in json.loads((data / "python.json").read_text())["benchmarks"]:
            row(value["name"], value["time_ns"], value["batch"])
            rows[-1][9:14] = [value.get("timing_boundary", ""), value.get("operation", ""),
                             value.get("prf", ""), value.get("domain_size", ""), "keys"]
    else:
        return [], []
    if not rows:
        errors.append("no usable benchmark results; filter matched no benchmarks or all benchmarks failed")
    return rows, errors


def summarize(directory):
    directory = directory.resolve()
    manifest = json.loads((directory / "run.json").read_text())
    rows = []
    for entry in manifest["commands"]:
        data = directory / f"{entry['library']}-{entry['device']}"
        try:
            parsed, errors = result_rows(entry, data)
            rows.extend(parsed)
            if errors and entry["status"] == "ok":
                entry.update(status="failed", error="; ".join(errors))
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
            if entry["status"] == "ok":
                entry.update(status="failed", error=str(error))
    (directory / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    headings = ["library", "platform", "benchmark", "batch", "time_ns", "ns_per_key",
                "items_per_iteration", "items_per_second", "statistic", "timing_boundary",
                "operation", "prf", "domain_size", "item_unit", "cpu_time_ns",
                "throughput_time_basis", "native_items_per_second",
                "logical_outputs_per_iteration", "output_bytes_per_iteration",
                "proofs_per_iteration", "proof_bytes_per_iteration",
                "point_hash_bytes_per_iteration",
                "host_wall_ns_per_iteration", "work_schema"]
    (directory / "summary.json").write_text(json.dumps([dict(zip(headings, row)) for row in rows], indent=2) + "\n")
    with (directory / "summary.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(headings)
        writer.writerows(rows)
    lines = ["# Benchmark results", "", "See `run.json` for hardware, versions, governor, and commands.",
             "Times are nanoseconds. EvalAll throughput counts domain outputs. Other rows count keys",
             "or raw PRG calls. EzPC EvalAll and GPU-DPF GPU rows count keys for table reductions.",
             "Items/iteration comes from the operation and key count. The JSON/CSV field `batch`",
             "records keys per iteration or call, or PRG calls for raw AES rows. It is retained",
             "for compatibility and does not record CUDA threads per block. Native Google Benchmark throughput",
             "uses CPU time on CPU and manual CUDA time on GPU. The CSV/JSON records its time basis",
             "and preserves the native counter. Rows without counters use real time.",
             "Schema v1 rows carry per-iteration work counters: they divide time per iteration by",
             "work per iteration only (keys, queries, domain outputs, prepared leaves, or proofs),",
             "keep logical output counts, output bytes (logical storage bytes, not DRAM traffic),",
             "proof counts, proof bytes, point-hash bytes (raw corrected hash material, not",
             "proofs), and GPU host wall time separate from CUDA event time, and are marked v1.",
             "Their items/s column stays wall-normalized while the native rate column keeps its",
             "own recorded time basis. Rows without the marker keep the legacy name-based",
             "reading. Keyless rows leave the key cells empty; the Work unit column names each",
             "row's counted unit.", "",
             "| Library | Platform | Benchmark | Keys/iteration | Work/iteration | Work unit | Time (ns) | ns/key | Items/s |",
             "| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |"]
    for row in rows:
        per_key = f"{row[5]:.3f}" if row[5] != "" else ""
        keys = str(int(row[3])) if row[5] != "" else ""
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {keys} | {int(row[6])} | {row[13]} | "
                     f"{row[4]:.3f} | {per_key} | {row[7]:.3f} |")
    failures = [entry for entry in manifest["commands"] if entry["status"] != "ok"]
    if manifest.get("run_error"):
        lines += ["", "Run failed: " + manifest["run_error"]]
    if failures:
        lines += ["", "## Failed processes", ""]
        lines += [f"- {e['library']}/{e['device']}: `{e['log']}` ({e['error']})" for e in failures]
    (directory / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"saved {len(rows)} rows to {directory / 'summary.csv'}")
    return 1 if failures or manifest.get("run_error") or not rows else 0


def run(args, selected):
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("benchmark runs require Linux x86_64")
    if args.cpu not in os.sched_getaffinity(0):
        raise RuntimeError(f"cpu {args.cpu} is outside this process's allowed affinity")
    run_id = args.run_id or datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise RuntimeError("run id must contain only letters, numbers, dots, underscores, or hyphens")
    directory = BUILD / "results" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    commands = []
    manifest = metadata(args, "not started")
    manifest["commands"] = commands
    try:
        if not args.no_build:
            for name in dict.fromkeys(n for n, _ in selected):
                try:
                    build(args, [(n, p) for n, p in selected if n == name])
                except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
                    commands.append({"library": name, "device": "build", "framework": "build",
                                     "command": [], "log": "../../logs", "status": "failed", "error": str(error)})
        manifest["build_configuration"] = build_configuration(selected)
        with governor(args.cpu, args.governor) as active:
            manifest["governor"] = active
            for name, device in selected:
                if any(e["library"] == name and e["framework"] == "build" for e in commands):
                    continue
                data = directory / f"{name}-{device}"
                data.mkdir()
                record(args, name, device, data, commands)
    except BaseException as error:
        manifest["run_error"] = str(error) or type(error).__name__
        raise
    finally:
        (directory / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
        summarize(directory)
    return 1 if any(e["status"] != "ok" for e in commands) or not commands else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("list")
    summary = subparsers.add_parser("summarize")
    summary.add_argument("directory", type=Path)
    for action in ("prepare", "build", "run"):
        command = subparsers.add_parser(action)
        command.add_argument("--libraries", default=None if action == "run" else "all",
                             help="comma-separated names from list")
        command.add_argument("--platform", choices=("cpu", "gpu", "all"), default="all")
        command.add_argument("--jobs", type=int, default=int(os.environ.get("JOBS", "4")))
        command.add_argument("--cuda-arch", default=os.environ.get("CUDA_ARCH"))
        command.add_argument("--gpu", default=os.environ.get("GPU_ID", "0"))
        if action == "run":
            command.add_argument("--cpu", type=int, default=int(os.environ.get("CPU_ID", "0")))
            command.add_argument("--governor", choices=("performance", "keep"), default="performance")
            command.add_argument("--no-build", action="store_true")
            command.add_argument("--smoke", action="store_true",
                                 help="alias for --profile smoke; conflicts with --profile")
            command.add_argument("--profile", choices=benchmark_policy.PROFILES,
                                 help="run profile; explicit timing and selection arguments override it")
            command.add_argument("--workload", choices=benchmark_policy.WORKLOADS,
                                 help="full API calls per iteration for the main GPU batch full-domain rows")
            command.add_argument("--full-launches", type=int,
                                 help="explicit full API calls per iteration, 1 to 1024")
            command.add_argument("--plan", action="store_true",
                                 help="print the resolved run plan without building or touching hardware")
            command.add_argument("--run-id")
            command.add_argument("--filter")
            command.add_argument("--min-time", type=float)
            command.add_argument("--repetitions", type=int)
            command.add_argument("native_args", nargs=argparse.REMAINDER, help="native benchmark arguments after --")
    args = parser.parse_args()
    if args.action == "list":
        for name, (_, tool, devices, _) in LIBRARIES.items():
            print(f"{name:12} {','.join(devices):8} {tool}")
        return 0
    if args.action == "summarize":
        return summarize(args.directory)
    if not 1 <= args.jobs <= max(1, os.cpu_count() or 1):
        raise RuntimeError("jobs must be positive and no greater than the host CPU count")
    if args.action == "run":
        policy = benchmark_policy.resolve_policy(args, "bench")
        args.profile = policy["profile"]
        args.workload = policy["workload"]
        args.full_launches = policy["full_launches"]
        args.min_time = policy["min_time_s"]
        args.repetitions = policy["repetitions"]
        args.warmup_s = policy["warmup_s"]
        args.smoke = policy["profile"] == "smoke"
        args.policy = policy
        benchmark_policy.apply_default_selection(args, "bench", args.profile, policy)
        if not math.isfinite(args.min_time) or args.min_time <= 0 or args.repetitions <= 0:
            raise RuntimeError("min-time and repetitions must be positive")
        if args.native_args[:1] == ["--"]:
            args.native_args.pop(0)
        args.explicit_interleaving = None
        for value in args.native_args:
            if value.startswith(("--benchmark_out", "--output")):
                raise RuntimeError("output paths are managed by the runner")
            if value.startswith("--benchmark_filter="):
                args.filter = value.split("=", 1)[1]
            if value.startswith(benchmark_policy.INTERLEAVING_FLAG + "="):
                setting = value.split("=", 1)[1]
                if setting not in ("true", "false"):
                    raise RuntimeError(f"{benchmark_policy.INTERLEAVING_FLAG} accepts only true or false")
                args.explicit_interleaving = setting == "true"
        args.native_args = [v for v in args.native_args if not v.startswith(
            ("--benchmark_filter=", benchmark_policy.INTERLEAVING_FLAG + "="))]
        if args.plan:
            print(json.dumps(plan_document(args, selection(args)), indent=2))
            return 0
    selected = selection(args)
    if args.action == "prepare":
        prepare(args, selected)
    elif args.action == "build":
        build(args, selected)
    else:
        return run(args, selected)
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        sys.exit(main())
    except (RuntimeError, subprocess.CalledProcessError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)

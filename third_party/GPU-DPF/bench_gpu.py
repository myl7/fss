#!/usr/bin/env python3
"""Benchmark the pinned GPU-DPF Python API with synchronous wall-clock timing.

Generation is a CPU operation. GPU evaluation reduces a public table using a
DPF share and returns CPU int32 results. The CHACHA20 enum selects 12 rounds.
Import paths are supplied by the runner for the copied source and extension.
"""

import argparse
import json
from pathlib import Path
import re
import statistics
import time

import torch
import dpf_cpp
from dpf import DPF


N = 1 << 20
PRF = "ChaCha12"


def release(dpf):
    if dpf.buffers is not None:
        dpf_cpp.eval_free(dpf.buffers)
        dpf.buffers = None


def validate():
    """Check reconstruction modulo 2^32 using CPU and GPU on a small domain."""
    domain = 128
    indices = [0, 1, 42, domain - 1]
    dpf = DPF(prf=dpf_cpp.PRF_CHACHA20)
    pairs = [dpf.gen(index, domain) for index in indices]
    first = [pair[0] for pair in pairs]
    second = [pair[1] for pair in pairs]
    # The upstream group is additive: party 0 minus party 1 reconstructs.
    cpu = dpf.eval_cpu(first, one_hot_only=True) - dpf.eval_cpu(
        second, one_hot_only=True
    )
    expected = torch.zeros((len(indices), domain), dtype=torch.int32)
    expected[torch.arange(len(indices)), indices] = 1
    if not torch.equal(cpu, expected):
        raise RuntimeError("cpu DPF one-hot reconstruction failed")
    table = torch.arange(1, domain * 3 + 1, dtype=torch.int32).reshape(domain, 3)
    try:
        dpf.eval_init(table)
        a = dpf.eval_gpu(first)
        b = dpf.eval_gpu(second)
        torch.cuda.synchronize()
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise RuntimeError("gpu DPF table reduction returned non-finite values")
        if not torch.equal(a - b, table[indices]):
            raise RuntimeError("gpu DPF table reconstruction failed")
    finally:
        release(dpf)
    return {"passed": True, "domain_size": domain, "keys": len(indices),
            "table_columns": 3, "group": "additive modulo 2^32",
            "reconstruction": "party0 - party1"}


def measure(name, operation, args, batch, gpu=False):
    for _ in range(args.warmup):
        operation()
    if gpu:
        torch.cuda.synchronize()
    samples = []
    for _ in range(args.repetitions):
        if gpu:
            torch.cuda.synchronize()
        start = time.perf_counter_ns()
        operation()
        if gpu:
            torch.cuda.synchronize()
        samples.append(time.perf_counter_ns() - start)
    median = statistics.median(samples)
    print(f"{name}: {median / 1e6:.3f} ms, batch={batch}, "
          f"{batch * 1e9 / median:.1f} keys/s (median)", flush=True)
    return {"name": name, "time_ns": median, "samples_ns": samples,
            "batch": batch, "domain_size": N, "prf": PRF,
            "repetitions": args.repetitions, "warmup": args.warmup,
            "statistic": "median", "timer": "perf_counter_ns",
            "timing_boundary": ("Python eval_gpu including key conversion, "
                                "host/device transfers, table reduction, CPU "
                                "result construction, and final CUDA synchronize"
                                if gpu else "Python gen including os.urandom, "
                                "CPU key generation, and tensor construction"),
            "operation": "table_reduce" if gpu else "key_generation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--filter", default=".*", help="benchmark name regex")
    args = parser.parse_args()
    if args.repetitions < 1 or args.warmup < 0:
        parser.error("repetitions must be positive and warmup must be nonnegative")
    try:
        pattern = re.compile(args.filter)
    except re.error as error:
        parser.error(f"invalid benchmark filter: {error}")
    gen_name = "GPU-DPF/CPU/DPF/Gen"
    eval_name = "GPU-DPF/GPU/DPF/TableReduce"
    if not any(pattern.search(name) for name in (gen_name, eval_name)):
        parser.error("benchmark filter matched no benchmarks")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the GPU-DPF benchmark")
    print(f"GPU-DPF: N={N}, batch={dpf_cpp.BATCH_SIZE}, PRF={PRF}, "
          f"GPU={torch.cuda.get_device_name()}", flush=True)
    correctness = validate()
    print("CPU one-hot and GPU nonzero-table reconstruction passed", flush=True)
    dpf = DPF(prf=dpf_cpp.PRF_CHACHA20)
    benchmarks = []
    if pattern.search(gen_name):
        benchmarks.append(measure(gen_name, lambda: dpf.gen(42, N), args, batch=1))
    if pattern.search(eval_name):
        keys = [dpf.gen(index, N)[0] for index in range(dpf_cpp.BATCH_SIZE)]
        table = torch.arange(1, N + 1, dtype=torch.int32).reshape(N, 1)
        try:
            dpf.eval_init(table)
            benchmarks.append(measure(eval_name, lambda: dpf.eval_gpu(keys), args,
                                      batch=dpf_cpp.BATCH_SIZE, gpu=True))
        finally:
            release(dpf)
    result = {"benchmarks": benchmarks, "correctness": correctness,
              "context": {"torch": torch.__version__,
                          "torch_cuda": torch.version.cuda,
                          "gpu": torch.cuda.get_device_name(),
                          "gpu_capability": list(torch.cuda.get_device_capability()),
                          "table_columns": 1, "table_dtype": "int32",
                          "upstream_commit": "ce23a06af884ee54300b5bc5fd5350e445f10b0b"}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

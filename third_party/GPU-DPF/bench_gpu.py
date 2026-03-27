#!/usr/bin/env python3
# Benchmark: GPU-DPF (facebookresearch/GPU-DPF) GPU gen/eval
# N=2^20, AES-128 PRF, batch=512 (BATCH_SIZE).
#
# Requires: pip install torch numpy
# Install GPU-DPF: cd GPU-DPF && CC=g++ python setup.py install
# Run: CUDA_VISIBLE_DEVICES=1 python bench_gpu.py

import sys
import os
import time
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "GPU-DPF"))

import torch
import dpf_cpp
import numpy as np

N = 1 << 20
REPS = 10
WARMUP = 2


def bench_gen():
    import dpf as dpf_mod
    dpf = dpf_mod.DPF(prf=dpf_cpp.PRF_CHACHA20)

    for _ in range(WARMUP):
        dpf.gen(42, N)

    times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        k1, k2 = dpf.gen(42, N)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)

    med = statistics.median(times)
    print(f"GPU-DPF/GPU/DPF/Gen   {med:12.0f} us  (CPU, {REPS} reps, median)")
    return k1, k2


def bench_eval_gpu(k1, k2):
    import dpf as dpf_mod
    dpf = dpf_mod.DPF(prf=dpf_cpp.PRF_CHACHA20)

    batch = dpf_cpp.BATCH_SIZE

    keys = []
    for i in range(batch):
        kk1, _ = dpf.gen(i % N, N)
        keys.append(kk1)

    table = torch.zeros((N, 1), dtype=torch.int32)
    dpf.eval_init(table)

    for _ in range(WARMUP):
        dpf.eval_gpu(keys)
    torch.cuda.synchronize()

    start_events = []
    end_events = []
    for _ in range(REPS):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        dpf.eval_gpu(keys)
        e.record()
        start_events.append(s)
        end_events.append(e)

    torch.cuda.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    med_ms = statistics.median(times_ms)
    throughput = batch / (med_ms / 1000)
    print(f"GPU-DPF/GPU/DPF/Eval  {med_ms:12.3f} ms  "
          f"({batch} keys, {throughput:.0f} evals/s, {REPS} reps, median)")


if __name__ == "__main__":
    print(f"N={N}, BATCH_SIZE={dpf_cpp.BATCH_SIZE}, PRF=CHACHA20")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()
    k1, k2 = bench_gen()
    bench_eval_gpu(k1, k2)

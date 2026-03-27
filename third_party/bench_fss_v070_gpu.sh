#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
CPU_ID="${CPU_ID:-0}"
CPU_SG="/sys/devices/system/cpu/cpu${CPU_ID}/cpufreq/scaling_governor"

cd "$(dirname "$0")/fss-v0.7.0"

cmake -B build -DCMAKE_BUILD_TYPE=Release -S .
cmake --build build -j

prev_gov=$(cat "$CPU_SG")
echo performance | sudo tee "$CPU_SG" > /dev/null
trap 'echo "$prev_gov" | sudo tee "$CPU_SG" > /dev/null' EXIT

CUDA_VISIBLE_DEVICES="$GPU_ID" taskset -c "$CPU_ID" ./build/bench_gpu_uint "$@"
CUDA_VISIBLE_DEVICES="$GPU_ID" taskset -c "$CPU_ID" ./build/bench_gpu_bytes "$@"

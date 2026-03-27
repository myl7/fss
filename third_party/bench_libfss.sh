#!/usr/bin/env bash
set -euo pipefail

CPU_ID="${CPU_ID:-0}"
CPU_SG="/sys/devices/system/cpu/cpu${CPU_ID}/cpufreq/scaling_governor"

cd "$(dirname "$0")/libfss"

cmake -B build -DCMAKE_BUILD_TYPE=Release -S .
cmake --build build -j

prev_gov=$(cat "$CPU_SG")
echo performance | sudo tee "$CPU_SG" > /dev/null
trap 'echo "$prev_gov" | sudo tee "$CPU_SG" > /dev/null' EXIT

export OMP_NUM_THREADS=1
taskset -c "$CPU_ID" ./build/bench "$@"

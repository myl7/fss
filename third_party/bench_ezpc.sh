#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
CPU_ID="${CPU_ID:-0}"
CPU_SG="/sys/devices/system/cpu/cpu${CPU_ID}/cpufreq/scaling_governor"

cd "$(dirname "$0")/EzPC"

if [ ! -f EzPC/GPU-MPC/ext/sytorch/build/libsytorch.a ]; then
    echo "error: EzPC sytorch libs not found." >&2
    echo "Build first: cd EzPC/GPU-MPC && bash setup.sh" >&2
    exit 1
fi

cmake -B build -DCMAKE_BUILD_TYPE=Release -S .
cmake --build build -j

prev_gov=$(cat "$CPU_SG")
echo performance | sudo tee "$CPU_SG" > /dev/null
trap 'echo "$prev_gov" | sudo tee "$CPU_SG" > /dev/null' EXIT

CUDA_VISIBLE_DEVICES="$GPU_ID" taskset -c "$CPU_ID" ./build/bench "$@"

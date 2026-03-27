#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
CPU_ID="${CPU_ID:-0}"
CPU_SG="/sys/devices/system/cpu/cpu${CPU_ID}/cpufreq/scaling_governor"

cd "$(dirname "$0")/GPU-DPF"

# Install PyTorch and build the CUDA extension using uv.
if ! uv run python -c "import dpf_cpp" 2>/dev/null; then
    uv venv --seed -p python3 .venv
    uv pip install torch numpy
    (cd GPU-DPF && CC=g++ uv run python setup.py install)
fi

prev_gov=$(cat "$CPU_SG")
echo performance | sudo tee "$CPU_SG" > /dev/null
trap 'echo "$prev_gov" | sudo tee "$CPU_SG" > /dev/null' EXIT

CUDA_VISIBLE_DEVICES="$GPU_ID" taskset -c "$CPU_ID" uv run python bench_gpu.py "$@"

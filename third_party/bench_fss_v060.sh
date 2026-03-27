#!/usr/bin/env bash
set -euo pipefail

CPU_ID="${CPU_ID:-0}"
CPU_SG="/sys/devices/system/cpu/cpu${CPU_ID}/cpufreq/scaling_governor"

cd "$(dirname "$0")/fss-v0.6.0"

prev_gov=$(cat "$CPU_SG")
echo performance | sudo tee "$CPU_SG" > /dev/null
trap 'echo "$prev_gov" | sudo tee "$CPU_SG" > /dev/null' EXIT

export RAYON_NUM_THREADS=1
taskset -c "$CPU_ID" cargo bench --bench bench_dpf -- "$@"

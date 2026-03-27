#!/usr/bin/env bash
set -euo pipefail

CPU_ID="${CPU_ID:-0}"
CPU_SG="/sys/devices/system/cpu/cpu${CPU_ID}/cpufreq/scaling_governor"

cd "$(dirname "$0")/distributed_point_functions"

bazel build :bench_dpf_google

prev_gov=$(cat "$CPU_SG")
echo performance | sudo tee "$CPU_SG" > /dev/null
trap 'echo "$prev_gov" | sudo tee "$CPU_SG" > /dev/null' EXIT

taskset -c "$CPU_ID" ./bazel-bin/bench_dpf_google "$@"

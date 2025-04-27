#!/bin/bash
set -euo pipefail

file=/tmp/fss_bench.log
timeout=300 # seconds
start_time=$(date +%s)
end_time=$((start_time + timeout))
wait_interval=10 # seconds

while [ $(date +%s) -lt $end_time ]; do
  if [ -f "$file" ]; then
    echo "File $file found"
    break
  fi
  sleep "$wait_interval"
  echo "Waiting for file $file to exist ($(( $(date +%s) - start_time ))s elapsed)"
done

if [ ! -f "$file" ]; then
  echo "Timeout. File $file did not occur in $timeout seconds."
  exit 1
fi

#!/usr/bin/env bash
set -euo pipefail
exec python3 "$(dirname "$0")/bench.py" run --libraries libdpf --platform cpu -- "$@"

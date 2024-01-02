#!/usr/bin/env zsh
[ "${ZSH_VERSION:-}" = "" ] && echo >&2 "Only works with zsh" && exit 1
set -euo pipefail

# See https://bheisler.github.io/criterion.rs/book/faq.html#cargo-bench-gives-unrecognized-option-errors-for-valid-command-line-options
# for the reason why this script is created.

bench_dirs=(dcf/benches dpf-fss/benches)
bench_args=($(find $bench_dirs -type f -exec basename -s .rs {} \; | sed 's/^/--bench /'))
cargo bench -F prg $bench_args $@

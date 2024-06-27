#!/usr/bin/env zsh
[ "${ZSH_VERSION:-}" = "" ] && echo >&2 "Only works with zsh" && exit 1
set -euo pipefail

# See https://bheisler.github.io/criterion.rs/book/faq.html#cargo-bench-gives-unrecognized-option-errors-for-valid-command-line-options
# for the reason why the script is created.

bench_dirs=(benches)
bench_args=($(find $bench_dirs -type f -exec basename -s .rs {} \; | sed 's/^/--bench /'))
cargo bench $bench_args $@

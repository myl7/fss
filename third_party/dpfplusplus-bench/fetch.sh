#!/usr/bin/env bash
# Fetches the pinned rh3nry/dpfplusplus snapshot into <builddir>/src and
# applies the three scripted text fixes that make the pinned headers build on
# the bench hosts. Upstream is "Copyright (C) 2019 Anonymous", a pre-release
# distributed for peer review with a "please do not redistribute" request (see
# include/dpf++/prg.hpp of the fetched tree and doc/bench-dpfplusplus.md), so
# nothing under <builddir>/src is ever committed; the build/ tree is
# gitignored. Usage: fetch.sh [builddir] (default: .). Idempotent: re-runs
# force-reset the tree to the pinned commit and re-apply the fixes.
set -euo pipefail

repo="https://github.com/rh3nry/dpfplusplus"
commit="e3f7f0fba75d1736b0da3edb0ce3b22d1b3c73d5"

builddir="${1:-.}"
src="$builddir/src"
header="$src/include/dpf.hpp"

mkdir -p "$builddir"
if [ ! -d "$src/.git" ]; then
    rm -rf "$src"
    git clone "$repo" "$src"
fi
git -C "$src" fetch origin "$commit" --depth 1
# --force restores any tree modified by an earlier run, so the fixes below
# always apply to pristine pinned sources and the exact-match counts turn
# upstream drift into a loud failure instead of a silent mismatch.
git -C "$src" checkout --detach --force "$commit"

python3 - "$header" <<'PYEOF'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()

def replace(text, old, new, expected):
    found = text.count(old)
    if found != expected:
        sys.exit(f"fix pattern matched {found} times, expected {expected}: {old[:64]!r}")
    return text.replace(old, new)

# (1) arc4random_buf lives in <stdlib.h> on Apple and in glibc >= 2.36; the
# unconditional <bsd/stdlib.h> include needs libbsd, which the bench hosts do
# not install.
text = replace(text,
    "#include <bsd/stdlib.h> // arc4random_buf\n",
    "#if defined(__APPLE__) || (defined(__GLIBC__) && defined(__GLIBC_PREREQ)"
    " && __GLIBC_PREREQ(2, 36))\n"
    "#include <stdlib.h>     // arc4random_buf\n"
    "#else\n"
    "#include <bsd/stdlib.h> // arc4random_buf\n"
    "#endif\n",
    1)

# (2) std::ceil is not a constant expression before C++23, so the double-based
# nodes_per_leaf initializer fails to compile under C++17; use an integer
# ceil-div with identical value.
text = replace(text,
    "dpf_key::is_packed ? 1 : std::ceil(static_cast<double>(bits_per_leaf)"
    " / (sizeof(node_t) * CHAR_BIT));",
    "dpf_key::is_packed ? 1 : (bits_per_leaf + sizeof(node_t) * CHAR_BIT - 1)"
    " / (sizeof(node_t) * CHAR_BIT); // std::ceil not constexpr pre-C++23",
    1)

# (3) "if constexpr(dpfkey.is_packed)" reads a static member through a
# function parameter, which is not a constant expression there; name the
# static class member instead (two occurrences: dpf::eval and dpf::__reeval).
text = replace(text,
    "if constexpr(dpfkey.is_packed)",
    "if constexpr(dpf_key<leaf_t, node_t, prg_t>::is_packed)",
    2)

path.write_text(text)
print("dpfplusplus: applied 3 fixes to " + str(path))
PYEOF

echo "dpfplusplus: $src at $commit; include dir: $src/include"

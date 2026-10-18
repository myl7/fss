- Follow Google C++ Style Guide
- Start error messages with a lowercase letter unless it is a proper noun or variable name
- Never reorder `#include`
- Build, save perf data, or save flamegraphs to ./build
- Try searching the FlameGraph lib in ../
- In GPU device code, registers are limited and memory access is expensive.
  Avoid use of `memcpy`, `memset`, and `reinterpret_cast`.
  Prefer plain assignments. Prefer `int4` for types larger than 8B.
- Write commit messages per the commit-msg skill in `.agents/skills/commit-msg`.
  Recommend a repo scope only for a recurring module with a real code boundary —
  a directory or namespace that many future commits touch — never for an
  execution path or an area one or two commits cover (gpu was rejected on this
  rule: the scheme sources serve host and device alike). Explicitly allowed
  scopes beyond the skill's predefined ones:
  - python: fss_crypto bindings and their tests
- Before finishing work on unpushed commits, tidy the history: squash fix-ups into
  their original commits (commit --fixup plus rebase --autosquash), drop unpushed
  change-and-revert pairs, and keep one logical change per commit. For nearby
  pushed commits, suggest the same tidy-up instead of acting.
- Public library code belongs under `include/fss`. `src` contains tests and
  benchmarks.
- When changing the public header layout, verify wheel packaging and JIT
  include discovery.
- When reporting benchmark throughput, check whether items count keys or
  domain outputs and whether timing covers one operation or a batch.

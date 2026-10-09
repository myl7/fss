- Follow Google C++ Style Guide
- Start error messages with a lowercase letter unless it is a proper noun or variable name
- Never reorder `#include`
- Build, save perf data, or save flamegraphs to ./build
- Try searching the FlameGraph lib in ../
- In GPU device code, registers are limited and memory access is expensive.
  Avoid use of `memcpy`, `memset`, and `reinterpret_cast`.
  Prefer plain assignments. Prefer `int4` for types larger than 8B.
- Write commit messages as [Scoped Commits](https://scopedcommits.com/)
- Public library code belongs under `include/fss`. `src` contains tests and
  benchmarks.
- When changing the public header layout, verify wheel packaging and JIT
  include discovery.
- When reporting benchmark throughput, check whether items count keys or
  domain outputs and whether timing covers one operation or a batch.

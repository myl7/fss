# Released consumer fixtures

The six sample `.cu` files come unchanged from `samples/` at tag `v1.2.0`.
Keep their API calls intact when changing the library. CMake compiles them
against the current public headers and checks the valid examples' reported results.

The released VDMPF sample supplies eight points, although v1.2.0 requires at
least 30. Its calls remain a compile fixture. `legacy_oracle.cu` compares the
initialized correction fields, outputs, and chain proofs with the old tag
using valid inputs, including VDMPF inputs with 30 points.

`gpu_sample_runner.cu` adds a device-availability check around the unchanged
GPU sample. CPU samples run with one OpenMP worker because their released
OpenSSL contexts are shared by scheme copies.

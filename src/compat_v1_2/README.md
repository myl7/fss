# Historical released-consumer fixtures

The six sample `.cu` files are snapshots from `samples/` at tag `v1.2.0`.
Keep their original API calls as historical fixtures.
They do not define the current public API or require compatibility aliases.
The root CMake project does not register or compile these six snapshots.

`consumer/CMakeLists.txt` is a separate installed-package consumer project.
It finds `fss` through `find_package` and builds only the historical `dpf_dcf_cpu.cu` sample.
It requires a package with a compatible API.

`legacy_oracle.cu` records initialized correction fields, outputs, and chain proofs for comparison with the old tag.
It uses valid inputs, including 30-point VDMPF generation, whereas the released VDMPF sample contains eight points.
The oracle is a historical comparison source, not a current-API test entry point.
Do not restore removed library interfaces merely to compile it.

`gpu_sample_runner.cu` adds a device-availability check around the GPU snapshot.
`check_sample.cmake` checks an explicitly supplied sample executable's reported results.
Neither utility is registered by the root CMake project.
When running historical CPU samples with their compatible package, use one OpenMP worker because their OpenSSL contexts are shared by scheme copies.

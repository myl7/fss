# API capabilities for v1.3.0

This reference covers the C++ v1.3.0 candidate and the Python 0.2.0 candidate. C++ v1.2.0 is the compatibility baseline. The tables describe available operations. Backend restrictions, index limits, and memory requirements also apply.

## C++ operations

Use the scheme headers under `include/fss` for host operations. Include `fss/point_eval_gpu.cuh` or `fss/eval_all_gpu.cuh` for CUDA launch helpers. A device-callable `Gen` can run inside a caller's kernel. The library does not provide a CUDA key-generation batch launcher.

| Scheme | Host Gen | Device-callable Gen | Host point or batch evaluation | CUDA point or batch evaluation | Host full domain | CUDA full domain |
| --- | --- | --- | --- | --- | --- | --- |
| DPF | `Gen` | `Gen` | `Eval` | `DpfEvalPointGpu` after repack | `EvalAll` | `DpfEvalAllGpu`, `DpfEvalAllGpuBatch` |
| DCF | `Gen` | `Gen` | `Eval`, both predicates | `DcfEvalPointGpu` after repack | `EvalAll` | `DcfEvalAllGpu`, `DcfEvalAllGpuBatch` |
| HalfTree DPF | `Gen` | `Gen` | `Eval` | `HalfTreeDpfEvalPointGpu` after repack | `EvalAll` | `HalfTreeDpfEvalAllGpu`, `HalfTreeDpfEvalAllGpuBatch` |
| Packed HalfTree DPF | `Gen` | `Gen` | `Eval`, then `Extract` | `PackedHalfTreeDpfEvalPointGpu` after repack | `EvalAll` returns packed blocks | `PackedHalfTreeDpfEvalAllGpu`, `PackedHalfTreeDpfEvalAllGpuBatch` |
| Grotto DCF | `Gen` | `Gen` | `Eval` along x's path, or `Preprocess` then `Eval` on the parity tree | `GrottoDcfEvalPointGpu` after repack, or `GrottoDcfPreprocessGpu` then device-callable `Eval` on its parity tree | `EvalAll` | `GrottoDcfEvalAllGpu` |
| HalfTree Grotto DCF | `Gen` | `Gen` | `Eval` | `HalfTreeGrottoDcfEvalPointGpu` after repack | No full-domain API | No full-domain API |
| VDPF | `Gen` | `Gen` | `Eval` returns a share and a corrected per-point hash | `VdpfEvalPointGpu` after repack | `EvalAll` with chain proof, `EvalAllMerkle` | `VdpfEvalAllMerkleGpu` |
| DMPF | `Gen` | No | `BatchEval` | `DmpfBatchEvalGpu` | `EvalAll` | `DmpfEvalAllGpu` |
| VDMPF | `Gen` | No | `BatchEval` with chain proof, `BatchEvalMerkle` | `VdmpfBatchEvalMerkleGpu` | No full-domain API | No full-domain API |

A HalfTree Grotto DCF key is a 1-bit Packed HalfTree DPF key with `beta = 1`, so `PackedHalfTreeDpf::Eval` on the same key gives the shares of `x == alpha`.

DPF, DCF, HalfTree, Packed HalfTree, Grotto DCF, HalfTree Grotto DCF, and VDPF also expose device-callable scalar evaluation through their scheme objects. Pass device-accessible keys and a device-callable backend. GPU helpers accept a CUDA stream and preserve the configured scheme type, including nondefault `par_depth`. DMPF and VDMPF helpers also preserve nondefault `ch_lambda`.

On x86-64 with SSE4.1, `fss/point_eval_cpu.cuh` provides `DpfPackPoint` / `DpfEvalPoint` and `HalfTreeDpfPackPoint` / `HalfTreeDpfEvalPoint`. Pack each key once before repeated queries. These helpers accept keys with nondefault depth. GPU `*RepackGpu` helpers produce the layouts for the GPU point launchers. Preparation cost and query cost are separate operations.

## Python operations

Python `gen` runs on CPU for every scheme. Move returned key tensors to CUDA before CUDA evaluation. Keep each evaluation's tensors on the same device. The native bindings select that device for copies, allocations, and launches, then restore the caller's previous CUDA device.

| Wrapper | CPU point or batch | CUDA point or batch | CPU full domain | CUDA full domain | Proof or preparation |
| --- | --- | --- | --- | --- | --- |
| `Dpf` | `eval` | `eval` | `eval_all` | `eval_all` | None |
| `Dcf` | `eval` | `eval` | `eval_all` | `eval_all` | None |
| `HalfTreeDpf` | `eval` | `eval` | `eval_all` | `eval_all` | None |
| `PackedHalfTreeDpf` | `eval` | `eval` | `eval_all` | `eval_all` | `extract` reads a lane from a packed output block |
| `GrottoDcf` | `eval(party, tree, x)` | No Python point method | `eval_all` | `eval_all` | `preprocess` accepts CPU or CUDA keys. Python `eval` consumes a CPU tree |
| `Vdpf` | `eval` | `eval` | `eval_all`, `eval_all_merkle` | `eval_all_merkle` | `prove(..., merkle=False/True)` on CPU, static `verify` |
| `Dmpf` | `batch_eval` | `batch_eval` | `eval_all` | `eval_all` | None |
| `Vdmpf` | `batch_eval`, `batch_eval_merkle` | `batch_eval`, `batch_eval_merkle` | No method | No method | Static `verify` |

`Vdpf.eval_all` and `Vdmpf.batch_eval` use chain accumulation on CPU. `Vdmpf.batch_eval` keeps its development API behavior on CUDA and produces a Merkle proof there. Use `Vdmpf.batch_eval_merkle` for the same Merkle accumulator on either device. The explicit C++ CUDA name is `VdmpfBatchEvalMerkleGpu`. `VdmpfBatchEvalGpu` remains a forwarding alias.

Chain and Merkle proofs have the same `(4, 4)` int32 representation but different values. Both parties must choose the same accumulator and matching queries. VDPF Merkle folding combines adjacent nodes of a fixed ordered sequence. VDMPF Merkle evaluation constructs a canonical per-bucket sequence and folds its chunks and bucket roots. CPU and CUDA Merkle results match for the same scheme configuration, key, and queries. See [the proof construction](vdpf_merkle_fold_security.md).

`Vdpf.verify(pi0, pi1)` and `Vdmpf.verify(pi0, pi1)` compare validated proof tensors. They require no instance or JIT compilation. Calling `scheme.verify(pi0, pi1)` also works. Keep both proof tensors on the same device.

## Host parallel execution

The eight C++ schemes with a `par_depth` template argument are DPF, DCF, HalfTree DPF, Packed HalfTree DPF, Grotto DCF, HalfTree Grotto DCF, VDPF, and DMPF. C++ defaults to `-1`. HalfTree Grotto DCF has no Python wrapper yet. The other Python constructors expose keyword-only `par_depth=0`, which preserves the sequential Python default.

A loaded Python extension initializes public ChaCha nonce or AES key parameters in its process. Generate and evaluate a key with the same initialized PRG parameters and extension configuration, including `par_depth`. Changing depth selects another extension whose public PRG values need not match. Raw key tensors and individual party shares need not match or interoperate across those extensions. The raw tensors do not export these public PRG parameters, so saving them alone does not provide a self-contained cross-process key format. Each configuration reconstructs the requested function from its own generated keys. Moving a key between CPU and CUDA keeps the same extension configuration when its backend supports both.

| Value | Host behavior |
| --- | --- |
| `0` | Execute the controlled PRG, mapping, conversion, hashing, and fold work on the calling thread, without starting an OpenMP worker team |
| `-1` | Resolve the split depth from the available OpenMP worker count and clamp it to useful work |
| Positive `d` | Use tree split depth `d`. For loop and fold stages, use at most `2^d` partitions, clamped to useful work |

The parameter controls partitioning. It does not change the process-wide OpenMP thread setting. Batched paths can use fewer partitions to preserve useful work per batch. Python rejects noninteger values and values below `-1`. It clamps oversized positive integers to the C++ `int` range before generating source.

| Scheme | Operations affected by depth | Remaining sequential work |
| --- | --- | --- |
| DPF, DCF, HalfTree, Packed HalfTree | `EvalAll` | Scalar point evaluation and generation |
| Grotto | `Preprocess`, `EvalAll` | Point queries and generation |
| HalfTree Grotto DCF | None. It selects the Packed HalfTree DPF key type | Point queries and generation |
| VDPF | `EvalAll` traversal, `EvalAllMerkle`, `ProveMerkle` | Chain conversion and accumulation in `EvalAll`, chain `Prove`, point evaluation, generation |
| DMPF | Inner DPF `EvalAll` | Mapping, gathering, generation, and point-batch evaluation |
| VDMPF | No exposed depth parameter | Host `BatchEval` and `BatchEvalMerkle` are sequential reference implementations |

The C++ toolchain must provide OpenMP headers. Compile and link parallel or automatic-depth evaluation with OpenMP support. For nvcc compilation, pass the OpenMP flag to the host pass and link its runtime, for example `-Xcompiler=-fopenmp`. Python probes compiler and linker support before compiling a requested parallel extension and supplies the host, nvcc-host, and linker flags. Unsupported parallel configurations fail before JIT compilation. Depth zero specifies sequential execution but does not guarantee that a toolchain without OpenMP headers can build the public library. GPU launches use their own tiling parameters and do not interpret the CPU depth as a CUDA launch configuration.

A C++ `Aes128Mmo` object borrows mutable OpenSSL contexts. Copies share those contexts. Use depth zero or provide a PRG with separate contexts for concurrent calls. Python `prg="aes128_mmo"` uses the actual `Aes128Mmo` through a binding adapter with one public key set per extension and owned contexts per host thread. The adapter supports the exposed host parallel modes and frees contexts when their owner thread exits.

## Backends and limits

| Component | Python option | Host | CUDA | Constraint |
| --- | --- | --- | --- | --- |
| ChaCha PRG | `prg="chacha"` | Yes | Yes | Keep the nonce and key configuration consistent between generation and evaluation |
| OpenSSL AES MMO PRG | `prg="aes128_mmo"` | Yes | No | C++ context ownership and concurrency rules apply |
| Raw AES MMO PRG | No Python option | Yes | No | C++ x86-64/AES-NI backend |
| OpenSSL Feistel PRP | `prp="aes128_feistel"` | Yes | No | Default PRP for Python DMPF and VDMPF |
| ChaCha Feistel PRP | `prp="chacha_feistel"` | Yes | Yes | Required for Python CUDA DMPF and VDMPF |
| Blake3 hash | `xor_hash="blake3"`, `hash="blake3"` | Yes | Yes | Required in both hash positions for Python CUDA VDPF and VDMPF |
| SHA256 hash | `xor_hash="sha256"`, `hash="sha256"` | Yes | No | Either SHA256 hash position rejects CUDA evaluation before launch |

Python CUDA evaluation also rejects the host-only AES PRG before launch. Generation can run on CPU with a CUDA-capable backend and produce keys for later CUDA evaluation. Python JIT compilation requires CUDA-enabled PyTorch and an nvcc toolchain even when callers evaluate on CPU. OpenSSL backends require its development headers and libraries.

Python uses `group="bytes"` or `group="uint"` for DPF, DCF, HalfTree, VDPF, DMPF, and VDMPF. Packed HalfTree uses XOR reconstruction with output width in `{1, 2, 4, 8, 16, 32, 64}`. It accepts the full unsigned beta range for its width, including `[0, 2^64)` for 64-bit outputs. Grotto returns boolean shares. Its `eval` arguments put `party` first, as do the other Python evaluators.

The tensor beta representation for byte and integer groups reserves the least significant bit of its fourth int32 word. Keep `beta[3] & 1 == 0`, including every row of multi-point betas. This is the existing C++ group precondition. The shape and dtype validators do not replace it. Packed HalfTree's integer beta interface has its own width check and accepts odd values.

| Python operation | Input-width or size limit |
| --- | --- |
| Scalar point evaluation for DPF, DCF, HalfTree, Packed HalfTree, VDPF | `1 <= in_bits <= 128`, subject to packed tree depth |
| Packed HalfTree construction | `depth = in_bits - log2(128 / out_bits) >= 1` |
| DMPF and VDMPF construction | `in_bits <= 63` because alphas and queries use int64 tensors. Bucket capacity also limits valid combinations |
| DMPF and VDMPF buckets | `1 <= bucket_bits <= 30`, `max_points >= 30`, `30 <= t <= max_points`. The runtime bucket domain must fit the declared width |
| Host full-domain bindings | At most 60 expanded bits before size calculation. Packed evaluation counts tree depth rather than the unpacked input width |
| DPF / DCF CUDA `eval_all` | `in_bits <= 41` |
| HalfTree CUDA `eval_all` | `in_bits <= 42` |
| Packed HalfTree CUDA `eval_all` | Expanded tree `depth <= 41` |
| Grotto CUDA `preprocess` / `eval_all` | `in_bits <= 38` |
| DMPF CUDA `eval_all` | `in_bits <= 38` |
| VDPF CUDA `eval_all_merkle` | `in_bits <= 22` |

These full-domain bounds describe representation and launch limits. An accepted width still requires enough memory for the outputs and scratch. VDPF checks its full-domain size before shifting or allocating, while retaining wide-domain point evaluation. C++ callers choose unsigned input types and CUDA tiling parameters that satisfy the selected helper's compile-time constraints. The Python limits above do not expand those C++ contracts.

C++ DMPF and VDMPF also constrain `bucket_bits` to `[1, 30]`. Their compile-time capacity check bounds the domain before multiplying it by the positive Cuckoo factor, preventing overflow even for 128-bit input types. For Python multi-point schemes, the validator uses the smallest supported runtime point count, `t=30`, when checking worst-case bucket capacity. With the Python defaults `kappa=3` and `ch_lambda=80`, the required capacity is `ceil(2^in_bits * 3 / ChBucket(30))`. C++ callers that customize these parameters must satisfy the corresponding compile-time and runtime checks.

DMPF and VDMPF queries must satisfy `0 <= xs < 2^in_bits` before unsigned conversion. Both Python and native extension entry points validate that range. Valid int64 values need no signed upper-bound comparison once `in_bits >= 63`. Duplicate and empty query tensors are supported. Python accepts noncontiguous query views and makes a contiguous input for native evaluation. Empty CUDA DMPF evaluation returns an empty output. Empty VDMPF Merkle evaluation returns the defined bucket-seed proof, matching the host reference.

## Released interfaces and retry behavior

The v1.3.0 candidate retains the v1.2.0 raw correction-word APIs alongside the aggregate `Key` APIs. The compatibility interfaces include caller-owned output buffers, Grotto's `Preprocess(ParityTree&, ...)`, VDMPF's `BucketKey` fields and `std::span` entry points, and the original template defaults. Existing DPF and DCF Python positional calls keep their argument order.

| Call family | Seed ownership and failure behavior |
| --- | --- |
| Released raw VDPF `Gen(cws, cs, ocw, const seed span, alpha, beta)` | Read-only seeds, one attempt, return `0` or `1` |
| Aggregate VDPF `Gen(k0, k1, mutable seed span, alpha, beta, retry)` | Defaults to retry. May perturb caller-owned seeds. Returned keys carry the effective seeds |
| Released VDMPF `Gen(..., sigma by value, const seed span, std::span alphas, std::span betas, t[, ch_retry])` | Read-only caller seeds and sigma, fail-fast behavior |
| Mutable VDMPF `Gen(..., sigma by reference, mutable seed span, cuda spans, t, ch_retry, retry)` | Both final arguments are explicit. Retrying can update seeds and sigma |
| New DMPF `Gen(..., mutable seeds, ..., ch_retry=1000, retry=true)` | Retrying can update seeds and sigma |

Use const seed views to request the released read-only contracts. For mutable VDMPF generation, pass `ch_retry` and `retry` so overload selection preserves the intended behavior. On successful retrying generation, use the returned key's effective seeds and sigma for evaluation. Python generation works on local copies and returns those effective values, so callers must keep the returned values or key object.

The legacy `ChaCha(const int*)` constructor remains a borrowed pointer API. `Gen` reads that pointer in its execution memory space. Keep the nonce storage alive, and use device-accessible storage for device calls. The two-integer ChaCha constructor owns its nonce values and supports capturing the scheme by value.

`DpfRelayoutGpu`, `DcfRelayoutGpu`, `HalfTreeDpfRelayoutGpu`, and `VdpfRelayoutGpu` forward to the current repack helpers. The released 32-bit query-buffer calls remain available for their supported domain widths. The typed input paths remain available for wider domains. Legacy adapters read 32-bit elements with their original stride before converting their values.

## GPU scratch lifetime

VDMPF batch Merkle, VDPF full-domain Merkle, and Grotto CUDA helpers cache owned scratch per host thread, helper specialization, and CUDA device. The cache reuses capacity on the same device. Thread exit releases its allocations. Workspace growth and release complete outstanding device work before freeing storage.

Do not overlap calls that share a workspace on different streams from one host thread. Use `fss::gpu::ReleaseGpuWorkspaces()` on each owning host thread before `cudaDeviceReset` or explicit runtime shutdown. Check its returned CUDA status. This function releases that thread's caches across devices and restores its previous current device. The Python API relies on the same native ownership but does not expose a separate workspace-release method.

## JIT and packaging

The JIT cache identity includes source and header contents, resolved CUDA architecture flags and PTX choices, depth and other scheme parameters, Python/Torch ABIs, compiler identities, and compile/link options. Each build stores `build_inputs.json`. A fresh process can import a matching binary without rebuilding. Changing an architecture, ABI, source, or supported build option selects a different extension.

Architecture resolution honors `TORCH_CUDA_ARCH_LIST` without assigning it. With no explicit target and no visible GPU, the headless fallback targets sm80. The installed wheel contains the public header tree, including internal helper headers, and binding `.cuh` files. Installed consumers discover the headers through the environment's data prefix, without relying on a repository checkout.

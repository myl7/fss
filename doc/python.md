# Python

`fss_crypto` binds the C++ schemes, Groups, PRGs, hashes and PRPs to PyTorch.
Scheme methods take caller-owned tensors for seeds, corrections, outputs and proofs.
They make one Core generation attempt; callers handle any reported retry status.

Install from a checkout with the declared dependencies:

```bash
uv sync --extra dev
```

The package requires Python >= 3.10 and PyTorch >= 2.6.
JIT compilation requires a CUDA toolkit compatible with the installed PyTorch CUDA major version, including for CPU operations.
OpenSSL backends also require development headers and libraries.

```python
import secrets
import torch
from fss_crypto import Dpf, group, prg

dpf = Dpf(8, group.Bytes(), prg.ChaCha(nonce=(1, 2), mul=2))
# A dealer samples independent seeds for both parties.
words = [secrets.randbits(32) - 2**31 for _ in range(8)]
s0s = torch.tensor(words, dtype=torch.int32).reshape(2, 4)
b_buf = torch.tensor([7, 0, 0, 0], dtype=torch.int32)
cws = torch.empty(dpf.cws_shape, dtype=torch.uint8)
dpf.gen(cws, s0s, a=42, b_buf=b_buf)
y0 = dpf.eval(b=False, s0=s0s[0], cws=cws, x=42)
y1 = dpf.eval(b=True, s0=s0s[1], cws=cws, x=42)
assert torch.equal(torch.bitwise_xor(y0, y1), b_buf)
```

Prepare compiled plugins before ordinary construction with the same constructor configuration, excluding runtime `instance_seed`:

```python
from fss_crypto import Dpf, HalfTreeDpf, group, hash, jit, prg

with jit.cache(write_root="./build/fss-plugins"):
    output_group = group.Bytes()
    chacha = prg.ChaCha(nonce=(0, 0), mul=2)
    Dpf.precompile(8, output_group, chacha)
    Dpf.precompile(8, output_group, chacha, threads_per_block=128)
    ccr = hash.Aes128Ccr(prg.Aes128Soft(mul=1, keys=(bytes(16),)))
    HalfTreeDpf.precompile(8, output_group, ccr)
    ccr.precompile()

with jit.cache(read_roots=("./build/fss-plugins",), write_root=None):
    dpf = Dpf(8, group.Bytes(), prg.ChaCha(nonce=(19, 27), mul=2))
```

Preparation reads backend template specifications without constructing native owners, tensors or device state, and does not serialize runtime keys or nonces.
It includes nested primitive dependencies and the default scheme launch specialization, plus any requested launch specialization.
With a write root, successful preparation guarantees reusable disk entries for the complete closure in that root, including when modules were already in memory.
With `write_root=None`, every prepared entry must be available from the read roots.
Loading still checks the current compiler, Python, PyTorch, flags and headers, so these plugins require the same toolchain.

Outside a scope, compiled plugins use `~/.cache/fss_crypto`.
Within a scope, lookups search `read_roots` in order, followed by `write_root` when present.
Read roots receive no files or locks and must not overlap the write root.
Read-only ordinary execution may reuse compatible modules already in memory, while a missing plugin raises an error without compiling.
Paths resolve on scope entry, nested scopes restore the preceding locations, and a new thread must establish its own scope.
A scheme retains the locations selected during construction for later launch specializations.
Ordinary constructors continue to load their default modules eagerly.

Run the complete Python examples after installation:

```bash
python samples/python/point_functions.py --device cpu
python samples/python/point_functions.py --device cuda:0
python samples/python/grotto.py --device cpu
python samples/python/grotto.py --device cuda:0
python samples/python/verifiable.py
python samples/python/verifiable.py --cuda
python samples/python/multi_point.py
python samples/python/multi_point.py --cuda
python samples/python/primitives.py --device cpu
python samples/python/primitives.py --device cuda:0
```

[Point and comparison functions](../samples/python/point_functions.py) cover DPF, DCF, HalfTreeDPF and PackedHalfTreeDPF, including full-domain evaluation, batched GPU keys and native Python Group conversions.
[Grotto trees](../samples/python/grotto.py) cover both Grotto variants, caller-provided preprocessing storage and CPU-owned trees.
[Verifiable evaluation](../samples/python/verifiable.py) covers VDPF generation, corrected hashes and proof aggregation.
[Multi-point functions](../samples/python/multi_point.py) cover DMPF and VDMPF generation, failure-dependent resampling and evaluation.
[Primitive operations](../samples/python/primitives.py) cover Groups, PRGs, hashes, permutations and seed control bits.
These examples keep both parties in one process to check reconstruction; a deployed party receives only its own seed and the shared key material.

DMPF and VDMPF generation take party-first seeds with shape `(2, scheme.kBucketCount, 4)`.
Allocate this contiguous `torch.int32` tensor directly in that order.
The view `seeds[party]` has shape `(scheme.kBucketCount, 4)` and is already contiguous for evaluation, without a transpose or copy.

All buffers of an operation must be contiguous and on the same device, including the CUDA device index.
`cws` is a `torch.uint8` tensor with `scheme.cws_shape`, containing complete native C++ correction records including padding.
It is not a serialized key format.
For CUDA, prepare the input and output tensors on the selected device or explicitly transfer them with `.to(device)` before the operation.
Bindings use PyTorch's current stream and retain native backend resource owners.
A backend that does not support CUDA is rejected; there is no CPU fallback.

`eval_all` dispatches from tensor devices.
On CPU, `par_depth=0` is sequential; positive depths require OpenMP and a parallel-safe backend.
On CUDA, `par_depth` is ignored.
GPU methods expose launch parameters only where their Core counterparts support them, using fixed Core defaults rather than hardware-dependent auto-tuning.
`eval_all_gpu_batch` accepts multiple independent keys for the same party and scheme configuration; it is distinct from querying multiple coordinates with `batch_eval`.

The first Core GPU launch on each device and launch specialization synchronizes to prepare its private host launch view.
Single-key GPU full evaluation and Grotto preprocessing also synchronize to read their seed value; multi-point full evaluation reads sigma.
The batched point-function GPU operation passes seed arrays directly.
Returning Python integers, booleans or bytes from CUDA likewise transfers the result to CPU.
These convenience paths are not promises of fully asynchronous scalar use.

For DCF, select `DcfPred.LT` or `DcfPred.GT` instead of a string.
Packed outputs retain multiple logical results per int4 block; use the scheme's `extract` operation and `kLanes` when reading them.
Chain and Merkle proofs remain separate protocols, and both parties must use the same protocol and ordered inputs.
VDMPF chain full evaluation remains CPU-only; its Merkle full evaluation supports CPU and CUDA.

// SPDX-License-Identifier: Apache-2.0
#include "binding_common.cuh"
#include "binding_gpu_eval.cuh"
#include <cuda/std/array>
#include <cuda/std/span>

namespace fss::python {
namespace {
using Proof = cuda::std::array<int4, 4>;

// Host-only descriptor selection keeps the wide Group tag out of the stub.
using VdpfLaunchDescriptor = fss::gpu::detail::VdpfGpuKernelDescriptor<kInBits,
    typename fss::detail::GpuGroupTag<GroupType>::Type, PrgType, XorHashType, HashType, InType>;

template <typename Descriptor>
__global__ void EvalKernel(void *scheme_storage, bool b, const int4 *seed,
    const void *cws_storage, const int4 *cs, InType x, int4 *y, Proof *pi) {
  using S = typename Descriptor::Scheme;
  S *scheme = static_cast<S *>(scheme_storage);
  const typename S::Cw *cws = static_cast<const typename S::Cw *>(cws_storage);
  *pi = scheme->Eval(b, *seed, {cws, kInBits + 1}, cuda::std::span<const int4,4>(cs,4), x, *y);
}

template <typename S = SchemeType>
int Gen(NativeState &state, Tensor cws, Tensor cs, Tensor seeds,
    py::int_ a, Tensor beta) {
  CheckSameDevice({cws, cs, seeds, beta});
  auto *words = CwPtr(cws, kInBits + 1);
  CheckBlocks(cs, 4); CheckBlocks(seeds, 2); CheckBlocks(beta, 1);
  InType alpha = ReadIn(a);
  if (cws.is_cpu()) {
    return state.Cpu().Gen(words, cuda::std::span<int4,4>(Blocks(cs),4),
        cuda::std::span<const int4,2>(ConstBlocks(seeds),2), alpha, *ConstBlocks(beta));
  }
  if constexpr (kGpuSupported) {
    c10::cuda::CUDAGuard guard(cws.device());
    auto stream = CurrentStream();
    auto status = at::empty({}, cws.options().dtype(at::kInt));
    CheckCuda(fss::gpu::GenGpu<S>(state.template Device<kGpuSupported, S>(stream),
        cuda::std::span<typename S::Cw,kInBits+1>(words,kInBits+1),
        cuda::std::span<int4,4>(Blocks(cs),4),
        cuda::std::span<const int4,2>(ConstBlocks(seeds),2),
        alpha, FetchBlock(beta), status.data_ptr<int>(), stream));
    state.RecordUse(stream); RecordTensors({cws, cs, seeds, beta, status}, stream);
    int result;
    CheckCuda(cudaMemcpyAsync(&result, status.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost, stream));
    CheckCuda(cudaStreamSynchronize(stream));
    return result;
  } else TORCH_CHECK(false, "configured backends do not support CUDA generation");
}

template <typename S = SchemeType>
Tensor Eval(NativeState &state, bool b, Tensor seed, Tensor cws, Tensor cs,
    py::int_ x, Tensor y) {
  CheckSameDevice({seed, cws, cs, y});
  CheckBlocks(seed, 1); CheckBlocks(cs, 4); CheckBlocks(y, 1);
  auto *words = ConstCwPtr(cws, kInBits + 1);
  auto pi = at::empty({4, 4}, y.options());
  auto *proof = reinterpret_cast<Proof *>(Blocks(pi));
  InType input = ReadIn(x);
  if (y.is_cpu()) *proof = state.Cpu().Eval(b, *ConstBlocks(seed),
      {words, kInBits + 1}, cuda::std::span<const int4,4>(ConstBlocks(cs),4), input, *Blocks(y));
  else if constexpr (kGpuSupported) {
    c10::cuda::CUDAGuard guard(y.device()); auto stream = CurrentStream();
    EvalKernel<VdpfLaunchDescriptor><<<1, 1, 0, stream>>>(
        state.template Device<kGpuSupported, S>(stream), b, ConstBlocks(seed),
        words, ConstBlocks(cs), input, Blocks(y), proof);
    CheckCuda(cudaGetLastError()); state.RecordUse(stream);
    RecordTensors({seed, cws, cs, y, pi}, stream);
  } else TORCH_CHECK(false, "configured backends do not support CUDA evaluation");
  return pi;
}

template <typename S = SchemeType>
void Prove(NativeState &state, Tensor corrected, Tensor cs, Tensor pi, bool merkle, int par_depth = 0) {
  TORCH_CHECK(pi.is_cuda() || par_depth >= 0, "par_depth must be nonnegative");
  CheckSameDevice({corrected, cs, pi}); CheckBlocks(cs, 4); CheckBlocks(pi, 4);
  CheckBlocks(corrected);
  TORCH_CHECK(corrected.numel() > 0, "corrected hashes must be nonempty");
  TORCH_CHECK(corrected.dim() == 3 && corrected.size(1) == 4,
      "corrected hashes must have shape [n, 4, 4]");
  auto inputs = cuda::std::span<const Proof>(reinterpret_cast<const Proof *>(ConstBlocks(corrected)), corrected.size(0));
  if (pi.is_cpu()) {
    if (merkle) state.CheckCpuParallel(par_depth, NativeState::kHashRole);
    if (merkle) state.Cpu().ProveMerkle(inputs, cuda::std::span<const int4,4>(ConstBlocks(cs),4), cuda::std::span<int4,4>(Blocks(pi),4), par_depth);
    else state.Cpu().Prove(inputs, cuda::std::span<const int4,4>(ConstBlocks(cs),4), cuda::std::span<int4,4>(Blocks(pi),4));
  } else if constexpr (kGpuSupported) {
    TORCH_CHECK(merkle, "chain proof accumulation is a CPU operation; use prove_merkle for CUDA");
    c10::cuda::CUDAGuard guard(pi.device()); auto stream = CurrentStream();
    CheckCuda(fss::gpu::ProveMerkleGpu<S>(state.template Device<kGpuSupported, S>(stream), inputs,
        cuda::std::span<const int4,4>(ConstBlocks(cs),4), cuda::std::span<int4,4>(Blocks(pi),4), stream));
    state.RecordUse(stream); RecordTensors({corrected, cs, pi}, stream);
  } else TORCH_CHECK(false, "configured backends do not support CUDA Merkle proofs");
}

template <typename S = SchemeType>
void Full(NativeState &state, bool b, Tensor seed, Tensor cws, Tensor cs,
    Tensor ys, Tensor pi, bool merkle, int par_depth = 0) {
  TORCH_CHECK(pi.is_cuda() || par_depth >= 0, "par_depth must be nonnegative");
  CheckSameDevice({seed, cws, cs, ys, pi});
  CheckBlocks(seed, 1); CheckBlocks(cs, 4); CheckBlocks(pi, 4);
  CheckBlocks(ys, AddressableCount(kInBits));
  auto *words = ConstCwPtr(cws, kInBits + 1);
  if constexpr (fss::detail::FitsFullDomain<int4, kInBits>()) {
    if (ys.is_cpu()) {
      state.CheckCpuParallel(par_depth, merkle
          ? NativeState::kPrgRole | NativeState::kXorHashRole | NativeState::kHashRole
          : NativeState::kPrgRole);
      if (merkle) {
        if constexpr (fss::detail::FitsFullDomain<Proof, kInBits>()) {
          static_cast<S &>(state.Cpu()).EvalAllMerkle(b, *ConstBlocks(seed), {words, kInBits + 1},
              cuda::std::span<const int4,4>(ConstBlocks(cs),4), {Blocks(ys), size_t(ys.numel()/4)}, cuda::std::span<int4,4>(Blocks(pi),4), par_depth);
        } else TORCH_CHECK(false, "full-domain proof scratch size overflows");
      } else {
        static_cast<S &>(state.Cpu()).EvalAll(b, *ConstBlocks(seed), {words, kInBits + 1},
            cuda::std::span<const int4,4>(ConstBlocks(cs),4), {Blocks(ys), size_t(ys.numel()/4)}, cuda::std::span<int4,4>(Blocks(pi),4), par_depth);
      }
    } else if constexpr (kGpuSupported && ValidCoreGpuLaunch<kInBits, kInBits, true>()) {
      TORCH_CHECK(merkle, "chain full evaluation is a CPU operation; use eval_all_merkle for CUDA");
      c10::cuda::CUDAGuard guard(ys.device()); auto stream = CurrentStream();
      auto &core = static_cast<S &>(state.template DeviceForLaunch<kGpuSupported>(stream));
      CheckCuda(core.template EvalAllMerkleGpu<kFrontierDepth, kThreadsPerBlock>(b,
          FetchBlock(seed), {words, kInBits + 1}, cuda::std::span<const int4,4>(ConstBlocks(cs),4),
          {Blocks(ys), size_t(ys.numel()/4)}, cuda::std::span<int4,4>(Blocks(pi),4), stream));
      state.RecordUse(stream); RecordTensors({seed, cws, cs, ys, pi}, stream);
    } else TORCH_CHECK(false, "configured full-domain CUDA evaluation is unavailable");
  } else TORCH_CHECK(false, "full domain does not fit a tensor allocation");
}

template <bool merkle, typename S = SchemeType>
void Range(NativeState &state, bool b, const py::int_ &first, const py::int_ &last,
    Tensor seed, Tensor cws, Tensor cs, Tensor ys, Tensor pi, int par_depth) {
  CheckSameDevice({seed, cws, cs, ys, pi});
  CheckBlocks(seed, 1); CheckBlocks(cs, 4); CheckBlocks(pi, 4);
  const InType left = ReadIn(first);
  const InType right = ReadIn(last);
  const int64_t count = RangeBlockCount(left, right);
  CheckBlockCapacity(ys, count);
  const auto *words = ConstCwPtr(cws, kInBits + 1);
  if (ys.is_cpu()) {
    if constexpr (merkle) {
      constexpr size_t max_bytes = std::numeric_limits<size_t>::max();
      constexpr size_t initial_size = sizeof(Proof) + sizeof(cuda::std::array<int4, 2>);
      const size_t next_count = count / 2 + count % 2;
      TORCH_CHECK(static_cast<size_t>(count) <= max_bytes / initial_size,
          "range proof scratch size overflows");
      const size_t initial_bytes = static_cast<size_t>(count) * initial_size;
      TORCH_CHECK(next_count <= (max_bytes - initial_bytes) / sizeof(cuda::std::array<int4, 2>),
          "range proof scratch size overflows");
    }
    state.CheckCpuParallel(par_depth, merkle
        ? NativeState::kPrgRole | NativeState::kXorHashRole | NativeState::kHashRole
        : NativeState::kPrgRole);
    auto &core = static_cast<S &>(state.Cpu());
    if constexpr (merkle) {
      core.EvalRangeMerkle(b, left, right, *ConstBlocks(seed),
          {words, kInBits + 1}, cuda::std::span<const int4, 4>(ConstBlocks(cs), 4),
          {Blocks(ys), static_cast<size_t>(count)}, cuda::std::span<int4, 4>(Blocks(pi), 4), par_depth);
    } else {
      core.EvalRange(b, left, right, *ConstBlocks(seed),
          {words, kInBits + 1}, cuda::std::span<const int4, 4>(ConstBlocks(cs), 4),
          {Blocks(ys), static_cast<size_t>(count)}, cuda::std::span<int4, 4>(Blocks(pi), 4), par_depth);
    }
  } else if constexpr (!merkle) {
    throw py::type_error("chain range evaluation requires CPU storage; use eval_range_merkle for CUDA");
  } else if constexpr (kGpuSupported && ValidRangeGpuLaunch()) {
    c10::cuda::CUDAGuard guard(ys.device());
    auto stream = CurrentStream();
    auto &core = static_cast<S &>(state.template DeviceForLaunch<kGpuSupported>(stream));
    CheckCuda(core.template EvalRangeMerkleGpu<kFrontierDepth, kThreadsPerBlock>(b,
        left, right, FetchBlock(seed), {words, kInBits + 1},
        cuda::std::span<const int4, 4>(ConstBlocks(cs), 4),
        {Blocks(ys), static_cast<size_t>(count)}, cuda::std::span<int4, 4>(Blocks(pi), 4), stream));
    state.RecordUse(stream); RecordTensors({seed, cws, cs, ys, pi}, stream);
  } else {
    throw py::value_error("configured backend or GPU launch shape is unavailable; check frontier_depth and threads_per_block");
  }
}

bool Verify(NativeState &, Tensor p0, Tensor p1) {
  CheckSameDevice({p0,p1}); CheckBlocks(p0,4); CheckBlocks(p1,4);
  Proof left, right;
  if (p0.is_cpu()) {
    for (int i=0;i<4;++i) { left[i]=ConstBlocks(p0)[i]; right[i]=ConstBlocks(p1)[i]; }
  } else {
    c10::cuda::CUDAGuard guard(p0.device()); auto stream=CurrentStream();
    CheckCuda(cudaMemcpyAsync(left.data(),ConstBlocks(p0),sizeof(Proof),cudaMemcpyDeviceToHost,stream));
    CheckCuda(cudaMemcpyAsync(right.data(),ConstBlocks(p1),sizeof(Proof),cudaMemcpyDeviceToHost,stream));
    CheckCuda(cudaStreamSynchronize(stream));
  }
  return SchemeType::Verify(cuda::std::span<const int4,4>(left),cuda::std::span<const int4,4>(right));
}
}  // namespace
}  // namespace fss::python

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace fss::python;
  auto cls = BindNative(m);
  cls.def("gen", [](NativeState &s, Tensor c, Tensor cs, Tensor seeds, py::int_ a, Tensor b) { return Gen(s,c,cs,seeds,a,b); });
  cls.def("eval", &Eval<>);
  cls.def("prove", [](NativeState &s, Tensor h, Tensor cs, Tensor pi) { Prove(s,h,cs,pi,false); });
  cls.def("prove_merkle", [](NativeState &s, Tensor h, Tensor cs, Tensor pi, int par_depth) { Prove(s,h,cs,pi,true,par_depth); });
  cls.def("eval_all", [](NativeState &s, bool b, Tensor seed, Tensor c, Tensor cs, Tensor ys, Tensor pi, int par_depth) { Full(s,b,seed,c,cs,ys,pi,false,par_depth); });
  cls.def("eval_all_merkle", [](NativeState &s, bool b, Tensor seed, Tensor c, Tensor cs, Tensor ys, Tensor pi, int par_depth) { Full(s,b,seed,c,cs,ys,pi,true,par_depth); });
  cls.def("eval_range", &Range<false>, py::arg("b"), py::arg("first"),
      py::arg("last"), py::arg("s0"), py::arg("cws"), py::arg("cs"),
      py::arg("ys"), py::arg("pi"), py::arg("par_depth") = 0);
  cls.def("eval_range_merkle", &Range<true>, py::arg("b"), py::arg("first"),
      py::arg("last"), py::arg("s0"), py::arg("cws"), py::arg("cs"),
      py::arg("ys"), py::arg("pi"), py::arg("par_depth") = 0);
  cls.def("verify", &Verify);
}

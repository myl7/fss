// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "binding_common.cuh"
#include "binding_gpu_core.cuh"
#include <cuda_runtime.h>
#include <limits>
#include <fss/util.cuh>

namespace fss::python {
namespace detail {

// Scalar Python coordinates are kernel arguments, not temporary input tensors.
template <typename Core>
__global__ void PointEvalOneKernel(Core *scheme, bool b, const int4 *s0,
    const typename Core::Cw *cws, InType x, int4 *y) {
  y[0] = scheme->Eval(b, s0[0], cws, x);
}

template <typename Core>
__global__ void PackedExtractOneKernel(const int4 *block, InType x, uint64_t *out) {
  out[0] = Core::Extract(block[0], x);
}

template <typename Core>
void PointGenBlock(NativeState &state, const Tensor &cws, const Tensor &s0s,
    const py::int_ &a, const Tensor &b_buf) {
  constexpr size_t count = CorrectionCount<Core>();
  CheckSameDevice({cws, s0s, b_buf});
  CheckBlocks(s0s, 2);
  CheckBlocks(b_buf, 1);
  auto *words = CwPtr(cws, count);
  InType point = ReadInput<InType>(a, kInBits);
  if (cws.is_cuda()) {
    if constexpr (kGpuSupported) {
      c10::cuda::CUDAGuard guard(cws.device());
      auto stream = CurrentStream();
      int4 payload = FetchBlock(b_buf);
      CheckCuda(fss::gpu::GenGpu<Core>(state.template Device<kGpuSupported, Core>(stream),
          cuda::std::span<typename Core::Cw, count>(words, count),
          cuda::std::span<const int4, 2>(ConstBlocks(s0s), 2), point, payload, stream));
      RecordTensors({cws, s0s, b_buf}, stream);
      state.RecordUse(stream);
    } else {
      throw py::type_error("configured backend does not support CUDA generation");
    }
  } else {
    CheckCpu({cws, s0s, b_buf});
    Core &scheme = state.Cpu();
    scheme.Gen(words, ConstBlocks(s0s), point, FetchBlock(b_buf));
  }
}

template <typename Core>
void PointGenPacked(NativeState &state, const Tensor &cws, const Tensor &s0s,
    const py::int_ &a, uint64_t beta) {
  constexpr size_t count = CorrectionCount<Core>();
  CheckSameDevice({cws, s0s});
  CheckBlocks(s0s, 2);
  auto *words = CwPtr(cws, count);
  constexpr int payload_bits = 128 / Core::kLanes;
  if constexpr (payload_bits < 64) {
    TORCH_CHECK(beta < (uint64_t{1} << payload_bits), "beta exceeds the packed output width");
  }
  InType point = ReadInput<InType>(a, kInBits);
  if (cws.is_cuda()) {
    if constexpr (kGpuSupported) {
      c10::cuda::CUDAGuard guard(cws.device());
      auto stream = CurrentStream();
      CheckCuda(fss::gpu::GenGpu<Core>(state.template Device<kGpuSupported, Core>(stream),
          cuda::std::span<typename Core::Cw, count>(words, count),
          cuda::std::span<const int4, 2>(ConstBlocks(s0s), 2), point, beta, stream));
      RecordTensors({cws, s0s}, stream);
      state.RecordUse(stream);
    } else {
      throw py::type_error("configured backend does not support CUDA generation");
    }
  } else {
    CheckCpu({cws, s0s});
    Core &scheme = state.Cpu();
    scheme.Gen(words, ConstBlocks(s0s), point, beta);
  }
}

template <typename Core>
Tensor PointEval(NativeState &state, bool b, const Tensor &s0, const Tensor &cws,
    const py::int_ &x) {
  constexpr size_t count = CorrectionCount<Core>();
  CheckSameDevice({s0, cws});
  CheckBlocks(s0, 1);
  const auto *words = ConstCwPtr(cws, count);
  InType point = ReadInput<InType>(x, kInBits);
  auto y = at::empty({4}, s0.options().dtype(at::kInt));
  if (s0.is_cuda()) {
    if constexpr (kGpuSupported) {
      c10::cuda::CUDAGuard guard(s0.device());
      auto stream = CurrentStream();
      PointEvalOneKernel<Core><<<1, 1, 0, stream>>>(
          state.template Device<kGpuSupported, Core>(stream), b, ConstBlocks(s0), words, point, Blocks(y));
      CheckCuda(cudaGetLastError());
      RecordTensors({s0, cws, y}, stream);
      state.RecordUse(stream);
    } else {
      throw py::type_error("configured backend does not support CUDA evaluation");
    }
  } else {
    Core &scheme = state.Cpu();
    Blocks(y)[0] = scheme.Eval(b, ConstBlocks(s0)[0], words, point);
  }
  return y;
}

template <typename Core>
constexpr int FullOutputBits() {
  if constexpr (kSchemeKind == 4) return Core::kDepth;
  else return kInBits;
}

template <typename Core>
void PointEvalAll(NativeState &state, bool b, const Tensor &s0, const Tensor &cws,
    const Tensor &ys, int par_depth) {
  TORCH_CHECK(ys.is_cuda() || par_depth >= 0, "par_depth must be nonnegative");
  constexpr int bits = FullOutputBits<Core>();
  if constexpr (bits >= 63 || !fss::detail::FitsFullDomain<int4, bits>()) {
    throw py::value_error("full-domain output is not addressable by a tensor");
  } else {
    int64_t outputs = AddressableCount(bits);
    TORCH_CHECK(outputs <= std::numeric_limits<int64_t>::max() / 4,
        "full-domain output element count overflows");
    CheckSameDevice({s0, cws, ys});
    CheckBlocks(s0, 1);
    CheckBlocks(ys, outputs);
    constexpr size_t count = CorrectionCount<Core>();
    const auto *words = ConstCwPtr(cws, count);
    if (ys.is_cuda()) {
      constexpr int frontier_limit = kSchemeKind == 3 ? bits - 1 : bits;
      if constexpr (kGpuSupported && ValidCoreGpuLaunch<bits, frontier_limit>()) {
        c10::cuda::CUDAGuard guard(ys.device());
        auto stream = CurrentStream();
        auto &scheme = static_cast<Core &>(state.template DeviceForLaunch<kGpuSupported>(stream));
        CheckCuda(scheme.template EvalAllGpu<kFrontierDepth, kThreadsPerBlock>(b,
            FetchBlock(s0), cuda::std::span<const typename Core::Cw>(words, count),
            cuda::std::span<int4>(Blocks(ys), outputs), stream));
        RecordTensors({s0, cws, ys}, stream);
        state.RecordUse(stream);
      } else {
        throw py::value_error("configured backend or GPU launch shape is unavailable; check frontier_depth and threads_per_block");
      }
    } else {
      state.CheckCpuParallel(par_depth);
      Core &scheme = state.Cpu();
      scheme.EvalAll(b, ConstBlocks(s0)[0], words, Blocks(ys), par_depth);
    }
  }
}

template <typename Core>
void PointEvalRange(NativeState &state, bool b, const py::int_ &first,
    const py::int_ &last, const Tensor &s0, const Tensor &cws,
    const Tensor &ys, int par_depth) {
  CheckSameDevice({s0, cws, ys});
  CheckBlocks(s0, 1);
  constexpr size_t corrections = CorrectionCount<Core>();
  const auto *words = ConstCwPtr(cws, corrections);
  const InType left = ReadInput<InType>(first, kInBits);
  const InType right = ReadInput<InType>(last, kInBits);
  constexpr size_t lanes = [] {
    if constexpr (kSchemeKind == 4) return Core::kLanes;
    else return 1;
  }();
  const int64_t count = RangeBlockCount(left, right, lanes);
  CheckBlockCapacity(ys, count);
  if (ys.is_cuda()) {
    if constexpr (kSchemeKind == 4) {
      throw py::type_error("packed range evaluation requires CPU storage");
    } else if constexpr (kGpuSupported && ValidRangeGpuLaunch()) {
      c10::cuda::CUDAGuard guard(ys.device());
      auto stream = CurrentStream();
      auto &scheme = static_cast<Core &>(state.template DeviceForLaunch<kGpuSupported>(stream));
      CheckCuda(scheme.template EvalRangeGpu<kFrontierDepth, kThreadsPerBlock>(b,
          left, right, FetchBlock(s0),
          cuda::std::span<const typename Core::Cw>(words, corrections),
          cuda::std::span<int4>(Blocks(ys), count), stream));
      RecordTensors({s0, cws, ys}, stream);
      state.RecordUse(stream);
    } else {
      throw py::value_error("configured backend or GPU launch shape is unavailable; check frontier_depth and threads_per_block");
    }
  } else {
    constexpr int roles = kSchemeKind == 3 || kSchemeKind == 4
        ? NativeState::kHashRole : NativeState::kPrgRole;
    state.CheckCpuParallel(par_depth, roles);
    static_cast<Core &>(state.Cpu()).EvalRange(b, left, right, ConstBlocks(s0)[0],
        words, cuda::std::span<int4>(Blocks(ys), count), par_depth);
  }
}

template <typename Core>
void PointEvalAllGpuBatch(NativeState &state, bool b, const Tensor &s0s,
    const Tensor &cws, const Tensor &ys) {
  CheckSameDevice({s0s, cws, ys});
  TORCH_CHECK(ys.is_cuda(), "eval_all_gpu_batch requires CUDA storage");
  CheckBlocks(s0s);
  TORCH_CHECK(s0s.dim() == 2 && s0s.size(1) == 4,
      "s0s must have shape [key_count, 4]");
  constexpr int bits = FullOutputBits<Core>();
  constexpr int64_t corrections = CorrectionCount<Core>();
  if constexpr (bits >= 63 || !fss::detail::FitsFullDomain<int4, bits>()) {
    throw py::value_error("full-domain output is not addressable by a tensor");
  } else if constexpr (!kGpuSupported ||
      !ValidCoreGpuLaunch<bits, kSchemeKind == 3 ? bits - 1 : bits>()) {
    throw py::value_error("configured backend or GPU launch shape is unavailable; check frontier_depth and threads_per_block");
  } else {
    int64_t keys = s0s.size(0);
    int64_t outputs = AddressableCount(bits);
    constexpr int64_t max_elements = std::numeric_limits<int64_t>::max();
    TORCH_CHECK(keys <= max_elements / outputs / 4 &&
        keys <= max_elements / corrections / sizeof(typename Core::Cw),
        "full-domain batch storage size overflows");
    TORCH_CHECK(cws.dim() == 3 && cws.size(0) == keys &&
        cws.size(1) == corrections && cws.size(2) == sizeof(typename Core::Cw),
        "cws has the wrong key-major correction-word shape");
    const auto *words = ConstCwPtr(cws, keys * corrections);
    CheckBlocks(ys, keys * outputs);
    TORCH_CHECK(ys.dim() == 3 && ys.size(0) == keys &&
        ys.size(1) == outputs && ys.size(2) == 4,
        "ys has the wrong key-major full-domain shape");
    if (keys == 0) return;
    c10::cuda::CUDAGuard guard(ys.device());
    auto stream = CurrentStream();
    auto &scheme = static_cast<Core &>(state.template DeviceForLaunch<kGpuSupported>(stream));
    CheckCuda(scheme.template EvalAllGpuBatch<kFrontierDepth, kThreadsPerBlock>(b,
        cuda::std::span<const int4>(ConstBlocks(s0s), keys),
        cuda::std::span<const typename Core::Cw>(words, keys * corrections),
        cuda::std::span<int4>(Blocks(ys), keys * outputs), stream));
    RecordTensors({s0s, cws, ys}, stream);
    state.RecordUse(stream);
  }
}

template <typename Core>
uint64_t PackedExtract(NativeState &, const Tensor &block, const py::int_ &x) {
  CheckBlocks(block, 1);
  InType point = ReadInput<InType>(x, kInBits);
  if (block.is_cuda()) {
    if constexpr (kGpuSupported) {
      c10::cuda::CUDAGuard guard(block.device());
      auto stream = CurrentStream();
      auto value = at::empty({}, block.options().dtype(at::ScalarType::UInt64));
      PackedExtractOneKernel<Core><<<1, 1, 0, stream>>>(ConstBlocks(block), point, value.data_ptr<uint64_t>());
      CheckCuda(cudaGetLastError());
      RecordTensors({block, value}, stream);
      auto host_value = value.cpu();
      return host_value.data_ptr<uint64_t>()[0];
    } else {
      throw py::type_error("configured backend does not support CUDA execution");
    }
  }
  return Core::Extract(ConstBlocks(block)[0], point);
}

template <typename Core = SchemeType>
void BindPointScheme(py::module_ &module) {
  auto cls = BindNative(module);
  if constexpr (kSchemeKind == 4) {
    cls.def("gen", &PointGenPacked<Core>, py::arg("cws"), py::arg("s0s"), py::arg("a"), py::arg("beta"));
    cls.def("extract", &PackedExtract<Core>, py::arg("block"), py::arg("x"));
  } else {
    cls.def("gen", &PointGenBlock<Core>, py::arg("cws"), py::arg("s0s"), py::arg("a"), py::arg("b_buf"));
  }
  cls.def("eval", &PointEval<Core>, py::arg("b"), py::arg("s0"), py::arg("cws"), py::arg("x"));
  cls.def("eval_all", &PointEvalAll<Core>, py::arg("b"), py::arg("s0"), py::arg("cws"), py::arg("ys"), py::arg("par_depth") = 0);
  cls.def("eval_range", &PointEvalRange<Core>, py::arg("b"), py::arg("first"),
      py::arg("last"), py::arg("s0"), py::arg("cws"), py::arg("ys"), py::arg("par_depth") = 0);
  cls.def("eval_all_gpu_batch", &PointEvalAllGpuBatch<Core>, py::arg("b"),
      py::arg("s0s"), py::arg("cws"), py::arg("ys"));
}

}  // namespace detail
}  // namespace fss::python

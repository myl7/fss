// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "binding_tensor.cuh"
#include "binding_materials.cuh"
#include "binding_device_init.cuh"
#include <cuda/std/tuple>
#include <memory>
#include <mutex>
#include "native_group_value.cuh"

namespace fss::python {

#define FSS_PRIMITIVE_SPACE __host__
#define FSS_PRIMITIVE_APPLY ApplyPrimitiveHost
#include "primitive_operations.cuh"
#undef FSS_PRIMITIVE_APPLY
#undef FSS_PRIMITIVE_SPACE
#define FSS_PRIMITIVE_SPACE __device__
#define FSS_PRIMITIVE_APPLY ApplyPrimitiveDevice
#include "primitive_operations.cuh"
#undef FSS_PRIMITIVE_APPLY
#undef FSS_PRIMITIVE_SPACE

template <typename Tag>
__global__ void PrimitiveKernel(void *backend, const int4 *lhs,
    int4 *out, int mode, __uint128_t x, __uint128_t domain) {
  using T = typename Tag::Type;
  ApplyPrimitiveDevice<T, kPrimitiveKind>(
      *static_cast<T *>(backend), lhs, out, mode, x, domain);
}

template <typename T>
class PrimitiveBinding {
public:
  explicit PrimitiveBinding(const py::dict &config) : material_(std::make_shared<BackendMaterial<T>>(config)) {}

  BackendHandle<T> Backend() const { return {material_}; }

  Tensor Run(const Tensor &lhs, int64_t count, int mode = 0,
      __uint128_t x = 0, __uint128_t domain = 0) {
    std::lock_guard<std::mutex> lock(mutex_);
    CheckBlocks(lhs);
    auto out = EmptyBlocks(count, lhs.options());
    if (!lhs.is_cuda()) {
      ApplyPrimitiveHost<T, kPrimitiveKind>(material_->Cpu(),
          ConstBlocks(lhs), Blocks(out), mode, x, domain);
    } else {
      if constexpr (kGpuSupported) {
        c10::cuda::CUDAGuard guard(lhs.device());
        auto stream = CurrentStream();
        int device = lhs.get_device();
        auto *object = material_->Device(device, stream);
        PrimitiveKernel<NativePrimitiveTag><<<1, 1, 0, stream>>>(object, ConstBlocks(lhs),
            Blocks(out), mode, x, domain);
        CheckCuda(cudaGetLastError());
        RecordTensors({lhs, out}, stream);
        material_->RecordUse(device, stream);
      } else TORCH_CHECK(false, "backend supports CPU tensors only");
    }
    return out;
  }

  Tensor Gen(const Tensor &seed) { CheckBlocks(seed, 1); return Run(seed, kPrgMul); }
  Tensor Gen8(const Tensor &seeds) {
    CheckBlocks(seeds, 8); CheckCpu({seeds});
    std::lock_guard<std::mutex> lock(mutex_);
    auto result = material_->Cpu().Gen8(cuda::std::span<const int4, 8>(ConstBlocks(seeds), 8));
    auto out = EmptyBlocks(8 * kPrgMul, seeds.options());
    for (int j = 0; j < 8; ++j)
      for (int i = 0; i < kPrgMul; ++i) Blocks(out)[j * kPrgMul + i] = result[j][i];
    return out.view({8, kPrgMul, 4});
  }
  Tensor Hash(const Tensor &msg) {
    CheckBlocks(msg);
    if constexpr (requires(T &backend) { backend.Hash(int4{}); }) {
      CheckBlocks(msg, 1); return Run(msg, 1).view({4});
    } else {
      int64_t count = msg.numel() / 4;
      TORCH_CHECK(count == 2 || count == 4, "hash requires two or four blocks");
      return Run(msg, count == 4 ? 2 : 4, static_cast<int>(count));
    }
  }
  py::int_ Permu(const Tensor &seed, const py::int_ &input, const py::int_ &domain_size) {
    CheckBlocks(seed, 1);
    auto x = ReadU128(input), domain = ReadU128(domain_size);
    TORCH_CHECK(domain >= 2 && x < domain, "permutation requires domain_size >= 2 and x < domain_size");
    auto out = FetchBlock(Run(seed, 1, 0, x, domain));
    __uint128_t result = static_cast<uint32_t>(out.x);
    result |= static_cast<__uint128_t>(static_cast<uint32_t>(out.y)) << 32;
    result |= static_cast<__uint128_t>(static_cast<uint32_t>(out.z)) << 64;
    result |= static_cast<__uint128_t>(static_cast<uint32_t>(out.w)) << 96;
    return ToPyU128(result);
  }

private:
  std::shared_ptr<BackendMaterial<T>> material_;
  std::mutex mutex_;
};

template <typename T>
void BindPrimitive(py::module_ &module) {
  using Binding = PrimitiveBinding<T>;
  py::class_<BackendHandle<T>>(module, "_BackendHandle", py::module_local());
  auto cls = py::class_<Binding>(module, "NativePrimitive", py::module_local()).def(py::init<const py::dict &>());
  cls.def("_backend", &Binding::Backend);
  if constexpr (kPrimitiveKind == 1) {
    BindGroupValue<T>(module, cls);
  } else if constexpr (kPrimitiveKind == 2) {
    cls.def("gen", &Binding::Gen);
    if constexpr (HasGen8<T, kPrgMul>)
      cls.def("gen8", &Binding::Gen8);
  } else if constexpr (kPrimitiveKind == 3) cls.def("hash", &Binding::Hash);
  else if constexpr (kPrimitiveKind == 4) cls.def("permu", &Binding::Permu);
}

}  // namespace fss::python

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  fss::python::BindPrimitive<PrimitiveType>(module);
}

// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/core/Tensor.h>
#include <c10/core/TensorOptions.h>
#include <ATen/ops/empty.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda/std/array>
#include <cuda/std/span>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <type_traits>

namespace fss::python {
namespace py = pybind11;
using Tensor = at::Tensor;

inline void CheckCuda(cudaError_t status) {
  TORCH_CHECK(status == cudaSuccess, "cuda operation failed: ", cudaGetErrorString(status));
}

inline cudaStream_t CurrentStream() { return c10::cuda::getCurrentCUDAStream().stream(); }

inline void CheckTensor(const Tensor &tensor) {
  TORCH_CHECK(tensor.defined(), "tensor must be defined");
  TORCH_CHECK(tensor.device().is_cpu() || tensor.is_cuda(), "tensor device must be CPU or CUDA");
  TORCH_CHECK(tensor.is_contiguous(), "tensor must be contiguous");
}

inline void CheckCpu(std::initializer_list<Tensor> tensors) {
  for (const auto &tensor : tensors) {
    CheckTensor(tensor);
    TORCH_CHECK(tensor.device().is_cpu(), "operation requires CPU storage");
  }
}

inline void CheckSameDevice(std::initializer_list<Tensor> tensors) {
  c10::optional<c10::Device> device;
  for (const auto &tensor : tensors) {
    CheckTensor(tensor);
    if (!device.has_value()) device = tensor.device();
    TORCH_CHECK(tensor.device() == *device, "all component tensors must use the same device");
  }
}

inline void CheckAlignment(const Tensor &tensor, size_t alignment) {
  TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % alignment == 0,
      "tensor storage is not aligned for the native type");
}

inline void CheckBlocks(const Tensor &tensor, int64_t count = -1) {
  CheckTensor(tensor);
  TORCH_CHECK(tensor.scalar_type() == at::kInt, "int4 storage must use int32");
  TORCH_CHECK(tensor.dim() >= 1 && tensor.size(-1) == 4, "int4 storage must end in dimension 4");
  if (count >= 0) {
    TORCH_CHECK(count <= std::numeric_limits<int64_t>::max() / 4, "int4 storage size overflows");
    TORCH_CHECK(tensor.numel() == count * 4, "int4 storage has the wrong number of blocks");
  }
  CheckAlignment(tensor, alignof(int4));
}

inline int4 *Blocks(const Tensor &tensor) {
  CheckBlocks(tensor);
  return reinterpret_cast<int4 *>(tensor.data_ptr<int32_t>());
}
inline const int4 *ConstBlocks(const Tensor &tensor) { return Blocks(tensor); }

inline void RecordTensors(std::initializer_list<Tensor> tensors, cudaStream_t stream) {
  for (const auto &tensor : tensors) {
    if (tensor.is_cuda()) {
      c10::cuda::CUDACachingAllocator::recordStream(tensor.storage().data_ptr(),
          c10::cuda::getStreamFromExternal(stream, tensor.get_device()));
    }
  }
}

inline __uint128_t ReadU128(const py::int_ &value) {
  TORCH_CHECK(!PyBool_Check(value.ptr()), "unsigned input must be an integer, not bool");
  py::int_ zero(0), shift(64);
  TORCH_CHECK(PyObject_RichCompareBool(value.ptr(), zero.ptr(), Py_LT) == 0,
      "unsigned input must be nonnegative");
  auto low = PyLong_AsUnsignedLongLongMask(value.ptr());
  if (PyErr_Occurred()) throw py::error_already_set();
  py::object high_object = py::reinterpret_steal<py::object>(PyNumber_Rshift(value.ptr(), shift.ptr()));
  if (!high_object) throw py::error_already_set();
  auto high = PyLong_AsUnsignedLongLong(high_object.ptr());
  if (PyErr_Occurred()) throw py::error_already_set();
  return static_cast<__uint128_t>(low) | (static_cast<__uint128_t>(high) << 64);
}

inline py::int_ ToPyU128(__uint128_t value) {
  py::int_ low(static_cast<uint64_t>(value)), high(static_cast<uint64_t>(value >> 64)), shift(64);
  py::object upper = py::reinterpret_steal<py::object>(PyNumber_Lshift(high.ptr(), shift.ptr()));
  if (!upper) throw py::error_already_set();
  py::object result = py::reinterpret_steal<py::object>(PyNumber_Or(upper.ptr(), low.ptr()));
  if (!result) throw py::error_already_set();
  return py::reinterpret_borrow<py::int_>(result);
}

template <typename In>
In ReadInput(const py::int_ &value, int bits) {
  __uint128_t result = ReadU128(value);
  TORCH_CHECK(bits > 0 && bits <= static_cast<int>(sizeof(In) * 8), "input type does not cover the configured domain");
  TORCH_CHECK(bits == 128 || result < (__uint128_t{1} << bits), "input is outside the configured domain");
  return static_cast<In>(result);
}

template <typename In>
void CheckInputs(const Tensor &tensor) {
  CheckTensor(tensor);
  auto dtype = tensor.scalar_type();
  if constexpr (sizeof(In) == 1) {
    TORCH_CHECK(dtype == at::kByte || dtype == at::kChar, "input storage must use 8-bit words");
  } else if constexpr (sizeof(In) == 2) {
    TORCH_CHECK(dtype == at::ScalarType::UInt16 || dtype == at::kShort, "input storage must use 16-bit words");
  } else if constexpr (sizeof(In) == 4) {
    TORCH_CHECK(dtype == at::ScalarType::UInt32 || dtype == at::kInt, "input storage must use 32-bit words");
  } else if constexpr (sizeof(In) == 8) {
    TORCH_CHECK(dtype == at::ScalarType::UInt64 || dtype == at::kLong, "input storage must use 64-bit words");
  } else {
    TORCH_CHECK((dtype == at::kLong || dtype == at::ScalarType::UInt64) && tensor.dim() >= 1 && tensor.size(-1) == 2,
        "128-bit input storage must use low/high 64-bit limbs");
  }
  CheckAlignment(tensor, alignof(In));
}

template <typename In>
In *InputData(const Tensor &tensor) {
  CheckInputs<In>(tensor);
  return reinterpret_cast<In *>(tensor.data_ptr());
}

template <typename In>
int64_t InputCount(const Tensor &tensor) {
  CheckInputs<In>(tensor);
  return tensor.numel() / (sizeof(In) == 16 ? 2 : 1);
}

template <typename In>
void CheckCpuInputsInDomain(const Tensor &tensor, int bits) {
  CheckCpu({tensor});
  const int64_t count = InputCount<In>(tensor);
  if (bits < sizeof(In) * 8) {
    const In *inputs = InputData<In>(tensor);
    for (int64_t i = 0; i < count; ++i) {
      TORCH_CHECK((static_cast<__uint128_t>(inputs[i]) >> bits) == 0,
          "query point is outside the input domain");
    }
  }
}

inline int4 FetchBlock(const Tensor &tensor) {
  CheckBlocks(tensor, 1);
  int4 result;
  if (tensor.is_cuda()) {
    c10::cuda::CUDAGuard guard(tensor.device());
    CheckCuda(cudaMemcpyAsync(&result, ConstBlocks(tensor), sizeof(result), cudaMemcpyDeviceToHost, CurrentStream()));
    CheckCuda(cudaStreamSynchronize(CurrentStream()));
  } else result = ConstBlocks(tensor)[0];
  return result;
}

inline void CheckBool(const Tensor &tensor, int64_t count = -1) {
  CheckTensor(tensor);
  TORCH_CHECK(tensor.scalar_type() == at::kBool, "boolean storage must use bool");
  if (count >= 0) TORCH_CHECK(tensor.numel() == count, "boolean storage has the wrong size");
}

inline bool *Bools(const Tensor &tensor) { CheckBool(tensor); return tensor.data_ptr<bool>(); }

inline int64_t AddressableCount(int bits) {
  TORCH_CHECK(bits >= 0 && bits < 63, "full-domain output is not addressable by a tensor");
  return int64_t{1} << bits;
}

inline Tensor EmptyBlocks(int64_t count, const c10::TensorOptions &options) {
  TORCH_CHECK(count >= 0 && count <= std::numeric_limits<int64_t>::max() / 4, "output allocation size overflows");
  return at::empty({count, 4}, options.dtype(at::kInt));
}

}  // namespace fss::python

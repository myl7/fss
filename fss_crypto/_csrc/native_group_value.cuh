// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "binding_tensor.cuh"
#include <memory>
#include <new>
#include <optional>
#include <type_traits>
#include <utility>
#include <fss/group/bytes.cuh>

namespace fss::python {
inline namespace FSS_NATIVE_ABI {

// Primitive type is selected by the module; the tag stays private to this translation unit.
namespace {
struct NativePrimitiveTag {
  using Type = PrimitiveType;
};
}  // namespace

template <typename Tag>
__global__ void GroupValueZeroKernel(void *out) {
  using T = typename Tag::Type;
  new (static_cast<T *>(out)) T{};
}

template <typename Tag>
__global__ void GroupValueFromKernel(const int4 *input, void *out) {
  using T = typename Tag::Type;
  new (static_cast<T *>(out)) T(T::From(input[0]));
}

template <typename Tag>
__global__ void GroupValueAddKernel(const void *lhs, const void *rhs, void *out) {
  using T = typename Tag::Type;
  new (static_cast<T *>(out)) T(*static_cast<const T *>(lhs) + *static_cast<const T *>(rhs));
}

template <typename Tag>
__global__ void GroupValueNegateKernel(const void *input, void *out) {
  using T = typename Tag::Type;
  new (static_cast<T *>(out)) T(-*static_cast<const T *>(input));
}

template <typename Tag>
__global__ void GroupValueIntoKernel(const void *input, int4 *out) {
  using T = typename Tag::Type;
  out[0] = static_cast<const T *>(input)->Into();
}

template <typename Tag>
__global__ void GroupValueReadKernel(const void *input, void *out) {
  using T = typename Tag::Type;
  using Result = decltype(std::declval<const T &>().Value());
  static_cast<Result *>(out)[0] = static_cast<const T *>(input)->Value();
}

// Each handle owns one constructed group element in its execution space.
template <typename T>
class NativeGroupValue {
public:
  using Handle = std::shared_ptr<NativeGroupValue>;

  static Handle Zero(const Tensor &prototype) {
    CheckBlocks(prototype, 1);
    if (!prototype.is_cuda()) return Host(T{});
    c10::cuda::CUDAGuard guard(prototype.device());
    auto stream = CurrentStream();
    auto result = Device(prototype.options());
    GroupValueZeroKernel<NativePrimitiveTag><<<1, 1, 0, stream>>>(result->DeviceData());
    result->Finish(stream);
    return result;
  }

  static Handle From(const Tensor &input) {
    CheckBlocks(input, 1);
    // Check Core's precondition before its device assertion can run.
    auto block = FetchBlock(input);
    TORCH_CHECK((block.w & 1) == 0, "group input must be clamped");
    if (!input.is_cuda()) return Host(T::From(block));
    c10::cuda::CUDAGuard guard(input.device());
    auto stream = CurrentStream();
    auto result = Device(input.options());
    GroupValueFromKernel<NativePrimitiveTag><<<1, 1, 0, stream>>>(ConstBlocks(input), result->DeviceData());
    result->Finish(stream);
    RecordTensors({input}, stream);
    return result;
  }

  ~NativeGroupValue() {
    if (ready_ == nullptr) return;
    int previous = 0;
    if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(storage_.get_device()) != cudaSuccess) return;
    cudaEventDestroy(ready_);
    cudaSetDevice(previous);
  }

  NativeGroupValue(const NativeGroupValue &) = delete;
  NativeGroupValue &operator=(const NativeGroupValue &) = delete;

  Tensor Into() const {
    if (host_) {
      auto output = EmptyBlocks(1, c10::TensorOptions().device(c10::kCPU));
      Blocks(output)[0] = host_->Into();
      return output.view({4});
    }
    c10::cuda::CUDAGuard guard(storage_.device());
    auto stream = CurrentStream();
    Await(stream);
    auto output = EmptyBlocks(1, storage_.options());
    GroupValueIntoKernel<NativePrimitiveTag><<<1, 1, 0, stream>>>(DeviceData(), Blocks(output));
    CheckCuda(cudaGetLastError());
    RecordTensors({storage_, output}, stream);
    return output.view({4});
  }

  py::object Value() const {
    if constexpr (std::is_same_v<T, fss::group::Bytes>) {
      return py::cast(Into());
    } else {
      using Result = decltype(std::declval<const T &>().Value());
      if (host_) return ToPyU128(static_cast<__uint128_t>(host_->Value()));
      c10::cuda::CUDAGuard guard(storage_.device());
      auto stream = CurrentStream();
      Await(stream);
      auto output = at::empty({static_cast<int64_t>(sizeof(Result))}, storage_.options());
      auto *output_data = static_cast<Result *>(output.data_ptr());
      GroupValueReadKernel<NativePrimitiveTag><<<1, 1, 0, stream>>>(DeviceData(), output_data);
      CheckCuda(cudaGetLastError());
      RecordTensors({storage_, output}, stream);
      Result result{};
      CheckCuda(cudaMemcpyAsync(&result, output_data, sizeof(Result), cudaMemcpyDeviceToHost, stream));
      CheckCuda(cudaStreamSynchronize(stream));
      return ToPyU128(static_cast<__uint128_t>(result));
    }
  }

  Handle Add(const NativeGroupValue &rhs) const {
    TORCH_CHECK(static_cast<bool>(host_) == static_cast<bool>(rhs.host_), "group values must use the same device");
    if (host_) return Host(*host_ + *rhs.host_);
    CheckSameDevice({storage_, rhs.storage_});
    c10::cuda::CUDAGuard guard(storage_.device());
    auto stream = CurrentStream();
    Await(stream);
    rhs.Await(stream);
    auto result = Device(storage_.options());
    GroupValueAddKernel<NativePrimitiveTag><<<1, 1, 0, stream>>>(DeviceData(), rhs.DeviceData(), result->DeviceData());
    result->Finish(stream);
    RecordTensors({storage_, rhs.storage_}, stream);
    return result;
  }

  Handle Negate() const {
    if (host_) return Host(-*host_);
    c10::cuda::CUDAGuard guard(storage_.device());
    auto stream = CurrentStream();
    Await(stream);
    auto result = Device(storage_.options());
    GroupValueNegateKernel<NativePrimitiveTag><<<1, 1, 0, stream>>>(DeviceData(), result->DeviceData());
    result->Finish(stream);
    RecordTensors({storage_}, stream);
    return result;
  }

private:
  static_assert(std::is_trivially_destructible_v<T>);
  static_assert(std::is_same_v<T, NativePrimitiveTag::Type>);

  explicit NativeGroupValue(T value) : host_(value) {}

  explicit NativeGroupValue(const c10::TensorOptions &options)
      : storage_(at::empty({static_cast<int64_t>(sizeof(T))}, options.dtype(at::kByte))) {
    CheckAlignment(storage_, alignof(T));
    CheckCuda(cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming));
  }

  static Handle Host(T value) { return Handle(new NativeGroupValue(value)); }
  static Handle Device(const c10::TensorOptions &options) { return Handle(new NativeGroupValue(options)); }
  T *DeviceData() const { return static_cast<T *>(storage_.data_ptr()); }

  void Finish(cudaStream_t stream) {
    CheckCuda(cudaGetLastError());
    RecordTensors({storage_}, stream);
    CheckCuda(cudaEventRecord(ready_, stream));
  }

  void Await(cudaStream_t stream) const { CheckCuda(cudaStreamWaitEvent(stream, ready_, 0)); }

  std::optional<T> host_;
  // Tensor allocator stream records retain storage through every device use.
  Tensor storage_;
  cudaEvent_t ready_ = nullptr;
};

template <typename T, typename Binding, typename... Options>
void BindGroupValue(py::module_ &module, py::class_<Binding, Options...> &primitive) {
  using Value = NativeGroupValue<T>;
  py::class_<Value, typename Value::Handle>(module, "NativeGroupValue", py::module_local())
      .def("into", &Value::Into)
      .def("value", &Value::Value)
      .def("add", &Value::Add)
      .def("negate", &Value::Negate);
  primitive.def("zero", [](Binding &, const Tensor &prototype) { return Value::Zero(prototype); })
      .def("from_", [](Binding &, const Tensor &input) { return Value::From(input); });
}

}  // namespace FSS_NATIVE_ABI
}  // namespace fss::python

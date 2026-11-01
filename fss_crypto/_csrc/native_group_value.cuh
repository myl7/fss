// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "binding_tensor.cuh"
#include <memory>
#include <new>
#include <optional>
#include <type_traits>
#include <utility>

namespace fss::python {
inline namespace FSS_NATIVE_ABI {

template <typename T>
__global__ void GroupValueZeroKernel(T *out) {
  new (out) T{};
}

template <typename T>
__global__ void GroupValueFromKernel(const int4 *input, T *out) {
  new (out) T(T::From(input[0]));
}

template <typename T>
__global__ void GroupValueAddKernel(const T *lhs, const T *rhs, T *out) {
  new (out) T(*lhs + *rhs);
}

template <typename T>
__global__ void GroupValueNegateKernel(const T *input, T *out) {
  new (out) T(-*input);
}

template <typename T>
__global__ void GroupValueIntoKernel(const T *input, int4 *out) {
  out[0] = input->Into();
}

template <typename T, typename Result>
__global__ void GroupValueReadKernel(const T *input, Result *out) {
  out[0] = input->Value();
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
    GroupValueZeroKernel<T><<<1, 1, 0, stream>>>(result->DeviceData());
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
    GroupValueFromKernel<T><<<1, 1, 0, stream>>>(ConstBlocks(input), result->DeviceData());
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
      auto output = EmptyBlocks(1, torch::TensorOptions().device(torch::kCPU));
      Blocks(output)[0] = host_->Into();
      return output.view({4});
    }
    c10::cuda::CUDAGuard guard(storage_.device());
    auto stream = CurrentStream();
    Await(stream);
    auto output = EmptyBlocks(1, storage_.options());
    GroupValueIntoKernel<T><<<1, 1, 0, stream>>>(DeviceData(), Blocks(output));
    CheckCuda(cudaGetLastError());
    RecordTensors({storage_, output}, stream);
    return output.view({4});
  }

  py::object Value() const {
    using Result = decltype(std::declval<const T &>().Value());
    if constexpr (std::is_same_v<Result, int4>) {
      if (host_) {
        auto output = EmptyBlocks(1, torch::TensorOptions().device(torch::kCPU));
        Blocks(output)[0] = host_->Value();
        return py::cast(output.view({4}));
      }
      c10::cuda::CUDAGuard guard(storage_.device());
      auto stream = CurrentStream();
      Await(stream);
      auto output = EmptyBlocks(1, storage_.options());
      GroupValueReadKernel<T, Result><<<1, 1, 0, stream>>>(DeviceData(), Blocks(output));
      CheckCuda(cudaGetLastError());
      RecordTensors({storage_, output}, stream);
      return py::cast(output.view({4}));
    } else {
      if (host_) return ToPyU128(static_cast<__uint128_t>(host_->Value()));
      c10::cuda::CUDAGuard guard(storage_.device());
      auto stream = CurrentStream();
      Await(stream);
      auto output = torch::empty({static_cast<int64_t>(sizeof(Result))}, storage_.options());
      auto *output_data = static_cast<Result *>(output.data_ptr());
      GroupValueReadKernel<T, Result><<<1, 1, 0, stream>>>(DeviceData(), output_data);
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
    GroupValueAddKernel<T><<<1, 1, 0, stream>>>(DeviceData(), rhs.DeviceData(), result->DeviceData());
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
    GroupValueNegateKernel<T><<<1, 1, 0, stream>>>(DeviceData(), result->DeviceData());
    result->Finish(stream);
    RecordTensors({storage_}, stream);
    return result;
  }

private:
  static_assert(std::is_trivially_destructible_v<T>);

  explicit NativeGroupValue(T value) : host_(value) {}

  explicit NativeGroupValue(const torch::TensorOptions &options)
      : storage_(torch::empty({static_cast<int64_t>(sizeof(T))}, options.dtype(torch::kUInt8))) {
    CheckAlignment(storage_, alignof(T));
    CheckCuda(cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming));
  }

  static Handle Host(T value) { return Handle(new NativeGroupValue(value)); }
  static Handle Device(const torch::TensorOptions &options) { return Handle(new NativeGroupValue(options)); }
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

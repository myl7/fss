// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "binding_materials.cuh"
#include "binding_device_init.cuh"
#include "binding_multi_descriptor.cuh"
#include <fss/dpf.cuh>
#include <fss/dcf.cuh>
#include <fss/half_tree_dpf.cuh>
#include <fss/vdpf.cuh>
#include <fss/dmpf.cuh>
#include <fss/vdmpf.cuh>
#include <cstddef>
#include <map>
#include <memory>
#include <type_traits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace fss::python {

template <typename T>
std::shared_ptr<BackendMaterial<T>> NativeRole(const py::dict &backends, const char *name) {
  if constexpr (std::is_same_v<T, NoBackend>) return std::make_shared<BackendMaterial<T>>(py::dict{});
  else {
    TORCH_CHECK(backends.contains(name), "native backend instance requires ", name);
    return ExtractBackend<T>(backends[name]);
  }
}

template <typename Core>
constexpr size_t CorrectionCount() {
  if constexpr (kSchemeKind == 4 || kSchemeKind == 6) return Core::kDepth + 1;
  else if constexpr (kSchemeKind == 8 || kSchemeKind == 9)
    return sizeof(std::declval<typename Core::BucketKey>().cws) / sizeof(typename Core::Cw);
  else return kInBits + 1;
}

template <typename Core>
constexpr size_t CorrectionRows() {
  if constexpr (kSchemeKind == 8 || kSchemeKind == 9) return Core::kBucketCount;
  else return 1;
}

using Cw = typename SchemeType::Cw;
inline constexpr size_t kNativeCorrectionCount = CorrectionCount<SchemeType>();
inline constexpr size_t kNativeCorrectionRows = CorrectionRows<SchemeType>();

inline Cw *CwPtr(const Tensor &tensor, int64_t expected_words = -1) {
  CheckTensor(tensor);
  TORCH_CHECK(tensor.scalar_type() == at::kByte, "correction storage must use uint8 native bytes");
  TORCH_CHECK(tensor.dim() >= 2 && tensor.size(-1) == sizeof(Cw),
      "correction storage has the wrong native record size");
  TORCH_CHECK(tensor.numel() % sizeof(Cw) == 0, "correction storage contains an incomplete record");
  if (expected_words >= 0) TORCH_CHECK(tensor.numel() / sizeof(Cw) == expected_words,
      "correction storage has the wrong number of records");
  CheckAlignment(tensor, alignof(Cw));
  return reinterpret_cast<Cw *>(tensor.data_ptr<uint8_t>());
}
inline const Cw *ConstCwPtr(const Tensor &tensor, int64_t expected_words = -1) {
  return CwPtr(tensor, expected_words);
}
inline InType ReadIn(const py::int_ &value) { return ReadInput<InType>(value, kInBits); }
inline const InType *ConstInPtr(const Tensor &tensor) { return InputData<InType>(tensor); }

struct NativeDeviceState {
  int device;
  Tensor object;
  std::unique_ptr<SchemeType> launch_object;
  cudaEvent_t last_use = nullptr;
  cudaStream_t stream = nullptr;
  bool pending_use = false;

  explicit NativeDeviceState(int ordinal) : device(ordinal) {
    CheckCuda(cudaEventCreateWithFlags(&last_use, cudaEventDisableTiming));
  }
  ~NativeDeviceState() {
    int previous = 0;
    if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(device) != cudaSuccess) return;
    if (last_use != nullptr) {
      if (pending_use && cudaEventRecord(last_use, stream) != cudaSuccess) cudaStreamSynchronize(stream);
      cudaEventSynchronize(last_use);
    }
    if (last_use != nullptr) cudaEventDestroy(last_use);
    cudaSetDevice(previous);
  }
};

template <typename DescriptorType, typename Prg = PrgType, typename XorHash = XorHashType,
    typename Hash = HashType, typename Prp = PrpType>
struct NativeDeviceFactory {
  using Descriptor = DescriptorType;
  using Core = typename DescriptorType::Scheme;
  Prg *prg;
  XorHash *xor_hash;
  Hash *hash;
  Prp *prp;
  int4 instance_seed;

  __device__ Core operator()() const {
    if constexpr (kSchemeKind == 1 || kSchemeKind == 2 || kSchemeKind == 5) return Core{*prg};
    else if constexpr (kSchemeKind == 3 || kSchemeKind == 4 || kSchemeKind == 6) return Core{*hash, instance_seed};
    else if constexpr (kSchemeKind == 7) return Core{*prg, *xor_hash, *hash};
    else if constexpr (kSchemeKind == 8) return Core{*prg, *prp};
    else return Core{*prg, *xor_hash, *hash, *prp};
  }
};

// Host-only overloads keep the raw scheme type out of the factory type.
template <typename Core>
inline NativeDeviceFactory<fss::gpu::detail::CoreLaunchIdentityDescriptor<Core>>
MakeNativeDeviceFactory(Core *, PrgType *prg, XorHashType *xor_hash, HashType *hash,
    PrpType *prp, int4 instance_seed) {
  return {prg, xor_hash, hash, prp, instance_seed};
}

template <int n, typename Group, typename In>
inline NativeDeviceFactory<fss::gpu::detail::DpfGpuKernelDescriptor<n,
    typename fss::detail::GpuGroupTag<Group>::Type, PrgType, In>> MakeNativeDeviceFactory(
    Dpf<n, Group, PrgType, In> *, PrgType *prg, XorHashType *xor_hash, HashType *hash,
    PrpType *prp, int4 instance_seed) {
  return {prg, xor_hash, hash, prp, instance_seed};
}

template <int n, typename Group, typename In, DcfPred pred>
inline NativeDeviceFactory<fss::gpu::detail::DcfGpuKernelDescriptor<n,
    typename fss::detail::GpuGroupTag<Group>::Type, PrgType, In, pred>> MakeNativeDeviceFactory(
    Dcf<n, Group, PrgType, In, pred> *, PrgType *prg, XorHashType *xor_hash, HashType *hash,
    PrpType *prp, int4 instance_seed) {
  return {prg, xor_hash, hash, prp, instance_seed};
}

template <int n, typename Group, typename In>
inline NativeDeviceFactory<fss::gpu::detail::HalfTreeGpuKernelDescriptor<n,
    typename fss::detail::GpuGroupTag<Group>::Type, HashType, In>> MakeNativeDeviceFactory(
    HalfTreeDpf<n, Group, HashType, In> *, PrgType *prg, XorHashType *xor_hash, HashType *hash,
    PrpType *prp, int4 instance_seed) {
  return {prg, xor_hash, hash, prp, instance_seed};
}

template <int n, typename Group, typename In>
inline NativeDeviceFactory<fss::gpu::detail::VdpfGpuKernelDescriptor<n,
    typename fss::detail::GpuGroupTag<Group>::Type, PrgType, XorHashType, HashType, In>>
MakeNativeDeviceFactory(
    Vdpf<n, Group, PrgType, XorHashType, HashType, In> *, PrgType *prg, XorHashType *xor_hash,
    HashType *hash, PrpType *prp, int4 instance_seed) {
  return {prg, xor_hash, hash, prp, instance_seed};
}

template <int in_bits, int max_points, int bucket_bits, typename Group, typename In, int ch_lambda>
inline NativeDeviceFactory<fss::gpu::detail::DmpfLaunchDescriptor<in_bits, max_points, bucket_bits,
    typename fss::detail::GpuGroupTag<Group>::Type, PrgType, PrpType, In, ch_lambda>>
MakeNativeDeviceFactory(
    Dmpf<in_bits, max_points, bucket_bits, Group, PrgType, PrpType, In, ch_lambda> *,
    PrgType *prg, XorHashType *xor_hash, HashType *hash, PrpType *prp, int4 instance_seed) {
  return {prg, xor_hash, hash, prp, instance_seed};
}

template <int in_bits, int max_points, int bucket_bits, typename Group, typename In, int ch_lambda>
inline NativeDeviceFactory<fss::gpu::detail::VdmpfLaunchDescriptor<in_bits, max_points, bucket_bits,
    typename fss::detail::GpuGroupTag<Group>::Type, PrgType, XorHashType, HashType, PrpType, In,
    ch_lambda>>
MakeNativeDeviceFactory(
    Vdmpf<in_bits, max_points, bucket_bits, Group, PrgType, XorHashType, HashType, PrpType, In,
        ch_lambda> *,
    PrgType *prg, XorHashType *xor_hash, HashType *hash, PrpType *prp, int4 instance_seed) {
  return {prg, xor_hash, hash, prp, instance_seed};
}

class NativeState {
public:
  explicit NativeState(const py::dict &backends, const py::object &instance_seed)
      : group_(NativeRole<GroupType>(backends, "group")), prg_(NativeRole<PrgType>(backends, "prg")),
        xor_hash_(NativeRole<XorHashType>(backends, "xor_hash")), hash_(NativeRole<HashType>(backends, "hash")),
        prp_(NativeRole<PrpType>(backends, "prp")) {
    if (!instance_seed.is_none()) instance_seed_ = ConfigBlock(instance_seed);
    cpu_ = std::make_unique<SchemeType>(MakeCpuScheme<SchemeType>());
  }
  ~NativeState() { devices_.clear(); }
  NativeState(const NativeState &) = delete;
  NativeState &operator=(const NativeState &) = delete;

  SchemeType &Cpu() { return *cpu_; }

  static constexpr int kPrgRole = 1;
  static constexpr int kXorHashRole = 2;
  static constexpr int kHashRole = 4;
  static constexpr int kPrpRole = 8;
  static constexpr int kAllRoles = kPrgRole | kXorHashRole | kHashRole | kPrpRole;

  void CheckCpuParallel(int par_depth, int roles = kAllRoles) const {
    TORCH_CHECK(par_depth >= 0, "par_depth must be nonnegative");
#ifdef _OPENMP
    bool parallel = par_depth != 0 && omp_get_max_threads() > 1;
#else
    TORCH_CHECK(par_depth == 0, "positive par_depth requires a module built with OpenMP support");
    bool parallel = false;
#endif
    TORCH_CHECK(!parallel || ((!(roles & kPrgRole) || prg_->CPUParallelSafe()) &&
        (!(roles & kXorHashRole) || xor_hash_->CPUParallelSafe()) &&
        (!(roles & kHashRole) || hash_->CPUParallelSafe()) &&
        (!(roles & kPrpRole) || prp_->CPUParallelSafe())),
        "parallel execution requires a backend without shared mutable contexts; use par_depth=0 for OpenSSL contexts");
  }

  template <bool Enabled = kGpuSupported, typename Core = SchemeType>
  Core *Device(cudaStream_t stream) {
    if constexpr (!Enabled) {
      throw py::type_error("configured backend combination does not support CUDA execution");
    } else {
      int device = 0;
      CheckCuda(cudaGetDevice(&device));
      auto &state = EnsureDeviceState(device, stream);
      auto factory = MakeNativeDeviceFactory(static_cast<Core *>(nullptr),
          prg_->Device(device, stream), xor_hash_->Device(device, stream),
          hash_->Device(device, stream), prp_->Device(device, stream), instance_seed_);
      if (!state.object.defined()) {
        Tensor object_storage = at::empty({static_cast<int64_t>(sizeof(Core))},
            c10::TensorOptions().dtype(at::kByte).device(c10::Device(c10::kCUDA, device)));
        CheckAlignment(object_storage, alignof(Core));
        auto *object = reinterpret_cast<Core *>(object_storage.data_ptr<uint8_t>());
        CheckCuda(detail::InitializeObjectGpu<Core>(object, factory, stream));
        CheckCuda(cudaEventRecord(state.last_use, stream));
        state.object = std::move(object_storage);
      }
      return reinterpret_cast<Core *>(state.object.data_ptr<uint8_t>());
    }
  }

  void RecordUse(cudaStream_t stream) {
    int device = 0;
    CheckCuda(cudaGetDevice(&device));
    auto it = devices_.find(device);
    TORCH_CHECK(it != devices_.end(), "device scheme has not been initialized");
    if (it->second->object.defined()) RecordTensors({it->second->object}, stream);
    prg_->RecordUse(device, stream);
    xor_hash_->RecordUse(device, stream);
    hash_->RecordUse(device, stream);
    prp_->RecordUse(device, stream);
    CheckCuda(cudaEventRecord(it->second->last_use, stream));
    it->second->pending_use = false;
  }

  template <bool Enabled = kGpuSupported>
  SchemeType &DeviceForLaunch(cudaStream_t stream) {
    static_assert(!Enabled || std::is_trivially_copyable_v<SchemeType>);
    auto *object = Device<Enabled>(stream);
    int device = 0;
    CheckCuda(cudaGetDevice(&device));
    auto &state = *devices_.at(device);
    if (!state.launch_object) {
      auto mirror = std::make_unique<SchemeType>(Cpu());
      // Host launch members need the actual device backend's pointer values.
      CheckCuda(cudaMemcpyAsync(mirror.get(), object, sizeof(SchemeType),
          cudaMemcpyDeviceToHost, stream));
      CheckCuda(cudaStreamSynchronize(stream));
      state.launch_object = std::move(mirror);
    }
    return *state.launch_object;
  }

private:
  NativeDeviceState &EnsureDeviceState(int device, cudaStream_t stream) {
    auto it = devices_.find(device);
    if (it == devices_.end()) {
      auto state = std::make_unique<NativeDeviceState>(device);
      CheckCuda(cudaEventRecord(state->last_use, stream));
      it = devices_.emplace(device, std::move(state)).first;
    } else {
      if (it->second->pending_use) CheckCuda(cudaEventRecord(it->second->last_use, it->second->stream));
      CheckCuda(cudaStreamWaitEvent(stream, it->second->last_use, 0));
    }
    it->second->stream = stream;
    it->second->pending_use = true;
    return *it->second;
  }

  template <typename Core>
  Core MakeCpuScheme() {
    if constexpr (kSchemeKind == 1 || kSchemeKind == 2 || kSchemeKind == 5) return Core{prg_->Cpu()};
    else if constexpr (kSchemeKind == 3 || kSchemeKind == 4 || kSchemeKind == 6)
      return Core{hash_->Cpu(), instance_seed_};
    else if constexpr (kSchemeKind == 7) return Core{prg_->Cpu(), xor_hash_->Cpu(), hash_->Cpu()};
    else if constexpr (kSchemeKind == 8) return Core{prg_->Cpu(), prp_->Cpu()};
    else return Core{prg_->Cpu(), xor_hash_->Cpu(), hash_->Cpu(), prp_->Cpu()};
  }

  std::shared_ptr<BackendMaterial<GroupType>> group_;
  std::shared_ptr<BackendMaterial<PrgType>> prg_;
  std::shared_ptr<BackendMaterial<XorHashType>> xor_hash_;
  std::shared_ptr<BackendMaterial<HashType>> hash_;
  std::shared_ptr<BackendMaterial<PrpType>> prp_;
  int4 instance_seed_{};
  std::unique_ptr<SchemeType> cpu_;
  std::map<int, std::unique_ptr<NativeDeviceState>> devices_;
};

template <typename Core>
void AddDimensions(py::module_ &module, py::class_<NativeState> &cls) {
  if constexpr (requires { Core::kLanes; }) {
    module.attr("kLanes") = Core::kLanes;
    cls.def_property_readonly("kLanes", [](const NativeState &) { return Core::kLanes; });
  }
  if constexpr (requires { Core::kDepth; }) {
    module.attr("kDepth") = Core::kDepth;
    cls.def_property_readonly("kDepth", [](const NativeState &) { return Core::kDepth; });
  }
  if constexpr (requires { Core::kBucketCount; }) {
    module.attr("kMinPoints") = Core::kMinPoints;
    module.attr("kBucketCount") = Core::kBucketCount;
    module.attr("kDomainSize") = ToPyU128(Core::kDomainSize);
    module.attr("kBucketSize") = ToPyU128(Core::kBucketSize);
    cls.def_property_readonly("kMinPoints", [](const NativeState &) { return Core::kMinPoints; });
    cls.def_property_readonly("kBucketCount", [](const NativeState &) { return Core::kBucketCount; });
    cls.def_property_readonly("kDomainSize", [](const NativeState &) { return ToPyU128(Core::kDomainSize); });
    cls.def_property_readonly("kBucketSize", [](const NativeState &) { return ToPyU128(Core::kBucketSize); });
  }
}

inline py::class_<NativeState> BindNative(py::module_ &module) {
  py::class_<NativeState> cls(module, "NativeScheme", py::module_local());
  cls.def(py::init<const py::dict &, const py::object &>(), py::arg("backends"), py::arg("instance_seed") = py::none());
  cls.def_property_readonly("cw_count", [](const NativeState &) { return kNativeCorrectionCount; });
  cls.def_property_readonly("cw_rows", [](const NativeState &) { return kNativeCorrectionRows; });
  cls.def_property_readonly("cw_size", [](const NativeState &) { return sizeof(Cw); });
  cls.def_property_readonly("cw_alignment", [](const NativeState &) { return alignof(Cw); });
  AddDimensions<SchemeType>(module, cls);
  return cls;
}

}  // namespace fss::python

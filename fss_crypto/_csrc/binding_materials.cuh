// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "binding_types.cuh"
#include "binding_tensor.cuh"
#include "binding_aes_material.cuh"
#include <fss/hash/aes128_ccr.cuh>
#include <array>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include "binding_device_init.cuh"

struct evp_cipher_ctx_st;

namespace fss::python {
inline namespace FSS_NATIVE_ABI {

template <typename T>
inline constexpr bool kHasGen = requires(T &value) { value.Gen(int4{}); };

template <typename T>
constexpr int BackendMultiplier() {
  if constexpr (kHasGen<T>) return cuda::std::tuple_size<decltype(std::declval<T &>().Gen(int4{}))>::value;
  else return 1;
}

template <typename T>
inline constexpr bool kSoftBackend = std::is_constructible_v<T, const uint8_t (*)[16], const uint32_t *, const uint8_t *>;

template <typename T>
inline constexpr bool kOpenSslPrg = requires(const unsigned char **keys) { T::CreateCtxs(keys); };

template <typename>
inline constexpr bool kUnsupportedDeviceBackend = false;

template <typename T, bool Enabled = kOpenSslPrg<T>>
struct BackendContexts {};

template <typename T>
struct BackendContexts<T, true> {
  static constexpr int kMultiplier = BackendMultiplier<T>();
  cuda::std::array<evp_cipher_ctx_st *, kMultiplier> values{};
  ~BackendContexts() { T::FreeCtxs(values); }
};

inline std::string ConfigBytes(const py::dict &config, const char *name, size_t count) {
  TORCH_CHECK(config.contains(name), "backend configuration requires ", name);
  TORCH_CHECK(py::isinstance<py::bytes>(config[name]), name, " must be bytes");
  auto bytes = py::cast<std::string>(config[name]);
  TORCH_CHECK(bytes.size() == count, name, " has the wrong byte length");
  return bytes;
}

inline int ConfigWord(const py::handle &value) {
  int64_t integer = py::cast<int64_t>(value);
  TORCH_CHECK(integer >= std::numeric_limits<int32_t>::min() && integer <= std::numeric_limits<uint32_t>::max(),
      "backend word must fit 32 bits");
  uint32_t word = static_cast<uint32_t>(integer);
  return word <= static_cast<uint32_t>(INT32_MAX) ? static_cast<int>(word)
      : -1 - static_cast<int>(UINT32_MAX - word);
}

inline int4 ConfigBlock(const py::handle &value) {
  py::sequence words = py::reinterpret_borrow<py::sequence>(value);
  TORCH_CHECK(py::len(words) == 4, "int4 configuration needs four words");
  return {ConfigWord(words[0]), ConfigWord(words[1]), ConfigWord(words[2]), ConfigWord(words[3])};
}

inline int4 BytesBlock(const char *bytes) {
  unsigned int words[4]{};
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) words[i] |= static_cast<unsigned int>(static_cast<unsigned char>(bytes[4 * i + j])) << (8 * j);
  }
  return {static_cast<int>(words[0]), static_cast<int>(words[1]), static_cast<int>(words[2]), static_cast<int>(words[3])};
}

template <typename T>
struct DeviceArgument {
  int nonce0 = 0;
  int nonce1 = 0;
  int4 iv[2]{};
  const uint8_t (*keys)[16] = nullptr;
  const uint32_t *te0 = nullptr;
  const uint8_t *sbox = nullptr;

  __device__ T Make() const {
    if constexpr (std::is_same_v<T, NoBackend>) return {};
    else if constexpr (std::is_constructible_v<T, int, int>) return T{nonce0, nonce1};
    else if constexpr (kSoftBackend<T>) return T{keys, te0, sbox};
    else if constexpr (std::is_constructible_v<T, cuda::std::span<const int4, 2>>)
      return T{cuda::std::span<const int4, 2>(iv, 2)};
    else if constexpr (!kHasGen<T> && std::is_default_constructible_v<T>) return T{};
    else static_assert(kUnsupportedDeviceBackend<T>, "backend does not support device construction");
  }
};

template <typename T>
struct BackendDeviceFactory {
  DeviceArgument<T> argument;
  __device__ T operator()() const { return argument.Make(); }
};

template <typename Aes>
struct CcrDeviceFactory {
  Aes *aes;
  __device__ hash::Aes128Ccr<Aes> operator()() const { return hash::Aes128Ccr<Aes>{*aes}; }
};

struct DeviceBackendStorage {
  int device;
  Tensor object;
  Tensor keys;
  Tensor te0;
  Tensor sbox;
  cudaEvent_t last_use = nullptr;
  cudaStream_t stream = nullptr;
  bool pending_use = false;

  explicit DeviceBackendStorage(int ordinal) : device(ordinal) {
    CheckCuda(cudaEventCreateWithFlags(&last_use, cudaEventDisableTiming));
  }
  ~DeviceBackendStorage() {
    int previous = 0;
    if (cudaGetDevice(&previous) != cudaSuccess) return;
    if (cudaSetDevice(device) != cudaSuccess) return;
    if (last_use != nullptr) {
      if (pending_use && cudaEventRecord(last_use, stream) != cudaSuccess) cudaStreamSynchronize(stream);
      cudaEventSynchronize(last_use);
      cudaEventDestroy(last_use);
    }
    cudaSetDevice(previous);
  }
};

template <typename T>
class BackendDevices {
public:
  template <typename MakeFactory>
  T *Get(int device, cudaStream_t stream, MakeFactory make_factory) {
    auto it = states_.find(device);
    if (it == states_.end()) {
      auto state = std::make_unique<DeviceBackendStorage>(device);
      state->stream = stream;
      state->pending_use = true;
      it = states_.emplace(device, std::move(state)).first;
    } else {
      if (it->second->pending_use) CheckCuda(cudaEventRecord(it->second->last_use, it->second->stream));
      CheckCuda(cudaStreamWaitEvent(stream, it->second->last_use, 0));
    }
    auto &state = *it->second;
    state.stream = stream;
    state.pending_use = true;
    if (!state.object.defined()) {
      auto storage = at::empty({static_cast<int64_t>(sizeof(T))},
          c10::TensorOptions().dtype(at::kByte).device(c10::Device(c10::kCUDA, device)));
      CheckAlignment(storage, alignof(T));
      auto factory = make_factory(state);
      CheckCuda(detail::InitializeObjectGpu(static_cast<T *>(storage.data_ptr()), factory, stream));
      CheckCuda(cudaEventRecord(state.last_use, stream));
      state.object = std::move(storage);
    }
    return static_cast<T *>(state.object.data_ptr());
  }

  void RecordUse(int device, cudaStream_t stream) {
    auto it = states_.find(device);
    if (it == states_.end()) return;
    auto &state = *it->second;
    for (const auto &tensor : {state.object, state.keys, state.te0, state.sbox}) {
      if (tensor.defined()) RecordTensors({tensor}, stream);
    }
    CheckCuda(cudaEventRecord(state.last_use, stream));
    state.pending_use = false;
  }

private:
  std::map<int, std::unique_ptr<DeviceBackendStorage>> states_;
};

template <typename T>
class BackendMaterial;

template <typename T>
struct BackendHandle {
  std::shared_ptr<BackendMaterial<T>> material;
};

template <typename T>
std::shared_ptr<BackendMaterial<T>> ExtractBackend(const py::handle &value) {
  // Foreign module-local loading transfers a typed reference, not a holder.
  const auto &handle = py::cast<const BackendHandle<T> &>(value);
  TORCH_CHECK(handle.material != nullptr, "native backend owner is empty");
  return handle.material;
}

template <typename T>
class BackendMaterial {
public:
  static constexpr int kMultiplier = BackendMultiplier<T>();

  explicit BackendMaterial(const py::dict &config) : config_(config) {
    if constexpr (std::is_same_v<T, NoBackend>) {
    } else if constexpr (std::is_constructible_v<T, int, int>) {
      TORCH_CHECK(config.contains("nonce"), "ChaCha configuration requires nonce");
      py::sequence nonce = py::reinterpret_borrow<py::sequence>(config["nonce"]);
      TORCH_CHECK(py::len(nonce) == 2, "nonce needs two words");
      nonce_[0] = ConfigWord(nonce[0]);
      nonce_[1] = ConfigWord(nonce[1]);
    } else if constexpr (kHasGen<T>) {
      TORCH_CHECK(config.contains("keys"), "AES configuration requires keys");
      py::sequence keys = py::reinterpret_borrow<py::sequence>(config["keys"]);
      TORCH_CHECK(py::len(keys) == kMultiplier, "AES key count must match the PRG multiplier");
      for (int i = 0; i < kMultiplier; ++i) {
        TORCH_CHECK(py::isinstance<py::bytes>(keys[i]), "each AES key must be bytes");
        auto bytes = py::cast<std::string>(keys[i]);
        TORCH_CHECK(bytes.size() == 16, "each AES key must contain 16 bytes");
        for (int j = 0; j < 16; ++j) keys_[i][j] = static_cast<uint8_t>(bytes[j]);
      }
      if constexpr (kSoftBackend<T>) {
        prg::aes_detail::InitTe0(te0_.data());
        prg::aes_detail::InitSbox(sbox_.data());
      } else if constexpr (kOpenSslPrg<T>) {
        const unsigned char *pointers[kMultiplier];
        for (int i = 0; i < kMultiplier; ++i) pointers[i] = keys_[i];
        contexts_.values = T::CreateCtxs(pointers);
      }
    } else if constexpr (std::is_constructible_v<T, cuda::std::span<const int4, 2>>) {
      auto bytes = ConfigBytes(config, "iv", 32);
      iv_[0] = BytesBlock(bytes.data());
      iv_[1] = BytesBlock(bytes.data() + 16);
    } else if constexpr (std::is_constructible_v<T, int4> && !std::is_default_constructible_v<T>) {
      auto bytes = ConfigBytes(config, "key", 16);
      key_ = BytesBlock(bytes.data());
    }
    cpu_ = std::make_unique<T>(BuildCpu());
  }

  BackendMaterial(const BackendMaterial &) = delete;
  BackendMaterial &operator=(const BackendMaterial &) = delete;

  T &Cpu() { return *cpu_; }
  bool CPUParallelSafe() const {
    if constexpr (kOpenSslPrg<T>) return false;
    else if constexpr (!kHasGen<T> && std::is_default_constructible_v<T> && !std::is_same_v<T, NoBackend>) {
      return !config_.contains("kind") || py::cast<std::string>(config_["kind"]) != "aes128_feistel";
    } else return true;
  }

  T *Device(int device, cudaStream_t stream) {
    if constexpr (std::is_same_v<T, NoBackend>) return nullptr;
    else return devices_.Get(device, stream, [&](DeviceBackendStorage &state) {
      DeviceArgument<T> argument;
      argument.nonce0 = nonce_[0];
      argument.nonce1 = nonce_[1];
      argument.iv[0] = iv_[0];
      argument.iv[1] = iv_[1];
      if constexpr (kSoftBackend<T>) {
        auto options = c10::TensorOptions().device(c10::Device(c10::kCUDA, device));
        state.keys = at::empty({kMultiplier, 16}, options.dtype(at::kByte));
        state.te0 = at::empty({256}, options.dtype(at::kInt));
        state.sbox = at::empty({256}, options.dtype(at::kByte));
        auto *keys = reinterpret_cast<uint8_t (*)[16]>(state.keys.data_ptr<uint8_t>());
        auto *te0 = reinterpret_cast<uint32_t *>(state.te0.data_ptr<int32_t>());
        auto *sbox = state.sbox.data_ptr<uint8_t>();
        detail::Aes128SoftTables tables{
            cuda::std::span<uint32_t, 256>(te0, 256), cuda::std::span<uint8_t, 256>(sbox, 256)};
        CheckCuda(detail::InitializeAes128SoftMaterialGpu<kMultiplier>(
            cuda::std::span<const uint8_t, static_cast<size_t>(kMultiplier) * 16>(&keys_[0][0], kMultiplier * 16),
            keys, tables, stream));
        argument.keys = keys;
        argument.te0 = te0;
        argument.sbox = sbox;
      }
      return BackendDeviceFactory<T>{argument};
    });
  }

  void RecordUse(int device, cudaStream_t stream) { devices_.RecordUse(device, stream); }

private:
  T BuildCpu() {
    if constexpr (std::is_same_v<T, NoBackend>) return {};
    else if constexpr (std::is_constructible_v<T, int, int>) return T{nonce_[0], nonce_[1]};
    else if constexpr (kSoftBackend<T>) return T{keys_, te0_.data(), sbox_.data()};
    else if constexpr (kOpenSslPrg<T>) return T{cuda::std::span<evp_cipher_ctx_st *, kMultiplier>(contexts_.values)};
    else if constexpr (kHasGen<T>) return T{keys_};
    else if constexpr (std::is_constructible_v<T, cuda::std::span<const int4, 2>>)
      return T{cuda::std::span<const int4, 2>(iv_, 2)};
    else if constexpr (std::is_constructible_v<T, int4> && !std::is_default_constructible_v<T>) return T{key_};
    else if constexpr (std::is_default_constructible_v<T>) {
      if constexpr (std::is_constructible_v<T, int4>) {
        if (config_.contains("seed")) return T{ConfigBlock(config_["seed"])};
      }
      return T{};
    } else static_assert(kUnsupportedDeviceBackend<T>, "unsupported native backend constructor");
  }

  py::dict config_;
  int nonce_[2]{};
  int4 iv_[2]{};
  int4 key_{};
  uint8_t keys_[kMultiplier][16]{};
  std::array<uint32_t, 256> te0_{};
  std::array<uint8_t, 256> sbox_{};
  BackendContexts<T> contexts_;
  std::unique_ptr<T> cpu_;
  BackendDevices<T> devices_;
};

template <typename Aes>
class BackendMaterial<hash::Aes128Ccr<Aes>> {
public:
  explicit BackendMaterial(const py::dict &config)
      : aes_(ExtractBackend<Aes>(config["aes_mmo"])), cpu_(aes_->Cpu()) {}
  hash::Aes128Ccr<Aes> &Cpu() { return cpu_; }
  bool CPUParallelSafe() const { return aes_->CPUParallelSafe(); }
  hash::Aes128Ccr<Aes> *Device(int device, cudaStream_t stream) {
    auto *aes = aes_->Device(device, stream);
    return devices_.Get(device, stream, [&](DeviceBackendStorage &) { return CcrDeviceFactory<Aes>{aes}; });
  }
  void RecordUse(int device, cudaStream_t stream) {
    devices_.RecordUse(device, stream);
    aes_->RecordUse(device, stream);
  }
private:
  std::shared_ptr<BackendMaterial<Aes>> aes_;
  hash::Aes128Ccr<Aes> cpu_;
  BackendDevices<hash::Aes128Ccr<Aes>> devices_;
};

}  // namespace FSS_NATIVE_ABI
}  // namespace fss::python

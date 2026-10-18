// SPDX-License-Identifier: Apache-2.0
#ifndef FSS_INTERNAL_GPU_WORKSPACE_CUH_
#define FSS_INTERNAL_GPU_WORKSPACE_CUH_

#include <map>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>

namespace fss::gpu::detail {

inline void CheckCudaWorkspace(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("workspace operation failed: ") + cudaGetErrorString(status));
  }
}

// One specialization's scratch allocations on one device. Growth allocates
// before replacing the old buffer, so a failed allocation preserves capacity.
class GpuWorkspace {
public:
  explicit GpuWorkspace(int device) : device_(device) {}
  GpuWorkspace(const GpuWorkspace &) = delete;
  GpuWorkspace &operator=(const GpuWorkspace &) = delete;
  ~GpuWorkspace() {
    Release();
  }

  template <typename T>
  T *Ensure(size_t slot, size_t count) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
      throw std::length_error("workspace size exceeds addressable memory");
    }
    const size_t bytes = count * sizeof(T);
    if (slot >= buffers_.size()) buffers_.resize(slot + 1);
    Buffer &buffer = buffers_[slot];
    if (bytes <= buffer.capacity) return static_cast<T *>(buffer.data);
    void *replacement = nullptr;
    CheckCudaWorkspace(cudaMalloc(&replacement, bytes));
    if (buffer.data != nullptr) {
      // The previous call can still be using the old capacity on a stream.
      cudaError_t status = cudaDeviceSynchronize();
      if (status == cudaSuccess) status = cudaFree(buffer.data);
      if (status != cudaSuccess) {
        cudaFree(replacement);
        CheckCudaWorkspace(status);
      }
    }
    buffer = {replacement, bytes};
    return static_cast<T *>(buffer.data);
  }

  // Select the allocation's device for both synchronization and cleanup.
  // Runtime shutdown can prevent cleanup. The runtime owns the allocations
  // in that case, and the destructor must not throw.
  cudaError_t Release() noexcept {
    if (buffers_.empty()) return cudaSuccess;
    int previous = 0;
    cudaError_t status = cudaGetDevice(&previous);
    if (status != cudaSuccess) return status;
    if (previous != device_) {
      status = cudaSetDevice(device_);
      if (status != cudaSuccess) return status;
    }
    cudaError_t result = cudaDeviceSynchronize();
    if (result == cudaSuccess) {
      for (Buffer &buffer : buffers_) {
        if (buffer.data == nullptr) continue;
        status = cudaFree(buffer.data);
        if (status == cudaSuccess) buffer = {};
        else if (result == cudaSuccess) result = status;
      }
    }
    if (previous != device_) {
      status = cudaSetDevice(previous);
      if (result == cudaSuccess) result = status;
    }
    if (result == cudaSuccess) buffers_.clear();
    return result;
  }

private:
  struct Buffer {
    void *data = nullptr;
    size_t capacity = 0;
  };
  int device_;
  std::vector<Buffer> buffers_;
};

class GpuWorkspaceCache {
public:
  GpuWorkspace &Get(const void *tag) {
    int device = 0;
    CheckCudaWorkspace(cudaGetDevice(&device));
    auto &entry = workspaces_[device][tag];
    if (!entry) entry = std::make_unique<GpuWorkspace>(device);
    return *entry;
  }

  cudaError_t Release() noexcept {
    cudaError_t result = cudaSuccess;
    for (auto &device : workspaces_) {
      for (auto &entry : device.second) {
        cudaError_t status = entry.second->Release();
        if (result == cudaSuccess) result = status;
      }
    }
    if (result == cudaSuccess) workspaces_.clear();
    return result;
  }

private:
  std::map<int, std::map<const void *, std::unique_ptr<GpuWorkspace>>> workspaces_;
};

inline GpuWorkspaceCache &GetGpuWorkspaceCache() {
  thread_local GpuWorkspaceCache cache;
  return cache;
}

}  // namespace fss::gpu::detail

namespace fss::gpu {

/**
 * Release this host thread's cached scratch on every CUDA device. Completes
 * outstanding device work and restores the current device. Call before
 * cudaDeviceReset or explicit CUDA runtime shutdown. Thread exit also releases
 * scratch. Calls sharing a workspace must not overlap on different streams.
 */
inline cudaError_t ReleaseGpuWorkspaces() noexcept {
  return detail::GetGpuWorkspaceCache().Release();
}

}  // namespace fss::gpu

#endif  // FSS_INTERNAL_GPU_WORKSPACE_CUH_

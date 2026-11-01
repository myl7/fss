// SPDX-License-Identifier: Apache-2.0
// Shared representation and per-instance resources for generated bindings.
#pragma once
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <type_traits>
#include "binding_tensor.cuh"
#include "binding_state.cuh"
#include "binding_gpu_launch.cuh"

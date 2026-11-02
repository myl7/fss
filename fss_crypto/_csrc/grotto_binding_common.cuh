// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <ATen/ops/from_blob.h>
#include <fss/grotto_dcf.cuh>
#include <fss/half_tree_grotto_dcf.cuh>
#include "binding_gpu_core.cuh"
#include <memory>
#include <utility>
#include <fss/util.cuh>

namespace fss::python {

template <typename Scheme>
__global__ void GrottoScalarKernel(Scheme *scheme, bool b, const int4 *s0,
    const typename Scheme::Cw *cws, InType x, bool *result) {
  result[0] = scheme->Eval(b, s0[0], cws, x);
}

template <typename Scheme>
__global__ void GrottoTreeScalarKernel(typename Scheme::ParityTreeView tree,
    InType x, bool *result) {
  result[0] = Scheme::Eval(tree, x);
}

inline bool FetchGrottoBool(const Tensor &result, cudaStream_t stream) {
  bool value;
  CheckCuda(cudaMemcpyAsync(&value, result.data_ptr<bool>(), sizeof(bool), cudaMemcpyDeviceToHost, stream));
  CheckCuda(cudaStreamSynchronize(stream));
  return value;
}

template <typename Scheme, bool Packed>
struct GrottoBinding {
  static constexpr int kTreeBits = [] {
    if constexpr (Packed) return Scheme::kDepth;
    else return kInBits;
  }();
  static constexpr size_t kCws = CwCount<Scheme>();
  static constexpr bool kTreeAddressable = kTreeBits < 63 &&
      fss::detail::FitsBinaryTree<bool, kTreeBits>() &&
      (!Packed || fss::detail::FitsFullDomain<int4, kTreeBits>());
  static constexpr bool kFullOutputAddressable = kInBits < 63 &&
      fss::detail::FitsFullDomain<bool, kInBits>() &&
      (!Packed || fss::detail::FitsFullDomain<int4, kTreeBits>());

  static int64_t TreeLeaves() { return AddressableCount(kTreeBits); }
  static int64_t TreeEntries() { return TreeLeaves() + (TreeLeaves() - 1); }

  static void CheckKey(const Tensor &s0, const Tensor &cws) {
    CheckBlocks(s0, 1);
    ConstCwPtr(cws, kCws);
    CheckSameDevice({s0, cws});
  }

  static void CheckTreeStorage(const Tensor &blocks, const Tensor &p) {
    CheckBool(p, TreeEntries());
    if constexpr (Packed) {
      CheckBlocks(blocks, TreeLeaves());
      CheckSameDevice({blocks, p});
    }
  }

  static typename Scheme::ParityTreeView Tree(const Tensor &blocks, const Tensor &p) {
    if constexpr (kTreeAddressable) {
      CheckTreeStorage(blocks, p);
      if constexpr (Packed) return {ConstBlocks(blocks), p.data_ptr<bool>()};
      else return {p.data_ptr<bool>()};
    } else throw py::value_error("parity-tree storage is not addressable as a tensor");
  }

  struct NativeTreeView {
    Tensor blocks;
    Tensor p;
    typename Scheme::ParityTreeView value;
    c10::Device device;

    NativeTreeView(Tensor blocks_owner, Tensor parity_owner)
        : blocks(std::move(blocks_owner)), p(std::move(parity_owner)),
          value(Tree(blocks, p)), device(p.device()) {}

    const typename Scheme::ParityTreeView &View() const {
      CheckTreeStorage(blocks, p);
      TORCH_CHECK(p.device() == device && p.data_ptr<bool>() == value.p,
          "parity-tree storage changed after native view construction");
      if constexpr (Packed) TORCH_CHECK(ConstBlocks(blocks) == value.blocks,
          "packed leaf storage changed after native view construction");
      return value;
    }
  };

  static void Gen(NativeState &state, Tensor cws, Tensor s0s, py::int_ a) {
    auto *words = CwPtr(cws, kCws);
    CheckBlocks(s0s, 2);
    CheckSameDevice({cws, s0s});
    auto input = ReadIn(a);
    if (s0s.is_cuda()) {
      if constexpr (kGpuSupported) {
        c10::cuda::CUDAGuard guard(s0s.device());
        auto stream = CurrentStream();
        CheckCuda(gpu::GenGpu(state.Device(stream),
            cuda::std::span<typename Scheme::Cw, kCws>(words, kCws),
            cuda::std::span<const int4, 2>(ConstBlocks(s0s), 2), input, stream));
        state.RecordUse(stream);
        RecordTensors({cws, s0s}, stream);
      } else throw py::type_error("configured backend does not support CUDA generation");
    } else {
      CheckCpu({cws, s0s});
      state.Cpu().Gen(words, ConstBlocks(s0s), input);
    }
  }

  static bool Eval(NativeState &state, bool b, Tensor s0, Tensor cws, py::int_ x) {
    CheckKey(s0, cws);
    auto input = ReadIn(x);
    if (s0.is_cuda()) {
      if constexpr (kGpuSupported) {
        c10::cuda::CUDAGuard guard(s0.device());
        auto stream = CurrentStream();
        auto result = at::empty({}, s0.options().dtype(at::kBool));
        GrottoScalarKernel<Scheme><<<1, 1, 0, stream>>>(state.Device(stream), b,
            ConstBlocks(s0), ConstCwPtr(cws), input, result.data_ptr<bool>());
        CheckCuda(cudaGetLastError());
        state.RecordUse(stream);
        RecordTensors({s0, cws, result}, stream);
        return FetchGrottoBool(result, stream);
      } else throw py::type_error("configured backend does not support CUDA evaluation");
    }
    CheckCpu({s0, cws});
    return state.Cpu().Eval(b, ConstBlocks(s0)[0], ConstCwPtr(cws), input);
  }

  static void EvalAll(NativeState &state, bool b, Tensor s0, Tensor cws, Tensor ys, int par_depth) {
    TORCH_CHECK(ys.is_cuda() || par_depth >= 0, "par_depth must be nonnegative");
    if constexpr (kFullOutputAddressable) {
      CheckKey(s0, cws);
      CheckBool(ys, AddressableCount(kInBits));
      CheckSameDevice({s0, cws, ys});
      if (s0.is_cuda()) {
        if constexpr (kGpuSupported) {
          c10::cuda::CUDAGuard guard(s0.device());
          auto stream = CurrentStream();
          if constexpr (ValidCoreGpuLaunch<kTreeBits>()) {
            auto &core = static_cast<Scheme &>(state.template DeviceForLaunch<kGpuSupported>(stream));
            CheckCuda(core.template EvalAllGpu<kFrontierDepth, kThreadsPerBlock>(b,
                FetchBlock(s0), cuda::std::span<const typename Scheme::Cw>(ConstCwPtr(cws), kCws),
                cuda::std::span<bool>(Bools(ys), ys.numel()), stream));
          } else throw py::value_error("GPU launch shape is unavailable; check frontier_depth and threads_per_block");
          state.RecordUse(stream);
          RecordTensors({s0, cws, ys}, stream);
        } else throw py::type_error("configured backend does not support CUDA evaluation");
      } else {
        CheckCpu({s0, cws, ys});
        state.CheckCpuParallel(par_depth);
        state.Cpu().EvalAll(b, ConstBlocks(s0)[0], ConstCwPtr(cws), Bools(ys), par_depth);
      }
    } else throw py::value_error("full-domain output is not addressable as a tensor");
  }

  static void Preprocess(NativeState &state, bool b, Tensor s0, Tensor cws,
      Tensor blocks, Tensor p, int par_depth) {
    TORCH_CHECK(p.is_cuda() || par_depth >= 0, "par_depth must be nonnegative");
    if constexpr (kTreeAddressable) {
      CheckKey(s0, cws);
      Tree(blocks, p);
      CheckSameDevice({s0, cws, p});
      if (s0.is_cuda()) {
        if constexpr (kGpuSupported) {
          c10::cuda::CUDAGuard guard(s0.device());
          auto stream = CurrentStream();
          if constexpr (ValidCoreGpuLaunch<kTreeBits>()) {
            auto &core = static_cast<Scheme &>(state.template DeviceForLaunch<kGpuSupported>(stream));
            auto words = cuda::std::span<const typename Scheme::Cw>(ConstCwPtr(cws), kCws);
            if constexpr (Packed) {
              CheckCuda(core.template PreprocessGpu<kFrontierDepth, kThreadsPerBlock>(b,
                  FetchBlock(s0), words, cuda::std::span<int4>(Blocks(blocks), TreeLeaves()),
                  cuda::std::span<bool>(Bools(p), p.numel()), stream));
              RecordTensors({blocks}, stream);
            } else CheckCuda(core.template PreprocessGpu<kFrontierDepth, kThreadsPerBlock>(b,
                FetchBlock(s0), words, cuda::std::span<bool>(Bools(p), p.numel()), stream));
          } else throw py::value_error("GPU launch shape is unavailable; check frontier_depth and threads_per_block");
          state.RecordUse(stream);
          RecordTensors({s0, cws, p}, stream);
        } else throw py::type_error("configured backend does not support CUDA preprocessing");
      } else {
        CheckCpu({s0, cws, p});
        state.CheckCpuParallel(par_depth);
        Scheme &core = state.Cpu();
        if constexpr (Packed) core.Preprocess(b, ConstBlocks(s0)[0], ConstCwPtr(cws), Blocks(blocks), Bools(p), par_depth);
        else core.Preprocess(b, ConstBlocks(s0)[0], ConstCwPtr(cws), Bools(p), par_depth);
      }
    } else throw py::value_error("parity-tree storage is not addressable as a tensor");
  }

  static py::object PreprocessOwned(NativeState &state, bool b, Tensor s0, Tensor cws, int par_depth) {
    if constexpr (kTreeAddressable) {
      CheckKey(s0, cws);
      CheckCpu({s0, cws});
      state.CheckCpuParallel(par_depth);
      Scheme &core = state.Cpu();
      auto native_owner = core.PreprocessOwned(b, ConstBlocks(s0)[0], ConstCwPtr(cws), par_depth);
      using Owner = decltype(native_owner);
      auto owner = std::make_shared<Owner>(std::move(native_owner));
      auto view = owner->View();
      auto p = at::from_blob(const_cast<bool *>(view.p), {TreeEntries()},
          [owner](void *) {}, c10::TensorOptions().dtype(at::kBool).device(c10::kCPU));
      if constexpr (Packed) {
        auto blocks = at::from_blob(const_cast<int4 *>(view.blocks), {TreeLeaves(), 4},
            [owner](void *) {}, c10::TensorOptions().dtype(at::kInt).device(c10::kCPU));
        return py::make_tuple(blocks, p);
      } else return py::cast(p);
    } else throw py::value_error("parity-tree storage is not addressable as a tensor");
  }

  static bool EvalTree(NativeState &, const NativeTreeView &pt, py::int_ x) {
    if constexpr (kTreeAddressable) {
      const auto &tree = pt.View();
      const Tensor &p = pt.p;
      auto input = ReadIn(x);
      if (p.is_cuda()) {
        c10::cuda::CUDAGuard guard(p.device());
        auto stream = CurrentStream();
        auto result = at::empty({}, p.options());
        GrottoTreeScalarKernel<Scheme><<<1, 1, 0, stream>>>(tree, input, Bools(result));
        CheckCuda(cudaGetLastError());
        RecordTensors({p, result}, stream);
        if constexpr (Packed) RecordTensors({pt.blocks}, stream);
        return FetchGrottoBool(result, stream);
      }
      CheckCpu({p});
      return Scheme::Eval(tree, input);
    } else throw py::value_error("parity-tree storage is not addressable as a tensor");
  }

  static void EvalAllTree(NativeState &, const NativeTreeView &pt, Tensor ys) {
    if constexpr (kFullOutputAddressable && kTreeAddressable) {
      const auto &tree = pt.View();
      const Tensor &p = pt.p;
      CheckBool(ys, AddressableCount(kInBits));
      CheckSameDevice({p, ys});
      if (p.is_cuda()) {
        c10::cuda::CUDAGuard guard(p.device());
        auto stream = CurrentStream();
        if constexpr (Packed) {
          CheckCuda(Scheme::template EvalAllGpu<kThreadsPerBlock>(tree,
              cuda::std::span<bool>(Bools(ys), ys.numel()), stream));
        } else CheckCuda(Scheme::EvalAllGpu(tree,
            cuda::std::span<bool>(Bools(ys), ys.numel()), stream));
        RecordTensors({p, ys}, stream);
        if constexpr (Packed) RecordTensors({pt.blocks}, stream);
      } else {
        CheckCpu({p, ys});
        Scheme::EvalAll(tree, Bools(ys));
      }
    } else throw py::value_error("full-domain output is not addressable as a tensor");
  }
};

template <typename Scheme, bool Packed>
void BindGrotto(py::module_ &module) {
  using Binding = GrottoBinding<Scheme, Packed>;
  using NativeTreeView = typename Binding::NativeTreeView;
  py::class_<NativeTreeView> tree_cls(module, "NativeParityTreeView", py::module_local());
  if constexpr (Packed) tree_cls.def(py::init<Tensor, Tensor>(), py::arg("blocks"), py::arg("p"));
  else tree_cls.def(py::init([](Tensor p) { return NativeTreeView(Tensor{}, std::move(p)); }), py::arg("p"));
  auto cls = BindNative(module);
  cls.def("gen", &Binding::Gen, py::arg("cws"), py::arg("s0s"), py::arg("a"));
  cls.def("eval", &Binding::Eval, py::arg("b"), py::arg("s0"), py::arg("cws"), py::arg("x"));
  cls.def("eval_all", &Binding::EvalAll, py::arg("b"), py::arg("s0"), py::arg("cws"),
      py::arg("ys"), py::arg("par_depth") = 0);
  cls.def("preprocess_owned", &Binding::PreprocessOwned, py::arg("b"), py::arg("s0"), py::arg("cws"), py::arg("par_depth") = 0);
  cls.def("eval_tree", &Binding::EvalTree, py::arg("pt"), py::arg("x"));
  cls.def("eval_all_tree", &Binding::EvalAllTree, py::arg("pt"), py::arg("ys"));
  if constexpr (Packed) {
    cls.def("preprocess", &Binding::Preprocess, py::arg("b"), py::arg("s0"), py::arg("cws"),
        py::arg("blocks"), py::arg("p"), py::arg("par_depth") = 0);
  } else {
    cls.def("preprocess", [](NativeState &state, bool b, Tensor s0, Tensor cws, Tensor p, int par_depth) {
      Binding::Preprocess(state, b, s0, cws, Tensor{}, p, par_depth);
    }, py::arg("b"), py::arg("s0"), py::arg("cws"), py::arg("p"), py::arg("par_depth") = 0);
  }
}

}  // namespace fss::python

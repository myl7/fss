// SPDX-License-Identifier: Apache-2.0
#include "binding_common.cuh"
#include "binding_gpu_eval.cuh"
#include <cuda/std/span>
#include <fss/util.cuh>

namespace fss::python {
namespace {
constexpr int kRows = SchemeType::kBucketCount;
constexpr int kWords = kBucketBits + 1;

void CheckRuntime(int rows, __uint128_t size) {
  TORCH_CHECK(rows > 0 && rows <= kRows, "m_rt is outside the allocated bucket count");
  TORCH_CHECK(size > 0 && size <= (__uint128_t(1) << kBucketBits), "b_size_rt is outside the bucket domain");
}

py::tuple Gen(NativeState &state, Tensor cws, Tensor sigma, Tensor seeds,
    Tensor as, Tensor betas, int t, int evictions) {
  CheckCpu({cws, sigma, seeds, as, betas});
  TORCH_CHECK(t >= SchemeType::kMinPoints && t <= kMaxPoints,
      "t must be between ", SchemeType::kMinPoints, " and ", kMaxPoints);
  TORCH_CHECK(evictions >= 0, "max_evictions must be nonnegative");
  CheckBlocks(sigma,1); CheckBlocks(seeds,int64_t{kRows}*2); CheckBlocks(betas);
  TORCH_CHECK(seeds.dim() == 3 && seeds.size(0) == 2 && seeds.size(1) == kRows && seeds.size(2) == 4,
      "s0s must have shape [2, ", kRows, ", 4]");
  TORCH_CHECK(betas.numel()/4 >= t, "b_bufs must contain at least t blocks");
  TORCH_CHECK(InputCount<InType>(as) >= t, "as must contain at least t inputs");
  const InType *points=ConstInPtr(as);
  for (int i=0;i<t;++i) {
    if constexpr (kInBits < sizeof(InType)*8)
      TORCH_CHECK((__uint128_t(points[i]) >> kInBits)==0, "point is outside the input domain");
  }
  auto *words=CwPtr(cws,kRows*kWords);
  auto seed_view=cuda::std::span<const int4>(ConstBlocks(seeds),size_t(kRows)*2);
  auto corrections=cuda::std::span<Cw>(words,size_t(kRows)*kWords);
  auto input=cuda::std::span<const InType>(points,t);
  auto payloads=cuda::std::span<const int4>(ConstBlocks(betas),t);
  int runtime_rows=fss::cuckoo_hash::ChBucket(t,kChLambda);
  __uint128_t required_size=((__uint128_t{3} << kInBits)+runtime_rows-1)/runtime_rows;
  TORCH_CHECK(required_size <= (__uint128_t{1} << kBucketBits), "runtime point count requires a larger bucket_bits");
  for (int i=0;i<t;++i) for (int j=0;j<i;++j)
    TORCH_CHECK(points[i] != points[j], "secret points must be distinct");
  int m_rt=0; __uint128_t b_size_rt=0;
  int status=state.Cpu().Gen(corrections, m_rt,b_size_rt,
      *ConstBlocks(sigma),seed_view,input,payloads,t,evictions);
  return py::make_tuple(status,m_rt,ToPyU128(b_size_rt));
}

template <typename S=SchemeType>
void Batch(NativeState &state,bool b,Tensor sigma,int m_rt,py::int_ size,
    Tensor seeds,Tensor cws, Tensor xs,Tensor ys) {
  CheckSameDevice({sigma,seeds,cws, xs,ys});
  CheckBlocks(sigma,1); CheckBlocks(seeds,kRows);
  int64_t count=InputCount<InType>(xs); CheckBlocks(ys,count);
  auto domain=ReadU128(size); CheckRuntime(m_rt,domain);
  auto *words=ConstCwPtr(cws,kRows*kWords);

  if (ys.is_cpu()) {
    CheckCpuInputsInDomain<InType>(xs, kInBits);
    state.Cpu().BatchEval(b,*ConstBlocks(sigma),m_rt,domain,
        cuda::std::span<const int4,kRows>(ConstBlocks(seeds),kRows),
        cuda::std::span<const Cw>(words,size_t(kRows)*kWords),
        {ConstInPtr(xs),size_t(count)},{Blocks(ys),size_t(count)});
  } else if constexpr (kGpuSupported) {

    c10::cuda::CUDAGuard guard(ys.device()); auto stream=CurrentStream();
    fss::gpu::MultiComponents<S> key{FetchBlock(sigma),m_rt,domain,
        {ConstBlocks(seeds),kRows},{words,kRows*kWords}};
    CheckCuda(fss::gpu::BatchEvalGpu<S>(state.template Device<kGpuSupported, S>(stream),b,key,
        {ConstInPtr(xs),size_t(count)},{Blocks(ys),size_t(count)},stream));
    state.RecordUse(stream); RecordTensors({sigma,seeds,cws, xs,ys},stream);
  } else TORCH_CHECK(false,"configured backends do not support CUDA evaluation");
}
template <typename S=SchemeType>
void Full(NativeState &state,bool b,Tensor sigma,int m_rt,py::int_ size,
    Tensor seeds,Tensor cws,Tensor ys,int par_depth) {
  TORCH_CHECK(ys.is_cuda() || par_depth >= 0,"par_depth must be nonnegative");
  CheckSameDevice({sigma,seeds,cws,ys}); CheckBlocks(sigma,1); CheckBlocks(seeds,kRows);
  CheckBlocks(ys,AddressableCount(kInBits));
  auto domain=ReadU128(size); CheckRuntime(m_rt,domain);
  auto *words=ConstCwPtr(cws,kRows*kWords);
  if (ys.is_cpu()) {
    if constexpr (fss::detail::FitsFullDomain<int4, kInBits>() &&
        fss::detail::FitsFullDomain<int4, kBucketBits>()) {
      state.CheckCpuParallel(par_depth, NativeState::kPrgRole);
      static_cast<S &>(state.Cpu()).EvalAll(b,*ConstBlocks(sigma),m_rt,domain,
          cuda::std::span<const int4,kRows>(ConstBlocks(seeds),kRows),
          cuda::std::span<const Cw>(words,size_t(kRows)*kWords),
          {Blocks(ys),size_t(ys.numel()/4)},par_depth);
    } else TORCH_CHECK(false,"CPU full evaluation requires addressable input and bucket domains");
  } else if constexpr (kGpuSupported && kInBits <= 38 &&
      fss::detail::FitsFullDomain<int4, kInBits>() &&
      fss::detail::FitsFullDomain<int4, kBucketBits>() && ValidCoreGpuLaunch<kBucketBits>()) {
    c10::cuda::CUDAGuard guard(ys.device());
    auto stream=CurrentStream();
    auto &core = static_cast<S &>(state.template DeviceForLaunch<kGpuSupported>(stream));
    CheckCuda(core.template EvalAllGpu<kFrontierDepth, kThreadsPerBlock>(b,
        FetchBlock(sigma), m_rt, domain, {ConstBlocks(seeds), kRows},
        {words, kRows * kWords}, {Blocks(ys), size_t(ys.numel() / 4)}, stream));
    state.RecordUse(stream);
    RecordTensors({sigma,seeds,cws,ys},stream);
  } else TORCH_CHECK(false,"configured full-domain CUDA evaluation is unavailable");
}

}  // namespace
}  // namespace fss::python

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  using namespace fss::python;
  auto cls=BindNative(m);
  cls.def("gen",[](NativeState &s,Tensor c, Tensor sigma,Tensor seeds,Tensor as,Tensor bs,int t,int e) { return Gen(s,c, sigma,seeds,as,bs,t,e); });
  cls.def("batch_eval",&Batch<>);
  cls.def("eval_all",&Full<>);
}

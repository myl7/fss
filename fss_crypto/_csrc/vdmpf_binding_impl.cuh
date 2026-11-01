// SPDX-License-Identifier: Apache-2.0
#include "binding_common.cuh"
#include "binding_gpu_eval.cuh"
#include <cuda/std/array>
#include <cuda/std/span>
#include <utility>

namespace fss::python {
namespace {
constexpr int kRows = SchemeType::kBucketCount;
constexpr int kWords = kBucketBits + 1;
using Proof = cuda::std::array<int4, 4>;

void CheckRuntime(int rows, __uint128_t size) {
  TORCH_CHECK(rows > 0 && rows <= kRows, "m_rt is outside the allocated bucket count");
  TORCH_CHECK(size > 0 && size <= (__uint128_t(1) << kBucketBits), "b_size_rt is outside the bucket domain");
}

py::tuple Gen(NativeState &state, Tensor cws, Tensor cs, Tensor sigma, Tensor seeds,
    Tensor as, Tensor betas, int t, int evictions) {
  CheckCpu({cws, cs, sigma, seeds, as, betas});
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
  CheckBlocks(cs,kRows*4);
  auto checks=cuda::std::span<int4>(Blocks(cs),size_t(kRows)*4);
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
  int status=state.Cpu().Gen(corrections, checks, m_rt,b_size_rt,
      *ConstBlocks(sigma),seed_view,input,payloads,t,evictions);
  return py::make_tuple(status,m_rt,ToPyU128(b_size_rt));
}

template <typename S=SchemeType>
void Batch(NativeState &state,bool b,Tensor sigma,int m_rt,py::int_ size,
    Tensor seeds,Tensor cws,Tensor cs, Tensor xs,Tensor ys,Tensor pi,bool merkle) {
  CheckSameDevice({sigma,seeds,cws,cs, xs,ys,pi});
  CheckBlocks(sigma,1); CheckBlocks(seeds,kRows);
  int64_t count=InputCount<InType>(xs); CheckBlocks(ys,count);
  auto domain=ReadU128(size); CheckRuntime(m_rt,domain);
  auto *words=ConstCwPtr(cws,kRows*kWords);
  CheckBlocks(cs,kRows*4);
  CheckBlocks(pi,4);
  if (ys.is_cpu()) {
    CheckCpuInputsInDomain<InType>(xs, kInBits);
    const auto seed_view=cuda::std::span<const int4,kRows>(ConstBlocks(seeds),kRows);
    const auto corrections=cuda::std::span<const Cw>(words,size_t(kRows)*kWords);
    const auto checks=cuda::std::span<const int4>(ConstBlocks(cs),size_t(kRows)*4);
    if (merkle) state.Cpu().BatchEvalMerkle(b,*ConstBlocks(sigma),m_rt,domain,
        seed_view,corrections,checks,
        {ConstInPtr(xs),size_t(count)},{Blocks(ys),size_t(count)},cuda::std::span<int4,4>(Blocks(pi),4));
    else state.Cpu().BatchEval(b,*ConstBlocks(sigma),m_rt,domain,
        seed_view,corrections,checks,
        {ConstInPtr(xs),size_t(count)},{Blocks(ys),size_t(count)},cuda::std::span<int4,4>(Blocks(pi),4));
  } else if constexpr (kGpuSupported) {
    TORCH_CHECK(merkle,"chain batch evaluation is a CPU operation; use batch_eval_merkle for CUDA");
    c10::cuda::CUDAGuard guard(ys.device()); auto stream=CurrentStream();
    fss::gpu::VerifiableMultiComponents<S> key{FetchBlock(sigma),m_rt,domain,
        {ConstBlocks(seeds),kRows},{words,kRows*kWords},{ConstBlocks(cs),kRows*4}};
    CheckCuda(fss::gpu::BatchEvalMerkleGpu<S>(state.template Device<kGpuSupported, S>(stream),b,key,
        {ConstInPtr(xs),size_t(count)},{Blocks(ys),size_t(count)},
        cuda::std::span<int4,4>(Blocks(pi),4),stream));
    state.RecordUse(stream); RecordTensors({sigma,seeds,cws,cs, xs,ys,pi},stream);
  } else TORCH_CHECK(false,"configured backends do not support CUDA evaluation");
}

bool Verify(NativeState &,Tensor p0,Tensor p1) {
  CheckSameDevice({p0,p1}); CheckBlocks(p0,4); CheckBlocks(p1,4);
  Proof left,right;
  if (p0.is_cpu()) for (int i=0;i<4;++i) { left[i]=ConstBlocks(p0)[i]; right[i]=ConstBlocks(p1)[i]; }
  else {
    c10::cuda::CUDAGuard guard(p0.device()); auto stream=CurrentStream();
    CheckCuda(cudaMemcpyAsync(left.data(),ConstBlocks(p0),sizeof(Proof),cudaMemcpyDeviceToHost,stream));
    CheckCuda(cudaMemcpyAsync(right.data(),ConstBlocks(p1),sizeof(Proof),cudaMemcpyDeviceToHost,stream));
    CheckCuda(cudaStreamSynchronize(stream));
  }
  return SchemeType::Verify(cuda::std::span<const int4,4>(left),cuda::std::span<const int4,4>(right));
}

template <typename S = SchemeType>
void Full(NativeState &state, bool b, Tensor sigma, int m_rt, py::int_ size,
    Tensor seeds, Tensor cws, Tensor cs, Tensor ys, Tensor pi, bool merkle) {
  CheckSameDevice({sigma, seeds, cws, cs, ys, pi});
  if constexpr (fss::detail::FitsFullDomain<int4, kInBits>()) {
    const int64_t count = AddressableCount(kInBits);
    CheckBlocks(sigma, 1);
    CheckBlocks(seeds, kRows);
    CheckBlocks(cs, kRows * 4);
    CheckBlocks(ys, count);
    CheckBlocks(pi, 4);
    const __uint128_t domain = ReadU128(size);
    CheckRuntime(m_rt, domain);
    if (ys.is_cuda()) {
      TORCH_CHECK(merkle, "chain full evaluation is a CPU operation; use eval_all_merkle for CUDA");
      if constexpr (kGpuSupported) {
        const auto *words = ConstCwPtr(cws, kRows * kWords);
        c10::cuda::CUDAGuard guard(ys.device());
        auto stream = CurrentStream();
        auto &core = static_cast<S &>(state.template DeviceForLaunch<kGpuSupported>(stream));
        CheckCuda(core.template EvalAllMerkleGpu<kThreadsPerBlock>(b,
            FetchBlock(sigma), m_rt, domain, {ConstBlocks(seeds), kRows},
            {words, kRows * kWords}, {ConstBlocks(cs), kRows * 4},
            {Blocks(ys), size_t(count)}, cuda::std::span<int4, 4>(Blocks(pi), 4), stream));
        state.RecordUse(stream);
        RecordTensors({sigma, seeds, cws, cs, ys, pi}, stream);
        return;
      } else throw py::type_error("configured backends do not support CUDA evaluation");
    }
    using Record = std::pair<__uint128_t, size_t>;
    const __uint128_t total_records = S::kDomainSize * 3;
    const __uint128_t max_bucket_records = domain < total_records ? domain : total_records;
    TORCH_CHECK(max_bucket_records <= std::numeric_limits<size_t>::max() / sizeof(Record),
        "full-domain bucket record storage size overflows");
    if (merkle) {
      using Node = cuda::std::array<int4, 2>;
      TORCH_CHECK(max_bucket_records <= std::numeric_limits<size_t>::max() / sizeof(Node),
          "full-domain bucket proof storage size overflows");
    }

    const auto *words = ConstCwPtr(cws, kRows * kWords);
    const auto seed_view = cuda::std::span<const int4, kRows>(ConstBlocks(seeds), kRows);
    const auto correction_view = cuda::std::span<const Cw>(words, size_t(kRows) * kWords);
    const auto check_view = cuda::std::span<const int4>(ConstBlocks(cs), size_t(kRows) * 4);
    const auto output = cuda::std::span<int4>(Blocks(ys), static_cast<size_t>(count));
    const auto proof = cuda::std::span<int4, 4>(Blocks(pi), 4);
    const int4 permutation_seed = *ConstBlocks(sigma);
    if (merkle) {
      static_cast<S &>(state.Cpu()).EvalAllMerkle(b, permutation_seed, m_rt, domain,
          seed_view, correction_view, check_view, output, proof);
    } else {
      static_cast<S &>(state.Cpu()).EvalAll(b, permutation_seed, m_rt, domain,
          seed_view, correction_view, check_view, output, proof);
    }
  } else {
    TORCH_CHECK(false, "full-domain output count and byte size are not addressable");
  }
}
}  // namespace
}  // namespace fss::python

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  using namespace fss::python;
  auto cls=BindNative(m);
  cls.def("gen",[](NativeState &s,Tensor c,Tensor cs, Tensor sigma,Tensor seeds,Tensor as,Tensor bs,int t,int e) { return Gen(s,c,cs, sigma,seeds,as,bs,t,e); });
  cls.def("batch_eval",[](NativeState &s,bool b,Tensor sigma,int m,py::int_ size,Tensor seeds,Tensor c,Tensor cs,Tensor xs,Tensor ys,Tensor pi) { Batch(s,b,sigma,m,size,seeds,c,cs,xs,ys,pi,false); });
  cls.def("batch_eval_merkle",[](NativeState &s,bool b,Tensor sigma,int m,py::int_ size,Tensor seeds,Tensor c,Tensor cs,Tensor xs,Tensor ys,Tensor pi) { Batch(s,b,sigma,m,size,seeds,c,cs,xs,ys,pi,true); });
  cls.def("eval_all", [](NativeState &s, bool b, Tensor sigma, int m, py::int_ size,
      Tensor seeds, Tensor c, Tensor cs, Tensor ys, Tensor pi) {
    Full(s, b, sigma, m, size, seeds, c, cs, ys, pi, false);
  });
  cls.def("eval_all_merkle", [](NativeState &s, bool b, Tensor sigma, int m, py::int_ size,
      Tensor seeds, Tensor c, Tensor cs, Tensor ys, Tensor pi) {
    Full(s, b, sigma, m, size, seeds, c, cs, ys, pi, true);
  });
  cls.def("verify",&Verify);
}

// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda/std/array>
#include <cuda/std/span>
#include <cuda/std/tuple>
#include <array>
#include <bit>
#include <climits>
#include <cstdint>
#include <string_view>
#include <vector>
#include <fss/prg/chacha.cuh>
#include <fss/hash/blake3.cuh>
#include <fss/dpf.cuh>
#include <fss/group/bytes.cuh>
#include <omp.h>

namespace {

using Words = std::array<uint32_t, 16>;

void ExpectBlock(int4 actual, int4 expected) {
  EXPECT_EQ(actual.x, expected.x);
  EXPECT_EQ(actual.y, expected.y);
  EXPECT_EQ(actual.z, expected.z);
  EXPECT_EQ(actual.w, expected.w);
}

std::array<uint32_t, 4> Unpack(int4 value) {
  return {static_cast<uint32_t>(value.x), static_cast<uint32_t>(value.y),
      static_cast<uint32_t>(value.z), static_cast<uint32_t>(value.w)};
}

int4 Pack(const Words &words, int offset) {
  return {static_cast<int>(words[offset]), static_cast<int>(words[offset + 1]),
      static_cast<int>(words[offset + 2]), static_cast<int>(words[offset + 3])};
}

void ReferenceQuarter(Words &s, int a, int b, int c, int d) {
  s[a] += s[b];
  s[d] = std::rotl(s[d] ^ s[a], 16);
  s[c] += s[d];
  s[b] = std::rotl(s[b] ^ s[c], 12);
  s[a] += s[b];
  s[d] = std::rotl(s[d] ^ s[a], 8);
  s[c] += s[d];
  s[b] = std::rotl(s[b] ^ s[c], 7);
}

// Independent unsigned word-array oracle for this library's XOR feed-forward
// PRG. The multiplier selects the documented 128-bit or repeated 256-bit key.
template <int mul, int rounds>
std::array<int4, mul> ReferenceChaCha(int4 seed, int nonce0, int nonce1) {
  Words initial = {0x61707865u, mul <= 2 ? 0x3120646eu : 0x3320646eu,
      mul <= 2 ? 0x79622d36u : 0x79622d32u, 0x6b206574u};
  auto key = Unpack(seed);
  for (int i = 0; i < 4; ++i) initial[4 + i] = initial[8 + i] = key[i];
  initial[14] = static_cast<uint32_t>(nonce0);
  initial[15] = static_cast<uint32_t>(nonce1);
  Words state = initial;
  for (int round = 0; round < rounds; round += 2) {
    for (int i = 0; i < 4; ++i) ReferenceQuarter(state, i, i + 4, i + 8, i + 12);
    for (int i = 0; i < 4; ++i) ReferenceQuarter(state, i, 4 + (i + 1) % 4, 8 + (i + 2) % 4, 12 + (i + 3) % 4);
  }
  for (int i = 0; i < 16; ++i) state[i] ^= initial[i];
  std::array<int4, mul> result;
  for (int i = 0; i < mul; ++i) result[i] = Pack(state, mul == 1 ? 4 : 4 * i);
  return result;
}

void ReferenceG(Words &state, int a, int b, int c, int d, uint32_t x, uint32_t y) {
  state[a] += state[b] + x;
  state[d] = std::rotr(state[d] ^ state[a], 16);
  state[c] += state[d];
  state[b] = std::rotr(state[b] ^ state[c], 12);
  state[a] += state[b] + y;
  state[d] = std::rotr(state[d] ^ state[a], 8);
  state[c] += state[d];
  state[b] = std::rotr(state[b] ^ state[c], 7);
}

std::array<int4, 2> ReferenceBlake3(const int4 iv[2], const int4 message[4], int bytes) {
  Words state;
  for (int block = 0; block < 2; ++block) {
    auto words = Unpack(iv[block]);
    for (int i = 0; i < 4; ++i) state[4 * block + i] = words[i];
  }
  state[8] = 0x6a09e667u;
  state[9] = 0xbb67ae85u;
  state[10] = 0x3c6ef372u;
  state[11] = 0xa54ff53au;
  state[12] = state[13] = 0;
  state[14] = static_cast<uint32_t>(bytes);
  state[15] = 27;  // CHUNK_START | CHUNK_END | ROOT | KEYED_HASH.
  Words msg;
  for (int block = 0; block < 4; ++block) {
    auto words = Unpack(message[block]);
    for (int i = 0; i < 4; ++i) msg[4 * block + i] = words[i];
  }
  constexpr int permutation[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
  for (int round = 0; round < 7; ++round) {
    for (int i = 0; i < 4; ++i) ReferenceG(state, i, i + 4, i + 8, i + 12, msg[2 * i], msg[2 * i + 1]);
    for (int i = 0; i < 4; ++i) {
      ReferenceG(state, i, 4 + (i + 1) % 4, 8 + (i + 2) % 4, 12 + (i + 3) % 4,
          msg[8 + 2 * i], msg[9 + 2 * i]);
    }
    Words next;
    for (int i = 0; i < 16; ++i) next[i] = msg[permutation[i]];
    msg = next;
  }
  for (int i = 0; i < 8; ++i) state[i] ^= state[i + 8];
  return {Pack(state, 0), Pack(state, 4)};
}

struct ChaChaInput {
  int4 seed;
  int nonce0;
  int nonce1;
};

std::vector<ChaChaInput> ChaChaCases() {
  std::vector<ChaChaInput> result = {
      {{0, 0, 0, 0}, 0, 0}, {{INT_MAX, INT_MIN, -1, 1}, INT_MIN, INT_MAX},
      {{-1, -1, -1, -1}, -1, -1}, {{967459772, 685770617, 5688182, 142619765}, 1, 2},
      {{1958864317, -1157351853, 618433185, -451532952}, 1, 2}};
  uint32_t state = 0x9e3779b9u;
  auto next = [&] {
    state = state * 1664525u + 1013904223u;
    return static_cast<int>(state);
  };
  for (int i = 0; i < 128; ++i) result.push_back({{next(), next(), next(), next()}, next(), next()});
  return result;
}

struct BlakeInput {
  int4 iv[2];
  int4 message[4];
};

BlakeInput OfficialBlakeVector() {
  BlakeInput input{};
  // BLAKE3 1.8.2 official keyed_hash vector with input_len=64.
  // https://github.com/BLAKE3-team/BLAKE3/blob/1.8.2/test_vectors/test_vectors.json
  constexpr std::string_view key = "whats the Elvish word for friend";
  Words iv_words{};
  Words msg_words{};
  for (int i = 0; i < 32; ++i) iv_words[i / 4] |= static_cast<uint32_t>(key[i]) << (8 * (i % 4));
  for (int i = 0; i < 64; ++i) msg_words[i / 4] |= static_cast<uint32_t>(i) << (8 * (i % 4));
  input.iv[0] = Pack(iv_words, 0);
  input.iv[1] = Pack(iv_words, 4);
  for (int i = 0; i < 4; ++i) input.message[i] = Pack(msg_words, 4 * i);
  return input;
}

constexpr uint32_t kOfficialBlakeDigest[8] = {
    0x36ed8cbau, 0x0d7027f3u, 0x0b123f21u, 0x3b7a201au,
    0x0533048cu, 0x416f5828u, 0xf7f2094du, 0xe6b7ccd9u};

void ExpectOfficialBlakeDigest(const int4 actual[2]) {
  for (int block = 0; block < 2; ++block) {
    auto words = Unpack(actual[block]);
    for (int i = 0; i < 4; ++i) EXPECT_EQ(words[i], kOfficialBlakeDigest[4 * block + i]);
  }
}

template <int mul, int rounds>
void CheckChaChaHost() {
  for (const auto &input : ChaChaCases()) {
    auto expected = ReferenceChaCha<mul, rounds>(input.seed, input.nonce0, input.nonce1);
    fss::prg::ChaCha<mul, rounds> prg(input.nonce0, input.nonce1);
    auto actual = prg.Gen(input.seed);
    for (int i = 0; i < mul; ++i) ExpectBlock(actual[i], expected[i]);
  }
}

#ifdef __CUDACC__

template <int mul, int rounds>
__global__ void ChaChaKernel(const ChaChaInput *inputs, int4 *outputs, int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  ChaChaInput input = inputs[index];
  fss::prg::ChaCha<mul, rounds> prg(input.nonce0, input.nonce1);
  auto result = prg.Gen(input.seed);
  for (int i = 0; i < mul; ++i) outputs[index * mul + i] = result[i];
}

__global__ void BlakeKernel(const BlakeInput *inputs, int4 *outputs, int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  BlakeInput input = inputs[index];
  fss::hash::Blake3 hash(cuda::std::span<const int4, 2>(input.iv, 2));
  auto full = hash.Hash(cuda::std::span<const int4, 4>(input.message, 4));
  auto tuple = hash.Hash(cuda::std::tuple<int4, const int4>{input.message[0], input.message[1]});
  outputs[6 * index] = full[0];
  outputs[6 * index + 1] = full[1];
  for (int i = 0; i < 4; ++i) outputs[6 * index + 2 + i] = tuple[i];
}

bool HasCudaDevice() {
  int count = 0;
  cudaError_t status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess) cudaGetLastError();
  return status == cudaSuccess && count > 0;
}

template <int mul, int rounds>
void CheckChaChaDevice() {
  auto inputs = ChaChaCases();
  void *input_storage = nullptr;
  void *output_storage = nullptr;
  ASSERT_EQ(cudaMalloc(&input_storage, inputs.size() * sizeof(ChaChaInput)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&output_storage, inputs.size() * mul * sizeof(int4)), cudaSuccess);
  auto *device_inputs = static_cast<ChaChaInput *>(input_storage);
  auto *device_outputs = static_cast<int4 *>(output_storage);
  ASSERT_EQ(cudaMemcpy(device_inputs, inputs.data(), inputs.size() * sizeof(ChaChaInput), cudaMemcpyHostToDevice), cudaSuccess);
  ChaChaKernel<mul, rounds><<<(inputs.size() + 127) / 128, 128>>>(device_inputs, device_outputs, inputs.size());
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  std::vector<int4> actual(inputs.size() * mul);
  ASSERT_EQ(cudaMemcpy(actual.data(), device_outputs, actual.size() * sizeof(int4), cudaMemcpyDeviceToHost), cudaSuccess);
  for (size_t index = 0; index < inputs.size(); ++index) {
    auto expected = ReferenceChaCha<mul, rounds>(inputs[index].seed, inputs[index].nonce0, inputs[index].nonce1);
    for (int i = 0; i < mul; ++i) ExpectBlock(actual[index * mul + i], expected[i]);
  }
  EXPECT_EQ(cudaFree(device_outputs), cudaSuccess);
  EXPECT_EQ(cudaFree(device_inputs), cudaSuccess);
}

#endif

}  // namespace

TEST(WordArithmetic, ChaChaMatchesUnsignedReference) {
  CheckChaChaHost<1, 8>();
  CheckChaChaHost<2, 8>();
  CheckChaChaHost<4, 8>();
  CheckChaChaHost<1, 12>();
  CheckChaChaHost<2, 12>();
  CheckChaChaHost<4, 12>();
  CheckChaChaHost<1, 20>();
  CheckChaChaHost<2, 20>();
  CheckChaChaHost<4, 20>();
}

TEST(WordArithmetic, Blake3MatchesOfficialKeyedVector) {
  BlakeInput input = OfficialBlakeVector();
  fss::hash::Blake3 hash(input.iv);
  auto actual = hash.Hash(cuda::std::span<const int4, 4>(input.message));
  ExpectOfficialBlakeDigest(actual.data());
  auto reference = ReferenceBlake3(input.iv, input.message, 64);
  ExpectOfficialBlakeDigest(reference.data());
}

TEST(WordArithmetic, Blake3HighWordsAndTupleMatchUnsignedReference) {
  for (const auto &test_case : ChaChaCases()) {
    int4 iv[2] = {test_case.seed, {INT_MAX, INT_MIN, -1, 1}};
    int4 message[4] = {test_case.seed, {test_case.nonce0, test_case.nonce1, INT_MAX, INT_MIN},
        {INT_MAX, INT_MAX, -1, 1}, {INT_MIN, -1, 0, INT_MAX}};
    fss::hash::Blake3 hash(iv);
    auto actual = hash.Hash(cuda::std::span<const int4, 4>(message));
    auto reference = ReferenceBlake3(iv, message, 64);
    for (int i = 0; i < 2; ++i) ExpectBlock(actual[i], reference[i]);
    auto tuple = hash.Hash(cuda::std::tuple<int4, const int4>{message[0], message[1]});
    int4 padded[4] = {message[0], message[1], {0, 0, 0, 0}, {0, 0, 0, 0}};
    for (int bit = 0; bit < 2; ++bit) {
      padded[0].w = (padded[0].w & ~1) | bit;
      auto expected = ReferenceBlake3(iv, padded, 32);
      for (int i = 0; i < 2; ++i) ExpectBlock(tuple[2 * bit + i], expected[i]);
    }
  }
}

TEST(WordArithmetic, ParallelChaChaDpfFullDomainMatchesPoint) {
  omp_set_num_threads(4);
  using Scheme = fss::Dpf<8, fss::group::Bytes, fss::prg::ChaCha<2>, uint32_t, 2>;
  Scheme scheme{fss::prg::ChaCha<2>{1, 2}};
  int4 seeds[2] = {{967459772, 685770617, 5688182, 142619765},
      {1958864317, -1157351853, 618433185, -451532952}};
  Scheme::Key k0, k1;
  scheme.Gen(k0, k1, seeds, 7, {604, 0, 0, 0});
  std::vector<int4> full(256);
  for (bool party : {false, true}) {
    const auto &key = party ? k1 : k0;
    scheme.EvalAll(party, key, full);
    for (int x = 0; x < 256; ++x) ExpectBlock(full[x], scheme.Eval(party, key, x));
  }
}

TEST(WordArithmetic, ChaChaGpuMatchesUnsignedReference) {
#ifdef __CUDACC__
  if (!HasCudaDevice()) GTEST_SKIP() << "no CUDA device";
  CheckChaChaDevice<1, 8>();
  CheckChaChaDevice<2, 8>();
  CheckChaChaDevice<4, 8>();
  CheckChaChaDevice<1, 12>();
  CheckChaChaDevice<2, 12>();
  CheckChaChaDevice<4, 12>();
  CheckChaChaDevice<1, 20>();
  CheckChaChaDevice<2, 20>();
  CheckChaChaDevice<4, 20>();
#else
  GTEST_SKIP() << "CUDA compiler required";
#endif
}

TEST(WordArithmetic, Blake3GpuMatchesOfficialVectorAndUnsignedTupleReference) {
#ifdef __CUDACC__
  if (!HasCudaDevice()) GTEST_SKIP() << "no CUDA device";
  std::vector<BlakeInput> inputs = {OfficialBlakeVector()};
  for (const auto &test_case : ChaChaCases()) {
    BlakeInput input = {{test_case.seed, {INT_MAX, INT_MIN, -1, 1}},
        {test_case.seed, {test_case.nonce0, test_case.nonce1, INT_MAX, INT_MIN},
            {INT_MAX, INT_MAX, -1, 1}, {INT_MIN, -1, 0, INT_MAX}}};
    inputs.push_back(input);
  }
  void *input_storage = nullptr;
  void *output_storage = nullptr;
  ASSERT_EQ(cudaMalloc(&input_storage, inputs.size() * sizeof(BlakeInput)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&output_storage, inputs.size() * 6 * sizeof(int4)), cudaSuccess);
  auto *device_inputs = static_cast<BlakeInput *>(input_storage);
  auto *device_outputs = static_cast<int4 *>(output_storage);
  ASSERT_EQ(cudaMemcpy(device_inputs, inputs.data(), inputs.size() * sizeof(BlakeInput), cudaMemcpyHostToDevice), cudaSuccess);
  BlakeKernel<<<(inputs.size() + 127) / 128, 128>>>(device_inputs, device_outputs, inputs.size());
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  std::vector<int4> actual(inputs.size() * 6);
  ASSERT_EQ(cudaMemcpy(actual.data(), device_outputs, actual.size() * sizeof(int4), cudaMemcpyDeviceToHost), cudaSuccess);
  ExpectOfficialBlakeDigest(actual.data());
  for (size_t index = 0; index < inputs.size(); ++index) {
    auto expected = ReferenceBlake3(inputs[index].iv, inputs[index].message, 64);
    for (int i = 0; i < 2; ++i) ExpectBlock(actual[6 * index + i], expected[i]);
    int4 padded[4] = {inputs[index].message[0], inputs[index].message[1], {0, 0, 0, 0}, {0, 0, 0, 0}};
    for (int bit = 0; bit < 2; ++bit) {
      padded[0].w = (padded[0].w & ~1) | bit;
      auto tuple = ReferenceBlake3(inputs[index].iv, padded, 32);
      for (int i = 0; i < 2; ++i) ExpectBlock(actual[6 * index + 2 + 2 * bit + i], tuple[i]);
    }
  }
  EXPECT_EQ(cudaFree(device_outputs), cudaSuccess);
  EXPECT_EQ(cudaFree(device_inputs), cudaSuccess);
#else
  GTEST_SKIP() << "CUDA compiler required";
#endif
}

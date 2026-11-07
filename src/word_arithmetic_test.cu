// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda/std/array>
#include <cuda/std/span>
#include <cuda/std/tuple>
#include <array>
#include <bit>
#include <climits>
#include <cstdio>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <fss/prg/chacha.cuh>
#include <fss/hash/blake3.cuh>
#include <fss/hash/sha256.cuh>
#include <fss/dpf.cuh>
#include <fss/group/bytes.cuh>
#include <omp.h>
#include "core_api_test_helpers.cuh"

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

// Declaration-only shape used to check rejection of a wrong-sized Hash4.
struct WrongHash4Shape {
  cuda::std::array<int4, 2> Hash(cuda::std::span<const int4, 4>);
  cuda::std::array<cuda::std::array<int4, 2>, 3> Hash4(
      cuda::std::span<const cuda::std::array<int4, 4>, 4>);
};

static_assert(Hashable<WrongHash4Shape>);
static_assert(!HasHash4<WrongHash4Shape>);
static_assert(HasHash4<fss::hash::Blake3>);

// Golden fixtures for the tuple hash message domain: the first tuple element is
// a full 128-bit coordinate, the second is a seed whose lowest bit of w is a
// reserved tag. S_t = (seed.x, seed.y, seed.z, (seed.w & 0xfffffffe) | t) and
// M_t = W(coordinate) || W(S_t), where W writes the words x, y, z, w in this
// order, each little-endian. bit96-lo/hi hold the integers 0 and 2^96.
//
// The digests were generated independently with blake3 1.0.11 and hashlib
// (Python 3.12.3, OpenSSL 3.0.13), without importing fss code. BLAKE3 golden = keyed(W(iv), M_0) || keyed(W(iv), M_1)
// with 32B messages; Sha256 golden = SHA256(W(key) || M_0) || SHA256(W(key) || M_1)
// with 48B messages. The same package reproduced the official BLAKE3 1.8.2
// keyed vector with input_len=64 (kOfficialBlakeDigest) before digesting.
struct TupleGoldenCase {
  int4 coordinate;
  int4 seed;
  uint32_t blake3_golden[16];
  uint32_t sha256_golden[16];
};

const int4 kTupleGoldenIv[2] = {
    {static_cast<int>(0x6A09E667u), static_cast<int>(0xBB67AE85u), static_cast<int>(0x3C6EF372u), static_cast<int>(0xA54FF53Au)},
    {static_cast<int>(0x510E527Fu), static_cast<int>(0x9B05688Cu), static_cast<int>(0x1F83D9ABu), static_cast<int>(0x5BE0CD19u)}};
const int4 kTupleGoldenShaKey = {static_cast<int>(0xDEADBEEFu), static_cast<int>(0xFEEDFACEu), static_cast<int>(0x0BADCAFEu), static_cast<int>(0x0C0FFEE1u)};
const TupleGoldenCase kTupleGoldenCases[] = {
    // asym-a
    {{static_cast<int>(0x243F6A88u), static_cast<int>(0x85A308D3u), static_cast<int>(0x13198A2Eu), static_cast<int>(0x03707345u)}, {static_cast<int>(0xA4093822u), static_cast<int>(0x299F31D0u), static_cast<int>(0x082EFA98u), static_cast<int>(0xEC4E6C44u)},
        {0x8A6822E2u, 0x535A2295u, 0x131D17B4u, 0x178D2BF0u, 0x346F943Bu, 0xFB35ECF2u, 0xAF51FA37u, 0x6743472Au, 0x05994585u, 0xB35DC332u, 0xAF7690DEu, 0xCE5E851Fu, 0x1BC3B921u, 0x00D0496Du, 0x586188F7u, 0x4BBE9AA3u},
        {0xCF7B5054u, 0x585B130Du, 0x9731C753u, 0x29CF96CBu, 0x4C8D7D09u, 0xB900D9DBu, 0x983688EBu, 0xEF75DCC6u, 0xD955A769u, 0xEB6A8789u, 0x3C29FED4u, 0xA45E9D37u, 0xC3AAA4B4u, 0x493742C9u, 0x2ED1EB54u, 0x6D5EAD6Bu}},
    // asym-b
    {{static_cast<int>(0x452821E6u), static_cast<int>(0x38D01377u), static_cast<int>(0xBE5466CFu), static_cast<int>(0x34E90C6Cu)}, {static_cast<int>(0xC0AC29B7u), static_cast<int>(0xC97C50DDu), static_cast<int>(0x3F84D5B5u), static_cast<int>(0xB5470917u)},
        {0x498A2AF1u, 0xF5D83BA7u, 0x880EA565u, 0x5F522283u, 0x9A9B206Au, 0x55682F30u, 0x0C04CE10u, 0xBB68FA0Au, 0x515743D8u, 0x0EA88C6Eu, 0xFA7E0C7Cu, 0x6C429E6Cu, 0xD80EAE88u, 0x64B8E63Bu, 0x7E10FF43u, 0x8121AF53u},
        {0x821597A8u, 0x34A3C06Cu, 0x4D413DC0u, 0xDA4B076Au, 0x05A64F9Bu, 0x3CAD2BCCu, 0x88430F32u, 0x26ED4F8Du, 0xB25EF07Au, 0xA337451Du, 0x35F1AD17u, 0x311C032Au, 0x39EA07E8u, 0x49EB3D32u, 0x5E924A3Cu, 0x50CFF815u}},
    // bit96-lo
    {{static_cast<int>(0x00000000u), static_cast<int>(0x00000000u), static_cast<int>(0x00000000u), static_cast<int>(0x00000000u)}, {static_cast<int>(0x1F2E3D4Cu), static_cast<int>(0x5B6A7988u), static_cast<int>(0x99AABBCCu), static_cast<int>(0xD0E0F010u)},
        {0x4A6B5FC5u, 0x652C4ADBu, 0x1DA3429Du, 0x4250CB63u, 0xAA4FA6D6u, 0xC0E1C41Au, 0xEDEA7527u, 0xC11FE225u, 0x73E95ADCu, 0xA7422F2Au, 0x2C178C3Bu, 0xAC9828E1u, 0xB670944Du, 0x784B3A2Eu, 0xC9570776u, 0xE062CA60u},
        {0xE4EF1ECDu, 0xBEFE6E53u, 0x95BCC75Au, 0xFAF0069Bu, 0xCEEDAA22u, 0x7E82B3ACu, 0xE95D522Fu, 0xA8D66F51u, 0xF78D3615u, 0x06543C76u, 0x4C215F14u, 0x89A746BFu, 0xF217499Au, 0x7DFB1809u, 0x73BA9D37u, 0xD6460FBBu}},
    // bit96-hi
    {{static_cast<int>(0x00000000u), static_cast<int>(0x00000000u), static_cast<int>(0x00000000u), static_cast<int>(0x00000001u)}, {static_cast<int>(0x1F2E3D4Cu), static_cast<int>(0x5B6A7988u), static_cast<int>(0x99AABBCCu), static_cast<int>(0xD0E0F010u)},
        {0x3F7C0C7Au, 0xBE774891u, 0xD5554624u, 0xD6D5C5A9u, 0x6EFA4555u, 0xF49C9CFCu, 0x54FA07C7u, 0xDA91003Eu, 0xBB205DEEu, 0x3B36E4AAu, 0x059A98B3u, 0x974C1784u, 0x7215E6EEu, 0x1749E516u, 0x8E764D66u, 0x163614EEu},
        {0x7B779B0Bu, 0xDEB202C9u, 0x11D36DA1u, 0x8AE13496u, 0x56D6C8A8u, 0xC39F8915u, 0x8F298B9Du, 0xE92DAB9Fu, 0xB2CEFBA9u, 0xB2E5B7B3u, 0xAFB2CAAAu, 0xBC5346C8u, 0x63F2B0B5u, 0x6F3672F6u, 0x1B24C675u, 0x3D1EA17Eu}},
};

bool SameTupleWords(const int4 a[4], const int4 b[4]) {
  for (int block = 0; block < 4; ++block) {
    auto wa = Unpack(a[block]);
    auto wb = Unpack(b[block]);
    for (int i = 0; i < 4; ++i)
      if (wa[i] != wb[i]) return false;
  }
  return true;
}

std::string HexTupleWords(const int4 blocks[4]) {
  std::string out;
  char buf[16];
  for (int block = 0; block < 4; ++block) {
    auto words = Unpack(blocks[block]);
    for (int i = 0; i < 4; ++i) {
      std::snprintf(buf, sizeof(buf), "%08x ", words[i]);
      out += buf;
    }
  }
  return out;
}

void ExpectTupleGolden(const char *backend, const int4 actual[4], const uint32_t golden[16]) {
  for (int block = 0; block < 4; ++block) {
    auto words = Unpack(actual[block]);
    for (int i = 0; i < 4; ++i)
      EXPECT_EQ(words[i], golden[4 * block + i]) << backend << " digest word " << 4 * block + i;
  }
}

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
      padded[1].w = (padded[1].w & ~1) | bit;
      auto expected = ReferenceBlake3(iv, padded, 32);
      for (int i = 0; i < 2; ++i) ExpectBlock(tuple[2 * bit + i], expected[i]);
    }
  }
}

TEST(WordArithmetic, Blake3Hash4MatchesScalarWithCustomIvs) {
  const int4 ivs[2][2] = {
      {{1, 2, 3, 4}, {5, 6, 7, 8}},
      {{INT_MIN, INT_MAX, -1, 0}, {static_cast<int>(0x80000000u), 0x7fffffff, 0x55555555,
          static_cast<int>(0xaaaaaaaau)}}};
  const cuda::std::array<cuda::std::array<int4, 4>, 4> messages = {{
      {{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}},
      {{{-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}}},
      {{{INT_MIN, INT_MAX, -1, 1}, {INT_MAX, INT_MIN, 0, -1},
          {-1, 1, INT_MIN, INT_MAX}, {0, -1, INT_MAX, INT_MIN}}},
      {{{static_cast<int>(0x80000000u), static_cast<int>(0x7fffffffu),
            static_cast<int>(0xaaaaaaaau), static_cast<int>(0x55555555u)},
          {static_cast<int>(0xdeadbeefu), static_cast<int>(0x80000001u), 1, 0},
          {-1, 0, static_cast<int>(0x80000000u), static_cast<int>(0x7fffffffu)},
          {static_cast<int>(0xfffffff0u), 15, -15, static_cast<int>(0xffffff00u)}}},
  }};
  for (const auto &iv : ivs) {
    fss::hash::Blake3 hash(cuda::std::span<const int4, 2>(iv, 2));
    cuda::std::array<cuda::std::array<int4, 4>, 4> input = messages;
    cuda::std::array<cuda::std::array<int4, 2>, 4> expected;
    for (int i = 0; i < 4; ++i)
      expected[i] = hash.Hash(cuda::std::span<const int4, 4>(input[i].data(), 4));

    auto actual = hash.Hash4(
        cuda::std::span<const cuda::std::array<int4, 4>, 4>(input.data(), 4));
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 2; ++j) ExpectBlock(actual[i][j], expected[i][j]);

    for (int i = 0; i < 4; ++i) {
      auto scalar = hash.Hash(cuda::std::span<const int4, 4>(input[i].data(), 4));
      for (int j = 0; j < 2; ++j) ExpectBlock(scalar[j], expected[i][j]);
    }

    auto retained = actual;
    input[0][0] = {-1, 0, 1, INT_MIN};
    input[3][2].w = INT_MAX;
    [[maybe_unused]] auto mutated = hash.Hash4(
        cuda::std::span<const cuda::std::array<int4, 4>, 4>(input.data(), 4));
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 2; ++j) ExpectBlock(retained[i][j], expected[i][j]);
  }
}

TEST(WordArithmetic, ParallelChaChaDpfFullDomainMatchesPoint) {
  omp_set_num_threads(4);
  using Scheme = fss::Dpf<8, fss::group::Bytes, fss::prg::ChaCha<2>, uint32_t>;
  Scheme scheme{fss::prg::ChaCha<2>{1, 2}};
  int4 seeds[2] = {{967459772, 685770617, 5688182, 142619765},
      {1958864317, -1157351853, 618433185, -451532952}};
  fss_test::ComponentKey<Scheme> k0, k1;
  fss_test::GenPair(scheme, k0, k1, seeds, 7, int4{604, 0, 0, 0});
  std::vector<int4> full(256);
  for (bool party : {false, true}) {
    const auto &key = party ? k1 : k0;
    fss_test::EvalAllKey<2>(scheme, party, key, full);
    for (int x = 0; x < 256; ++x) ExpectBlock(full[x], fss_test::EvalKey(scheme, party, key, x));
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
      padded[1].w = (padded[1].w & ~1) | bit;
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

TEST(WordArithmetic, Blake3TupleMatchesCoordinateSeedGolden) {
  fss::hash::Blake3 hash(cuda::std::span<const int4, 2>(kTupleGoldenIv, 2));
  for (const auto &test_case : kTupleGoldenCases) {
    auto actual = hash.Hash(cuda::std::tuple<int4, const int4>{test_case.coordinate, test_case.seed});
    ExpectTupleGolden("Blake3", actual.data(), test_case.blake3_golden);
  }
}

TEST(WordArithmetic, Sha256TupleMatchesCoordinateSeedGolden) {
  fss::hash::Sha256 hash(kTupleGoldenShaKey);
  for (const auto &test_case : kTupleGoldenCases) {
    auto actual = hash.Hash(cuda::std::tuple<int4, const int4>{test_case.coordinate, test_case.seed});
    ExpectTupleGolden("Sha256", actual.data(), test_case.sha256_golden);
  }
}

TEST(WordArithmetic, TupleHashBindsCoordinateBit96) {
  const int4 coordinate_lo = {0, 0, 0, 0};
  const int4 coordinate_hi = {0, 0, 0, 1};
  fss::hash::Blake3 blake3(cuda::std::span<const int4, 2>(kTupleGoldenIv, 2));
  auto blake3_lo = blake3.Hash(cuda::std::tuple<int4, const int4>{coordinate_lo, kTupleGoldenCases[2].seed});
  auto blake3_hi = blake3.Hash(cuda::std::tuple<int4, const int4>{coordinate_hi, kTupleGoldenCases[2].seed});
  EXPECT_FALSE(SameTupleWords(blake3_lo.data(), blake3_hi.data()))
      << "Blake3: coordinate bit 96 changed (block w bit 0 of the first tuple element), digest unchanged\n"
      << "  digest coordinate=0:    " << HexTupleWords(blake3_lo.data()) << "\n"
      << "  digest coordinate=2^96: " << HexTupleWords(blake3_hi.data());
  fss::hash::Sha256 sha256(kTupleGoldenShaKey);
  auto sha_lo = sha256.Hash(cuda::std::tuple<int4, const int4>{coordinate_lo, kTupleGoldenCases[2].seed});
  auto sha_hi = sha256.Hash(cuda::std::tuple<int4, const int4>{coordinate_hi, kTupleGoldenCases[2].seed});
  EXPECT_FALSE(SameTupleWords(sha_lo.data(), sha_hi.data()))
      << "Sha256: coordinate bit 96 changed (block w bit 0 of the first tuple element), digest unchanged\n"
      << "  digest coordinate=0:    " << HexTupleWords(sha_lo.data()) << "\n"
      << "  digest coordinate=2^96: " << HexTupleWords(sha_hi.data());
  // The lowest bit of the seed's fourth word is the reserved tag, not seed
  // data, so toggling it must not change either digest.
  for (int i = 0; i < 2; ++i) {
    int4 flipped = kTupleGoldenCases[i].seed;
    flipped.w ^= 1;
    auto blake3_base = blake3.Hash(
        cuda::std::tuple<int4, const int4>{kTupleGoldenCases[i].coordinate, kTupleGoldenCases[i].seed});
    auto blake3_flipped = blake3.Hash(cuda::std::tuple<int4, const int4>{kTupleGoldenCases[i].coordinate, flipped});
    EXPECT_TRUE(SameTupleWords(blake3_base.data(), blake3_flipped.data()))
        << "Blake3: reserved seed bit (seed.w bit 0 of the second tuple element) changed the digest, case "
        << i << "\n  base:    " << HexTupleWords(blake3_base.data()) << "\n  flipped: "
        << HexTupleWords(blake3_flipped.data());
    auto sha_base = sha256.Hash(
        cuda::std::tuple<int4, const int4>{kTupleGoldenCases[i].coordinate, kTupleGoldenCases[i].seed});
    auto sha_flipped = sha256.Hash(cuda::std::tuple<int4, const int4>{kTupleGoldenCases[i].coordinate, flipped});
    EXPECT_TRUE(SameTupleWords(sha_base.data(), sha_flipped.data()))
        << "Sha256: reserved seed bit (seed.w bit 0 of the second tuple element) changed the digest, case "
        << i << "\n  base:    " << HexTupleWords(sha_base.data()) << "\n  flipped: "
        << HexTupleWords(sha_flipped.data());
  }
}

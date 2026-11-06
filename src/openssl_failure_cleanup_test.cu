// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <cuda/std/array>
#include <openssl/evp.h>
#include <bit>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <fss/prg/aes128_mmo.cuh>
#include <fss/prp/aes128_feistel.cuh>
#include <fss/hash/sha256.cuh>

namespace {

enum class UpdateFault {
  kFailNoWrite,
  kFailFullWrite,
  kZeroLength,
  kUnchangedLength,
  kPartialOne,
  kPartialFifteen,
  kFullWriteShortLength,
  kFullWriteUnchangedLength,
};

enum class FinalFault {
  kFailNoWrite,
  kFailFullWriteLength32,
  kFullWriteLength31,
  kFullWriteUnchangedLength,
  kPartial16Length16,
};

struct OpenSslFault {
  int fail_new_at = 0;
  int fail_init_at = 0;
  int fail_padding_at = 0;
  int update_fault_at = 0;
  UpdateFault update_fault = UpdateFault::kFailNoWrite;
  int new_calls = 0;
  int init_calls = 0;
  int padding_calls = 0;
  int update_calls = 0;
  int free_calls = 0;
  int invalid_frees = 0;
  int digest_fault_at = 0;
  int digest_calls = 0;
  bool digest_capture = false;
  bool digest_input_changed = false;
  size_t digest_length[2] = {};
  unsigned char digest_input[2][48] = {};
  std::set<EVP_CIPHER_CTX *> live;

  int mdnew_fault_at = 0;
  int mdinit_fault_at = 0;
  int mdfinal_fault_at = 0;
  FinalFault mdfinal_fault = FinalFault::kFailNoWrite;
  int mdnew_calls = 0;
  int mdreset_calls = 0;
  int mdinit_calls = 0;
  int mdfinal_calls = 0;
  int mdfree_calls = 0;
  int md_invalid_frees = 0;
  bool md_oneshot[2] = {};
  bool md_init_md_null[2] = {};
  bool md_init_type_null[2] = {};
  unsigned int md_final_entry_len[2] = {};
  std::set<EVP_MD_CTX *> live_md;
};

OpenSslFault *g_fault = nullptr;

bool FailAt(int &calls, int fail_at) {
  ++calls;
  return calls == fail_at;
}

}  // namespace

// The library is header-only, so linker wrapping gives deterministic OpenSSL
// failures without an injection hook in the public code.
extern "C" EVP_CIPHER_CTX *__real_EVP_CIPHER_CTX_new(void);
extern "C" int __real_EVP_EncryptInit_ex2(
    EVP_CIPHER_CTX *, const EVP_CIPHER *, const unsigned char *, const unsigned char *, const OSSL_PARAM *);
extern "C" int __real_EVP_CIPHER_CTX_set_padding(EVP_CIPHER_CTX *, int);
extern "C" void __real_EVP_CIPHER_CTX_free(EVP_CIPHER_CTX *);
extern "C" int __real_EVP_EncryptUpdate(EVP_CIPHER_CTX *, unsigned char *, int *, const unsigned char *, int);

extern "C" EVP_CIPHER_CTX *__wrap_EVP_CIPHER_CTX_new(void) {
  if (g_fault != nullptr && FailAt(g_fault->new_calls, g_fault->fail_new_at)) return nullptr;
  EVP_CIPHER_CTX *ctx = __real_EVP_CIPHER_CTX_new();
  if (g_fault != nullptr && ctx != nullptr) g_fault->live.insert(ctx);
  return ctx;
}

extern "C" int __wrap_EVP_EncryptInit_ex2(EVP_CIPHER_CTX *ctx, const EVP_CIPHER *cipher, const unsigned char *key,
    const unsigned char *iv, const OSSL_PARAM *params) {
  if (g_fault != nullptr && FailAt(g_fault->init_calls, g_fault->fail_init_at)) return 0;
  return __real_EVP_EncryptInit_ex2(ctx, cipher, key, iv, params);
}

extern "C" int __wrap_EVP_CIPHER_CTX_set_padding(EVP_CIPHER_CTX *ctx, int pad) {
  if (g_fault != nullptr && FailAt(g_fault->padding_calls, g_fault->fail_padding_at)) return 0;
  return __real_EVP_CIPHER_CTX_set_padding(ctx, pad);
}

extern "C" void __wrap_EVP_CIPHER_CTX_free(EVP_CIPHER_CTX *ctx) {
  if (g_fault != nullptr && ctx != nullptr) {
    ++g_fault->free_calls;
    if (g_fault->live.erase(ctx) == 0) ++g_fault->invalid_frees;
  }
  __real_EVP_CIPHER_CTX_free(ctx);
}

extern "C" int __wrap_EVP_EncryptUpdate(
    EVP_CIPHER_CTX *ctx, unsigned char *out, int *outl, const unsigned char *in, int inl) {
  if (g_fault != nullptr) {
    ++g_fault->update_calls;
    if (g_fault->update_calls == g_fault->update_fault_at) {
      switch (g_fault->update_fault) {
        case UpdateFault::kFailNoWrite: return 0;
        case UpdateFault::kUnchangedLength: return 1;
        case UpdateFault::kZeroLength: *outl = 0; return 1;
        case UpdateFault::kFailFullWrite:
        case UpdateFault::kFullWriteShortLength:
        case UpdateFault::kFullWriteUnchangedLength:
          for (int i = 0; i < 16; ++i) out[i] = 0xa5;
          if (g_fault->update_fault != UpdateFault::kFullWriteUnchangedLength)
            *outl = g_fault->update_fault == UpdateFault::kFailFullWrite ? 16 : 15;
          return g_fault->update_fault == UpdateFault::kFailFullWrite ? 0 : 1;
        case UpdateFault::kPartialOne:
        case UpdateFault::kPartialFifteen:
          *outl = g_fault->update_fault == UpdateFault::kPartialOne ? 1 : 15;
          for (int i = 0; i < *outl; ++i) out[i] = 0xa5;
          return 1;
      }
    }
  }
  return __real_EVP_EncryptUpdate(ctx, out, outl, in, inl);
}

extern "C" int __real_EVP_Digest(
    const void *data, size_t count, unsigned char *md, unsigned int *size, const EVP_MD *type, ENGINE *impl);

extern "C" int __wrap_EVP_Digest(const void *data, size_t count, unsigned char *md, unsigned int *size,
    const EVP_MD *type, ENGINE *impl) {
  if (g_fault == nullptr) return __real_EVP_Digest(data, count, md, size, type, impl);

  ++g_fault->digest_calls;
  if (g_fault->digest_calls == g_fault->digest_fault_at) return 0;
  if (!g_fault->digest_capture || g_fault->digest_calls > 2)
    return __real_EVP_Digest(data, count, md, size, type, impl);

  const int index = g_fault->digest_calls - 1;
  const auto *input = static_cast<const unsigned char *>(data);
  g_fault->digest_length[index] = count;
  for (size_t i = 0; i < count && i < 48; ++i) g_fault->digest_input[index][i] = input[i];
  const int ret = __real_EVP_Digest(data, count, md, size, type, impl);
  for (size_t i = 0; i < count && i < 48; ++i)
    if (g_fault->digest_input[index][i] != input[i]) g_fault->digest_input_changed = true;
  return ret;
}

extern "C" EVP_MD_CTX *__real_EVP_MD_CTX_new(void);
extern "C" int __real_EVP_MD_CTX_reset(EVP_MD_CTX *);
extern "C" void __real_EVP_MD_CTX_free(EVP_MD_CTX *);
extern "C" int __real_EVP_DigestInit_ex(EVP_MD_CTX *, const EVP_MD *, ENGINE *);
extern "C" int __real_EVP_DigestUpdate(EVP_MD_CTX *, const void *, size_t);
extern "C" int __real_EVP_DigestFinal_ex(EVP_MD_CTX *, unsigned char *, unsigned int *);

extern "C" EVP_MD_CTX *__wrap_EVP_MD_CTX_new(void) {
  if (g_fault != nullptr) {
    ++g_fault->mdnew_calls;
    if (g_fault->mdnew_calls == g_fault->mdnew_fault_at) return nullptr;
  }
  EVP_MD_CTX *ctx = __real_EVP_MD_CTX_new();
  if (g_fault != nullptr && ctx != nullptr) g_fault->live_md.insert(ctx);
  return ctx;
}

extern "C" int __wrap_EVP_MD_CTX_reset(EVP_MD_CTX *ctx) {
  if (g_fault != nullptr) {
    ++g_fault->mdreset_calls;
  }
  return __real_EVP_MD_CTX_reset(ctx);
}

extern "C" void __wrap_EVP_MD_CTX_free(EVP_MD_CTX *ctx) {
  if (g_fault != nullptr && ctx != nullptr) {
    ++g_fault->mdfree_calls;
    if (g_fault->live_md.erase(ctx) == 0) ++g_fault->md_invalid_frees;
  }
  __real_EVP_MD_CTX_free(ctx);
}

extern "C" int __wrap_EVP_DigestInit_ex(EVP_MD_CTX *ctx, const EVP_MD *type, ENGINE *impl) {
  if (g_fault != nullptr) {
    ++g_fault->mdinit_calls;
    const int index = g_fault->mdinit_calls - 1;
    if (index >= 0 && index < 2) {
      g_fault->md_oneshot[index] = EVP_MD_CTX_test_flags(ctx, EVP_MD_CTX_FLAG_ONESHOT) != 0;
      g_fault->md_init_md_null[index] = EVP_MD_CTX_get0_md(ctx) == nullptr;
      g_fault->md_init_type_null[index] = type == nullptr;
    }
    if (g_fault->mdinit_calls == g_fault->mdinit_fault_at) return 0;
  }
  return __real_EVP_DigestInit_ex(ctx, type, impl);
}

extern "C" int __wrap_EVP_DigestUpdate(EVP_MD_CTX *ctx, const void *data, size_t count) {
  if (g_fault == nullptr) return __real_EVP_DigestUpdate(ctx, data, count);

  ++g_fault->digest_calls;
  if (g_fault->digest_calls == g_fault->digest_fault_at) return 0;
  if (!g_fault->digest_capture || g_fault->digest_calls > 2)
    return __real_EVP_DigestUpdate(ctx, data, count);

  const int index = g_fault->digest_calls - 1;
  const auto *input = static_cast<const unsigned char *>(data);
  g_fault->digest_length[index] = count;
  for (size_t i = 0; i < count && i < 48; ++i) g_fault->digest_input[index][i] = input[i];
  const int ret = __real_EVP_DigestUpdate(ctx, data, count);
  for (size_t i = 0; i < count && i < 48; ++i)
    if (g_fault->digest_input[index][i] != input[i]) g_fault->digest_input_changed = true;
  return ret;
}

extern "C" int __wrap_EVP_DigestFinal_ex(EVP_MD_CTX *ctx, unsigned char *md, unsigned int *size) {
  if (g_fault == nullptr) return __real_EVP_DigestFinal_ex(ctx, md, size);

  ++g_fault->mdfinal_calls;
  const int index = g_fault->mdfinal_calls - 1;
  const unsigned int entry_length = size == nullptr ? 0 : *size;
  if (index >= 0 && index < 2) g_fault->md_final_entry_len[index] = entry_length;
  if (g_fault->mdfinal_calls == g_fault->mdfinal_fault_at) {
    switch (g_fault->mdfinal_fault) {
      case FinalFault::kFailNoWrite: return 0;
      case FinalFault::kFailFullWriteLength32: {
        unsigned int real_length = 0;
        const int ret = __real_EVP_DigestFinal_ex(ctx, md, &real_length);
        EXPECT_EQ(ret, 1);
        EXPECT_EQ(real_length, 32u);
        if (size != nullptr) *size = 32;
        return 0;
      }
      case FinalFault::kFullWriteLength31: {
        unsigned int real_length = 0;
        const int ret = __real_EVP_DigestFinal_ex(ctx, md, &real_length);
        EXPECT_EQ(ret, 1);
        EXPECT_EQ(real_length, 32u);
        if (size != nullptr) *size = 31;
        return 1;
      }
      case FinalFault::kFullWriteUnchangedLength: {
        unsigned int real_length = 0;
        const int ret = __real_EVP_DigestFinal_ex(ctx, md, &real_length);
        EXPECT_EQ(ret, 1);
        EXPECT_EQ(real_length, 32u);
        if (size != nullptr) *size = entry_length;
        return 1;
      }
      case FinalFault::kPartial16Length16: {
        unsigned char block[32] = {};
        unsigned int real_length = 0;
        const int ret = __real_EVP_DigestFinal_ex(ctx, block, &real_length);
        EXPECT_EQ(ret, 1);
        EXPECT_EQ(real_length, 32u);
        for (int i = 0; i < 16; ++i) md[i] = block[i];
        if (size != nullptr) *size = 16;
        return 1;
      }
    }
  }
  return __real_EVP_DigestFinal_ex(ctx, md, size);
}

namespace {

using fss::prg::Aes128Mmo;
using fss::prp::Aes128Feistel;

template <typename F>
void ExpectRuntimeError(F &&f, const std::string &message) {
  try {
    f();
    FAIL() << "expected std::runtime_error";
  } catch (const std::runtime_error &e) {
    EXPECT_NE(std::string(e.what()).find(message), std::string::npos) << e.what();
  }
}

template <int mul>
void FillKeys(unsigned char (&storage)[mul][16], const unsigned char *(&keys)[mul]) {
  for (int i = 0; i < mul; ++i) keys[i] = storage[i];
}

template <int mul>
struct MmoCtxsGuard {
  cuda::std::array<EVP_CIPHER_CTX *, mul> ctxs;

  explicit MmoCtxsGuard(cuda::std::array<EVP_CIPHER_CTX *, mul> owned) : ctxs(owned) {}

  ~MmoCtxsGuard() { Aes128Mmo<mul>::FreeCtxs(ctxs); }

  MmoCtxsGuard(const MmoCtxsGuard &) = delete;
  MmoCtxsGuard &operator=(const MmoCtxsGuard &) = delete;
};

const int4 kAllZeroAesBlock{
    static_cast<int>(0xd44be966u), static_cast<int>(0x3b2c8aefu),
    static_cast<int>(0x59fa4c88u), static_cast<int>(0x2e2b34cau)};

void ExpectAllZeroAesBlock(const int4 &actual) {
  EXPECT_EQ(actual.x, kAllZeroAesBlock.x);
  EXPECT_EQ(actual.y, kAllZeroAesBlock.y);
  EXPECT_EQ(actual.z, kAllZeroAesBlock.z);
  EXPECT_EQ(actual.w, kAllZeroAesBlock.w);
}

enum class UpdateEntry { kGen, kGenHalfLeft, kGenHalfRight };

struct UpdateFailureParam {
  UpdateEntry entry;
  UpdateFault fault;
  int position;
  std::string name;
  int arity = 4;
};

template <int mul>
void CallUpdateEntry(UpdateEntry entry, Aes128Mmo<mul> &prg) {
  const int4 seed{0, 0, 0, 0};
  switch (entry) {
    case UpdateEntry::kGen:
      (void)prg.Gen(seed);
      break;
    case UpdateEntry::kGenHalfLeft:
      (void)prg.GenHalf(seed, false);
      break;
    case UpdateEntry::kGenHalfRight:
      (void)prg.GenHalf(seed, true);
      break;
  }
}

std::string UpdateFailureName(const ::testing::TestParamInfo<UpdateFailureParam> &info) {
  return info.param.name;
}

class OpenSslFailureCleanupTest : public ::testing::Test {
 protected:
  void SetUp() override { g_fault = &fault_; }

  void TearDown() override {
    EXPECT_EQ(fault_.invalid_frees, 0);
    EXPECT_EQ(fault_.md_invalid_frees, 0);
    EXPECT_TRUE(fault_.live.empty());
    EXPECT_TRUE(fault_.live_md.empty());
    g_fault = nullptr;
  }

  void ExpectMdCalls(int mdnew, int mdfree, int reset, int init, int update, int final) {
    EXPECT_EQ(fault_.mdnew_calls, mdnew);
    EXPECT_EQ(fault_.mdfree_calls, mdfree);
    EXPECT_EQ(fault_.mdreset_calls, reset);
    EXPECT_EQ(fault_.mdinit_calls, init);
    EXPECT_EQ(fault_.digest_calls, update);
    EXPECT_EQ(fault_.mdfinal_calls, final);
    EXPECT_EQ(fault_.md_invalid_frees, 0);
    EXPECT_TRUE(fault_.live_md.empty());
  }

  OpenSslFault fault_;
};

class OpenSslUpdateFailureTest : public OpenSslFailureCleanupTest,
                                 public ::testing::WithParamInterface<UpdateFailureParam> {};

TEST_F(OpenSslFailureCleanupTest, CreateCtxsFreesNothingWhenFirstAllocationFails) {
  unsigned char storage[4][16] = {};
  const unsigned char *keys[4];
  FillKeys<4>(storage, keys);

  fault_.fail_new_at = 1;
  ExpectRuntimeError([&] { Aes128Mmo<4>::CreateCtxs(keys); }, "EVP_CIPHER_CTX_new failed");

  EXPECT_EQ(fault_.new_calls, 1);
  EXPECT_EQ(fault_.free_calls, 0);
}

TEST_F(OpenSslFailureCleanupTest, CreateCtxsFreesEarlierContextsWhenAllocationFails) {
  unsigned char storage[4][16] = {};
  const unsigned char *keys[4];
  FillKeys<4>(storage, keys);

  fault_.fail_new_at = 3;
  ExpectRuntimeError([&] { Aes128Mmo<4>::CreateCtxs(keys); }, "EVP_CIPHER_CTX_new failed");

  EXPECT_EQ(fault_.new_calls, 3);
  EXPECT_EQ(fault_.free_calls, 2);
}

TEST_F(OpenSslFailureCleanupTest, CreateCtxsFreesAllContextsWhenInitFails) {
  unsigned char storage[4][16] = {};
  const unsigned char *keys[4];
  FillKeys<4>(storage, keys);

  fault_.fail_init_at = 2;
  ExpectRuntimeError([&] { Aes128Mmo<4>::CreateCtxs(keys); }, "EVP_EncryptInit_ex2 failed");

  EXPECT_EQ(fault_.new_calls, 2);
  EXPECT_EQ(fault_.free_calls, 2);
}

TEST_F(OpenSslFailureCleanupTest, CreateCtxsFreesAllContextsWhenPaddingFails) {
  unsigned char storage[4][16] = {};
  const unsigned char *keys[4];
  FillKeys<4>(storage, keys);

  fault_.fail_padding_at = 2;
  ExpectRuntimeError([&] { Aes128Mmo<4>::CreateCtxs(keys); }, "EVP_CIPHER_CTX_set_padding failed");

  EXPECT_EQ(fault_.new_calls, 2);
  EXPECT_EQ(fault_.free_calls, 2);
}

TEST_F(OpenSslFailureCleanupTest, CreateCtxsTransfersOwnershipAndFreesEachContextOnce) {
  unsigned char storage[4][16] = {};
  const unsigned char *keys[4];
  FillKeys<4>(storage, keys);

  auto ctxs = Aes128Mmo<4>::CreateCtxs(keys);
  EXPECT_EQ(fault_.new_calls, 4);
  EXPECT_EQ(fault_.live.size(), 4u);
  for (auto *ctx : ctxs) EXPECT_NE(ctx, nullptr);

  {
    Aes128Mmo<4> prg(ctxs);
    prg.Gen({1, 2, 3, 4});
  }

  Aes128Mmo<4>::FreeCtxs(ctxs);
  EXPECT_EQ(fault_.free_calls, 4);
  EXPECT_TRUE(fault_.live.empty());
}

TEST_F(OpenSslFailureCleanupTest, FeistelConstructorFreesNothingWhenAllocationFails) {
  fault_.fail_new_at = 1;
  ExpectRuntimeError([] { Aes128Feistel prp; }, "EVP_CIPHER_CTX_new failed");

  EXPECT_EQ(fault_.free_calls, 0);
}

TEST_F(OpenSslFailureCleanupTest, FeistelConstructorFreesContextWhenInitFails) {
  fault_.fail_init_at = 1;
  ExpectRuntimeError([] { Aes128Feistel prp; }, "EVP_EncryptInit_ex2 failed");

  EXPECT_EQ(fault_.new_calls, 1);
  EXPECT_EQ(fault_.free_calls, 1);
}

TEST_F(OpenSslFailureCleanupTest, FeistelConstructorFreesContextWhenPaddingFails) {
  fault_.fail_padding_at = 1;
  ExpectRuntimeError([] { Aes128Feistel prp; }, "EVP_CIPHER_CTX_set_padding failed");

  EXPECT_EQ(fault_.new_calls, 1);
  EXPECT_EQ(fault_.free_calls, 1);
}

TEST_F(OpenSslFailureCleanupTest, FeistelCopyConstructorFailureKeepsSourceContext) {
  {
    Aes128Feistel source;
    EXPECT_EQ(fault_.live.size(), 1u);

    fault_.fail_init_at = 2;
    ExpectRuntimeError([&] { Aes128Feistel copy(source); }, "EVP_EncryptInit_ex2 failed");

    EXPECT_EQ(fault_.free_calls, 1);
    EXPECT_EQ(fault_.live.size(), 1u);
  }

  EXPECT_EQ(fault_.free_calls, 2);
  EXPECT_TRUE(fault_.live.empty());
}

TEST_F(OpenSslFailureCleanupTest, FeistelSuccessFreesEachContextOnce) {
  {
    Aes128Feistel source;
    EXPECT_EQ(fault_.live.size(), 1u);

    Aes128Feistel copy(source);
    EXPECT_EQ(fault_.live.size(), 2u);
    EXPECT_TRUE(copy.Permu({1, 2, 3, 4}, 5, 100) < 100);
  }

  EXPECT_EQ(fault_.free_calls, 2);
  EXPECT_TRUE(fault_.live.empty());
}

TEST_F(OpenSslFailureCleanupTest, FeistelCopyAssignmentIntoMovedFromRestoresContext) {
  {
    Aes128Feistel a;
    Aes128Feistel b;
    EXPECT_EQ(fault_.new_calls, 2);

    b = a;
    EXPECT_EQ(fault_.new_calls, 2);

    Aes128Feistel sink(std::move(a));
    EXPECT_EQ(fault_.live.size(), 2u);

    a = b;
    EXPECT_EQ(fault_.new_calls, 3);
    EXPECT_EQ(fault_.live.size(), 3u);
    EXPECT_EQ(a.Permu({1, 2, 3, 4}, 5, 100), b.Permu({1, 2, 3, 4}, 5, 100));
  }

  EXPECT_EQ(fault_.free_calls, 3);
  EXPECT_TRUE(fault_.live.empty());
}

TEST_F(OpenSslFailureCleanupTest, FeistelCopyAssignmentAllocationFailurePropagates) {
  {
    Aes128Feistel a;
    Aes128Feistel b;
    Aes128Feistel sink(std::move(a));
    EXPECT_EQ(fault_.live.size(), 2u);

    fault_.fail_new_at = 3;
    ExpectRuntimeError([&] { a = b; }, "EVP_CIPHER_CTX_new failed");

    EXPECT_EQ(fault_.free_calls, 0);
    EXPECT_TRUE(b.Permu({1, 2, 3, 4}, 5, 100) < 100);
  }

  EXPECT_EQ(fault_.free_calls, 2);
  EXPECT_TRUE(fault_.live.empty());
}

template <int mul>
void RunUpdateFailure(const UpdateFailureParam &param, OpenSslFault &fault) {
  unsigned char storage[mul][16] = {};
  const unsigned char *keys[mul];
  FillKeys<mul>(storage, keys);

  fault.update_fault_at = param.position;
  fault.update_fault = param.fault;

  MmoCtxsGuard<mul> guard{Aes128Mmo<mul>::CreateCtxs(keys)};
  Aes128Mmo<mul> prg(guard.ctxs);

  std::string message = param.entry == UpdateEntry::kGen ? "Aes128Mmo::Gen:" : "Aes128Mmo::GenHalf:";
  const bool failed = param.fault == UpdateFault::kFailNoWrite ||
      param.fault == UpdateFault::kFailFullWrite;
  message += failed ? " EVP_EncryptUpdate failed" : " EVP_EncryptUpdate returned a partial block";

  ExpectRuntimeError([&] { CallUpdateEntry<mul>(param.entry, prg); }, message);

  EXPECT_EQ(fault.update_calls, param.position);
}

TEST_P(OpenSslUpdateFailureTest, RejectsFailureAtConfiguredUpdateCall) {
  const UpdateFailureParam &param = GetParam();
  if (param.arity == 2) {
    RunUpdateFailure<2>(param, fault_);
  } else {
    RunUpdateFailure<4>(param, fault_);
  }
}

std::vector<UpdateFailureParam> UpdateFailureParams() {
  struct Entry { UpdateEntry entry; int last; const char *name; };
  const Entry entries[] = {{UpdateEntry::kGen, 4, "Gen"},
      {UpdateEntry::kGenHalfLeft, 2, "GenHalfLeft"},
      {UpdateEntry::kGenHalfRight, 2, "GenHalfRight"}};
  struct Fault { UpdateFault fault; const char *name; };
  const Fault faults[] = {{UpdateFault::kFailNoWrite, "FailNoWrite"},
      {UpdateFault::kFailFullWrite, "FailFullWrite"},
      {UpdateFault::kZeroLength, "ZeroLength"},
      {UpdateFault::kUnchangedLength, "UnchangedLength"},
      {UpdateFault::kPartialOne, "PartialOne"},
      {UpdateFault::kPartialFifteen, "PartialFifteen"},
      {UpdateFault::kFullWriteShortLength, "FullWriteShortLength"},
      {UpdateFault::kFullWriteUnchangedLength, "FullWriteUnchangedLength"}};
  std::vector<UpdateFailureParam> params;
  for (const auto &entry : entries) {
    for (const auto &fault : faults) {
      for (int position : {1, entry.last}) {
        params.push_back({entry.entry, fault.fault, position,
            std::string(entry.name) + fault.name + (position == 1 ? "First" : "Last")});
      }
    }
  }
  const Entry arity2_entries[] = {{UpdateEntry::kGen, 2, "Gen"},
      {UpdateEntry::kGenHalfLeft, 1, "GenHalfLeft"},
      {UpdateEntry::kGenHalfRight, 1, "GenHalfRight"}};
  for (const auto &entry : arity2_entries) {
    for (const auto &fault : faults) {
      if (entry.last == 1) {
        params.push_back({entry.entry, fault.fault, 1,
            std::string(entry.name) + "Arity2" + fault.name, 2});
      } else {
        for (int position : {1, entry.last}) {
          params.push_back({entry.entry, fault.fault, position,
              std::string(entry.name) + "Arity2" + fault.name + (position == 1 ? "First" : "Last"), 2});
        }
      }
    }
  }
  return params;
}

INSTANTIATE_TEST_SUITE_P(AllEntrypoints, OpenSslUpdateFailureTest,
    ::testing::ValuesIn(UpdateFailureParams()), UpdateFailureName);

TEST_F(OpenSslFailureCleanupTest, GenMatchesKnownAllZeroAesBlock) {
  unsigned char storage[4][16] = {};
  const unsigned char *keys[4];
  FillKeys<4>(storage, keys);

  MmoCtxsGuard<4> guard{Aes128Mmo<4>::CreateCtxs(keys)};
  Aes128Mmo<4> prg(guard.ctxs);

  EXPECT_EQ(fault_.update_fault_at, 0);
  auto out = prg.Gen({0, 0, 0, 0});
  EXPECT_EQ(fault_.update_calls, 4);
  for (const auto &block : out) ExpectAllZeroAesBlock(block);
}

TEST_F(OpenSslFailureCleanupTest, GenHalfLeftMatchesKnownAllZeroAesBlock) {
  unsigned char storage[4][16] = {};
  const unsigned char *keys[4];
  FillKeys<4>(storage, keys);

  MmoCtxsGuard<4> guard{Aes128Mmo<4>::CreateCtxs(keys)};
  Aes128Mmo<4> prg(guard.ctxs);

  EXPECT_EQ(fault_.update_fault_at, 0);
  auto out = prg.GenHalf({0, 0, 0, 0}, false);
  EXPECT_EQ(fault_.update_calls, 2);
  for (const auto &block : out) ExpectAllZeroAesBlock(block);
}

TEST_F(OpenSslFailureCleanupTest, GenHalfRightMatchesKnownAllZeroAesBlock) {
  unsigned char storage[4][16] = {};
  const unsigned char *keys[4];
  FillKeys<4>(storage, keys);

  MmoCtxsGuard<4> guard{Aes128Mmo<4>::CreateCtxs(keys)};
  Aes128Mmo<4> prg(guard.ctxs);

  EXPECT_EQ(fault_.update_fault_at, 0);
  auto out = prg.GenHalf({0, 0, 0, 0}, true);
  EXPECT_EQ(fault_.update_calls, 2);
  for (const auto &block : out) ExpectAllZeroAesBlock(block);
}

// NIST AES_Core128.pdf, first block. MMO expectations are the literal XOR of
// the published ciphertext and plaintext, independent of all fss backends.
// Published ciphertext: 3ad77bb40d7a3660a89ecaf32466ef97.
TEST_F(OpenSslFailureCleanupTest, MatchesNistMmoAcrossRepeatedCalls) {
  static_assert(sizeof(int) == 4 && sizeof(int4) == 16);
  static_assert(std::is_trivially_copyable_v<int4> &&
      std::is_trivially_destructible_v<int4>);
  static_assert(std::endian::native == std::endian::little);
  const unsigned char key[16] = {
      0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
      0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};
  const unsigned char *keys[4] = {key, key, key, key};
  const int4 seed{
      static_cast<int>(0xe2bec16bu),
      static_cast<int>(0x969f402eu),
      static_cast<int>(0x117e3de9u),
      static_cast<int>(0x2a179373u)};
  const int4 expected{
      static_cast<int>(0x56c51651u),
      static_cast<int>(0xf6a93a23u),
      static_cast<int>(0xe2b4a341u),
      static_cast<int>(0xbdf8f557u)};
  MmoCtxsGuard<4> guard{Aes128Mmo<4>::CreateCtxs(keys)};
  Aes128Mmo<4> prg(guard.ctxs);
  auto check = [&](int4 value) {
    EXPECT_EQ(value.x, expected.x);
    EXPECT_EQ(value.y, expected.y);
    EXPECT_EQ(value.z, expected.z);
    EXPECT_EQ(value.w, expected.w);
  };
  for (int repeat = 0; repeat < 8; ++repeat) {
    SCOPED_TRACE(repeat);
    for (int4 value : prg.Gen(seed)) check(value);
    for (int4 value : prg.GenHalf(seed, false)) check(value);
    for (int4 value : prg.GenHalf(seed, true)) check(value);
  }
}

// Same published vector as above, on the arity-2 instance the DPF path uses.
// The expected value is still the literal XOR of published ciphertext and
// plaintext, not a result computed through the tested backend.
TEST_F(OpenSslFailureCleanupTest, MatchesNistMmoAcrossRepeatedCallsArity2) {
  static_assert(sizeof(int) == 4 && sizeof(int4) == 16);
  static_assert(std::is_trivially_copyable_v<int4> &&
      std::is_trivially_destructible_v<int4>);
  static_assert(std::endian::native == std::endian::little);
  const unsigned char key[16] = {
      0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
      0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};
  const unsigned char *keys[2] = {key, key};
  const int4 seed{
      static_cast<int>(0xe2bec16bu),
      static_cast<int>(0x969f402eu),
      static_cast<int>(0x117e3de9u),
      static_cast<int>(0x2a179373u)};
  const int4 expected{
      static_cast<int>(0x56c51651u),
      static_cast<int>(0xf6a93a23u),
      static_cast<int>(0xe2b4a341u),
      static_cast<int>(0xbdf8f557u)};
  MmoCtxsGuard<2> guard{Aes128Mmo<2>::CreateCtxs(keys)};
  Aes128Mmo<2> prg(guard.ctxs);
  auto check = [&](int4 value) {
    EXPECT_EQ(value.x, expected.x);
    EXPECT_EQ(value.y, expected.y);
    EXPECT_EQ(value.z, expected.z);
    EXPECT_EQ(value.w, expected.w);
  };
  for (int repeat = 0; repeat < 8; ++repeat) {
    SCOPED_TRACE(repeat);
    for (int4 value : prg.Gen(seed)) check(value);
    for (int4 value : prg.GenHalf(seed, false)) check(value);
    for (int4 value : prg.GenHalf(seed, true)) check(value);
  }
}

// Fixture bytes are literal Python hashlib inputs, not results from the tested
// backend. The tuple is (coordinate, seed): coordinate is a full 128-bit value
// whose .w LSB stays 1, and seed carries the reserved bit in .w.
const int4 kShaKey{
    static_cast<int>(0x03020100u), static_cast<int>(0x07060504u), static_cast<int>(0x0b0a0908u), static_cast<int>(0x0f0e0d0cu)};
const int4 kShaCoordinate{
    static_cast<int>(0xd3d2d1d0u), static_cast<int>(0xd7d6d5d4u), static_cast<int>(0xdbdad9d8u), static_cast<int>(0xdfdeddddu)};
const int4 kShaSeed{
    static_cast<int>(0xa3a2a1a0u), static_cast<int>(0xa7a6a5a4u), static_cast<int>(0xabaaa9a8u), static_cast<int>(0xafaeadacu)};

const unsigned char kShaMessage0[48] = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    0xd0, 0xd1, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7,
    0xd8, 0xd9, 0xda, 0xdb, 0xdd, 0xdd, 0xde, 0xdf,
    0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xab, 0xac, 0xad, 0xae, 0xaf,
};
const unsigned char kShaMessage1[48] = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    0xd0, 0xd1, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7,
    0xd8, 0xd9, 0xda, 0xdb, 0xdd, 0xdd, 0xde, 0xdf,
    0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xab, 0xad, 0xad, 0xae, 0xaf,
};
const unsigned char kShaExpected[64] = {
    0x66, 0x8e, 0xda, 0xf0, 0x43, 0x4f, 0x69, 0x7e,
    0x8e, 0x05, 0xb9, 0x15, 0xc4, 0xb7, 0x29, 0x20,
    0x47, 0xb7, 0x18, 0x04, 0x64, 0x56, 0xec, 0x0a,
    0x61, 0xf8, 0x70, 0xfc, 0xac, 0x18, 0xbc, 0xe7,
    0x1c, 0xd9, 0x45, 0x98, 0xab, 0x6d, 0x16, 0xcd,
    0x61, 0x89, 0xaa, 0x3c, 0x96, 0xae, 0xf8, 0xac,
    0x61, 0xa6, 0x30, 0xcb, 0xf0, 0xf1, 0xb3, 0x93,
    0x1d, 0x35, 0xda, 0x8d, 0x91, 0x69, 0xb1, 0xe0,
};

cuda::std::tuple<int4, const int4> ShaTuple() {
  return cuda::std::tuple<int4, const int4>{kShaCoordinate, kShaSeed};
}

TEST_F(OpenSslFailureCleanupTest, Sha256TupleHashesTwoTripleWordFrames) {
  fss::hash::Sha256 sha(kShaKey);
  fault_.digest_capture = true;

  const auto digest = sha.Hash(ShaTuple());

  ExpectMdCalls(1, 1, 0, 2, 2, 2);
  EXPECT_TRUE(fault_.md_oneshot[0]);
  EXPECT_TRUE(fault_.md_oneshot[1]);
  EXPECT_TRUE(fault_.md_init_md_null[0]);
  EXPECT_FALSE(fault_.md_init_md_null[1]);
  EXPECT_FALSE(fault_.md_init_type_null[0]);
  EXPECT_TRUE(fault_.md_init_type_null[1]);
  EXPECT_EQ(fault_.md_final_entry_len[0], 0u);
  EXPECT_EQ(fault_.md_final_entry_len[1], 0u);
  ASSERT_EQ(fault_.digest_length[0], 48u);
  ASSERT_EQ(fault_.digest_length[1], 48u);
  EXPECT_FALSE(fault_.digest_input_changed);
  for (size_t i = 0; i < 48; ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(fault_.digest_input[0][i], kShaMessage0[i]);
    EXPECT_EQ(fault_.digest_input[1][i], kShaMessage1[i]);
  }
  EXPECT_EQ(fault_.digest_input[0][28], 0xdd);
  EXPECT_EQ(fault_.digest_input[0][44], 0xac);
  EXPECT_EQ(fault_.digest_input[1][28], 0xdd);
  EXPECT_EQ(fault_.digest_input[1][44], 0xad);

  const auto *actual = reinterpret_cast<const unsigned char *>(digest.data());
  for (size_t i = 0; i < 64; ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(actual[i], kShaExpected[i]);
  }
}

TEST_F(OpenSslFailureCleanupTest, Sha256TupleFirstDigestFailurePropagates) {
  fss::hash::Sha256 sha(kShaKey);
  fault_.digest_fault_at = 1;

  ExpectRuntimeError([&] { (void)sha.Hash(ShaTuple()); }, "Sha256::Hash: EVP_Digest failed");

  ExpectMdCalls(1, 1, 0, 1, 1, 0);
}

TEST_F(OpenSslFailureCleanupTest, Sha256TupleSecondDigestFailurePropagates) {
  fss::hash::Sha256 sha(kShaKey);
  fault_.digest_fault_at = 2;

  ExpectRuntimeError([&] { (void)sha.Hash(ShaTuple()); }, "Sha256::Hash: EVP_Digest failed");

  ExpectMdCalls(1, 1, 0, 2, 2, 1);
}

TEST_F(OpenSslFailureCleanupTest, Sha256TupleMdCtxNewFailurePropagates) {
  fss::hash::Sha256 sha(kShaKey);
  fault_.mdnew_fault_at = 1;

  ExpectRuntimeError([&] { (void)sha.Hash(ShaTuple()); }, "Sha256::Hash: EVP_MD_CTX_new failed");

  ExpectMdCalls(1, 0, 0, 0, 0, 0);
}

TEST_F(OpenSslFailureCleanupTest, Sha256TupleFirstInitFailurePropagates) {
  fss::hash::Sha256 sha(kShaKey);
  fault_.mdinit_fault_at = 1;

  ExpectRuntimeError([&] { (void)sha.Hash(ShaTuple()); }, "Sha256::Hash: EVP_Digest failed");

  ExpectMdCalls(1, 1, 0, 1, 0, 0);
}

TEST_F(OpenSslFailureCleanupTest, Sha256TupleSecondInitFailurePropagates) {
  fss::hash::Sha256 sha(kShaKey);
  fault_.mdinit_fault_at = 2;

  ExpectRuntimeError([&] { (void)sha.Hash(ShaTuple()); }, "Sha256::Hash: EVP_Digest failed");

  ExpectMdCalls(1, 1, 0, 2, 1, 1);
}

TEST_F(OpenSslFailureCleanupTest, Sha256TupleFirstFinalFailurePropagates) {
  fss::hash::Sha256 sha(kShaKey);
  fault_.mdfinal_fault_at = 1;
  fault_.mdfinal_fault = FinalFault::kFailNoWrite;

  ExpectRuntimeError([&] { (void)sha.Hash(ShaTuple()); }, "Sha256::Hash: EVP_Digest failed");

  ExpectMdCalls(1, 1, 0, 1, 1, 1);
}

TEST_F(OpenSslFailureCleanupTest, Sha256TupleSecondFinalFailurePropagates) {
  fss::hash::Sha256 sha(kShaKey);
  fault_.mdfinal_fault_at = 2;
  fault_.mdfinal_fault = FinalFault::kFailNoWrite;

  ExpectRuntimeError([&] { (void)sha.Hash(ShaTuple()); }, "Sha256::Hash: EVP_Digest failed");

  ExpectMdCalls(1, 1, 0, 2, 2, 2);
}

TEST_F(OpenSslFailureCleanupTest, Sha256TupleFirstFinalFullWriteFailurePropagates) {
  fss::hash::Sha256 sha(kShaKey);
  fault_.mdfinal_fault_at = 1;
  fault_.mdfinal_fault = FinalFault::kFailFullWriteLength32;

  ExpectRuntimeError([&] { (void)sha.Hash(ShaTuple()); }, "Sha256::Hash: EVP_Digest failed");

  ExpectMdCalls(1, 1, 0, 1, 1, 1);
}

TEST_F(OpenSslFailureCleanupTest, Sha256TupleSecondFinalFullWriteFailurePropagates) {
  fss::hash::Sha256 sha(kShaKey);
  fault_.mdfinal_fault_at = 2;
  fault_.mdfinal_fault = FinalFault::kFailFullWriteLength32;

  ExpectRuntimeError([&] { (void)sha.Hash(ShaTuple()); }, "Sha256::Hash: EVP_Digest failed");

  ExpectMdCalls(1, 1, 0, 2, 2, 2);
}

TEST_F(OpenSslFailureCleanupTest, Sha256TupleFirstFinalWrongLength31Propagates) {
  fss::hash::Sha256 sha(kShaKey);
  fault_.mdfinal_fault_at = 1;
  fault_.mdfinal_fault = FinalFault::kFullWriteLength31;

  ExpectRuntimeError(
      [&] { (void)sha.Hash(ShaTuple()); }, "Sha256::Hash: EVP_Digest returned a partial digest");

  ExpectMdCalls(1, 1, 0, 1, 1, 1);
  EXPECT_EQ(fault_.md_final_entry_len[0], 0u);
}

TEST_F(OpenSslFailureCleanupTest, Sha256TupleSecondFinalWrongLength31Propagates) {
  fss::hash::Sha256 sha(kShaKey);
  fault_.mdfinal_fault_at = 2;
  fault_.mdfinal_fault = FinalFault::kFullWriteLength31;

  ExpectRuntimeError(
      [&] { (void)sha.Hash(ShaTuple()); }, "Sha256::Hash: EVP_Digest returned a partial digest");

  ExpectMdCalls(1, 1, 0, 2, 2, 2);
  EXPECT_EQ(fault_.md_final_entry_len[0], 0u);
  EXPECT_EQ(fault_.md_final_entry_len[1], 0u);
}

TEST_F(OpenSslFailureCleanupTest, Sha256TupleFirstFinalUnchangedLengthPropagates) {
  fss::hash::Sha256 sha(kShaKey);
  fault_.mdfinal_fault_at = 1;
  fault_.mdfinal_fault = FinalFault::kFullWriteUnchangedLength;

  ExpectRuntimeError(
      [&] { (void)sha.Hash(ShaTuple()); }, "Sha256::Hash: EVP_Digest returned a partial digest");

  ExpectMdCalls(1, 1, 0, 1, 1, 1);
  EXPECT_EQ(fault_.md_final_entry_len[0], 0u);
}

TEST_F(OpenSslFailureCleanupTest, Sha256TupleSecondFinalUnchangedLengthPropagates) {
  fss::hash::Sha256 sha(kShaKey);
  fault_.mdfinal_fault_at = 2;
  fault_.mdfinal_fault = FinalFault::kFullWriteUnchangedLength;

  ExpectRuntimeError(
      [&] { (void)sha.Hash(ShaTuple()); }, "Sha256::Hash: EVP_Digest returned a partial digest");

  ExpectMdCalls(1, 1, 0, 2, 2, 2);
  EXPECT_EQ(fault_.md_final_entry_len[0], 0u);
  EXPECT_EQ(fault_.md_final_entry_len[1], 0u);
}

TEST_F(OpenSslFailureCleanupTest, Sha256TupleFirstFinalPartial16Propagates) {
  fss::hash::Sha256 sha(kShaKey);
  fault_.mdfinal_fault_at = 1;
  fault_.mdfinal_fault = FinalFault::kPartial16Length16;

  ExpectRuntimeError(
      [&] { (void)sha.Hash(ShaTuple()); }, "Sha256::Hash: EVP_Digest returned a partial digest");

  ExpectMdCalls(1, 1, 0, 1, 1, 1);
}

TEST_F(OpenSslFailureCleanupTest, Sha256TupleSecondFinalPartial16Propagates) {
  fss::hash::Sha256 sha(kShaKey);
  fault_.mdfinal_fault_at = 2;
  fault_.mdfinal_fault = FinalFault::kPartial16Length16;

  ExpectRuntimeError(
      [&] { (void)sha.Hash(ShaTuple()); }, "Sha256::Hash: EVP_Digest returned a partial digest");

  ExpectMdCalls(1, 1, 0, 2, 2, 2);
}


TEST_F(OpenSslFailureCleanupTest, Sha256SpanMatchesIndependentGolden) {
  fss::hash::Sha256 sha(kShaKey);
  const cuda::std::array<int4, 4> message{{
      {static_cast<int>(0x03020100u), static_cast<int>(0x07060504u), static_cast<int>(0x0b0a0908u), static_cast<int>(0x0f0e0d0cu)},
      {static_cast<int>(0x13121110u), static_cast<int>(0x17161514u), static_cast<int>(0x1b1a1918u), static_cast<int>(0x1f1e1d1cu)},
      {static_cast<int>(0x23222120u), static_cast<int>(0x27262524u), static_cast<int>(0x2b2a2928u), static_cast<int>(0x2f2e2d2cu)},
      {static_cast<int>(0x33323130u), static_cast<int>(0x37363534u), static_cast<int>(0x3b3a3938u), static_cast<int>(0x3f3e3d3cu)}}};
  const unsigned char expected[32] = {0xed, 0x53, 0x26, 0x2a, 0xa1, 0x0a, 0x5f, 0xc4, 0xbe, 0x99, 0x00, 0xff, 0x73, 0x79, 0xd9, 0x1f, 0xba, 0x97, 0x4d, 0x6e, 0x34, 0xed, 0xa6, 0xaa, 0xd2, 0x6b, 0x54, 0xa9, 0x7b, 0x4f, 0x70, 0xf0};
  const auto digest = sha.Hash(cuda::std::span<const int4, 4>(message));
  EXPECT_EQ(fault_.digest_calls, 1);
  const auto *actual = reinterpret_cast<const unsigned char *>(digest.data());
  for (size_t i = 0; i < 32; ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(actual[i], expected[i]);
  }
}

TEST_F(OpenSslFailureCleanupTest, Sha256SpanDigestFailurePropagates) {
  fss::hash::Sha256 sha(kShaKey);
  const cuda::std::array<int4, 4> message{};
  fault_.digest_fault_at = 1;
  ExpectRuntimeError(
      [&] { (void)sha.Hash(cuda::std::span<const int4, 4>(message)); },
      "Sha256::Hash: EVP_Digest failed");
  EXPECT_EQ(fault_.digest_calls, 1);
}

}  // namespace

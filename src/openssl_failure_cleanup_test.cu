// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <cuda/std/array>
#include <openssl/evp.h>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <fss/prg/aes128_mmo.cuh>
#include <fss/prp/aes128_feistel.cuh>

namespace {

struct OpenSslFault {
  int fail_new_at = 0;
  int fail_init_at = 0;
  int fail_padding_at = 0;
  int new_calls = 0;
  int init_calls = 0;
  int padding_calls = 0;
  int free_calls = 0;
  int invalid_frees = 0;
  std::set<EVP_CIPHER_CTX *> live;
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

namespace {

using fss::prg::Aes128Mmo;
using fss::prp::Aes128Feistel;

template <typename F>
void ExpectRuntimeError(F &&f, const char *message) {
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

class OpenSslFailureCleanupTest : public ::testing::Test {
 protected:
  void SetUp() override { g_fault = &fault_; }

  void TearDown() override {
    EXPECT_EQ(fault_.invalid_frees, 0);
    EXPECT_TRUE(fault_.live.empty());
    g_fault = nullptr;
  }

  OpenSslFault fault_;
};

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

}  // namespace

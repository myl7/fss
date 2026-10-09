// Release-safe checks for the historical benchmark adapters.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifdef FSS070_BENCH_DPF
using CheckedKey = DpfKey;
constexpr int kCheckedCwLen = kDpfCwLen;
constexpr int kCheckedGenBytes = 6 * kLambda;
constexpr int kCheckedEvalBytes = 3 * kLambda;
#else
using CheckedKey = DcfKey;
constexpr int kCheckedCwLen = kDcfCwLen;
constexpr int kCheckedGenBytes = 10 * kLambda;
constexpr int kCheckedEvalBytes = 6 * kLambda;
#endif

HOST_DEVICE static void CheckedGen(CheckedKey key, Bits alpha, uint8_t *beta,
                                    uint8_t *scratch) {
#ifdef FSS070_BENCH_DPF
  dpf_gen(key, PointFunc{alpha, beta}, scratch);
#else
  dcf_gen(key, CmpFunc{alpha, beta, kLtAlpha}, scratch);
#endif
}

HOST_DEVICE static void CheckedEval(uint8_t *scratch, uint8_t party,
                                     CheckedKey key, Bits query) {
#ifdef FSS070_BENCH_DPF
  dpf_eval(scratch, party, key, query);
#else
  dcf_eval(scratch, party, key, query);
#endif
}

#ifndef FSS_BENCH_DOMAIN_BITS
#define FSS_BENCH_DOMAIN_BITS 20
#endif
constexpr uint32_t kCheckedDomainMask = (1u << FSS_BENCH_DOMAIN_BITS) - 1;
constexpr uint32_t kCheckedMiddle = 12345 & kCheckedDomainMask;
constexpr uint32_t kCheckedAlpha[] = {0, 0, 1, 1, 1, kCheckedMiddle,
                                    kCheckedMiddle, kCheckedMiddle,
                                    kCheckedDomainMask, kCheckedDomainMask};
constexpr uint32_t kCheckedQuery[] = {0, 1, 0, 1, 2, kCheckedMiddle - 1,
                                    kCheckedMiddle, kCheckedMiddle + 1,
                                    kCheckedDomainMask - 1, kCheckedDomainMask};

inline void CheckShares(uint32_t alpha, uint32_t query, const uint8_t *first,
                        const uint8_t *second, const uint8_t *beta) {
#ifdef FSS070_BENCH_DPF
  const bool selected = query == alpha;
#else
  const bool selected = query < alpha;
#endif
  if ((first[15] | second[15]) & 0x80) {
    fprintf(stderr, "historical output MSB is not zero\n");
    exit(EXIT_FAILURE);
  }
  unsigned carry = 0;
  for (int i = 0; i < kLambda; ++i) {
#ifdef FSS070_UINT_GROUP
    unsigned sum = first[i] + second[i] + carry;
    carry = sum >> 8;
    uint8_t value = sum & 0xff;
#else
    uint8_t value = first[i] ^ second[i];
#endif
    if (i == kLambda - 1) value &= 0x7f;
    if (value != (selected ? beta[i] : 0)) {
      fprintf(stderr, "historical reconstruction failed for alpha=%u, query=%u\n",
              alpha, query);
      exit(EXIT_FAILURE);
    }
  }
}

struct CheckedBuffer {
  std::vector<uint8_t> storage;
  explicit CheckedBuffer(size_t size) : storage(size + 32, 0xa5) {}
  uint8_t *data() { return storage.data() + 16; }
  void Check() const {
    for (int i = 0; i < 16; ++i) {
      if (storage[i] != 0xa5 || storage[storage.size() - 16 + i] != 0xa5) {
        fprintf(stderr, "historical scratch or key buffer canary changed\n");
        exit(EXIT_FAILURE);
      }
    }
  }
};

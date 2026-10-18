// SPDX-License-Identifier: Apache-2.0
// Standalone binding factory acceptance. Build with nvcc and -lcrypto.

#include "binding_prg.cuh"
#include <barrier>
#include <cassert>
#include <cstdint>
#include <thread>
#include <vector>

namespace {

bool Equal(int4 a, int4 b) {
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

template <int mul>
void CheckFactory() {
  using Adapter = fss::python::ThreadLocalAes128Mmo<mul>;
  const auto &keys = Adapter::PublicKeys();
  const unsigned char *pointers[mul];
  for (int i = 0; i < mul; ++i) pointers[i] = keys[i].data();
  auto contexts = fss::prg::Aes128Mmo<mul>::CreateCtxs(pointers);
  fss::prg::Aes128Mmo<mul> direct(contexts);
  Adapter adapter;
  int4 seed{7, -11, 13, -17};
  auto reference = direct.Gen(seed);
  auto actual = adapter.Gen(seed);
  for (int i = 0; i < mul; ++i) assert(Equal(reference[i], actual[i]));
  auto *owner = &Adapter::ThreadContexts();
  assert(owner == &Adapter::ThreadContexts());
  fss::prg::Aes128Mmo<mul>::FreeCtxs(contexts);

  constexpr int kThreads = 4;
  std::barrier barrier(kThreads);
  uintptr_t owners[kThreads]{};
  std::vector<std::thread> workers;
  for (int thread = 0; thread < kThreads; ++thread) {
    workers.emplace_back([&, thread] {
      owners[thread] = reinterpret_cast<uintptr_t>(&Adapter::ThreadContexts());
      barrier.arrive_and_wait();
      for (int iteration = 0; iteration < 100; ++iteration) {
        auto result = adapter.Gen(seed);
        for (int i = 0; i < mul; ++i) assert(Equal(reference[i], result[i]));
      }
    });
  }
  for (auto &worker : workers) worker.join();
  for (int i = 0; i < kThreads; ++i) {
    for (int j = i + 1; j < kThreads; ++j) assert(owners[i] != owners[j]);
  }
  // Repeated short-lived workers exercise TLS destructors. Address/Leak
  // Sanitizer checks the corresponding EVP_CIPHER_CTX releases.
  for (int i = 0; i < 20; ++i) {
    std::thread([&] {
      auto result = adapter.Gen(seed);
      for (int j = 0; j < mul; ++j) assert(Equal(reference[j], result[j]));
    }).join();
  }
}

}  // namespace

int main() {
  CheckFactory<1>();
  CheckFactory<2>();
  CheckFactory<4>();
}

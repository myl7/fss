// SPDX-License-Identifier: Apache-2.0
#include "binding_common.cuh"
#include "grotto_binding_common.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  fss::python::BindGrotto<SchemeType, false>(m);
}

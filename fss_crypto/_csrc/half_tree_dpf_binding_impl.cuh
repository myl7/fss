// SPDX-License-Identifier: Apache-2.0
#include "binding_point_scheme.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  fss::python::detail::BindPointScheme(module);
}

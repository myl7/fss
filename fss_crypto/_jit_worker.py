"""Build or validate canonical extensions in a fresh Torch process."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

import torch
from torch.utils.cpp_extension import load

_RESULT_PREFIX = "FSS_ARTIFACT_VALIDATION="


def _torch_identity():
    return {"version": str(torch.__version__), "cuda": torch.version.cuda,
            "cxx11_abi": getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", None),
            "pybind_abi": [getattr(torch._C, name, None) for name in
                           ("_PYBIND11_COMPILER_TYPE", "_PYBIND11_STDLIB", "_PYBIND11_BUILD_ABI")]}


def _check_torch_identity(expected):
    if _torch_identity() != expected:
        raise RuntimeError("worker Torch identity differs from preparation; use the same toolchain")


def _validate(name, path, expected_torch):
    try:
        _check_torch_identity(expected_torch)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as error:
        traceback.print_exc()
        result = {"kind": "miss" if isinstance(error, (ImportError, OSError)) else "error",
                  "type": type(error).__name__, "module": type(error).__module__,
                  "args": error.args, "message": str(error)}
        print(_RESULT_PREFIX + json.dumps(result, default=repr), flush=True)
        return 1
    print(_RESULT_PREFIX + json.dumps({"kind": "ok"}), flush=True)
    return 0


def main():
    if sys.argv[1] == "--validate":
        return _validate(sys.argv[2], sys.argv[3], json.loads(sys.argv[4]))
    request = json.loads(Path(sys.argv[1]).read_text())
    _check_torch_identity(request.pop("torch_identity"))
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", request.pop("arch_list"))
    load(**request)
    return 0


if __name__ == "__main__":
    sys.exit(main())

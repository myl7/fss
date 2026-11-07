"""Python-layer argument forwarding for the sparse batch and split methods.

The native layer is replaced with a recorder, so these tests pin the wrapper
contract only; Core behavior lives in the integration test files.
"""

import pytest
import torch

from conftest import make_multi_scheme, make_point_scheme


class Recorder:
    METHODS = ("batch_eval", "batch_eval_merkle", "eval", "prove", "prove_merkle")

    def __init__(self, real, result=None):
        self._real = real
        self.result = result
        self.calls = []

    def __getattr__(self, name):
        if name in Recorder.METHODS:
            def call(*args):
                self.calls.append((name, args))
                return self.result
            return call
        return getattr(self._real, name)


@pytest.fixture(scope="module")
def vdpf():
    return make_point_scheme("Vdpf", bits=8)


@pytest.fixture(scope="module")
def vdmpf():
    return make_multi_scheme(True)


def vdpf_args(vdpf):
    s0 = torch.zeros((4,), dtype=torch.int32)
    cws = torch.zeros(vdpf.cws_shape, dtype=torch.uint8)
    cs = torch.zeros((4, 4), dtype=torch.int32)
    xs = torch.tensor([1, 2, 3], dtype=torch.uint8)
    ys = torch.empty((3, 4), dtype=torch.int32)
    pi = torch.empty((4, 4), dtype=torch.int32)
    return s0, cws, cs, xs, ys, pi


def vdmpf_args(vdmpf):
    sigma = torch.zeros((4,), dtype=torch.int32)
    s0s = torch.zeros((vdmpf.kBucketCount, 4), dtype=torch.int32)
    cws = torch.zeros(vdmpf.cws_shape, dtype=torch.uint8)
    cs = torch.zeros((vdmpf.kBucketCount, 4, 4), dtype=torch.int32)
    xs = torch.zeros(3, dtype=torch.int64)
    values = torch.zeros((3, 3, 4, 4), dtype=torch.int32)
    pi = torch.empty((4, 4), dtype=torch.int32)
    return sigma, s0s, cws, cs, xs, values, pi


class TestVdpfForwarding:
    def test_batch_eval_forwards_caller_buffers(self, vdpf, monkeypatch):
        s0, cws, cs, xs, ys, pi = vdpf_args(vdpf)
        native = Recorder(vdpf._native)
        monkeypatch.setattr(vdpf, "_native", native)
        assert vdpf.batch_eval(False, s0, cws, cs, xs, ys, pi) is None
        assert native.calls == [("batch_eval", (False, s0, cws, cs, xs, ys, pi))]

    def test_batch_eval_merkle_routes_threads_per_block(self, vdpf, monkeypatch):
        s0, cws, cs, xs, ys, pi = vdpf_args(vdpf)
        native = Recorder(vdpf._native)
        monkeypatch.setattr(vdpf, "_native", native)
        routed = []
        monkeypatch.setattr(vdpf, "_execution_native",
                            lambda output, **kwargs: routed.append(kwargs) or native)
        vdpf.batch_eval_merkle(True, s0, cws, cs, xs, ys, pi)
        vdpf.batch_eval_merkle(True, s0, cws, cs, xs, ys, pi,
                               threads_per_block=512)
        assert routed == [{"threads_per_block": 256}, {"threads_per_block": 512}]
        assert native.calls == [("batch_eval_merkle", (True, s0, cws, cs, xs, ys, pi))] * 2


class TestVdmpfForwarding:
    def test_eval_returns_the_native_proof_tensor(self, vdmpf, monkeypatch):
        sigma, s0s, cws, cs, _xs, _values, _pi = vdmpf_args(vdmpf)
        sentinel = torch.empty((3, 4, 4), dtype=torch.int32)
        native = Recorder(vdmpf._native, result=sentinel)
        monkeypatch.setattr(vdmpf, "_native", native)
        y = torch.empty((4,), dtype=torch.int32)
        assert vdmpf.eval(True, sigma, 1 << 40, s0s, cws, cs, 5, y) is sentinel
        assert native.calls == [("eval", (True, sigma, 1 << 40, s0s, cws, cs, 5, y))]

    @pytest.mark.parametrize("method", ["prove", "prove_merkle"])
    def test_prove_forwards_caller_buffers(self, vdmpf, monkeypatch, method):
        sigma, _s0s, _cws, cs, xs, values, pi = vdmpf_args(vdmpf)
        native = Recorder(vdmpf._native)
        monkeypatch.setattr(vdmpf, "_native", native)
        wrapper = getattr(vdmpf, method)
        assert wrapper(sigma, 1 << 40, cs, xs, values, pi) is None
        assert native.calls == [(method, (sigma, 1 << 40, cs, xs, values, pi))]

"""Current component-buffer integration coverage; execution is opt-in."""
import pytest
import torch
from conftest import make_point_scheme, generate_point, evaluate_all

@pytest.fixture
def dpf():
    return make_point_scheme('PackedHalfTreeDpf', bits=16, out_bits=8)

@pytest.fixture
def dpf_1bit():
    return make_point_scheme('PackedHalfTreeDpf', bits=16, out_bits=1)

@pytest.fixture
def s0s():
    return torch.randint(-2 ** 31, 2 ** 31, (2, 4), dtype=torch.int32)

class TestPackedHalfTreeDpfGenShape:

    def test_key_shapes(self, dpf, s0s):
        cws = generate_point(dpf, s0s, alpha=107, beta=171)
        assert cws.shape == dpf.cws_shape
        assert cws.dtype == torch.uint8

    def test_key_device_is_cpu(self, dpf, s0s):
        cws = generate_point(dpf, s0s, alpha=107, beta=171)
        assert cws.device.type == 'cpu'

    def test_rejects_beta_out_of_range(self, dpf, s0s):
        with pytest.raises(RuntimeError, match='packed output width'):
            generate_point(dpf, s0s, alpha=107, beta=256)

class TestPackedHalfTreeDpfEvalShape:

    def test_output_shape_cpu(self, dpf, s0s):
        cws = generate_point(dpf, s0s, alpha=107, beta=171)
        out = dpf.eval(b=0, s0=s0s[0], cws=cws, x=50)
        assert out.shape == (4,)
        assert out.dtype == torch.int32
        assert out.device.type == 'cpu'

    def test_rejects_out_of_domain_x(self, dpf, s0s):
        cws = generate_point(dpf, s0s, alpha=107, beta=171)
        with pytest.raises(RuntimeError, match='outside'):
            dpf.eval(b=0, s0=s0s[0], cws=cws, x=2 ** 16)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
    def test_output_shape_cuda(self, dpf, s0s):
        cws = generate_point(dpf, s0s, alpha=107, beta=171)
        out = dpf.eval(b=0, s0=s0s[0].to('cuda'), cws=cws.to('cuda'), x=50)
        assert out.shape == (4,)
        assert out.dtype == torch.int32
        assert out.device.type == 'cuda'

class TestPackedHalfTreeDpfEvalAllShape:

    def test_output_shape_out_bits_8(self, dpf, s0s):
        cws = generate_point(dpf, s0s, alpha=107, beta=171)
        out = evaluate_all(dpf, party=0, s0=s0s[0], cws=cws)
        assert out.shape == (2 ** 12, 4)
        assert out.dtype == torch.int32
        assert out.device.type == 'cpu'

    def test_output_shape_out_bits_1(self, dpf_1bit, s0s):
        cws = generate_point(dpf_1bit, s0s, alpha=107, beta=1)
        out = evaluate_all(dpf_1bit, party=0, s0=s0s[0], cws=cws)
        assert out.shape == (2 ** 9, 4)
        assert out.dtype == torch.int32
        assert out.device.type == 'cpu'

class TestPackedHalfTreeDpfReconstruction:

    def test_eval_xor_reconstruction(self, dpf, s0s):
        alpha, beta = (107, 171)
        cws = generate_point(dpf, s0s, alpha=alpha, beta=beta)
        block0 = dpf.eval(b=0, s0=s0s[0], cws=cws, x=alpha)
        block1 = dpf.eval(b=1, s0=s0s[1], cws=cws, x=alpha)
        diff = torch.bitwise_xor(block0, block1)
        assert dpf.extract(diff, alpha) == beta

    def test_eval_all_xor_reconstruction(self, dpf, s0s):
        alpha, beta = (107, 171)
        cws = generate_point(dpf, s0s, alpha=alpha, beta=beta)
        first = evaluate_all(dpf, party=0, s0=s0s[0], cws=cws)
        second = evaluate_all(dpf, party=1, s0=s0s[1], cws=cws)
        diff = torch.bitwise_xor(first, second)
        # One int4 block holds 16 consecutive 8-bit output lanes.
        block_index = alpha >> 4
        mask = torch.ones(diff.shape[0], dtype=torch.bool)
        mask[block_index] = False
        assert not torch.any(diff[mask])
        block = diff[block_index]
        for lane in range(dpf.kLanes):
            x = (block_index << 4) + lane
            assert dpf.extract(block, x) == (beta if x == alpha else 0)

class TestPackedHalfTreeDpfCuda:

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
    def test_eval_matches_cpu(self, dpf, s0s):
        cws = generate_point(dpf, s0s, alpha=107, beta=171)
        for party in (0, 1):
            cpu = dpf.eval(b=party, s0=s0s[party], cws=cws, x=107)
            gpu = dpf.eval(b=party, s0=s0s[party].to('cuda'), cws=cws.to('cuda'), x=107)
            assert gpu.device.type == 'cuda'
            assert torch.equal(cpu, gpu.cpu())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
    def test_eval_all_matches_cpu(self, dpf, s0s):
        cws = generate_point(dpf, s0s, alpha=107, beta=171)
        for party in (0, 1):
            cpu = evaluate_all(dpf, party=party, s0=s0s[party], cws=cws)
            gpu = evaluate_all(dpf, party=party, s0=s0s[party].to('cuda'), cws=cws.to('cuda'))
            assert gpu.device.type == 'cuda'
            assert torch.equal(cpu, gpu.cpu())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
    def test_above_24_bits_matches_cpu(self):
        dpf = make_point_scheme('PackedHalfTreeDpf', bits=25, out_bits=8)
        s0s = torch.randint(-2 ** 31, 2 ** 31, (2, 4), dtype=torch.int32)
        cws = generate_point(dpf, s0s, alpha=107, beta=171)
        for party in (0, 1):
            cpu = evaluate_all(dpf, party=party, s0=s0s[party], cws=cws, frontier_depth=17)
            gpu = evaluate_all(dpf, party=party, s0=s0s[party].to('cuda'), cws=cws.to('cuda'), frontier_depth=17)
            assert gpu.device.type == 'cuda'
            assert torch.equal(cpu, gpu.cpu())

class TestPackedHalfTreeDpfExtract:

    def test_lane_5_out_bits_1(self, dpf_1bit):
        # Input x selects a lane within the containing packed block.
        block = torch.tensor([1 << 5, 0, 0, 0], dtype=torch.int32)
        assert dpf_1bit.extract(block, x=5) == 1
        assert dpf_1bit.extract(block, x=128 + 5) == 1
        assert dpf_1bit.extract(block, x=6) == 0

    def test_lane_3_out_bits_8(self, dpf):
        # Wrap the high byte of the first word into signed int32 storage.
        word0 = (171 << 24) - 2 ** 32
        block = torch.tensor([word0, 0, 0, 0], dtype=torch.int32)
        assert dpf.extract(block, x=3) == 171
        assert dpf.extract(block, x=19) == 171
        assert dpf.extract(block, x=2) == 0

    def test_lane_1_out_bits_64(self):
        dpf64 = make_point_scheme('PackedHalfTreeDpf', bits=16, out_bits=64)
        block = torch.tensor([0, 0, 1432778632, 287454020], dtype=torch.int32)
        assert dpf64.extract(block, x=1) == 1234605616436508552
        assert dpf64.extract(block, x=0) == 0

    def test_rejects_bad_block(self, dpf):
        with pytest.raises(RuntimeError, match='dimension 4'):
            dpf.extract(torch.zeros(5, dtype=torch.int32), x=0)

    def test_rejects_out_of_domain_x(self, dpf):
        block = torch.zeros(4, dtype=torch.int32)
        with pytest.raises(RuntimeError, match='outside'):
            dpf.extract(block, x=2 ** 16)

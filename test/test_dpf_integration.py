"""Current component-buffer integration coverage; execution is opt-in."""
import pytest
import torch
from conftest import make_point_scheme, generate_point, evaluate_all

@pytest.fixture
def dpf():
    return make_point_scheme('Dpf', bits=16)

@pytest.fixture
def s0s():
    return torch.randint(-2 ** 31, 2 ** 31, (2, 4), dtype=torch.int32)

@pytest.fixture
def beta():
    return torch.tensor([0, 0, 0, 604], dtype=torch.int32)

class TestDpfGenShape:

    def test_cws_shape(self, dpf, s0s, beta):
        cws = generate_point(dpf, s0s, alpha=107, beta=beta)
        assert cws.shape == dpf.cws_shape
        assert cws.dtype == torch.uint8

    def test_cws_device_is_cpu(self, dpf, s0s, beta):
        cws = generate_point(dpf, s0s, alpha=107, beta=beta)
        assert cws.device.type == 'cpu'

class TestDpfEvalShape:

    def test_output_shape_cpu(self, dpf, s0s, beta):
        cws = generate_point(dpf, s0s, alpha=107, beta=beta)
        out = dpf.eval(b=0, s0=s0s[0], cws=cws, x=50)
        assert out.shape == (4,)
        assert out.dtype == torch.int32
        assert out.device.type == 'cpu'

    def test_rejects_out_of_domain_x(self, dpf, s0s, beta):
        cws = generate_point(dpf, s0s, alpha=107, beta=beta)
        with pytest.raises(RuntimeError, match='outside'):
            dpf.eval(b=0, s0=s0s[0], cws=cws, x=2 ** 16)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
    def test_output_shape_cuda(self, dpf, s0s, beta):
        cws = generate_point(dpf, s0s, alpha=107, beta=beta)
        out = dpf.eval(b=0, s0=s0s[0].to('cuda'), cws=cws.to('cuda'), x=50)
        assert out.shape == (4,)
        assert out.dtype == torch.int32
        assert out.device.type == 'cuda'

class TestDpfEvalAllShape:

    def test_output_shape(self, dpf, s0s, beta):
        cws = generate_point(dpf, s0s, alpha=107, beta=beta)
        out = evaluate_all(dpf, party=0, s0=s0s[0], cws=cws)
        assert out.shape == (2 ** 16, 4)
        assert out.dtype == torch.int32
        assert out.device.type == 'cpu'

class TestDpfEvalAllCuda:

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
    def test_matches_cpu_cuda(self, dpf, s0s, beta):
        cws = generate_point(dpf, s0s, alpha=107, beta=beta)
        for party in (0, 1):
            cpu = evaluate_all(dpf, party=party, s0=s0s[party], cws=cws)
            gpu = evaluate_all(dpf, party=party, s0=s0s[party].to('cuda'), cws=cws.to('cuda'))
            assert gpu.device.type == 'cuda'
            assert torch.equal(cpu, gpu.cpu())
        first = evaluate_all(dpf, party=0, s0=s0s[0].to('cuda'), cws=cws.to('cuda')).cpu()
        second = evaluate_all(dpf, party=1, s0=s0s[1].to('cuda'), cws=cws.to('cuda')).cpu()
        diff = torch.bitwise_xor(first, second)
        assert torch.equal(diff[107], beta)
        mask = torch.ones(diff.shape[0], dtype=torch.bool)
        mask[107] = False
        assert not torch.any(diff[mask])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
    def test_above_24_bits_matches_cpu(self):
        dpf = make_point_scheme('Dpf', bits=25)
        s0s = torch.randint(-2 ** 31, 2 ** 31, (2, 4), dtype=torch.int32)
        beta = torch.tensor([0, 0, 0, 604], dtype=torch.int32)
        cws = generate_point(dpf, s0s, alpha=107, beta=beta)
        for party in (0, 1):
            cpu = evaluate_all(dpf, party=party, s0=s0s[party], cws=cws, frontier_depth=17)
            gpu = evaluate_all(dpf, party=party, s0=s0s[party].to('cuda'), cws=cws.to('cuda'), frontier_depth=17)
            assert gpu.device.type == 'cuda'
            assert torch.equal(cpu, gpu.cpu())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
    def test_gpu_batch_matches_cpu_for_one_and_two_keys(self, dpf, s0s, beta):
        # A one-key batch stays three-dimensional; its key-major cws block must
        # not be mistaken for a single-key two-dimensional tensor.
        count = 1 << dpf.in_bits
        for keys in (1, 2):
            stacked_s0s = s0s[:keys].to('cuda')
            stacked_cws = generate_point(dpf, s0s, alpha=107, beta=beta) \
                .unsqueeze(0).repeat(keys, 1, 1).to('cuda')
            ys = torch.empty((keys, count, 4), dtype=torch.int32, device='cuda')
            dpf.eval_all_gpu_batch(0, stacked_s0s, stacked_cws, ys)
            for k in range(keys):
                cpu = evaluate_all(dpf, party=0, s0=s0s[k], cws=stacked_cws[k].cpu())
                assert torch.equal(ys[k].cpu(), cpu)

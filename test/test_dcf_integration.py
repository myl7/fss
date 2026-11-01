"""Current component-buffer integration coverage; execution is opt-in."""
import pytest
import torch
import fss_crypto
from conftest import make_point_scheme, generate_point, evaluate_all

@pytest.fixture
def dcf():
    return make_point_scheme('Dcf', bits=16, pred=fss_crypto.DcfPred.LT)

@pytest.fixture
def s0s():
    return torch.randint(-2 ** 31, 2 ** 31, (2, 4), dtype=torch.int32)

@pytest.fixture
def beta():
    return torch.tensor([0, 0, 0, 604], dtype=torch.int32)

class TestDcfGenShape:

    def test_cws_shape(self, dcf, s0s, beta):
        cws = generate_point(dcf, s0s, alpha=107, beta=beta)
        assert cws.shape == dcf.cws_shape
        assert cws.dtype == torch.uint8

    def test_cws_device_is_cpu(self, dcf, s0s, beta):
        cws = generate_point(dcf, s0s, alpha=107, beta=beta)
        assert cws.device.type == 'cpu'

class TestDcfEvalShape:

    def test_output_shape_cpu(self, dcf, s0s, beta):
        cws = generate_point(dcf, s0s, alpha=107, beta=beta)
        out = dcf.eval(b=0, s0=s0s[0], cws=cws, x=50)
        assert out.shape == (4,)
        assert out.dtype == torch.int32
        assert out.device.type == 'cpu'

    def test_rejects_out_of_domain_x(self, dcf, s0s, beta):
        cws = generate_point(dcf, s0s, alpha=107, beta=beta)
        with pytest.raises(RuntimeError, match='outside'):
            dcf.eval(b=0, s0=s0s[0], cws=cws, x=2 ** 16)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
    def test_output_shape_cuda(self, dcf, s0s, beta):
        cws = generate_point(dcf, s0s, alpha=107, beta=beta)
        out = dcf.eval(b=0, s0=s0s[0].to('cuda'), cws=cws.to('cuda'), x=50)
        assert out.shape == (4,)
        assert out.dtype == torch.int32
        assert out.device.type == 'cuda'

class TestDcfEvalAllShape:

    def test_output_shape(self, dcf, s0s, beta):
        cws = generate_point(dcf, s0s, alpha=107, beta=beta)
        out = evaluate_all(dcf, party=0, s0=s0s[0], cws=cws)
        assert out.shape == (2 ** 16, 4)
        assert out.dtype == torch.int32
        assert out.device.type == 'cpu'

class TestDcfPredVariant:

    def test_gt_eval_all_semantics_cpu(self, s0s, beta):
        dcf_gt = make_point_scheme('Dcf', bits=16, pred=fss_crypto.DcfPred.GT)
        cws = generate_point(dcf_gt, s0s, alpha=107, beta=beta)
        assert cws.shape == dcf_gt.cws_shape
        first = evaluate_all(dcf_gt, party=0, s0=s0s[0], cws=cws)
        second = evaluate_all(dcf_gt, party=1, s0=s0s[1], cws=cws)
        diff = torch.bitwise_xor(first, second)
        assert not torch.any(diff[:108])
        assert torch.equal(diff[108:], beta.expand(2 ** 16 - 108, 4))

    def test_gt_alpha_zero(self, s0s, beta):
        dcf_gt = make_point_scheme('Dcf', bits=16, pred=fss_crypto.DcfPred.GT)
        cws = generate_point(dcf_gt, s0s, alpha=0, beta=beta)
        first = evaluate_all(dcf_gt, party=0, s0=s0s[0], cws=cws)
        second = evaluate_all(dcf_gt, party=1, s0=s0s[1], cws=cws)
        diff = torch.bitwise_xor(first, second)
        assert not torch.any(diff[:1])
        assert torch.equal(diff[1:], beta.expand(2 ** 16 - 1, 4))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
    def test_gt_eval_all_matches_cpu_cuda(self, s0s, beta):
        dcf_gt = make_point_scheme('Dcf', bits=16, pred=fss_crypto.DcfPred.GT)
        cws = generate_point(dcf_gt, s0s, alpha=107, beta=beta)
        for party in (0, 1):
            cpu = evaluate_all(dcf_gt, party=party, s0=s0s[party], cws=cws)
            gpu = evaluate_all(dcf_gt, party=party, s0=s0s[party].to('cuda'), cws=cws.to('cuda'))
            assert gpu.device.type == 'cuda'
            assert torch.equal(cpu, gpu.cpu())

class TestDcfEvalAllCuda:

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
    def test_matches_cpu_cuda(self, dcf, s0s, beta):
        cws = generate_point(dcf, s0s, alpha=107, beta=beta)
        for party in (0, 1):
            cpu = evaluate_all(dcf, party=party, s0=s0s[party], cws=cws)
            gpu = evaluate_all(dcf, party=party, s0=s0s[party].to('cuda'), cws=cws.to('cuda'))
            assert gpu.device.type == 'cuda'
            assert torch.equal(cpu, gpu.cpu())
        first = evaluate_all(dcf, party=0, s0=s0s[0].to('cuda'), cws=cws.to('cuda')).cpu()
        second = evaluate_all(dcf, party=1, s0=s0s[1].to('cuda'), cws=cws.to('cuda')).cpu()
        diff = torch.bitwise_xor(first, second)
        assert torch.equal(diff[:107], beta.expand(107, 4))
        assert not torch.any(diff[107:])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
    def test_above_24_bits_matches_cpu(self):
        dcf = make_point_scheme('Dcf', bits=25)
        s0s = torch.randint(-2 ** 31, 2 ** 31, (2, 4), dtype=torch.int32)
        beta = torch.tensor([0, 0, 0, 604], dtype=torch.int32)
        cws = generate_point(dcf, s0s, alpha=107, beta=beta)
        for party in (0, 1):
            cpu = evaluate_all(dcf, party=party, s0=s0s[party], cws=cws, frontier_depth=17)
            gpu = evaluate_all(dcf, party=party, s0=s0s[party].to('cuda'), cws=cws.to('cuda'), frontier_depth=17)
            assert gpu.device.type == 'cuda'
            assert torch.equal(cpu, gpu.cpu())

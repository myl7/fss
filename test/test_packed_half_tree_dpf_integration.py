import pytest
import torch
import fss_crypto


@pytest.fixture
def dpf():
    return fss_crypto.PackedHalfTreeDpf(in_bits=16, out_bits=8, prg="chacha")


@pytest.fixture
def dpf_1bit():
    return fss_crypto.PackedHalfTreeDpf(in_bits=16, out_bits=1, prg="chacha")


@pytest.fixture
def s0s():
    return torch.randint(-2**31, 2**31, (2, 4), dtype=torch.int32)


class TestPackedHalfTreeDpfGenShape:
    def test_key_shapes(self, dpf, s0s):
        cws, fcw = dpf.gen(s0s, alpha=107, beta=171)
        assert cws.shape == (12, 4)  # depth = 16 - log2(128 // 8)
        assert cws.dtype == torch.int32
        assert fcw.shape == (4,)
        assert fcw.dtype == torch.int32

    def test_key_device_is_cpu(self, dpf, s0s):
        cws, fcw = dpf.gen(s0s, alpha=107, beta=171)
        assert cws.device.type == "cpu"
        assert fcw.device.type == "cpu"

    def test_rejects_beta_out_of_range(self, dpf, s0s):
        with pytest.raises(ValueError, match="beta must be"):
            dpf.gen(s0s, alpha=107, beta=256)


class TestPackedHalfTreeDpfEvalShape:
    def test_output_shape_cpu(self, dpf, s0s):
        cws, fcw = dpf.gen(s0s, alpha=107, beta=171)
        out = dpf.eval(party=0, s0=s0s[0], cws=cws, fcw=fcw, x=50)
        assert out.shape == (4,)
        assert out.dtype == torch.int32
        assert out.device.type == "cpu"

    def test_rejects_out_of_domain_x(self, dpf, s0s):
        cws, fcw = dpf.gen(s0s, alpha=107, beta=171)
        with pytest.raises(ValueError, match="x must be"):
            dpf.eval(party=0, s0=s0s[0], cws=cws, fcw=fcw, x=2**16)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_output_shape_cuda(self, dpf, s0s):
        cws, fcw = dpf.gen(s0s, alpha=107, beta=171)
        out = dpf.eval(
            party=0,
            s0=s0s[0].to("cuda"),
            cws=cws.to("cuda"),
            fcw=fcw.to("cuda"),
            x=50,
        )
        assert out.shape == (4,)
        assert out.dtype == torch.int32
        assert out.device.type == "cuda"


class TestPackedHalfTreeDpfEvalAllShape:
    def test_output_shape_out_bits_8(self, dpf, s0s):
        cws, fcw = dpf.gen(s0s, alpha=107, beta=171)
        out = dpf.eval_all(party=0, s0=s0s[0], cws=cws, fcw=fcw)
        assert out.shape == (2**12, 4)
        assert out.dtype == torch.int32
        assert out.device.type == "cpu"

    def test_output_shape_out_bits_1(self, dpf_1bit, s0s):
        cws, fcw = dpf_1bit.gen(s0s, alpha=107, beta=1)
        out = dpf_1bit.eval_all(party=0, s0=s0s[0], cws=cws, fcw=fcw)
        assert out.shape == (2**9, 4)
        assert out.dtype == torch.int32
        assert out.device.type == "cpu"


class TestPackedHalfTreeDpfReconstruction:
    def test_eval_xor_reconstruction(self, dpf, s0s):
        alpha, beta = 107, 171
        cws, fcw = dpf.gen(s0s, alpha=alpha, beta=beta)
        block0 = dpf.eval(party=0, s0=s0s[0], cws=cws, fcw=fcw, x=alpha)
        block1 = dpf.eval(party=1, s0=s0s[1], cws=cws, fcw=fcw, x=alpha)
        diff = torch.bitwise_xor(block0, block1)
        assert dpf.extract(diff, alpha) == beta

    def test_eval_all_xor_reconstruction(self, dpf, s0s):
        alpha, beta = 107, 171
        cws, fcw = dpf.gen(s0s, alpha=alpha, beta=beta)
        first = dpf.eval_all(party=0, s0=s0s[0], cws=cws, fcw=fcw)
        second = dpf.eval_all(party=1, s0=s0s[1], cws=cws, fcw=fcw)
        diff = torch.bitwise_xor(first, second)
        # Beta at alpha's lane, zero at the other lanes of the alpha block,
        # and every other block zero.
        block_index = alpha >> 4  # log2(128 // 8) lane bits
        mask = torch.ones(diff.shape[0], dtype=torch.bool)
        mask[block_index] = False
        assert not torch.any(diff[mask])
        block = diff[block_index]
        for lane in range(dpf.lanes):
            x = (block_index << 4) + lane
            assert dpf.extract(block, x) == (beta if x == alpha else 0)


class TestPackedHalfTreeDpfCuda:

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_eval_matches_cpu(self, dpf, s0s):
        cws, fcw = dpf.gen(s0s, alpha=107, beta=171)
        for party in (0, 1):
            cpu = dpf.eval(party=party, s0=s0s[party], cws=cws, fcw=fcw, x=107)
            gpu = dpf.eval(
                party=party,
                s0=s0s[party].to("cuda"),
                cws=cws.to("cuda"),
                fcw=fcw.to("cuda"),
                x=107,
            )
            assert gpu.device.type == "cuda"
            assert torch.equal(cpu, gpu.cpu())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_eval_all_matches_cpu(self, dpf, s0s):
        cws, fcw = dpf.gen(s0s, alpha=107, beta=171)
        for party in (0, 1):
            cpu = dpf.eval_all(party=party, s0=s0s[party], cws=cws, fcw=fcw)
            gpu = dpf.eval_all(
                party=party,
                s0=s0s[party].to("cuda"),
                cws=cws.to("cuda"),
                fcw=fcw.to("cuda"),
            )
            assert gpu.device.type == "cuda"
            assert torch.equal(cpu, gpu.cpu())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_above_24_bits_matches_cpu(self):
        dpf = fss_crypto.PackedHalfTreeDpf(in_bits=25, out_bits=8, prg="chacha")
        s0s = torch.randint(-2**31, 2**31, (2, 4), dtype=torch.int32)
        cws, fcw = dpf.gen(s0s, alpha=107, beta=171)
        for party in (0, 1):
            cpu = dpf.eval_all(party=party, s0=s0s[party], cws=cws, fcw=fcw)
            gpu = dpf.eval_all(
                party=party,
                s0=s0s[party].to("cuda"),
                cws=cws.to("cuda"),
                fcw=fcw.to("cuda"),
            )
            assert gpu.device.type == "cuda"
            assert torch.equal(cpu, gpu.cpu())


class TestPackedHalfTreeDpfExtract:
    def test_lane_5_out_bits_1(self, dpf_1bit):
        # Lane 5 of 128 lives at bit 5 of the low word.
        block = torch.tensor([1 << 5, 0, 0, 0], dtype=torch.int32)
        assert dpf_1bit.extract(block, x=5) == 1
        assert dpf_1bit.extract(block, x=128 + 5) == 1
        assert dpf_1bit.extract(block, x=6) == 0

    def test_lane_3_out_bits_8(self, dpf):
        # Lane 3 of 16: byte lane 3 lives at bits 24..31 of the low word.
        word0 = (0xAB << 24) - 2**32  # 0xAB000000 wrapped to int32
        block = torch.tensor([word0, 0, 0, 0], dtype=torch.int32)
        assert dpf.extract(block, x=3) == 0xAB
        assert dpf.extract(block, x=19) == 0xAB  # same lane, later block
        assert dpf.extract(block, x=2) == 0

    def test_lane_1_out_bits_64(self):
        dpf64 = fss_crypto.PackedHalfTreeDpf(in_bits=16, out_bits=64)
        # Lane 1 of 2 occupies the high 64 bits of the 128-bit block: the
        # little-endian 64-bit read of words 2,3 is 0x1122334455667788.
        block = torch.tensor([0, 0, 0x55667788, 0x11223344], dtype=torch.int32)
        assert dpf64.extract(block, x=1) == 0x1122334455667788
        assert dpf64.extract(block, x=0) == 0

    def test_rejects_bad_block(self, dpf):
        with pytest.raises(TypeError, match="block must be"):
            dpf.extract(torch.zeros(5, dtype=torch.int32), x=0)

    def test_rejects_out_of_domain_x(self, dpf):
        block = torch.zeros(4, dtype=torch.int32)
        with pytest.raises(ValueError, match="x must be"):
            dpf.extract(block, x=2**16)

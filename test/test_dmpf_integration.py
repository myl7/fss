import pytest
import torch
import fss_crypto


IN_BITS = 12
MAX_POINTS = 30
# Worst-case bucket size is ceil(2^12 * 3 / 53) = 232, so bucket_bits >= 8
# passes validate_bucket_bits; use 10.
BUCKET_BITS = 10
# Cuckoo bucket count: ceil(30 * (80 + 130 + log2(30)) / 123.5).
M = 53
T = 30


@pytest.fixture
def dmpf():
    return fss_crypto.Dmpf(in_bits=IN_BITS, max_points=MAX_POINTS,
                           bucket_bits=BUCKET_BITS, group="bytes", prg="chacha")


@pytest.fixture
def s0s():
    return torch.randint(-2**31, 2**31, (M, 2, 4), dtype=torch.int32)


@pytest.fixture
def sigma():
    return torch.randint(-2**31, 2**31, (4,), dtype=torch.int32)


@pytest.fixture
def alphas():
    return torch.randperm(2**IN_BITS, dtype=torch.int64)[:T]


@pytest.fixture
def betas():
    betas = torch.randint(-2**31, 2**31, (T, 4), dtype=torch.int32)
    betas[:, 3] &= ~1  # the scheme clamps the last word's LSB (lambda = 127)
    return betas


@pytest.fixture
def key(dmpf, s0s, sigma, alphas, betas):
    return dmpf.gen(s0s, sigma, alphas, betas, t=T)


class TestDmpfParams:
    def test_m_and_bucket_bits(self, dmpf):
        assert dmpf.m == M
        assert dmpf.bucket_bits == BUCKET_BITS

    def test_rejects_in_bits_above_63(self):
        with pytest.raises(ValueError, match="in_bits must be at most 63"):
            fss_crypto.Dmpf(in_bits=64, max_points=MAX_POINTS, bucket_bits=13)

    def test_rejects_too_small_bucket_bits(self):
        with pytest.raises(ValueError, match="bucket_bits"):
            fss_crypto.Dmpf(in_bits=IN_BITS, max_points=MAX_POINTS, bucket_bits=5)

    def test_rejects_small_max_points(self):
        with pytest.raises(ValueError, match="max_points must be at least 30"):
            fss_crypto.Dmpf(in_bits=IN_BITS, max_points=29, bucket_bits=BUCKET_BITS)


class TestDmpfGenShape:
    def test_output_shapes(self, key):
        assert key.cws.shape == (M, BUCKET_BITS + 1, 8)
        assert key.cws.dtype == torch.int32
        assert key.s0s.shape == (2, M, 4)
        assert key.s0s.dtype == torch.int32
        assert key.sigma.shape == (4,)
        assert key.sigma.dtype == torch.int32
        assert key.meta.shape == (2,)
        assert key.meta.dtype == torch.int32

    def test_output_device_is_cpu(self, key):
        for tensor in (key.cws, key.s0s, key.sigma, key.meta):
            assert tensor.device.type == "cpu"

    def test_meta_holds_runtime_params(self, key):
        # t = 30 uses the full bucket count; b_size = ceil(2^12 * 3 / 53).
        assert key.meta.tolist() == [M, (2**IN_BITS * 3 + M - 1) // M]


class TestDmpfBatchEval:
    def test_output_shape(self, dmpf, key, alphas):
        ys0 = dmpf.batch_eval(0, key, alphas)
        ys1 = dmpf.batch_eval(1, key, alphas)
        for ys in (ys0, ys1):
            assert ys.shape == (T, 4)
            assert ys.dtype == torch.int32
            assert ys.device.type == "cpu"

    def test_reconstructs_betas_at_alphas(self, dmpf, key, alphas, betas):
        ys0 = dmpf.batch_eval(0, key, alphas)
        ys1 = dmpf.batch_eval(1, key, alphas)
        # XOR reconstruction (bytes group): beta at each alpha.
        assert torch.equal(torch.bitwise_xor(ys0, ys1), betas)

    def test_zero_away_from_alphas(self, dmpf, key, alphas):
        mask = torch.ones(2**IN_BITS, dtype=torch.bool)
        mask[alphas] = False
        xs = torch.arange(2**IN_BITS, dtype=torch.int64)[mask][:128]
        ys0 = dmpf.batch_eval(0, key, xs)
        ys1 = dmpf.batch_eval(1, key, xs)
        assert not torch.any(torch.bitwise_xor(ys0, ys1))


class TestDmpfEvalAll:
    def test_output_shape(self, dmpf, key):
        ys = dmpf.eval_all(0, key)
        assert ys.shape == (2**IN_BITS, 4)
        assert ys.dtype == torch.int32
        assert ys.device.type == "cpu"

    def test_reconstructs_full_domain(self, dmpf, key, alphas, betas):
        ys0 = dmpf.eval_all(0, key)
        ys1 = dmpf.eval_all(1, key)
        total = torch.bitwise_xor(ys0, ys1)
        expected = torch.zeros(2**IN_BITS, 4, dtype=torch.int32)
        expected[alphas] = betas
        assert torch.equal(total, expected)


class TestDmpfValidation:
    def test_rejects_bad_s0s_shape(self, dmpf, sigma, alphas, betas):
        s0s = torch.zeros((M, 2, 4), dtype=torch.int64)
        with pytest.raises(TypeError, match="s0s must be a"):
            dmpf.gen(s0s, sigma, alphas, betas, t=T)

    def test_rejects_bad_sigma_shape(self, dmpf, s0s, alphas, betas):
        with pytest.raises(TypeError, match="sigma must be a"):
            dmpf.gen(s0s, torch.zeros((5,), dtype=torch.int32), alphas, betas, t=T)

    def test_rejects_t_below_30(self, dmpf, s0s, sigma, alphas, betas):
        with pytest.raises(ValueError, match="t must be between"):
            dmpf.gen(s0s, sigma, alphas[:29], betas[:29], t=29)

    def test_rejects_t_above_max_points(self, dmpf, s0s, sigma):
        alphas = torch.randperm(2**IN_BITS, dtype=torch.int64)[:MAX_POINTS + 1]
        betas = torch.zeros((MAX_POINTS + 1, 4), dtype=torch.int32)
        with pytest.raises(ValueError, match="t must be between"):
            dmpf.gen(s0s, sigma, alphas, betas, t=MAX_POINTS + 1)

    def test_rejects_alpha_out_of_domain(self, dmpf, s0s, sigma, alphas, betas):
        alphas = alphas.clone()
        alphas[0] = 2**IN_BITS
        with pytest.raises(ValueError, match="alphas must be in"):
            dmpf.gen(s0s, sigma, alphas, betas, t=T)

    def test_rejects_duplicate_alphas(self, dmpf, s0s, sigma, alphas, betas):
        alphas = alphas.clone()
        alphas[1] = alphas[0]
        with pytest.raises(ValueError, match="distinct"):
            dmpf.gen(s0s, sigma, alphas, betas, t=T)

    def test_rejects_bad_xs_dtype(self, dmpf, key, alphas):
        with pytest.raises(TypeError, match="xs must be a 1-D int64"):
            dmpf.batch_eval(0, key, alphas.to(torch.int32))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_rejects_cuda_tensors(self, dmpf, s0s, sigma, alphas, betas):
        with pytest.raises(RuntimeError, match="on cpu"):
            dmpf.gen(s0s.to("cuda"), sigma, alphas, betas, t=T)

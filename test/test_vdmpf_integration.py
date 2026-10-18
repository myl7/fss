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
def vdmpf():
    return fss_crypto.Vdmpf(in_bits=IN_BITS, max_points=MAX_POINTS,
                            bucket_bits=BUCKET_BITS, group="bytes",
                            prg="chacha", xor_hash="blake3", prp="aes128_feistel")


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
def key(vdmpf, s0s, sigma, alphas, betas):
    return vdmpf.gen(s0s, sigma, alphas, betas, t=T)


class TestVdmpfParams:
    def test_m_and_bucket_bits(self, vdmpf):
        assert vdmpf.m == M
        assert vdmpf.bucket_bits == BUCKET_BITS

    def test_rejects_in_bits_above_63(self):
        with pytest.raises(ValueError, match="in_bits must be at most 63"):
            fss_crypto.Vdmpf(in_bits=64, max_points=MAX_POINTS, bucket_bits=13)

    def test_rejects_bad_hash(self):
        with pytest.raises(ValueError, match="hash must be one of"):
            fss_crypto.Vdmpf(in_bits=IN_BITS, max_points=MAX_POINTS,
                             bucket_bits=BUCKET_BITS, xor_hash="md5")


class TestVdmpfGenShape:
    def test_output_shapes(self, key):
        assert key.cws.shape == (M, BUCKET_BITS, 8)
        assert key.cws.dtype == torch.int32
        assert key.cs.shape == (M, 4, 4)
        assert key.cs.dtype == torch.int32
        assert key.ocw.shape == (M, 4)
        assert key.ocw.dtype == torch.int32
        assert key.s0s.shape == (2, M, 4)
        assert key.s0s.dtype == torch.int32
        assert key.sigma.shape == (4,)
        assert key.sigma.dtype == torch.int32
        assert key.meta.shape == (2,)
        assert key.meta.dtype == torch.int32

    def test_output_device_is_cpu(self, key):
        for tensor in (key.cws, key.cs, key.ocw, key.s0s, key.sigma, key.meta):
            assert tensor.device.type == "cpu"

    def test_meta_holds_runtime_params(self, key):
        # t = 30 uses the full bucket count; b_size = ceil(2^12 * 3 / 53).
        assert key.meta.tolist() == [M, (2**IN_BITS * 3 + M - 1) // M]


class TestVdmpfBatchEval:
    def test_output_shape(self, vdmpf, key, alphas):
        ys0, pi0 = vdmpf.batch_eval(0, key, alphas)
        ys1, pi1 = vdmpf.batch_eval(1, key, alphas)
        for ys in (ys0, ys1):
            assert ys.shape == (T, 4)
            assert ys.dtype == torch.int32
            assert ys.device.type == "cpu"
        for pi in (pi0, pi1):
            assert pi.shape == (4, 4)
            assert pi.dtype == torch.int32
            assert pi.device.type == "cpu"

    def test_reconstructs_betas_at_alphas(self, vdmpf, key, alphas, betas):
        ys0, _ = vdmpf.batch_eval(0, key, alphas)
        ys1, _ = vdmpf.batch_eval(1, key, alphas)
        # XOR reconstruction (bytes group): beta at each alpha.
        assert torch.equal(torch.bitwise_xor(ys0, ys1), betas)

    def test_zero_away_from_alphas(self, vdmpf, key, alphas):
        mask = torch.ones(2**IN_BITS, dtype=torch.bool)
        mask[alphas] = False
        xs = torch.arange(2**IN_BITS, dtype=torch.int64)[mask][:128]
        ys0, _ = vdmpf.batch_eval(0, key, xs)
        ys1, _ = vdmpf.batch_eval(1, key, xs)
        assert not torch.any(torch.bitwise_xor(ys0, ys1))


class TestVdmpfVerify:
    def test_honest_proofs_verify(self, vdmpf, key, alphas):
        _, pi0 = vdmpf.batch_eval(0, key, alphas)
        _, pi1 = vdmpf.batch_eval(1, key, alphas)
        assert vdmpf.verify(pi0, pi1) is True

    def test_rejects_tampered_proof(self, vdmpf, key, alphas):
        _, pi0 = vdmpf.batch_eval(0, key, alphas)
        _, pi1 = vdmpf.batch_eval(1, key, alphas)
        pi1 = pi1.clone()
        pi1[0, 0] ^= 1
        assert vdmpf.verify(pi0, pi1) is False


class TestVdmpfValidation:
    def test_rejects_bad_s0s_shape(self, vdmpf, sigma, alphas, betas):
        s0s = torch.zeros((M, 2, 8), dtype=torch.int32)
        with pytest.raises(TypeError, match="s0s must be a"):
            vdmpf.gen(s0s, sigma, alphas, betas, t=T)

    def test_rejects_bad_betas_shape(self, vdmpf, s0s, sigma, alphas):
        with pytest.raises(TypeError, match="betas must be a"):
            vdmpf.gen(s0s, sigma, alphas, torch.zeros((T, 5), dtype=torch.int32), t=T)

    def test_rejects_t_below_30(self, vdmpf, s0s, sigma, alphas, betas):
        with pytest.raises(ValueError, match="t must be between"):
            vdmpf.gen(s0s, sigma, alphas[:29], betas[:29], t=29)

    def test_rejects_alpha_out_of_domain(self, vdmpf, s0s, sigma, alphas, betas):
        alphas = alphas.clone()
        alphas[0] = -1
        with pytest.raises(ValueError, match="alphas must be in"):
            vdmpf.gen(s0s, sigma, alphas, betas, t=T)

    def test_rejects_bad_key_cs(self, vdmpf, key, alphas):
        bad = fss_crypto.VdmpfKey(cws=key.cws,
                                  cs=torch.zeros((M, 4), dtype=torch.int32),
                                  ocw=key.ocw, s0s=key.s0s, sigma=key.sigma,
                                  meta=key.meta)
        with pytest.raises(TypeError, match="cs must be a"):
            vdmpf.batch_eval(0, bad, alphas)

    def test_rejects_bad_proof_shape(self, vdmpf, key, alphas):
        _, pi0 = vdmpf.batch_eval(0, key, alphas)
        _, pi1 = vdmpf.batch_eval(1, key, alphas)
        with pytest.raises(TypeError, match="pi1 must be a"):
            vdmpf.verify(pi0, pi1.flatten())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_rejects_mixed_devices(self, vdmpf, key, alphas):
        with pytest.raises(RuntimeError, match="same device"):
            vdmpf.batch_eval(0, key, alphas.to("cuda"))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_rejects_cuda_tensors_with_default_prp(self, vdmpf, key, alphas):
        with pytest.raises(RuntimeError, match="chacha_feistel"):
            vdmpf.batch_eval(0, key.to("cuda"), alphas.to("cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestVdmpfGpu:
    @pytest.fixture
    def gpu_vdmpf(self):
        return fss_crypto.Vdmpf(in_bits=IN_BITS, max_points=MAX_POINTS,
                                bucket_bits=BUCKET_BITS, group="bytes",
                                prg="chacha", xor_hash="blake3",
                                prp="chacha_feistel")

    @pytest.fixture
    def key_cpu(self, gpu_vdmpf, s0s, sigma, alphas, betas):
        return gpu_vdmpf.gen(s0s, sigma, alphas, betas, t=T)

    @pytest.fixture
    def key_gpu(self, key_cpu):
        return key_cpu.to("cuda")

    def test_batch_eval_matches_cpu_outputs(self, gpu_vdmpf, key_cpu, key_gpu, alphas):
        ys_cpu, _ = gpu_vdmpf.batch_eval(0, key_cpu, alphas)
        ys_gpu, pi_gpu = gpu_vdmpf.batch_eval(0, key_gpu, alphas.to("cuda"))
        assert ys_gpu.device.type == "cuda"
        assert pi_gpu.device.type == "cuda"
        assert torch.equal(ys_gpu.cpu(), ys_cpu)

    def test_batch_eval_reconstructs_betas(self, gpu_vdmpf, key_gpu, alphas, betas):
        ys0, _ = gpu_vdmpf.batch_eval(0, key_gpu, alphas.to("cuda"))
        ys1, _ = gpu_vdmpf.batch_eval(1, key_gpu, alphas.to("cuda"))
        assert torch.equal(torch.bitwise_xor(ys0.cpu(), ys1.cpu()), betas)

    def test_merkle_fold_proofs_verify_on_device(self, gpu_vdmpf, key_gpu, alphas):
        _, pi0 = gpu_vdmpf.batch_eval(0, key_gpu, alphas.to("cuda"))
        _, pi1 = gpu_vdmpf.batch_eval(1, key_gpu, alphas.to("cuda"))
        assert gpu_vdmpf.verify(pi0, pi1) is True
        tampered = pi1.clone()
        tampered[0, 0] ^= 1
        assert gpu_vdmpf.verify(pi0, tampered) is False

    def test_accumulators_are_device_local(self, gpu_vdmpf, key_cpu, key_gpu, alphas):
        # CPU runs the sequential chains, CUDA the Merkle fold: identical
        # outputs but proofs that must not be mixed across devices.
        _, pi0_cpu = gpu_vdmpf.batch_eval(0, key_cpu, alphas)
        _, pi1_gpu = gpu_vdmpf.batch_eval(1, key_gpu, alphas.to("cuda"))
        assert gpu_vdmpf.verify(pi0_cpu, pi1_gpu.cpu()) is False

import pytest
import torch
import fss_crypto


@pytest.fixture
def vdpf():
    return fss_crypto.Vdpf(in_bits=16, group="bytes", prg="chacha", xor_hash="blake3")


@pytest.fixture
def s0s():
    return torch.randint(-2**31, 2**31, (2, 4), dtype=torch.int32)


@pytest.fixture
def beta():
    # LSB of the last word must be 0 (lambda = 127).
    return torch.tensor([0, 0, 0, 604], dtype=torch.int32)


class TestVdpfGen:
    def test_shapes(self, vdpf, s0s, beta):
        s0s_eff, cws, cs, ocw = vdpf.gen(s0s, alpha=107, beta=beta)
        assert s0s_eff.shape == (2, 4)
        assert s0s_eff.dtype == torch.int32
        assert s0s_eff.device.type == "cpu"
        assert cws.shape == (16, 8)
        assert cws.dtype == torch.int32
        assert cs.shape == (4, 4)
        assert cs.dtype == torch.int32
        assert ocw.shape == (4,)
        assert ocw.dtype == torch.int32

    def test_default_retry_succeeds(self, vdpf, s0s, beta):
        s0s_eff, cws, cs, ocw = vdpf.gen(s0s, alpha=5, beta=beta)
        # With the default retry, gen must produce usable keys.
        y, _ = vdpf.eval(party=0, s0=s0s_eff[0], cws=cws, cs=cs, ocw=ocw, x=5)
        assert y.shape == (4,)


class TestVdpfEval:
    def test_output_shapes(self, vdpf, s0s, beta):
        s0s_eff, cws, cs, ocw = vdpf.gen(s0s, alpha=107, beta=beta)
        y, pi_tilde = vdpf.eval(
            party=0, s0=s0s_eff[0], cws=cws, cs=cs, ocw=ocw, x=50
        )
        assert y.shape == (4,)
        assert y.dtype == torch.int32
        assert pi_tilde.shape == (4, 4)
        assert pi_tilde.dtype == torch.int32

    def test_rejects_out_of_domain_x(self, vdpf, s0s, beta):
        s0s_eff, cws, cs, ocw = vdpf.gen(s0s, alpha=107, beta=beta)
        with pytest.raises(ValueError, match="x must be"):
            vdpf.eval(
                party=0, s0=s0s_eff[0], cws=cws, cs=cs, ocw=ocw, x=2**16
            )


class TestVdpfEvalAll:
    def test_merkle_matches_chain(self, vdpf, s0s, beta):
        s0s_eff, cws, cs, ocw = vdpf.gen(s0s, alpha=107, beta=beta)
        for party in (0, 1):
            ys, _ = vdpf.eval_all(
                party=party, s0=s0s_eff[party], cws=cws, cs=cs, ocw=ocw
            )
            ys_m, pi_m = vdpf.eval_all_merkle(
                party=party, s0=s0s_eff[party], cws=cws, cs=cs, ocw=ocw
            )
            assert ys.shape == (2**16, 4)
            assert ys.dtype == torch.int32
            assert pi_m.shape == (4, 4)
            assert torch.equal(ys, ys_m)

    def test_reconstruction(self, vdpf, s0s, beta):
        alpha = 107
        s0s_eff, cws, cs, ocw = vdpf.gen(s0s, alpha=alpha, beta=beta)
        first, _ = vdpf.eval_all(
            party=0, s0=s0s_eff[0], cws=cws, cs=cs, ocw=ocw
        )
        second, _ = vdpf.eval_all(
            party=1, s0=s0s_eff[1], cws=cws, cs=cs, ocw=ocw
        )
        # XOR reconstruction: beta at alpha, zero everywhere else.
        recon = torch.bitwise_xor(first, second)
        assert torch.equal(recon[alpha], beta)
        mask = torch.ones(recon.shape[0], dtype=torch.bool)
        mask[alpha] = False
        assert not torch.any(recon[mask])


class TestVdpfProofs:

    @staticmethod
    def _point_hashes(vdpf, s0s_eff, cws, cs, ocw, xs):
        pi_tildes = []
        for party in (0, 1):
            acc = []
            for x in xs:
                _, pi_tilde = vdpf.eval(
                    party=party, s0=s0s_eff[party], cws=cws, cs=cs, ocw=ocw, x=x
                )
                acc.append(pi_tilde)
            pi_tildes.append(torch.stack(acc))
        return pi_tildes

    @pytest.mark.parametrize("merkle", [False, True])
    def test_prove_and_verify(self, vdpf, s0s, beta, merkle):
        s0s_eff, cws, cs, ocw = vdpf.gen(s0s, alpha=107, beta=beta)
        pi_tildes = self._point_hashes(
            vdpf, s0s_eff, cws, cs, ocw, xs=[0, 1, 106, 107, 108, 65535]
        )
        pi0 = vdpf.prove(pi_tildes[0], cs, merkle=merkle)
        pi1 = vdpf.prove(pi_tildes[1], cs, merkle=merkle)
        assert pi0.shape == (4, 4)
        assert pi0.dtype == torch.int32
        assert vdpf.verify(pi0, pi1) is True

    def test_tampered_proof_rejected(self, vdpf, s0s, beta):
        s0s_eff, cws, cs, ocw = vdpf.gen(s0s, alpha=107, beta=beta)
        pi_tildes = self._point_hashes(
            vdpf, s0s_eff, cws, cs, ocw, xs=[0, 1, 106, 107, 108, 65535]
        )
        pi0 = vdpf.prove(pi_tildes[0], cs)
        pi1 = vdpf.prove(pi_tildes[1], cs)
        assert vdpf.verify(pi0, pi1) is True
        bad = pi1.clone()
        bad[2, 1] = bad[2, 1] + 1
        assert vdpf.verify(pi0, bad) is False


class TestVdpfCuda:

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_eval_gpu_matches_cpu(self, vdpf, s0s, beta):
        s0s_eff, cws, cs, ocw = vdpf.gen(s0s, alpha=107, beta=beta)
        for party in (0, 1):
            y_cpu, pi_cpu = vdpf.eval(
                party=party, s0=s0s_eff[party], cws=cws, cs=cs, ocw=ocw, x=12345
            )
            y_gpu, pi_gpu = vdpf.eval(
                party=party,
                s0=s0s_eff[party].to("cuda"),
                cws=cws.to("cuda"),
                cs=cs.to("cuda"),
                ocw=ocw.to("cuda"),
                x=12345,
            )
            assert torch.equal(y_cpu, y_gpu.cpu())
            assert torch.equal(pi_cpu, pi_gpu.cpu())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_eval_all_merkle_gpu_matches_cpu(self, vdpf, s0s, beta):
        s0s_eff, cws, cs, ocw = vdpf.gen(s0s, alpha=107, beta=beta)
        for party in (0, 1):
            ys_cpu, pi_cpu = vdpf.eval_all_merkle(
                party=party, s0=s0s_eff[party], cws=cws, cs=cs, ocw=ocw
            )
            ys_gpu, pi_gpu = vdpf.eval_all_merkle(
                party=party,
                s0=s0s_eff[party].to("cuda"),
                cws=cws.to("cuda"),
                cs=cs.to("cuda"),
                ocw=ocw.to("cuda"),
            )
            assert torch.equal(ys_cpu, ys_gpu.cpu())
            assert torch.equal(pi_cpu, pi_gpu.cpu())

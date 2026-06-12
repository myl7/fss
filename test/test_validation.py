import pytest
import torch
from fss_crypto._validate import (
    validate_alpha,
    validate_beta,
    validate_cpu_only,
    validate_cws,
    validate_device_match,
    validate_domain_value,
    validate_group,
    validate_in_bits,
    validate_party,
    validate_pred,
    validate_prg,
    validate_s0,
    validate_s0s,
)


class TestValidateInBits:
    def test_valid(self):
        validate_in_bits(1)
        validate_in_bits(64)
        validate_in_bits(128)

    def test_zero(self):
        with pytest.raises(ValueError, match="in_bits must be between 1 and 128"):
            validate_in_bits(0)

    def test_too_large(self):
        with pytest.raises(ValueError, match="in_bits must be between 1 and 128"):
            validate_in_bits(129)


class TestValidateGroup:
    def test_valid(self):
        validate_group("bytes")
        validate_group("uint")

    def test_invalid(self):
        with pytest.raises(ValueError, match="group must be one of"):
            validate_group("invalid")


class TestValidatePrg:
    def test_valid(self):
        validate_prg("chacha", "dpf")
        validate_prg("aes128_mmo", "dpf")
        validate_prg("chacha", "dcf")
        validate_prg("aes128_mmo", "dcf")

    def test_invalid(self):
        with pytest.raises(ValueError, match="prg must be one of"):
            validate_prg("invalid", "dpf")

    def test_invalid_scheme(self):
        with pytest.raises(ValueError, match="scheme must be one of"):
            validate_prg("chacha", "invalid")


class TestValidatePred:
    def test_valid(self):
        validate_pred("lt")
        validate_pred("gt")

    def test_invalid(self):
        with pytest.raises(ValueError, match="pred must be one of"):
            validate_pred("eq")


class TestValidateParty:
    def test_valid(self):
        validate_party(0)
        validate_party(1)

    def test_invalid(self):
        with pytest.raises(ValueError, match="party must be 0 or 1"):
            validate_party(2)


class TestValidateS0s:
    def test_valid(self):
        validate_s0s(torch.zeros(2, 4, dtype=torch.int32))

    def test_wrong_shape(self):
        with pytest.raises(TypeError, match="s0s must be .* int32 tensor"):
            validate_s0s(torch.zeros(3, 4, dtype=torch.int32))

    def test_wrong_dtype(self):
        with pytest.raises(TypeError, match="s0s must be .* int32 tensor"):
            validate_s0s(torch.zeros(2, 4, dtype=torch.float32))


class TestValidateS0:
    def test_valid(self):
        validate_s0(torch.zeros(4, dtype=torch.int32))

    def test_wrong_shape(self):
        with pytest.raises(TypeError, match="s0 must be .* int32 tensor"):
            validate_s0(torch.zeros(2, 4, dtype=torch.int32))

    def test_wrong_dtype(self):
        with pytest.raises(TypeError, match="s0 must be .* int32 tensor"):
            validate_s0(torch.zeros(4, dtype=torch.float32))


class TestValidateBeta:
    def test_valid(self):
        validate_beta(torch.zeros(4, dtype=torch.int32))

    def test_wrong_shape(self):
        with pytest.raises(TypeError, match="beta must be .* int32 tensor"):
            validate_beta(torch.zeros(3, dtype=torch.int32))


class TestValidateCws:
    def test_valid(self):
        validate_cws(torch.zeros(17, 8, dtype=torch.int32), 16)

    def test_wrong_shape(self):
        with pytest.raises(TypeError, match="cws must be .* int32 tensor"):
            validate_cws(torch.zeros(16, 8, dtype=torch.int32), 16)

    def test_wrong_dtype(self):
        with pytest.raises(TypeError, match="cws must be .* int32 tensor"):
            validate_cws(torch.zeros(17, 8, dtype=torch.float32), 16)


class TestValidateAlpha:
    def test_valid(self):
        validate_alpha(0, 20)
        validate_alpha(2**20 - 1, 20)

    def test_negative(self):
        with pytest.raises(ValueError, match="alpha must be"):
            validate_alpha(-1, 20)

    def test_too_large(self):
        with pytest.raises(ValueError, match="alpha must be"):
            validate_alpha(2**20, 20)


class TestValidateDomainValue:
    def test_valid_x(self):
        validate_domain_value("x", 2**20 - 1, 20)

    def test_bool(self):
        with pytest.raises(TypeError, match="x must be an integer"):
            validate_domain_value("x", True, 20)

    def test_too_large(self):
        with pytest.raises(ValueError, match="x must be"):
            validate_domain_value("x", 2**20, 20)


class TestValidateDeviceMatch:
    def test_same_device(self):
        a = torch.zeros(4, dtype=torch.int32)
        b = torch.zeros(4, dtype=torch.int32)
        validate_device_match(a, b)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mismatch(self):
        a = torch.zeros(4, dtype=torch.int32)
        b = torch.zeros(4, dtype=torch.int32, device="cuda")
        with pytest.raises(RuntimeError, match="expected all tensors to be on the same device"):
            validate_device_match(a, b)


class TestValidateCpuOnly:
    def test_cpu(self):
        validate_cpu_only(torch.zeros(4), fn_name="eval_all")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda(self):
        with pytest.raises(RuntimeError, match="eval_all expects all tensors to be on cpu"):
            validate_cpu_only(torch.zeros(4, device="cuda"), fn_name="eval_all")

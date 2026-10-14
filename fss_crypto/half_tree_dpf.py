"""Python wrapper for the Half-Tree DPF (Distributed Point Function) scheme."""

import torch

from fss_crypto._int import split_uint128
from fss_crypto._jit import load
from fss_crypto._validate import (
    validate_alpha,
    validate_beta,
    validate_cpu_only,
    validate_device_match,
    validate_domain_value,
    validate_group,
    validate_in_bits,
    validate_party,
    validate_prg,
    validate_s0,
    validate_s0s,
)


def _validate_cws(cws: torch.Tensor, in_bits: int) -> None:
    if cws.shape != (in_bits, 8) or cws.dtype != torch.int32:
        raise TypeError(
            f"cws must be a ({in_bits}, 8) int32 tensor, "
            f"got shape {tuple(cws.shape)} dtype {cws.dtype}"
        )


def _validate_ocw(ocw: torch.Tensor) -> None:
    if ocw.shape != (4,) or ocw.dtype != torch.int32:
        raise TypeError(
            f"ocw must be a (4,) int32 tensor, "
            f"got shape {tuple(ocw.shape)} dtype {ocw.dtype}"
        )


class HalfTreeDpf:
    """2-party Distributed Point Function using the Half-Tree construction.

    Args:
        in_bits: Input domain bit size (1..128).
        group: Output group type, "bytes" or "uint".
        prg: PRG type, "chacha" or "aes128_mmo".
    """

    def __init__(self, in_bits: int, group: str = "bytes",
                 prg: str = "chacha"):
        validate_in_bits(in_bits)
        validate_group(group)
        validate_prg(prg, "half_tree_dpf")

        self.in_bits = in_bits
        self.group = group
        self.prg = prg
        self._ext = load("half_tree_dpf", in_bits, group, prg)

    def gen(self, s0s: torch.Tensor, alpha: int,
            beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate Half-Tree DPF keys.

        Args:
            s0s: (2, 4) int32 tensor of initial seeds.
            alpha: Point function input in [0, 2^in_bits).
            beta: (4,) int32 tensor of point function output.

        Returns:
            Tuple of the (in_bits, 8) int32 tensor of correction words and
            the (4,) int32 tensor output correction word. Both on CPU.
        """
        validate_s0s(s0s)
        validate_alpha(alpha, self.in_bits)
        validate_beta(beta)
        validate_cpu_only(s0s, beta, fn_name="gen")

        alpha_lo, alpha_hi = split_uint128(alpha)
        return self._ext.gen(s0s, alpha_lo, alpha_hi, beta)

    def eval(self, party: int, s0: torch.Tensor, cws: torch.Tensor,
             ocw: torch.Tensor, x: int) -> torch.Tensor:
        """Evaluate Half-Tree DPF on a single input.

        Args:
            party: Party index, 0 or 1.
            s0: (4,) int32 tensor, the party's initial seed.
            cws: (in_bits, 8) int32 tensor from gen().
            ocw: (4,) int32 output correction word from gen().
            x: Input to evaluate.

        Returns:
            (4,) int32 tensor output share.
        """
        validate_party(party)
        validate_s0(s0)
        _validate_cws(cws, self.in_bits)
        _validate_ocw(ocw)
        validate_device_match(s0, cws, ocw)
        validate_domain_value("x", x, self.in_bits)

        if self.prg == "aes128_mmo" and s0.device.type == "cuda":
            raise RuntimeError(
                "expected all tensors to be on cpu "
                "(aes128_mmo is CPU-only), "
                f"but found tensor on {s0.device}"
            )

        x_lo, x_hi = split_uint128(x)
        return self._ext.eval(party, s0, cws, ocw, x_lo, x_hi)

    def eval_all(self, party: int, s0: torch.Tensor, cws: torch.Tensor,
                 ocw: torch.Tensor) -> torch.Tensor:
        """Evaluate Half-Tree DPF on the full input domain.

        Args:
            party: Party index, 0 or 1.
            s0: (4,) int32 tensor, CPU or CUDA.
            cws: (in_bits, 8) int32 tensor on the same device.
            ocw: (4,) int32 output correction word on the same device.

        Returns:
            (2^in_bits, 4) int32 tensor of output shares on the same device.
            CUDA tensors need in_bits <= 24.
        """
        validate_party(party)
        validate_s0(s0)
        _validate_cws(cws, self.in_bits)
        _validate_ocw(ocw)
        validate_device_match(s0, cws, ocw)

        return self._ext.eval_all(party, s0, cws, ocw)

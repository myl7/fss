"""Python wrapper for the DPF (Distributed Point Function) scheme."""

import torch

from fss_crypto._jit import load
from fss_crypto._validate import (
    validate_alpha,
    validate_beta,
    validate_cpu_only,
    validate_device_match,
    validate_group,
    validate_in_bits,
    validate_party,
    validate_prg,
    validate_s0s,
)


def _split_int128(v: int) -> tuple[int, int]:
    """Split an integer into (lo, hi) 64-bit halves."""
    lo = v & ((1 << 64) - 1)
    hi = (v >> 64) & ((1 << 64) - 1)
    return lo, hi


class Dpf:
    """2-party Distributed Point Function.

    Args:
        in_bits: Input domain bit size (1..128).
        group: Output group type, "bytes" or "uint".
        prg: PRG type, "chacha" or "aes128_mmo".
    """

    def __init__(self, in_bits: int, group: str = "bytes",
                 prg: str = "chacha"):
        validate_in_bits(in_bits)
        validate_group(group)
        validate_prg(prg, "dpf")

        self.in_bits = in_bits
        self.group = group
        self.prg = prg
        self._ext = load("dpf", in_bits, group, prg)

    def gen(self, s0s: torch.Tensor, alpha: int,
            beta: torch.Tensor) -> torch.Tensor:
        """Generate DPF keys.

        Args:
            s0s: (2, 4) int32 tensor of initial seeds.
            alpha: Point function input in [0, 2^in_bits).
            beta: (4,) int32 tensor of point function output.

        Returns:
            (in_bits+1, 8) int32 tensor of correction words.
        """
        validate_s0s(s0s)
        validate_alpha(alpha, self.in_bits)
        validate_beta(beta)

        alpha_lo, alpha_hi = _split_int128(alpha)
        return self._ext.gen(s0s, alpha_lo, alpha_hi, beta)

    def eval(self, party: int, s0: torch.Tensor, cws: torch.Tensor,
             x: int) -> torch.Tensor:
        """Evaluate DPF on a single input.

        Args:
            party: Party index, 0 or 1.
            s0: (4,) int32 tensor, the party's initial seed.
            cws: (in_bits+1, 8) int32 tensor from gen().
            x: Input to evaluate.

        Returns:
            (4,) int32 tensor output share.
        """
        validate_party(party)
        validate_device_match(s0, cws)

        if self.prg == "aes128_mmo" and s0.device.type == "cuda":
            raise RuntimeError(
                "Expected all tensors to be on cpu "
                "(aes128_mmo is CPU-only), "
                f"but found tensor on {s0.device}"
            )

        x_lo, x_hi = _split_int128(x)
        return self._ext.eval(party, s0, cws, x_lo, x_hi)

    def eval_all(self, party: int, s0: torch.Tensor,
                 cws: torch.Tensor) -> torch.Tensor:
        """Evaluate DPF on the full input domain.

        Args:
            party: Party index, 0 or 1.
            s0: (4,) int32 tensor on CPU.
            cws: (in_bits+1, 8) int32 tensor on CPU.

        Returns:
            (2^in_bits, 4) int32 tensor of output shares.
        """
        validate_party(party)
        validate_cpu_only(s0, cws, fn_name="eval_all")

        return self._ext.eval_all(party, s0, cws)

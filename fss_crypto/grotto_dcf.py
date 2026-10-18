"""Python wrapper for the Grotto DCF (distributed comparison function) scheme."""

import torch

from fss_crypto._int import split_uint128
from fss_crypto._jit import load
from fss_crypto._validate import (
    validate_alpha,
    validate_cpu_only,
    validate_cws,
    validate_device_match,
    validate_domain_value,
    validate_in_bits,
    validate_par_depth,
    validate_party,
    validate_prg,
    validate_s0,
    validate_s0s,
)


class GrottoDcf:
    """2-party distributed comparison function over F2 (Grotto construction).

    Output shares are single bits: share_0 XOR share_1 = 1[alpha <= x].

    Args:
        in_bits: Input domain bit size (1..128).
        prg: PRG type, "chacha" or "aes128_mmo".
        par_depth: Host work partition depth. 0 runs sequentially, -1 selects
            automatically, and positive values bound the split depth.
    """

    def __init__(self, in_bits: int, prg: str = "chacha", *, par_depth: int = 0):
        validate_par_depth(par_depth)
        validate_in_bits(in_bits)
        validate_prg(prg, "grotto_dcf")

        self.par_depth = int(par_depth)
        self.in_bits = in_bits
        self.prg = prg
        self._ext = load("grotto_dcf", in_bits, prg=prg, par_depth=self.par_depth)

    def gen(self, s0s: torch.Tensor, alpha: int) -> torch.Tensor:
        """Generate DCF keys.

        Args:
            s0s: (2, 4) int32 tensor of initial seeds.
            alpha: Secret comparison threshold in [0, 2^in_bits).

        Returns:
            (in_bits+1, 8) int32 tensor of correction words.
        """
        validate_s0s(s0s)
        validate_alpha(alpha, self.in_bits)
        validate_cpu_only(s0s, fn_name="gen")

        alpha_lo, alpha_hi = split_uint128(alpha)
        return self._ext.gen(s0s, alpha_lo, alpha_hi)

    def preprocess(self, party: int, s0: torch.Tensor,
                   cws: torch.Tensor) -> torch.Tensor:
        """Preprocess a party's key into a parity segment tree.

        Args:
            party: Party index, 0 or 1.
            s0: (4,) int32 tensor, the party's initial seed.
            cws: (in_bits+1, 8) int32 tensor from gen().

        Returns:
            (2*2^in_bits-1,) bool tensor, the level-order parity segment tree,
            on the same device as the inputs. CUDA tensors need
            in_bits <= 38 (launch grid) and enough device memory for the tree.
        """
        validate_party(party)
        validate_s0(s0)
        validate_cws(cws, self.in_bits)
        validate_device_match(s0, cws)

        return self._ext.preprocess(party, s0, cws)

    def eval(self, party: int, tree: torch.Tensor, x: int) -> bool:
        """Evaluate a preprocessed tree on a single input (CPU only).

        The tree must come from preprocess() of the same party.

        Args:
            party: Party index the tree was preprocessed for, 0 or 1.
            tree: (2*2^in_bits-1,) bool tensor from preprocess().
            x: Input to evaluate.

        Returns:
            bool share such that share_0 XOR share_1 = 1[alpha <= x].
        """
        validate_party(party)
        self._validate_tree(tree)
        validate_cpu_only(tree, fn_name="eval")
        validate_domain_value("x", x, self.in_bits)

        x_lo, x_hi = split_uint128(x)
        return self._ext.eval(party, tree, x_lo, x_hi)

    def eval_all(self, party: int, s0: torch.Tensor,
                 cws: torch.Tensor) -> torch.Tensor:
        """Evaluate DCF on the full input domain.

        Args:
            party: Party index, 0 or 1.
            s0: (4,) int32 tensor, CPU or CUDA.
            cws: (in_bits+1, 8) int32 tensor on the same device.

        Returns:
            (2^in_bits,) bool tensor of output shares on the same device.
            CUDA tensors need in_bits <= 38 (launch grid) and enough device
            memory for the output.
        """
        validate_party(party)
        validate_s0(s0)
        validate_cws(cws, self.in_bits)
        validate_device_match(s0, cws)

        return self._ext.eval_all(party, s0, cws)

    def _validate_tree(self, tree: torch.Tensor) -> None:
        expected = (2 * (1 << self.in_bits) - 1,)
        if tree.shape != expected or tree.dtype != torch.bool:
            raise TypeError(
                f"tree must be a {expected} bool tensor, "
                f"got shape {tuple(tree.shape)} dtype {tree.dtype}"
            )

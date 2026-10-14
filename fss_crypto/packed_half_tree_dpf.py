"""Python wrapper for the packed Half-Tree DPF scheme."""

import torch

from fss_crypto._int import split_uint128
from fss_crypto._jit import load
from fss_crypto._validate import (
    validate_alpha,
    validate_cpu_only,
    validate_device_match,
    validate_domain_value,
    validate_in_bits,
    validate_out_bits,
    validate_party,
    validate_prg,
    validate_s0,
    validate_s0s,
)


def _validate_cws(cws: torch.Tensor, depth: int) -> None:
    if cws.shape != (depth, 4) or cws.dtype != torch.int32:
        raise TypeError(
            f"cws must be a ({depth}, 4) int32 tensor, "
            f"got shape {tuple(cws.shape)} dtype {cws.dtype}"
        )


def _validate_fcw(fcw: torch.Tensor) -> None:
    if fcw.shape != (4,) or fcw.dtype != torch.int32:
        raise TypeError(
            f"fcw must be a (4,) int32 tensor, "
            f"got shape {tuple(fcw.shape)} dtype {fcw.dtype}"
        )


class PackedHalfTreeDpf:
    """2-party DPF with packed sub-128-bit outputs, Half-Tree expansion.

    One (4,) int32 output block covers ``lanes = 128 // out_bits`` consecutive
    inputs: block j covers inputs j * lanes to (j + 1) * lanes - 1, and lane
    ``x % lanes`` of that block holds the out_bits-bit output share of input x,
    read with extract(). The two parties' blocks XOR-reconstruct to beta at
    alpha's lane and to the zero block everywhere else. Only the expanded tree
    depth ``depth = in_bits - log2(lanes)`` levels run.

    Args:
        in_bits: Input domain bit size (1..128). Must exceed log2(lanes) so
            that the tree has at least one level.
        out_bits: Output width in bits, one of (1, 2, 4, 8, 16, 32, 64).
        prg: PRG type, "chacha" or "aes128_mmo".

    Attributes:
        lanes: Number of consecutive inputs covered by one output block.
        depth: Expanded tree depth; eval_all() returns 2^depth blocks.
        k_beta_max: Largest beta that fits in out_bits bits.
    """

    def __init__(self, in_bits: int, out_bits: int, prg: str = "chacha"):
        validate_in_bits(in_bits)
        validate_out_bits(out_bits)
        validate_prg(prg, "packed_half_tree_dpf")

        self.in_bits = in_bits
        self.out_bits = out_bits
        self.prg = prg
        self.lanes = 128 // out_bits
        self.depth = in_bits - (self.lanes.bit_length() - 1)
        self.k_beta_max = 2**64 - 1 if out_bits == 64 else 2**out_bits - 1
        if self.depth < 1:
            raise ValueError(
                f"in_bits must exceed log2(128 // out_bits) so that the tree "
                f"has at least one level, got in_bits={in_bits} "
                f"out_bits={out_bits}"
            )
        self._ext = load("packed_half_tree_dpf", in_bits, prg=prg, out_bits=out_bits)

    def gen(self, s0s: torch.Tensor, alpha: int,
            beta: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate packed Half-Tree DPF keys.

        Args:
            s0s: (2, 4) int32 tensor of initial seeds.
            alpha: Point function input in [0, 2^in_bits). Its low
                log2(lanes) bits select the lane of beta.
            beta: Point function output, an integer in [0, k_beta_max].

        Returns:
            Tuple of the (depth, 4) int32 tensor of correction words and the
            (4,) int32 tensor final correction word. Both on CPU.
        """
        validate_s0s(s0s)
        validate_alpha(alpha, self.in_bits)
        validate_domain_value("beta", beta, self.out_bits)
        validate_cpu_only(s0s, fn_name="gen")

        alpha_lo, alpha_hi = split_uint128(alpha)
        # The extension takes a signed int64 and casts to uint64_t, so wrap
        # values above the int64 range to the same two's complement bits.
        beta_arg = beta - 2**64 if beta >= 2**63 else beta
        return self._ext.gen(s0s, alpha_lo, alpha_hi, beta_arg)

    def eval(self, party: int, s0: torch.Tensor, cws: torch.Tensor,
             fcw: torch.Tensor, x: int) -> torch.Tensor:
        """Evaluate packed Half-Tree DPF on a single input.

        Args:
            party: Party index, 0 or 1.
            s0: (4,) int32 tensor, the party's initial seed.
            cws: (depth, 4) int32 tensor from gen().
            fcw: (4,) int32 final correction word from gen().
            x: Input to evaluate.

        Returns:
            (4,) int32 tensor output share block containing x. extract()
            reads the lane of x from it.
        """
        validate_party(party)
        validate_s0(s0)
        _validate_cws(cws, self.depth)
        _validate_fcw(fcw)
        validate_device_match(s0, cws, fcw)
        validate_domain_value("x", x, self.in_bits)

        if self.prg == "aes128_mmo" and s0.device.type == "cuda":
            raise RuntimeError(
                "expected all tensors to be on cpu "
                "(aes128_mmo is CPU-only), "
                f"but found tensor on {s0.device}"
            )

        x_lo, x_hi = split_uint128(x)
        return self._ext.eval(party, s0, cws, fcw, x_lo, x_hi)

    def eval_all(self, party: int, s0: torch.Tensor, cws: torch.Tensor,
                 fcw: torch.Tensor) -> torch.Tensor:
        """Evaluate packed Half-Tree DPF on the full input domain.

        Args:
            party: Party index, 0 or 1.
            s0: (4,) int32 tensor, CPU or CUDA.
            cws: (depth, 4) int32 tensor on the same device.
            fcw: (4,) int32 final correction word on the same device.

        Returns:
            (2^depth, 4) int32 tensor of output share blocks on the same
            device: block j covers inputs j * lanes to (j + 1) * lanes - 1.
            CUDA tensors need depth <= 24.
        """
        validate_party(party)
        validate_s0(s0)
        _validate_cws(cws, self.depth)
        _validate_fcw(fcw)
        validate_device_match(s0, cws, fcw)

        return self._ext.eval_all(party, s0, cws, fcw)

    def extract(self, block: torch.Tensor, x: int) -> int:
        """Extract the out_bits-bit output share of x from a packed block.

        Args:
            block: (4,) int32 tensor returned by eval() or eval_all(), or the
                XOR of the two parties' blocks.
            x: Input whose lane is read, in [0, 2^in_bits). Only its low
                log2(lanes) bits select the lane.

        Returns:
            The unsigned lane value as an int.
        """
        if block.shape != (4,) or block.dtype != torch.int32:
            raise TypeError(
                f"block must be a (4,) int32 tensor, "
                f"got shape {tuple(block.shape)} dtype {block.dtype}"
            )
        validate_domain_value("x", x, self.in_bits)

        # The block is a little-endian 128-bit value; read the int32 words as
        # unsigned bit patterns.
        value = ((int(block[0]) & 0xFFFFFFFF)
                 | (int(block[1]) & 0xFFFFFFFF) << 32
                 | (int(block[2]) & 0xFFFFFFFF) << 64
                 | (int(block[3]) & 0xFFFFFFFF) << 96)
        return (value >> (self.out_bits * (x % self.lanes))) & self.k_beta_max

"""Python wrapper for the VDPF (verifiable distributed point function) scheme."""

import torch

from fss_crypto._int import split_uint128
from fss_crypto._jit import load
from fss_crypto._validate import (
    validate_alpha,
    validate_beta,
    validate_cpu_only,
    validate_cs,
    validate_device_match,
    validate_domain_value,
    validate_group,
    validate_hash,
    validate_in_bits,
    validate_ocw,
    validate_party,
    validate_pi,
    validate_pi_tildes,
    validate_prg,
    validate_retry,
    validate_s0,
    validate_s0s,
    validate_vdpf_cws,
)


class Vdpf:
    """2-party verifiable distributed point function.

    Evaluation produces output shares plus a corrected per-point hash; the
    hashes accumulate into a proof and two parties compare proofs to detect
    malicious key modification.

    Args:
        in_bits: Input domain bit size (1..128).
        group: Output group type, "bytes" or "uint".
        prg: PRG type, "chacha" or "aes128_mmo".
        xor_hash: XOR-collision-resistant hash, "blake3" or "sha256".
        hash: Proof hash, "blake3" or "sha256"; defaults to xor_hash.
    """

    def __init__(self, in_bits: int, group: str = "bytes", prg: str = "chacha",
                 xor_hash: str = "blake3", hash: str | None = None):
        validate_in_bits(in_bits)
        validate_group(group)
        validate_prg(prg, "vdpf")
        validate_hash(xor_hash)
        if hash is None:
            hash = xor_hash
        validate_hash(hash)

        self.in_bits = in_bits
        self.group = group
        self.prg = prg
        self.xor_hash = xor_hash
        self.hash = hash
        self._ext = load("vdpf", in_bits, group, prg, xor_hash=xor_hash, hash=hash)

    def gen(self, s0s: torch.Tensor, alpha: int, beta: torch.Tensor,
            retry: bool = True) -> tuple:
        """Generate VDPF keys.

        Args:
            s0s: (2, 4) int32 tensor of initial seeds.
            alpha: Point function input in [0, 2^in_bits).
            beta: (4,) int32 tensor of point function output.
            retry: Whether to retry with perturbed seeds on the rare
                bad-seed cases.

        Returns:
            Tuple (s0s_eff, cws, cs, ocw): the (2, 4) int32 tensor of
            effective seeds (the returned keys carry these; use them instead
            of s0s afterwards), the (in_bits, 8) int32 correction words, the
            (4, 4) int32 correction seeds, and the (4,) int32 output
            correction word.
        """
        validate_retry(retry)
        validate_s0s(s0s)
        validate_alpha(alpha, self.in_bits)
        validate_beta(beta)
        validate_cpu_only(s0s, beta, fn_name="gen")

        alpha_lo, alpha_hi = split_uint128(alpha)
        return self._ext.gen(s0s, alpha_lo, alpha_hi, beta, retry)

    def eval(self, party: int, s0: torch.Tensor, cws: torch.Tensor,
             cs: torch.Tensor, ocw: torch.Tensor,
             x: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate VDPF on a single input.

        Args:
            party: Party index, 0 or 1.
            s0: (4,) int32 tensor, the party's effective seed from gen().
            cws: (in_bits, 8) int32 tensor from gen().
            cs: (4, 4) int32 tensor from gen().
            ocw: (4,) int32 tensor from gen().
            x: Input to evaluate.

        Returns:
            Tuple (y, pi_tilde): the (4,) int32 tensor output share and the
            (4, 4) int32 tensor corrected per-point hash.
        """
        validate_party(party)
        validate_s0(s0)
        validate_vdpf_cws(cws, self.in_bits)
        validate_cs(cs)
        validate_ocw(ocw)
        validate_device_match(s0, cws, cs, ocw)
        validate_domain_value("x", x, self.in_bits)
        self._check_cpu_only_hashes(s0)

        x_lo, x_hi = split_uint128(x)
        return self._ext.eval(party, s0, cws, cs, ocw, x_lo, x_hi)

    def eval_all(self, party: int, s0: torch.Tensor, cws: torch.Tensor,
                 cs: torch.Tensor, ocw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate VDPF on the full input domain (chain proof).

        CPU only: the chain proof accumulation is inherently sequential.

        Args:
            party: Party index, 0 or 1.
            s0: (4,) int32 tensor, the party's effective seed from gen().
            cws: (in_bits, 8) int32 tensor from gen().
            cs: (4, 4) int32 tensor from gen().
            ocw: (4,) int32 tensor from gen().

        Returns:
            ((2^in_bits, 4) int32 output shares, (4, 4) int32 chain proof).
        """
        validate_party(party)
        validate_s0(s0)
        validate_vdpf_cws(cws, self.in_bits)
        validate_cs(cs)
        validate_ocw(ocw)
        validate_cpu_only(s0, cws, cs, ocw, fn_name="eval_all")

        return self._ext.eval_all(party, s0, cws, cs, ocw)

    def eval_all_merkle(self, party: int, s0: torch.Tensor,
                        cws: torch.Tensor, cs: torch.Tensor,
                        ocw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate VDPF on the full input domain (Merkle proof).

        Same output shares as eval_all(); the proof comes from a Merkle
        fold over the corrected per-point hashes.

        Args:
            party: Party index, 0 or 1.
            s0: (4,) int32 tensor, the party's effective seed from gen().
            cws: (in_bits, 8) int32 tensor from gen().
            cs: (4, 4) int32 tensor from gen().
            ocw: (4,) int32 tensor from gen().

        Returns:
            Tuple (ys, pi): the (2^in_bits, 4) int32 tensor of output
            shares and the (4, 4) int32 tensor proof, on the same device as
            the inputs. CUDA tensors need blake3 hashes and in_bits <= 22.
        """
        validate_party(party)
        validate_s0(s0)
        validate_vdpf_cws(cws, self.in_bits)
        validate_cs(cs)
        validate_ocw(ocw)
        validate_device_match(s0, cws, cs, ocw)

        if s0.device.type == "cuda":
            self._check_cpu_only_hashes(s0)
            if self.in_bits > 22:
                raise RuntimeError(
                    f"vdpf GPU eval_all_merkle supports in_bits <= 22, "
                    f"got {self.in_bits}"
                )

        return self._ext.eval_all_merkle(party, s0, cws, cs, ocw)

    def prove(self, pi_tildes: torch.Tensor, cs: torch.Tensor,
              merkle: bool = False) -> torch.Tensor:
        """Accumulate corrected per-point hashes into a single proof.

        Args:
            pi_tildes: (L, 4, 4) int32 tensor of corrected per-point hashes
                from eval(), one per evaluated input, in input order for the
                sequential chain. L must be at least 1.
            cs: (4, 4) int32 tensor from gen().
            merkle: Fold with the order-independent Merkle tree instead of
                the sequential chain. Both folds produce the same 64B proof
                shape, so verify() is shared. Parties must accumulate with
                the same flag.

        Returns:
            (4, 4) int32 tensor proof.
        """
        validate_pi_tildes(pi_tildes)
        validate_cs(cs)
        if not isinstance(merkle, bool):
            raise TypeError(
                f"merkle must be a bool, got {type(merkle).__name__}"
            )
        validate_cpu_only(pi_tildes, cs, fn_name="prove")

        return self._ext.prove(pi_tildes, cs, merkle)

    def verify(self, pi0: torch.Tensor, pi1: torch.Tensor) -> bool:
        """Verify a pair of accumulated proofs.

        Stateless: does not use key material or the instance configuration.

        Args:
            pi0: (4, 4) int32 tensor proof of party 0.
            pi1: (4, 4) int32 tensor proof of party 1.

        Returns:
            True if the proofs match (Accept), False otherwise (Reject).
        """
        validate_pi(pi0, "pi0")
        validate_pi(pi1, "pi1")
        validate_cpu_only(pi0, pi1, fn_name="verify")

        return self._ext.verify(pi0, pi1)

    def _check_cpu_only_hashes(self, t: torch.Tensor) -> None:
        if t.device.type != "cuda":
            return
        if self.xor_hash != "blake3" or self.hash != "blake3":
            raise RuntimeError(
                "expected all tensors to be on cpu "
                "(sha256 is CPU-only), "
                f"but found tensor on {t.device}"
            )

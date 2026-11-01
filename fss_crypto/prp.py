"""C++ small-domain permutations with explicit seeds."""

from ._config import (
    ChaChaFeistelConfig as _ChaChaFeistelConfig,
    Aes128FeistelConfig as _Aes128FeistelConfig,
    Aes128FeistelRawConfig as _Aes128FeistelRawConfig,
)
from ._primitive import PrpOperations as _PrpOperations


class ChaChaFeistel(_PrpOperations, _ChaChaFeistelConfig):
    """Host/device ChaCha Feistel permutation."""


class Aes128Feistel(_PrpOperations, _Aes128FeistelConfig):
    """Host OpenSSL AES Feistel permutation."""


class Aes128FeistelRaw(_PrpOperations, _Aes128FeistelRawConfig):
    """Host AES-NI Feistel permutation with optional round-key pre-expansion."""


__all__ = ["ChaChaFeistel", "Aes128Feistel", "Aes128FeistelRaw"]

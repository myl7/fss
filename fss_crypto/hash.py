"""C++ Hashable, XorHashable and CcrHashable convenience implementations."""

from ._config import (
    Blake3Config as _Blake3Config,
    Sha256Config as _Sha256Config,
    Aes128CcrConfig as _Aes128CcrConfig,
)
from ._primitive import HashOperations as _HashOperations


class Blake3(_HashOperations, _Blake3Config):
    """BLAKE3 using an explicitly supplied 32-byte IV."""


class Sha256(_HashOperations, _Sha256Config):
    """SHA256 using an explicitly supplied 16-byte key."""


class Aes128Ccr(_HashOperations, _Aes128CcrConfig):
    """Circular correlation robust hash using a configured AES-MMO backend."""


__all__ = ["Blake3", "Sha256", "Aes128Ccr"]

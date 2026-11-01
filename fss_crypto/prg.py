"""Explicitly configured C++ pseudorandom generators."""

from ._config import (
    ChaChaConfig as _ChaChaConfig,
    Aes128MmoConfig as _Aes128MmoConfig,
    Aes128MmoRawConfig as _Aes128MmoRawConfig,
    Aes128SoftConfig as _Aes128SoftConfig,
)
from ._primitive import PrgOperations as _PrgOperations, blocks as _blocks, native as _native


class ChaCha(_PrgOperations, _ChaChaConfig):
    """ChaCha with explicit nonce, expansion multiplier and round count."""


class Aes128Mmo(_PrgOperations, _Aes128MmoConfig):
    """Host OpenSSL AES-MMO with explicit 16-byte keys."""


class Aes128MmoRaw(_PrgOperations, _Aes128MmoRawConfig):
    """Host AES-NI AES-MMO, including the public eight-seed operation."""

    def gen8(self, seeds):
        """Expand exactly eight seeds, returning shape (8, mul, 4)."""
        return _native(self).gen8(_blocks(seeds, (8, 4)))


class Aes128Soft(_PrgOperations, _Aes128SoftConfig):
    """Host/device software AES-MMO with owned explicit key material."""


__all__ = ["ChaCha", "Aes128Mmo", "Aes128MmoRaw", "Aes128Soft"]

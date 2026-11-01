"""C++ Group implementations and their group elements."""

from ._config import BytesConfig as _BytesConfig, UintConfig as _UintConfig
from ._primitive import GroupOperations as _GroupOperations, Group


class Bytes(_GroupOperations, _BytesConfig):
    """Clamped 127-bit blocks with XOR addition."""


class Uint(_GroupOperations, _UintConfig):
    """Unsigned addition with the C++ template width and optional modulus."""


__all__ = ["Bytes", "Uint", "Group"]

"""Value configurations for the corresponding C++ backend constructors."""

from dataclasses import dataclass
from typing import ClassVar


def _integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _words(name: str, values, count: int) -> tuple[int, ...]:
    values = tuple(values)
    if len(values) != count:
        raise ValueError(f"{name} must contain {count} 32-bit words")
    normalized = []
    for value in values:
        _integer(name, value)
        if not -(1 << 31) <= value < (1 << 32):
            raise ValueError(f"{name} words must fit 32 bits")
        value &= (1 << 32) - 1
        normalized.append(value if value < (1 << 31) else value - (1 << 32))
    return tuple(normalized)


def _bytes(name: str, value: bytes, count: int) -> bytes:
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes")
    if len(value) != count:
        raise ValueError(f"{name} must contain {count} bytes")
    return value


class BackendConfig:
    """Explicit native constructor values, separated from template arguments."""

    _kind: ClassVar[str]

    def _spec(self) -> dict:
        return {"kind": self._kind}

    def _runtime(self) -> dict:
        return {"kind": self._kind}


@dataclass(frozen=True)
class BytesConfig(BackendConfig):
    _kind: ClassVar[str] = "bytes"


@dataclass(frozen=True)
class UintConfig(BackendConfig):
    bits: int
    mod: int = 0
    _kind: ClassVar[str] = "uint"

    def __post_init__(self):
        _integer("bits", self.bits)
        _integer("mod", self.mod)
        if self.bits not in (8, 16, 32, 64, 128):
            raise ValueError("bits must be 8, 16, 32, 64, or 128")
        if self.bits == 128:
            if not 0 < self.mod <= (1 << 127):
                raise ValueError("128-bit Uint requires 0 < mod <= 2^127")
        elif not 0 <= self.mod < (1 << self.bits):
            raise ValueError("mod must fit the selected unsigned type")

    def _spec(self) -> dict:
        return {"kind": self._kind, "bits": self.bits, "mod": self.mod}


@dataclass(frozen=True)
class ChaChaConfig(BackendConfig):
    nonce: tuple[int, int]
    mul: int
    rounds: int = 20
    _kind: ClassVar[str] = "chacha"

    def __post_init__(self):
        object.__setattr__(self, "nonce", _words("nonce", self.nonce, 2))
        _integer("mul", self.mul)
        _integer("rounds", self.rounds)
        if self.mul not in (1, 2, 4):
            raise ValueError("ChaCha mul must be 1, 2, or 4")
        if not 0 < self.rounds < (1 << 31) or self.rounds % 2:
            raise ValueError("rounds must be a positive even C++ int")

    def _spec(self) -> dict:
        return {"kind": self._kind, "mul": self.mul, "rounds": self.rounds}

    def _runtime(self) -> dict:
        return {"kind": self._kind, "nonce": self.nonce}


@dataclass(frozen=True)
class Aes128MmoConfig(BackendConfig):
    mul: int
    keys: tuple[bytes, ...]
    _kind: ClassVar[str] = "aes128_mmo"

    def __post_init__(self):
        _integer("mul", self.mul)
        if not 0 < self.mul < (1 << 31):
            raise ValueError("mul must be a positive C++ int")
        keys = tuple(_bytes("AES key", value, 16) for value in self.keys)
        if len(keys) != self.mul:
            raise ValueError("keys must contain exactly mul AES keys")
        object.__setattr__(self, "keys", keys)

    def _spec(self) -> dict:
        return {"kind": self._kind, "mul": self.mul}

    def _runtime(self) -> dict:
        return {"kind": self._kind, "keys": self.keys}


@dataclass(frozen=True)
class Aes128MmoRawConfig(Aes128MmoConfig):
    _kind: ClassVar[str] = "aes128_mmo_raw"


@dataclass(frozen=True)
class Aes128SoftConfig(Aes128MmoConfig):
    _kind: ClassVar[str] = "aes128_soft"


@dataclass(frozen=True)
class Blake3Config(BackendConfig):
    iv: bytes
    _kind: ClassVar[str] = "blake3"

    def __post_init__(self):
        _bytes("iv", self.iv, 32)

    def _runtime(self) -> dict:
        return {"kind": self._kind, "iv": self.iv}


@dataclass(frozen=True)
class Sha256Config(BackendConfig):
    key: bytes
    _kind: ClassVar[str] = "sha256"

    def __post_init__(self):
        _bytes("key", self.key, 16)

    def _runtime(self) -> dict:
        return {"kind": self._kind, "key": self.key}


@dataclass(frozen=True)
class Aes128CcrConfig(BackendConfig):
    aes_mmo: Aes128MmoConfig
    _kind: ClassVar[str] = "aes128_ccr"

    def __post_init__(self):
        if not isinstance(self.aes_mmo, Aes128MmoConfig) or self.aes_mmo.mul != 1:
            raise ValueError("aes_mmo must configure one of the supported AES backends with mul=1")

    def _spec(self) -> dict:
        return {"kind": self._kind, "aes_mmo": self.aes_mmo._spec()}


@dataclass(frozen=True)
class ChaChaFeistelConfig(BackendConfig):
    _kind: ClassVar[str] = "chacha_feistel"


@dataclass(frozen=True)
class Aes128FeistelConfig(BackendConfig):
    _kind: ClassVar[str] = "aes128_feistel"


@dataclass(frozen=True)
class Aes128FeistelRawConfig(BackendConfig):
    seed: tuple[int, int, int, int] | None = None
    _kind: ClassVar[str] = "aes128_feistel_raw"

    def __post_init__(self):
        if self.seed is not None:
            object.__setattr__(self, "seed", _words("seed", self.seed, 4))

    def _runtime(self) -> dict:
        result = {"kind": self._kind}
        if self.seed is not None:
            result["seed"] = self.seed
        return result


def _backend_spec(value: BackendConfig | None) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, BackendConfig):
        raise TypeError("backend must be an explicit backend configuration")
    return value._spec()

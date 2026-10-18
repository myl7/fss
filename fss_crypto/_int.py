"""Integer helpers shared by Python wrappers."""


def split_uint128(value: int) -> tuple[int, int]:
    """Split an unsigned integer into signed int64 transport bit patterns."""
    lo = value & ((1 << 64) - 1)
    hi = (value >> 64) & ((1 << 64) - 1)
    if lo >= 1 << 63:
        lo -= 1 << 64
    if hi >= 1 << 63:
        hi -= 1 << 64
    return lo, hi

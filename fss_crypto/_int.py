"""Integer helpers shared by Python wrappers."""


def split_uint128(value: int) -> tuple[int, int]:
    """Split an unsigned integer into low and high 64-bit halves."""
    lo = value & ((1 << 64) - 1)
    hi = (value >> 64) & ((1 << 64) - 1)
    return lo, hi

"""Scoped locations for same-toolchain compiled FSS plugins."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from os import PathLike
import os
import sys
import unicodedata
from pathlib import Path


@dataclass(frozen=True)
class _Locations:
    read_roots: tuple[Path, ...]
    write_root: Path | None


_CURRENT: ContextVar[_Locations | None] = ContextVar("fss_jit_locations", default=None)


def _current_locations() -> _Locations:
    locations = _CURRENT.get()
    if locations is None:
        return _Locations((), Path.home() / ".cache" / "fss_crypto")
    return locations


def _existing_parent(path: Path):
    remaining = []
    while True:
        try:
            path.stat()
            return path, tuple(reversed(remaining))
        except (FileNotFoundError, NotADirectoryError):
            parent = path.parent
            if parent == path:
                raise
            remaining.append(path.name)
            path = parent


def _case_sensitive(path: Path) -> bool:
    if sys.platform == "darwin":
        # Darwin defines _PC_CASE_SENSITIVE as 11, outside Python's name table.
        return bool(os.pathconf(path, os.pathconf_names.get("PC_CASE_SENSITIVE", 11)))
    return os.path.normcase("A") != os.path.normcase("a")


def _overlaps(left: Path, right: Path) -> bool:
    left_parent, left_tail = _existing_parent(left)
    right_parent, right_tail = _existing_parent(right)
    if left_parent.samefile(right_parent):
        if sys.platform == "darwin":
            left_tail = tuple(unicodedata.normalize("NFD", part) for part in left_tail)
            right_tail = tuple(unicodedata.normalize("NFD", part) for part in right_tail)
        if not _case_sensitive(left_parent):
            left_tail = tuple(part.casefold() for part in left_tail)
            right_tail = tuple(part.casefold() for part in right_tail)
        return (left_tail[:len(right_tail)] == right_tail or
                right_tail[:len(left_tail)] == left_tail)
    for root, tail, other in ((left_parent, left_tail, right_parent),
                              (right_parent, right_tail, left_parent)):
        if not tail and any(root.samefile(parent) for parent in other.parents):
            return True
    return False


@contextmanager
def cache(*, read_roots: Sequence[str | PathLike[str]] = (),
          write_root: str | PathLike[str] | None) -> Iterator[None]:
    """Select ordered read-only roots and an optional writable output root.

    Paths are fixed on entry. Nested scopes restore the previous locations.
    A new thread must establish its own scope. Read-only mode still checks
    the current compiler and header identities and requires the same toolchain.
    Explicit precompile calls materialize their full closure in write_root.
    """
    if isinstance(read_roots, (str, bytes, PathLike)) or not isinstance(read_roots, Sequence):
        raise TypeError("read_roots must be a sequence of paths")
    roots = tuple(dict.fromkeys(Path(root).resolve() for root in read_roots))
    output = None if write_root is None else Path(write_root).resolve()
    if output is not None:
        for root in roots:
            if _overlaps(root, output):
                raise ValueError("read roots and write root must not overlap")
    token = _CURRENT.set(_Locations(roots, output))
    try:
        yield
    finally:
        _CURRENT.reset(token)


__all__ = ["cache"]

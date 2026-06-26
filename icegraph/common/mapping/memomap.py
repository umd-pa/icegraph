# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from collections.abc import Mapping, Iterator, Hashable
from typing import TypeVar, Callable

__all__ = ["MemoMap"]


K = TypeVar("K", bound="Hashable")
V = TypeVar("V")


class MemoMap(Mapping[K, V]):
    """
    A lazy, memoizing mapping.

    Wraps a build function ``K -> V``. The first time a key is accessed
    the function is called and the result stored; every later access
    returns the stored result without calling the function again.

    Assumes ``build`` is effectively a pure function of the key: results
    are kept for the lifetime of the map and never recomputed.
    """

    def __init__(self, build: Callable[[K], V]) -> None:
        self._build: Callable[[K], V] = build
        self._store: dict[K, V] = {}

    def __getitem__(self, key: K) -> V:
        if key not in self._store:
            self._store[key] = self._build(key)
        return self._store[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: object) -> bool:
        # look at the store directly
        # the Mapping default would call
        # __getitem__ and trigger a build for any missing key
        return key in self._store

    def __repr__(self) -> str:
        return f"MemoMap({self._store!r})"

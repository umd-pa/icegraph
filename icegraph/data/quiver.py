# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Iterator, Iterable
from pathlib import Path
from collections.abc import Mapping
import shutil

import numpy as np
import polars as pl

__all__ = ["QuiverIPC", "QuiverSubset", "QuiverArrays"]


class QuiverIPC(Mapping[str, pl.DataFrame]):
    """
    Directories ("roots") of Arrow IPC files ("arrows"), one table per key in each.

    A key held by several roots reads as their tables stacked in root order, so the
    quivers of several files merge without copying. Tables are written uncompressed
    so reads can be zero-copy memory maps. Nested keys (e.g. ``"a/b"``) map to
    subdirectories.
    """

    def __init__(self, *roots: str | Path) -> None:
        self.roots = [Path(root) for root in roots]

    def _files(self, key: str) -> list[Path]:
        return [f for root in self.roots if (f := root / f"{key}.arrow").exists()]

    def __getitem__(self, key: str) -> pl.DataFrame:
        files = self._files(key)
        if not files:
            raise KeyError(key)

        # tables of one key must share a schema, concat raises if they do not
        return pl.concat([pl.read_ipc(f, memory_map=True) for f in files], how="vertical")

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and bool(self._files(key))

    def __iter__(self) -> Iterator[str]:
        yield from sorted({
            p.relative_to(root).with_suffix("").as_posix()
            for root in self.roots for p in root.rglob("*.arrow")
        })

    def __len__(self) -> int:
        return sum(1 for _ in self)

    @property
    def parts(self) -> list[QuiverIPC]:
        """One quiver per root, in order."""
        return [QuiverIPC(root) for root in self.roots]

    @property
    def nbytes(self) -> int:
        """Size of every table on disk."""
        return sum(p.stat().st_size for root in self.roots for p in root.rglob("*.arrow"))

    @classmethod
    def merge(cls, quivers: Iterable[QuiverIPC]) -> QuiverIPC:
        """One quiver over the roots of each, in order."""
        return cls(*(root for quiver in quivers for root in quiver.roots))

    def arrays(self) -> QuiverArrays:
        """Lazy column-dict view (tables as ``dict[str, np.ndarray]``)"""
        return QuiverArrays(self)

    def subset(self, keys: Iterable[str]) -> QuiverSubset:
        """Read-only view restricted to ``keys``, in the order given."""
        return QuiverSubset(self, keys)

    @classmethod
    def from_data(cls, data: Mapping[str, pl.DataFrame], root: str | Path) -> QuiverIPC:
        # a file without events may have no tables
        # its root still needs to exist
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)

        # write each table
        for key, df in data.items():
            f = root / f"{key}.arrow"
            f.parent.mkdir(parents=True, exist_ok=True)
            df.write_ipc(f, compression="uncompressed")  # uncompressed so reads can mmap

        return cls(root)

    def close(self) -> None:
        for root in self.roots:
            shutil.rmtree(root, ignore_errors=True)


class QuiverSubset(Mapping[str, pl.DataFrame]):
    """Read-only view of a QuiverIPC restricted to a set of keys."""

    def __init__(self, quiver: Mapping[str, pl.DataFrame], keys: Iterable[str]) -> None:
        self._quiver = quiver

        # preserve the caller's order, drop repeats
        self._keys: tuple[str, ...] = tuple(dict.fromkeys(keys))

        if missing := [key for key in self._keys if key not in quiver]:
            raise KeyError(
                f"Quiver holds no table(s) {missing}; available: {sorted(quiver)}."
            )

    def __getitem__(self, key: str) -> pl.DataFrame:
        if key not in self._keys:
            raise KeyError(key)
        return self._quiver[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def arrays(self) -> QuiverArrays:
        """Lazy column-dict view (tables as ``dict[str, np.ndarray]``)"""
        return QuiverArrays(self)


class QuiverArrays(Mapping[str, dict[str, np.ndarray]]):
    """
    Read-only view of a QuiverIPC exposing each table as a plain
    ``dict[str, np.ndarray]`` of columns. Tables load lazily on access.
    """

    def __init__(self, quiver: Mapping[str, pl.DataFrame]) -> None:
        self._quiver = quiver

    def __getitem__(self, key: str) -> dict[str, np.ndarray]:
        df = self._quiver[key]
        return {name: df.get_column(name).to_numpy() for name in df.columns}

    def __iter__(self) -> Iterator[str]:
        return iter(self._quiver)

    def __len__(self) -> int:
        return len(self._quiver)

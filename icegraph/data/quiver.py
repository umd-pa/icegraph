# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Iterator
from pathlib import Path
from collections.abc import Mapping
import shutil

import numpy as np
import polars as pl

__all__ = ["QuiverIPC", "QuiverArrays"]


class QuiverIPC(Mapping[str, pl.DataFrame]):
    """
    A directory of Arrow IPC files ("arrows"), one table per key.

    Tables are written uncompressed so reads can be zero-copy memory maps.
    Nested keys (e.g. ``"a/b"``) map to subdirectories.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path(self, key: str) -> Path:
        return self.root / f"{key}.arrow"

    def __getitem__(self, key: str) -> pl.DataFrame:
        f = self.path(key)
        if not f.exists():
            raise KeyError(key)
        return pl.read_ipc(f, memory_map=True)

    def __iter__(self) -> Iterator[str]:
        yield from sorted(
            p.relative_to(self.root).with_suffix("").as_posix()
            for p in self.root.rglob("*.arrow")
        )

    def __len__(self) -> int:
        return sum(1 for _ in self.root.rglob("*.arrow"))

    def arrays(self) -> QuiverArrays:
        """Lazy column-dict view (tables as ``dict[str, np.ndarray]``)"""
        return QuiverArrays(self)

    @classmethod
    def from_data(cls, data: Mapping[str, pl.DataFrame], root: str | Path) -> QuiverIPC:
        root = Path(root)

        # write each table
        for key, df in data.items():
            f = root / f"{key}.arrow"
            f.parent.mkdir(parents=True, exist_ok=True)
            df.write_ipc(f, compression="uncompressed")  # uncompressed so reads can mmap

        return cls(root)

    def close(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


class QuiverArrays(Mapping[str, dict[str, np.ndarray]]):
    """
    Read-only view of a QuiverIPC exposing each table as a plain
    ``dict[str, np.ndarray]`` of columns. Tables load lazily on access.
    """

    def __init__(self, quiver: QuiverIPC) -> None:
        self._quiver = quiver

    def __getitem__(self, key: str) -> dict[str, np.ndarray]:
        df = self._quiver[key]
        return {name: df.get_column(name).to_numpy() for name in df.columns}

    def __iter__(self) -> Iterator[str]:
        return iter(self._quiver)

    def __len__(self) -> int:
        return len(self._quiver)

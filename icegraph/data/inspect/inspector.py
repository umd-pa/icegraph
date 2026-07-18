# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any, Self
from pathlib import Path
from abc import ABC, abstractmethod

import polars as pl

__all__ = ["Inspector"]


class Inspector(ABC):

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

        # caches
        self._df:       pl.DataFrame | None     = None
        self._attrs:    dict[str, Any] | None   = None

        self.build()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @abstractmethod
    def build(self) -> None:
        ...

    @property
    def df(self) -> pl.DataFrame:
        if self._df is None:
            self._df = self._load_df()
        return self._df

    @property
    def attrs(self) -> dict[str, Any]:
        if self._attrs is None:
            self._attrs = self._load_attrs()
        return self._attrs

    @abstractmethod
    def _load_df(self) -> pl.DataFrame:
        ...

    @abstractmethod
    def _load_attrs(self) -> dict[str, Any]:
        ...

    def close(self) -> None:
        pass

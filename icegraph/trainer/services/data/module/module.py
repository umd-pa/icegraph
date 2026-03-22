# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TypeVar
from abc import abstractmethod
from collections.abc import Sized

from torch.utils.data import Dataset
from torch_geometric.data import Data

from icegraph.types.data import Split, ModelInputRole
from icegraph.types.common import ArrayUI
from icegraph.types.plugins import Plugin
from icegraph.trainer.services.data.store import Store

from .types import ModuleContext

__all__ = ["Module"]


C = TypeVar("C")


class Module(Plugin[C, ModuleContext], Dataset[Data], Sized):
    """The base dataset class for loading and managing IceCube data."""

    _split: Split
    _store: Store

    def on_attach(self) -> None:
        self._split = self._ctx.split
        self._store = self._ctx.store

    def __getitem__(self, index: int) -> Data:
        # ensure int index
        if not isinstance(index, int):
            raise TypeError(f"Parameter 'index' must be of type int, got '{type(index)}'")

        # normalize to key
        index = int(self.keys[index])

        # delegate to subclass
        return self.get(index)

    def __len__(self) -> int:
        """
        Return the number of events in the subset.

        Returns:
            int: Number of events.
        """
        return len(self.keys)

    @abstractmethod
    def get(self, index: int) -> Data:
        ...

    @property
    @abstractmethod
    def keys(self) -> ArrayUI:
        ...

    @abstractmethod
    def columns(self, role: ModelInputRole, aux: bool = False) -> list[str]:
        ...

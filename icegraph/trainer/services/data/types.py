# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Mapping, Iterator, TypeVar, Protocol, Any
from types import MappingProxyType
from dataclasses import dataclass
from collections.abc import Sized

from icegraph.statistics import StatisticService
from icegraph.types.data import AttributeDomain, ModelInputRole, Split

__all__ = ["Attributes", "GlobalAttributes", "SizedDataset"]


### PROTOCOLS

D = TypeVar("D")

class SizedDataset(Protocol[D], Sized):
    def __getitem__(self, index: int | slice) -> D: ...
    def __len__(self) -> int: ...


### ATTRIBUTES

@dataclass(frozen=True)
class Attributes(Mapping[AttributeDomain, dict[str, Any]]):
    _data: dict[AttributeDomain, str | dict[str, Any]]

    def __post_init__(self):
        # convert to read-only mapping
        object.__setattr__(self, "_data", MappingProxyType(self._data))

    def __getitem__(self, key: AttributeDomain) -> str | dict[str, Any]:
        return self._data[key]

    def __iter__(self) -> Iterator[AttributeDomain]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    @property
    def shard_id(self) -> str:
        shard_id = self._data[AttributeDomain.LOCAL].get("id")
        if shard_id is None:
            raise RuntimeError("Could not find key 'id' in LOCAL attrs. A shard ID is required for every shard.")

        return shard_id

    @property
    def checksum(self) -> str:
        checksum = self._data[AttributeDomain.GLOBAL].get("set_id")
        if checksum is None:
            raise RuntimeError("Could not find key 'set_id' in GLOBAL attrs. A checksum is required for every shard.")

        return checksum

    def stats(self, split: Split, role: ModelInputRole) -> StatisticService:
        stat_structs = self._data[AttributeDomain.LOCAL].get("stats")

        if stat_structs is None:
            raise RuntimeError(
                f"Key [{AttributeDomain.LOCAL.name}][stats] not found in attrs. "
                f"Problematic shard: ID={self.shard_id}."
            )

        struct = stat_structs.get(role.value, {}).get(str(split.to_int()))

        # ensure struct was found
        if struct is None:
            raise RuntimeError(
                f"No stats found for role={role.value}, split={str(split.to_int())}. "
                f"Problematic shard: ID={self.shard_id}."
            )

        return StatisticService.from_struct(struct)

@dataclass(frozen=True)
class GlobalAttributes(Mapping[str, Any]):
    _data: dict[str, Any]

    def __post_init__(self):
        # convert to read-only mapping
        object.__setattr__(self, "_data", MappingProxyType(self._data))

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    @property
    def checksum(self) -> str:
        # this should not be a problem, as this is built from Attributes which checks for checksum
        return self._data["checksum"]

    def columns(self, role: ModelInputRole) -> list[str]:
        """Get the column names for the given input role."""
        columns = self.get(role.value)
        if columns is None:
            # since this is missing from global, this is not isolated to one shard
            raise RuntimeError(
                f"Could not find key '{role.value}' in GLOBAL attrs. "
                f"This is likely a problem with the whole dataset and not one shard."
            )

        return columns

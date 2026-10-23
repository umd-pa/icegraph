# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from functools import cached_property
from typing import Annotated, Mapping, Iterator, Any, TypeAlias, TypeVar
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, BeforeValidator, InstanceOf, ValidationError

from icegraph.common.data import AttributeDomain

__all__ = ["Attributes", "GlobalAttributes", "Scalar", "ScalarList", "Array", "validate_attrs"]


T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)


@dataclass(frozen=True)
class Attributes(Mapping[AttributeDomain, dict[str, Any]]):
    _data: dict[AttributeDomain, dict[str, Any]]

    def __getitem__(self, key: AttributeDomain) -> dict[str, Any]:
        return self._data[key]

    def __iter__(self) -> Iterator[AttributeDomain]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    @cached_property
    def shard_id(self) -> str:
        shard_id = self._data[AttributeDomain.LOCAL].get("id")
        if shard_id is None:
            raise RuntimeError("Could not find key 'id' in LOCAL attrs. A shard ID is required for every shard.")

        return shard_id

    @cached_property
    def checksum(self) -> str:
        set_id = self._data[AttributeDomain.GLOBAL].get("set_id")
        if set_id is None:
            raise RuntimeError("Could not find key 'set_id' in GLOBAL attrs. A set ID is required for every shard.")

        return set_id


@dataclass(frozen=True)
class GlobalAttributes(Mapping[str, Any]):
    _data: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    @classmethod
    def from_attrs(cls, attrs: Iterator[Attributes], *, ignore_checksum: bool = False) -> GlobalAttributes:
        # get iterator over attributes
        it = iter(attrs)

        try:
            first_attr = next(it)
        except StopIteration:
            raise RuntimeError("Cannot build global attributes for an empty dataset.")

        # verify each checksum is identical to the first
        if not ignore_checksum:
            for attr in it:
                if attr.checksum != first_attr.checksum:
                    raise ValueError(
                        f"Checksums do not match across shards; expected {first_attr.checksum}, got {attr.checksum}.")

        # because each checksum is identical, just grab globals from first attribute
        global_attrs = first_attr.get(AttributeDomain.GLOBAL)
        if global_attrs is None:
            raise RuntimeError(f"Could not find key '{AttributeDomain.GLOBAL}' in dataset attributes.")

        return GlobalAttributes(global_attrs)

    @property
    def checksum(self) -> str:
        return self._data["set_id"]


def _scalar(value: Any) -> Any:
    """Unpack a numpy array to a scalar if possible."""
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.item()

    return value


def _sequence(value: Any) -> Any:
    """Unpack a numpy array to a list if possible."""
    if isinstance(value, np.ndarray):
        return value.tolist()

    return value


Scalar:     TypeAlias = Annotated[T, BeforeValidator(_scalar)]
ScalarList: TypeAlias = Annotated[list[T], BeforeValidator(_sequence)]
Array:      TypeAlias = InstanceOf[np.ndarray]


def validate_attrs(model: type[M], data: Any, *, source: str) -> M:
    """Validate restored attributes against the schema of whatever reads them."""
    try:
        return model.model_validate(data)
    except ValidationError as e:
        problems = "\n".join(
            f"  {'.'.join(str(part) for part in error['loc']) or '<root>'}: {error['msg']}"
            for error in e.errors()
        )

        raise ValueError(f"Invalid attributes in {source}:\n{problems}") from None

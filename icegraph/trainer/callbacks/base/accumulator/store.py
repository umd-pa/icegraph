# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Iterator, Self
from collections.abc import Mapping
from dataclasses import dataclass, field

from torch import Tensor

from .accumulator import Accumulator

__all__ = ["AccumulatorStore"]


@dataclass
class AccumulatorStore(Mapping[str, Accumulator]):
    accumulators: dict[str, Accumulator] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Accumulator:
        return self.accumulators[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.accumulators)

    def __len__(self) -> int:
        return len(self.accumulators)

    def __iadd__(self, other: Mapping[str, Tensor]) -> Self:
        self.accumulate(other)
        return self

    def require(self, name: str) -> Accumulator:
        try:
            return self.accumulators[name]
        except KeyError:
            raise KeyError(
                f"No accumulator with name {name} registered to {type(self).__name__}."
            ) from None

    def enum(self, name: str) -> Iterator[tuple[int, Tensor]]:
        # ensure key actually exists in the store and return iterator
        return self.require(name).enum()

    def enum_all(self) -> Iterator[tuple[int, tuple[Tensor | None, ...]]]:
        # build list of iterators
        iterators = [self.enum(name) for name in self]

        # convert to a list of dicts
        dicts = [dict(iterator) for iterator in iterators]

        # iterate over all indices present in any dict
        for index in set().union(*dicts):
            # yields None for any dict with missing index
            yield index, tuple(d.get(index) for d in dicts)

    def align_to(self, keys: list[str]) -> None:
        self.accumulators = {key: self.accumulators[key] for key in keys}

    def accumulate(self, other: Mapping[str, Tensor]) -> None:
        # iterate over other tensors
        for name, tensor in other.items():
            # ensure key actually exists in the store and accumulate
            self.require(name).accumulate(tensor)

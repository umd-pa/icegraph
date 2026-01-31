# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TypeAlias, Self, Callable, Iterator
from dataclasses import dataclass

from torch import Tensor

__all__ = ["Accumulator", "Combiner", "Enumerator"]

Combiner:   TypeAlias = Callable[[Tensor, Tensor], Tensor | None]
Enumerator: TypeAlias = Callable[[Tensor], Iterator[tuple[int, Tensor]]]


@dataclass
class Accumulator:
    tensor:     Tensor
    combiner:   Combiner
    enumerator: Enumerator

    def __iadd__(self, other: Tensor) -> Self:
        self.accumulate(other)
        return self

    def accumulate(self, other: Tensor) -> None:

        # ensure dtypes and devices match
        if other.dtype != self.tensor.dtype:
            raise TypeError(
                f"dtype mismatch in {type(self).__name__}: "
                f"expected {self.tensor.dtype}, got {other.dtype}."
            )
        if other.device != self.tensor.device:
            raise ValueError(
                f"device mismatch in {type(self).__name__}: "
                f"expected {self.tensor.device}, got {other.device}."
            )

        # in place combine if possible
        combined = self.combiner(self.tensor, other)
        if combined is not None:
            self.tensor = combined

    def enum(self) -> Iterator[tuple[int, Tensor]]:
        return self.enumerator(self.tensor)

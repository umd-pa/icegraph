# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from functools import cached_property
import dataclasses
from dataclasses import dataclass
from typing import Self

from icegraph.common.transforms import SpaceType, Space

__all__ = ["AxisLayout"]


@dataclass(frozen=True)
class AxisLayout:
    scale:      tuple[SpaceType, ...]   = ("linear",)
    bins:       tuple[int, ...]         = (100,)
    margins:    tuple[float, ...]       = (0.0,)
    log_count:  bool                    = False

    def __post_init__(self) -> None:
        if any(0 > m for m in self.margins):
            raise ValueError(f"{type(self).__name__}: 'margin' must be a sequence of non-negative numbers, got {self.margins!r}.")

    @cached_property
    def spaces(self) -> tuple[Space, ...]:
        return tuple(Space(s) for s in self.scale)

    def expand(self, ndim: int) -> Self:
        expanded: dict[str, tuple] = {}
        for field in ("scale", "bins", "margins"):
            data = getattr(self, field)

            # if len is 1, expand to ndim
            if len(data) == 1:
                data *= ndim

            # if it doesnt match ndim size now, raise
            if len(data) != ndim:
                raise ValueError(
                    f"Expected tuple of length 1 or ndim ({ndim}), got tuple of length {len(data)} for field {field!r}."
                )

            expanded[field] = data

        return dataclasses.replace(self, **expanded)

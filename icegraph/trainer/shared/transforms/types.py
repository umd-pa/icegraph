# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Protocol
from functools import cached_property
from dataclasses import dataclass

from icegraph.types.transforms import TransformSpace, TransformSpaceType

__all__ = ["TransformSpec", "GroupTransformSpec"]


class _TransformConfig(Protocol):
    space:  TransformSpaceType
    base:   int


@dataclass(frozen=True)
class TransformSpec:
    space:  TransformSpace
    base:   int | None


class GroupTransformSpec:

    def __init__(self, specs: list[TransformSpec]) -> None:
        self.specs = specs

    def __len__(self) -> int:
        return len(self.specs)

    @cached_property
    def groups(self) -> list[tuple[TransformSpace, list[int], list[int | None]]]:
        grouped: dict[TransformSpace, list[tuple[int, int | None]]] = {}

        for col, spec in enumerate(self.specs):
            grouped.setdefault(spec.space, []).append((col, spec.base))

        result: list[tuple[TransformSpace, list[int], list[int | None]]] = []

        for space, pairs in grouped.items():
            pairs.sort(key=lambda x: x[0])  # sort by column index

            cols = [col for col, _ in pairs]
            bases = [base for _, base in pairs]

            result.append((space, bases, cols))

        return result

    @classmethod
    def from_config(cls, columns: list[str], config: dict[str, _TransformConfig]) -> GroupTransformSpec:
        # build spec dict
        specs: list[TransformSpec] = []
        for column in columns:
            column_config = config.get(column)

            # if no config provided, use linear
            if not column_config:
                specs.append(TransformSpec(TransformSpace.LINEAR, None))
                continue

            # append the spec
            specs.append(
                TransformSpec(
                    TransformSpace(column_config.space), column_config.base
                )
            )

        return cls(specs)

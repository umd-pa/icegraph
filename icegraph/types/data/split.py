# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar
from enum import Enum

__all__ = ["Split"]


class Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    def to_int(self) -> int:
        return _SPLIT_TO_INT[self]

    @classmethod
    def from_int(cls, i: int) -> Split:
        try:
            return _INT_TO_SPLIT[i]
        except KeyError:
            raise ValueError(f"Invalid split int: {i}, expected {list(_INT_TO_SPLIT.keys())}")

    @classmethod
    def all(cls) -> tuple[Split, ...]:
        return tuple(cls)

    @classmethod
    def eval(cls) -> tuple[Split, ...]:
        return tuple(split for split in cls.all() if split != cls.TRAIN)


_SPLIT_TO_INT: dict[Split, int] = {
    Split.TRAIN: 0,
    Split.VAL: 1,
    Split.TEST: 2
}

_INT_TO_SPLIT: dict[int, Split] = {
    v: k for k, v in _SPLIT_TO_INT.items()
}

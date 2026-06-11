# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TypeAlias, Literal

from enum import StrEnum

__all__ = ["DataRole", "ColumnarRole", "TruthRole"]


class DataRole(StrEnum):
    FEATURES    = "features"
    TARGETS     = "targets"
    AUXILIARY   = "auxiliary"
    EDGE_INDEX  = "edge_index"
    EDGE_ATTR   = "edge_attr"
    SIMWEIGHT   = "simweights"
    BATCH       = "batch"

    @classmethod
    def core(cls) -> tuple[DataRole, ...]:
        return cls.FEATURES, cls.TARGETS

    @classmethod
    def all(cls) -> tuple[DataRole, ...]:
        return tuple(cls)

    @classmethod
    def values(cls) -> list[str]:
        return [role.value for role in cls.all()]

    @classmethod
    def names(cls) -> list[str]:
        return [role.name for role in cls.all()]

    @classmethod
    def truth(cls) -> frozenset[DataRole]:
        return TRUTH_DATA_ROLES

    @classmethod
    def columnar(cls) -> frozenset[DataRole]:
        return COLUMNAR_DATA_ROLES


ColumnarRole: TypeAlias = Literal[DataRole.TARGETS, DataRole.FEATURES, DataRole.AUXILIARY]
COLUMNAR_DATA_ROLES: frozenset[DataRole] = frozenset({
    DataRole.TARGETS, DataRole.FEATURES, DataRole.AUXILIARY
})

TruthRole: TypeAlias = Literal[DataRole.TARGETS, DataRole.AUXILIARY]
TRUTH_DATA_ROLES: frozenset[DataRole] = frozenset({
    DataRole.TARGETS, DataRole.AUXILIARY
})

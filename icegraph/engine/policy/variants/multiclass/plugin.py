# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any
from functools import cached_property

import torch

from icegraph.common.data import AttributeDomain, DataRole

from ...policy import Policy
from ...types import TaskSpec

from .config import MulticlassConfig

__all__ = ["Multiclass"]


class Multiclass(Policy[MulticlassConfig]):
    name: ClassVar[str] = "multiclass"
    version: ClassVar[int] = 1

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> MulticlassConfig:
        return MulticlassConfig(**config)

    def _extract_classlist(self, metadata: dict[str, Any], label: str) -> set[int]:
        cls = type(self).__name__

        column = metadata.get(label)
        if column is None:
            raise KeyError(
                f"{cls}: missing key 'columns.{label}' in dataset local attributes."
            )
        if not isinstance(column, dict):
            raise TypeError(
                f"{cls}: value at key 'columns.{label}' must be a dict."
            )

        uniques = column.get("unique")
        if uniques is None:
            raise KeyError(
                f"{cls}: missing key 'columns.{label}.unique' in dataset local attributes."
            )
        if not uniques:
            raise ValueError(
                f"{cls}: key 'columns.{label}.unique' must be non-empty."
            )
        if not isinstance(uniques, list):
            raise TypeError(
                f"{cls}: value at key 'columns.{label}.unique' must be a list."
            )
        for value in uniques:
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    f"{cls}: value at key 'columns.{label}.unique' must be a list[int]."
                )

        return set(uniques)

    @cached_property
    def classlists(self) -> tuple[set[int], ...]:
        cls = type(self).__name__

        # load services
        decoder = self._ctx.services.require("decode", required_by=type(self))
        record = self._ctx.services.require("record", required_by=type(self))

        # get target layout
        layout = decoder.get_segment_layout(DataRole.TARGETS, torch.device("cpu"))

        # one accumulator per label; unique values are stored per file (attr), so a
        # given file may only contain a subset of classes, need union across all files
        classlists: list[set[int]] = [set() for _ in layout.names]
        for attr in record.attrs():
            metadata = attr[AttributeDomain.LOCAL].get("columns")
            if metadata is None:
                raise KeyError(
                    f"{cls}: missing key 'columns' in dataset local attributes."
                )
            if not isinstance(metadata, dict):
                raise TypeError(
                    f"{cls}: value at key 'columns' must be a dict."
                )

            for index, label in enumerate(layout.names):
                classlists[index] |= self._extract_classlist(metadata, label)

        return tuple(classlists)

    def _build_task_spec(self) -> TaskSpec:
        # count classes for each head
        num_classes = tuple(max(cl) + 1 for cl in self.classlists)

        # compute offsets
        offsets = torch.cat([
            torch.zeros(1, dtype=torch.long),
            torch.cumsum(
                torch.as_tensor(num_classes, dtype=torch.long),
                dim=0
            )
        ])

        return TaskSpec(
            out_offsets=offsets,
            target_dtype=torch.long,
            norm_targets=False
        )

# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from abc import abstractmethod, ABC
from typing import TypeVar, Iterator, Callable, final

import numpy as np

from icegraph.common.plugins import Plugin
from icegraph.statistics import StatisticService
from icegraph.typing.common import ArrayI
from icegraph.common.record import GlobalAttributes, Attributes

from .types import AttributeDecoderContext

__all__ = ["AttributeDecoder"]


C = TypeVar("C")


class AttributeDecoder(Plugin[C, AttributeDecoderContext], ABC):
    """Provides methods for decoding dataset attributes."""

    @final
    def extract_columns(self, role: str) -> list[str]:
        columns = self._extract_columns(
            role,
            attrs=self._ctx.attrs,
            global_attrs=self._ctx.global_attrs
        )

        cls = type(self).__name__

        if not isinstance(columns, list):
            raise TypeError(
                f"{cls}._extract_columns() must return a list[str], "
                f"got {type(columns).__name__}."
            )
        for i, item in enumerate(columns):
            if not isinstance(item, str):
                raise TypeError(
                    f"{cls}._extract_columns() must return a list[str]; "
                    f"element [{i}] is {type(item).__name__}."
                )

        return columns

    @abstractmethod
    def _extract_columns(
            self, role: str, *,
            attrs: Callable[[], Iterator[Attributes]], global_attrs: GlobalAttributes
    ) -> list[str]:
        ...

    @final
    def extract_offsets(self, role: str) -> ArrayI:
        offsets = self._extract_offsets(
            role,
            attrs=self._ctx.attrs,
            global_attrs=self._ctx.global_attrs
        )

        cls = type(self).__name__

        if not isinstance(offsets, np.ndarray):
            raise TypeError(
                f"{cls}._extract_offsets() must return an ndarray, "
                f"got {type(offsets).__name__}."
            )

        if not np.issubdtype(offsets.dtype, np.integer):
            raise TypeError(
                f"{cls}._extract_offsets() must return an integer ndarray, "
                f"got dtype {offsets.dtype}."
            )

        if offsets.ndim != 1:
            raise ValueError(
                f"{cls}._extract_offsets() must return a 1D ndarray, "
                f"got {offsets.ndim}D with shape {offsets.shape}."
            )

        # cross-check length against the columns for this role: n columns -> n+1 offsets
        num_columns = len(self.extract_columns(role))
        expected = num_columns + 1
        if offsets.size != expected:
            raise ValueError(
                f"{cls}._extract_offsets() must return L+1 offsets for L columns; "
                f"role {role} has {num_columns} column(s), expected {expected} "
                f"offsets, got {offsets.size}."
            )

        if offsets[0] != 0:
            raise ValueError(
                f"{cls}._extract_offsets() must start at 0, "
                f"got {offsets[0]}."
            )

        diffs = np.diff(offsets)
        if not np.all(diffs > 0):
            bad = int(np.argmin(diffs))
            raise ValueError(
                f"{cls}._extract_offsets() must be strictly increasing; "
                f"offsets[{bad}]={offsets[bad]} >= offsets[{bad + 1}]={offsets[bad + 1]}."
            )

        return offsets

    @abstractmethod
    def _extract_offsets(
            self, role: str, *,
            attrs: Callable[[], Iterator[Attributes]], global_attrs: GlobalAttributes
    ) -> ArrayI:
        ...

    @final
    def extract_keys(self, split: int) -> ArrayI:
        splitmap = self._extract_keys(
            split,
            attrs=self._ctx.attrs,
            global_attrs=self._ctx.global_attrs
        )

        cls = type(self).__name__

        if not isinstance(splitmap, np.ndarray):
            raise TypeError(
                f"{cls}._extract_keys() must return a numpy ndarray, "
                f"got {type(splitmap).__name__}."
            )

        if not np.issubdtype(splitmap.dtype, np.integer):
            raise TypeError(
                f"{cls}._extract_keys() must return an integer array, "
                f"got dtype {splitmap.dtype.__name__}."
            )

        return splitmap

    @abstractmethod
    def _extract_keys(
            self, split: int, *,
            attrs: Callable[[], Iterator[Attributes]], global_attrs: GlobalAttributes
    ) -> ArrayI:
        ...

    @final
    def extract_stats(self, key: tuple[int, str]) -> StatisticService:
        split, role = key
        stats = self._extract_stats(
            split, role,
            attrs=self._ctx.attrs,
            global_attrs=self._ctx.global_attrs
        )

        cls = type(self).__name__

        if not isinstance(stats, StatisticService):
            raise TypeError(
                f"{cls}._extract_stats() must return a StatisticService instance, got {type(stats).__name__}"
            )

        return stats

    @abstractmethod
    def _extract_stats(
            self, split: int, role: str, *,
            attrs: Callable[[], Iterator[Attributes]], global_attrs: GlobalAttributes
    ) -> StatisticService:
        ...

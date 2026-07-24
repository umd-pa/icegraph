# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any, Self, Literal
from dataclasses import dataclass, field
from collections import defaultdict
from collections.abc import Mapping
import itertools

import polars as pl

from icegraph.common.data import AttributeDomain

from .quiver import QuiverIPC

__all__ = ["Envelope"]


def nested_dict():
    return defaultdict(nested_dict)


@dataclass
class Envelope:
    """Data wrapper passed between pipeline stages."""
    # raw data
    quiver:     QuiverIPC

    # tables and attrs
    tmp:        dict[str, pl.DataFrame]     = field(default_factory=dict)
    main:       pl.DataFrame                = field(default_factory=pl.DataFrame)
    attrs:      dict[str, dict[str, Any]]   = field(default_factory=nested_dict)

    # not persisted
    state:      dict[str, Any]              = field(default_factory=nested_dict)
    metrics:    dict[str, float]            = field(default_factory=dict)
    active:     str | None                  = None

    def resolve_cols(self, value: str | int | list[int] | list[str], *, _seen: set[str] | None = None) -> list[str]:
        _seen = set() if _seen is None else _seen

        if isinstance(value, list):
            # use recursion for list inputs
            return list(itertools.chain.from_iterable(self.resolve_cols(v, _seen=_seen) for v in value))

        # polars column names are always strings, normalize numeric refs
        value = str(value)

        # check for cyclic refs
        if value in _seen:
            raise RuntimeError(f"Cyclic group reference detected for '{value}'")
        _seen.add(value)

        # first check if it is column in the active frame (prioritize col over group, helps prevent infinite recursion)
        if self.active:
            if value in self.tmp[self.active].columns:
                return [value]

        # resolve group or raise if not found in groups
        groups = self.state.get("alias", {})
        if value not in groups:
            return [value]  # up to downstream to ensure this col actually exists

        return self.resolve_cols(groups[value], _seen=_seen)

    def set_local_attr(self, key: str, data: Any) -> None:
        self.attrs[AttributeDomain.LOCAL.name][key] = data

    def get_local_attr(self, key: str, default: Any | None = None) -> Any:
        return self.attrs[AttributeDomain.LOCAL.name].get(key, default)

    def set_global_attr(self, key: str, data: Any) -> None:
        self.attrs[AttributeDomain.GLOBAL.name][key] = data

    def get_global_attr(self, key: str, default: Any | None = None) -> Any:
        return self.attrs[AttributeDomain.GLOBAL.name].get(key, default)

    def set_column_attr(self, column: str | int, key: str, data: Any, *, domain: AttributeDomain) -> None:
        self.attrs[domain.name]["columns"][column][key] = data

    def sync_column_names(self, mapping: dict[str, str]) -> None:
        for domain in AttributeDomain.all():
            cols = self.attrs[domain.name]["columns"]
            renamed = {mapping.get(k, k): v for k, v in cols.items()}
            cols.clear()
            cols.update(renamed)

    def merge(
        self,
        df: pl.DataFrame, /, *,
        to: str,
        on: str | list[str],
        how: Literal['left', 'right', 'inner', 'full', 'cross', 'semi', 'anti'] = "left",
        **kwargs
    ) -> Self:
        if to not in self.tmp:
            self.tmp[to] = df
            return self

        self.tmp[to] = self.tmp[to].join(df, on=on, how=how, maintain_order="left", **kwargs)
        return self

    def commit(self, df: pl.DataFrame, /, on: str | list[str], **kwargs) -> Self:
        if self.main.is_empty():
            self.main = df
            return self

        self.main = self.main.join(df, on=on, how="left", maintain_order="left", **kwargs)
        return self

    def devivify(self) -> Self:
        """De-vivify attrs and state."""

        def _devivify(obj: Any) -> Any:
            if isinstance(obj, defaultdict):
                obj = dict(obj)
            if isinstance(obj, Mapping):
                return {k: _devivify(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_devivify(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(_devivify(v) for v in obj)
            return obj

        self.attrs = _devivify(self.attrs)
        self.state = _devivify(self.state)
        return self

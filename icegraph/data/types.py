# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any, Iterable, Self, Literal
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
from collections.abc import Mapping
import itertools

import pandas as pd

from icegraph.types.plugins import PluginContext

from .shared.queue import IterableQueue

__all__ = ["StageContext", "Envelope"]


def nested_dict():
    return defaultdict(nested_dict)


@dataclass(frozen=True)
class StageContext(PluginContext):
    src: IterableQueue[Envelope] | Iterable[Path]
    dst: IterableQueue[Envelope] | None
    scratch: Path

    # ordering and index
    index: int
    total: int

@dataclass
class Envelope:
    """
    Data wrapper passed between pipeline stages.

    Attributes:
        main: The payload DataFrame.
        tmp: Temp work space for processors. The selector moves frames to tmp, committer packs to main.
        data: Raw data from the extractor.
        attrs: Data attributes (auto-nesting until de-vivification). Must include 'GLOBAL' and 'LOCAL' domain.
        state: Internal dict for inter-stage communication.
    """
    data:       dict[str, pd.DataFrame]
    tmp:        dict[str, pd.DataFrame]     = field(default_factory=dict)
    main:       pd.DataFrame                = field(default_factory=pd.DataFrame)
    attrs:      dict[str, dict[str, Any]]   = field(default_factory=nested_dict)

    # not persisted
    state:      dict[str, Any]              = field(default_factory=nested_dict)

    @property
    def active(self) -> str | None:
        return self.state.get("active")

    @active.setter
    def active(self, key: str | None) -> None:
        if key is None:
            self.state.pop("active", None)
            return
        self.state["active"] = key

    def resolve_cols(self, value: str | list[str], *, _seen: set[str] | None = None) -> list[str]:
        _seen = set() if _seen is None else _seen

        if isinstance(value, list):
            # use recursion for list inputs
            return list(itertools.chain.from_iterable(self.resolve_cols(v, _seen=_seen) for v in value))

        # check for cyclic refs
        if value in _seen:
            raise RuntimeError(f"Cyclic group reference detected for '{value}'")
        _seen.add(value)

        # first check if it is column (prioritize col over group, helps prevent infinite recursion)
        if value in self.tmp[self.active].columns:
            return [value]

        # resolve group or raise if not found in groups
        groups = self.state.get("groups", {})
        if value not in groups:
            return [value]  # up to downstream to ensure this col actually exists

        return self.resolve_cols(groups[value], _seen=_seen)

    def merge(
        self,
        df: pd.DataFrame, /, *,
        to: str,
        on: str | list[str],
        how: Literal['left', 'right', 'inner', 'outer', 'cross'] = "left",
        **kwargs
    ) -> Self:
        if to not in self.tmp:
            self.tmp[to] = df.copy(deep=True)
            return self

        self.tmp[to] = self.tmp[to].merge(df, on=on, how=how, **kwargs)
        return self

    def commit(self, df: pd.DataFrame, /, on: str | list[str], **kwargs) -> Self:
        if self.main.empty:
            self.main = df.copy(deep=True)
            return self

        self.main = self.main.merge(df, on=on, how="left", **kwargs)
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


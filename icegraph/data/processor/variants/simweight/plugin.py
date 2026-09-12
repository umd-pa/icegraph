# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any, Callable
from collections.abc import Mapping

import numpy as np
import polars as pl
import simweights
from simweights import Weighter

from icegraph.data.processor import Processor
from icegraph.data.envelope import Envelope

# internal shim
from icegraph._internal import to_dict

from .config import SimWeightConfig
from .surface import describe

__all__ = ["SimWeighter"]

import logging
logger = logging.getLogger(__name__)


# simweights exposes one weighter per simulation type, each reading the generation
# parameters out of that type's weighting tables. resolved by name at build time so
# an installation missing one of them only fails if that one is configured
_WEIGHTERS: dict[str, str] = {
    "nugen":    "NuGenWeighter",
    "corsika":  "CorsikaWeighter",
    "genie":    "GenieWeighter"
}


class SimWeighter(Processor[SimWeightConfig]):
    """Extract the per-event generation quantities and generation surface used to weight simulation."""
    name: ClassVar[str] = "i3-simweight"
    version: ClassVar[int] = 1

    _weighter_constructor: Callable[..., Weighter]

    def build(self) -> None:
        weighter_constructor = getattr(simweights, _WEIGHTERS[self.config.weighter], None)

        # make sure simweights actually provides weighter
        if weighter_constructor is None:
            raise RuntimeError(
                f"The installed simweights does not provide {_WEIGHTERS[self.config.weighter]}, required by "
                f"weighter {self.config.weighter}."
            )

        # should be a callable, assert just in case
        assert callable(weighter_constructor)
        self._weighter_constructor = weighter_constructor

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> SimWeightConfig:
        return SimWeightConfig(**config)

    def _resolve_ids(
            self, frames: Mapping[str, pl.DataFrame], ids: list[str], height: int
    ) -> pl.DataFrame:
        """Take the ids keying the weight columns from the weighting tables themselves."""
        reference: pl.DataFrame | None = None
        source: str | None = None
        rejected: dict[str, str] = {}

        for key in frames:
            frame = frames[key]

            if not set(ids) <= set(frame.columns):
                rejected[key] = "missing id columns"
                continue

            if frame.height != height:
                rejected[key] = f"{frame.height} rows"
                continue

            frame = frame.select(ids)

            # a table repeating an id is indexed by something finer than the event
            # so it cannot be the source of a per-event weight
            if frame.is_duplicated().any():
                rejected[key] = "ids are not unique"
                continue

            if reference is None:
                reference, source = frame, key
                continue

            if not frame.equals(reference):
                raise RuntimeError(
                    f"{type(self).name}: tables {source!r} and {key!r} both look like the source "
                    f"of the weight columns but disagree on the id columns {ids}. Cannot determine "
                    f"which rows the weights belong to."
                )

        if reference is None:
            raise RuntimeError(
                f"{type(self).name}: no weighting table carries the id columns {ids} over "
                f"{height} unique rows, so the weight columns cannot be keyed. Tables "
                f"considered: {rejected}."
            )

        logger.debug("stage=%s: keyed %d generation rows on %s", type(self).name, height, source)

        return reference

    def _process(self, item: Envelope) -> Envelope | None:
        kwargs: dict[str, Any] = {}
        if self.config.nfiles is not None:
            kwargs["nfiles"] = self.config.nfiles

        # only the tables the weighter reads, so nothing unrelated is offered to it
        # or considered when the weight columns are keyed
        tables = item.resolve_cols(self.config.tables)
        quiver = item.quiver.subset(tables)

        assert self._weighter_constructor is not None
        weighter = self._weighter_constructor(quiver, **kwargs)

        # resolve all cols
        ids = item.resolve_cols(self.config.ids)
        cols = item.resolve_cols(self.config.cols)

        columns: list[pl.Series] = []
        for name in cols:

            # if weighter does not contain requested col, raise
            try:
                values = weighter.get_weight_column(name)
            except KeyError:
                raise KeyError(
                    f"{type(self).__name__}: unknown weight column '{name}'. "
                    f"Available: {weighter.colnames}"
                )

            # ensure one to one
            if values.ndim != 1:
                raise RuntimeError(
                    f"{type(self).__name__}: weight column {name!r} has shape {values.shape}, "
                    f"expected one value per event."
                )

            # convert to polars series and append as a new column
            columns.append(pl.Series(name, values))

        # ensure each column is of same size
        heights = {series.len() for series in columns}
        if len(heights) != 1:
            widths = {series.name: series.len() for series in columns}
            raise RuntimeError(
                f"{type(self).__name__}: weight columns disagree on row count: {widths}."
            )

        frame = self._resolve_ids(quiver, ids, heights.pop())

        item.tmp[self.config.to] = frame.with_columns(columns)

        # one surface per file, summed across shards on load
        item.set_local_attr(self.config.attr, to_dict(weighter.surface))

        return item

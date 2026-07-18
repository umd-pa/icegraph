# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any, Callable

import numpy as np
import polars as pl
import simweights

from icegraph.data.processor import Processor
from icegraph.data.envelope import Envelope

from .config import SimweighterConfig
from .model import FluxModelFactory, FluxModelContext

__all__ = ["Simweighter"]


class Simweighter(Processor[SimweighterConfig]):
    """Load a frame into tmp for processing."""
    name: ClassVar[str] = "simweights"
    version: ClassVar[int] = 1

    def build(self) -> None:
        return

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> SimweighterConfig:
        return SimweighterConfig(**config)

    def _process(self, item: Envelope) -> Envelope | None:
        active = self._require_active(item)
        main = item.tmp[active]

        # resolve flux model class and weighter class
        weighter_cls = self._get_weighter(self.config.weighter)

        # instantiate flux model
        flux_model = FluxModelFactory.create(self.config.flux.name, **self.config.flux.kwargs)
        flux_model.attach(FluxModelContext())

        # construct the weighter over the quiver's numpy view
        # (simweights consumes mappings of table -> dict of column arrays)
        # nfiles is a valid and required parameter, weighter was apparently not typed properly
        weighter = weighter_cls(item.quiver.arrays(), nfiles=1)  # pyright: ignore[reportCallIssue]

        # compute weights
        weights = weighter.get_weights(flux_model)

        # weights must align with the active frame row count
        if len(weights) != len(main):
            raise RuntimeError(
                f"simweights returned {len(weights)} weights but active frame "
                f"{item.active!r} has {len(main)} rows. The active frame likely "
                f"diverged from the quiver table ordering after some upstream processor."
            )

        # write to active frame
        item.tmp[active] = main.with_columns(
            pl.Series(self.config.out, np.asarray(weights, dtype=np.float64))
        )

        # cache weight_group to attrs
        item.set_local_attr("weight_group", self.config.weight_group)

        return item

    @staticmethod
    def _get_weighter(name: str) -> Callable[..., simweights.Weighter]:
        weighter = getattr(simweights, name, None)

        # ensure the user requested a valid weighter
        if weighter is None:
            raise ValueError(f"Requested weighter '{name}' does not exist.")

        if not callable(weighter):
            raise TypeError(
                f"{name!r} must be callable (a Weighter factory or subclass); "
                f"got {type(weighter).__name__}"
            )

        # again weighter was apparently not typed properly
        return weighter  # pyright: ignore[reportReturnType]

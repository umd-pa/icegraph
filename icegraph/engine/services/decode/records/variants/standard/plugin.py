# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, ClassVar, Sequence

import numpy as np
import torch
from torch import Tensor

from icegraph.common.record import RecordBlock, Column

from ...decoder import RecordDecoder

from .config import StandardI3DecoderConfig
from .flux import build_flux, expected_source
from .surface import SurfaceSet, build_surfaces

__all__ = ["StandardI3Decoder"]

import logging
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SimGroup:
    """Simulation group."""
    simulation: str
    surface:    Any
    flux:       Any

    # source codes standing for this type, which is what routes a record to it
    codes:      tuple[int, ...]


class StandardI3Decoder(RecordDecoder[StandardI3DecoderConfig]):
    """Record decoder for IceCube data written by the IceGraph pipeline.

    Decodes features, targets and auxiliary columns straight out of the block, and
    turns the per-event generation columns the ``i3-simweight`` processor wrote into
    final weights against a configured flux.

    A dataset may mix simulation types.
    """
    name: ClassVar[str] = "standard"
    version: ClassVar[int] = 1

    def build(self) -> None:
        return

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> StandardI3DecoderConfig:
        return StandardI3DecoderConfig(**config)

    def extract(self, block: RecordBlock, key: str) -> Column | None:
        return block.columns.get(key)

    ### WEIGHTS

    @cached_property
    def _surfaces(self) -> SurfaceSet:
        """The dataset's generation surfaces, and how a record is routed to one."""
        return build_surfaces(self._ctx.attrs, key=self.config.surface_attr)

    @cached_property
    def _groups(self) -> tuple[_SimGroup, ...]:
        """One weighting group per simulation type the dataset carries."""
        # dict of simulation keyed to its aggregate surface
        surfaces = self._surfaces.surfaces

        # surfaces missing flux config (fatal)
        missing = sorted(simulation for simulation in surfaces if simulation not in self.config.flux)
        if missing:
            raise ValueError(
                f"{type(self).__name__}: the dataset carries {missing} generation data with no "
                f"flux configured for it, so those records cannot be weighted. Every simulation "
                f"type in the data needs an entry under the decoder's 'flux' option, which "
                f"currently holds {sorted(self.config.flux)}."
            )

        # flux config missing surfaces (non-fatal)
        for simulation in sorted(set(self.config.flux) - set(surfaces)):
            logger.warning(
                "a flux is configured for %s but no shard was loaded with it", simulation
            )

        # construct sim groups
        groups: list[_SimGroup] = []
        for simulation, surface in surfaces.items():
            config = self.config.flux[simulation]

            # weighting neutrino generation with a cosmic ray flux (or the reverse)
            # produces numbers rather than an error
            # this only works if the simulation name is the default in shard attrs
            # mostly just a helpful check to catch simple mistakes
            expected = expected_source(simulation)
            if expected is not None and config.source != expected:
                logger.warning(
                    "%s data is normally weighted with a %s flux, got %s model %s",
                    simulation, expected, config.source, config.name
                )

            # construct the sim group for this simulation
            group = _SimGroup(
                simulation=simulation,
                surface=surface,
                flux=build_flux(config),
                codes=tuple(
                    code for code, owner in self._surfaces.codes.items() if owner == simulation
                )
            )

            groups.append(group)

        return tuple(groups)

    def _weight_columns(
            self, column: Column, key: str, height: int
    ) -> tuple[np.ndarray, list[str], list[str | None]]:
        """Validate a generation column and name the columns packed into it."""
        values = column.values

        # [B] -> [B, 1]
        if values.ndim == 1:
            values = values[:, None]

        # generation data cannot be ragged
        if values.ndim != 2 or values.shape[0] != height:
            raise ValueError(
                f"{type(self).__name__}: column {key!r} must hold exactly one row per record to "
                f"be generation data, got {values.shape[0]} row(s) of shape {values.shape[1:]} "
                f"for {height} records."
            )

        names = self._ctx.columns(key)

        if len(names) != values.shape[1]:
            raise ValueError(
                f"{type(self).__name__}: 'columns.{key}.names' lists {len(names)} column(s) "
                f"{names}, but received {values.shape[1]} columns."
            )

        # reassign correct dtype
        dtypes = self._ctx.dtypes(key) or [None] * len(names)

        return values, names, dtypes

    @staticmethod
    def _unpack(
            values: np.ndarray, keep: list[tuple[int, str, str | None]]
    ) -> dict[str, np.ndarray]:
        """Split a generation table into the arrays the weighter reads, by name."""
        return {
            name: values[:, i] if dtype is None else values[:, i].astype(dtype, copy=False)
            for i, name, dtype in keep
        }

    def _weights_for(self, group: _SimGroup, columns: dict[str, np.ndarray]) -> np.ndarray:
        """Weight the given generation columns against its group."""
        import simweights

        # no file objects, only using rebuilt surface
        weighter = simweights.Weighter([], group.surface)

        for name, values in columns.items():
            weighter.add_weight_column(name, values)

        return weighter.get_weights(group.flux).reshape(-1)

    def _routed_weights(
            self, groups: Sequence[_SimGroup], values: np.ndarray,
            keep: list[tuple[int, str, str | None]], source: np.ndarray
    ) -> np.ndarray:
        """Weight a block of mixed simulation, each type against its own surface."""
        weights = np.zeros(len(source), dtype=np.float32)
        routed = np.zeros(len(source), dtype=bool)

        for group in groups:
            mask = np.isin(source, group.codes)

            if not mask.any():
                continue

            # one gather over the table, rather than one per column
            weights[mask] = self._weights_for(group, self._unpack(values[mask], keep))
            routed |= mask

        if not routed.all():
            raise ValueError(
                f"{type(self).__name__}: {int((~routed).sum())} record(s) carry source code(s) "
                f"{sorted(np.unique(source[~routed]).tolist())} that no generation surface in the "
                f"dataset declares, so they cannot be weighted."
            )

        return weights

    def _extract_weights(self, block: RecordBlock, key: str) -> Tensor | None:
        column = self.extract(block, key)

        if column is None:
            return None

        groups = self._groups
        values, names, dtypes = self._weight_columns(column, key, block.height)

        # the source code labels the record
        routing = self._surfaces.column

        if routing not in names:
            raise KeyError(
                f"{type(self).__name__}: the shards declare their source code in column "
                f"{routing!r} of {key!r}, but it could not be found."
            )

        # the label is not a quantity the weighter reads
        keep = [(i, name, dtypes[i]) for i, name in enumerate(names) if name != routing]

        # one simulation type needs no routing
        if len(groups) == 1:
            weights = self._weights_for(groups[0], self._unpack(values, keep)).astype(np.float32)
        else:
            source = values[:, names.index(routing)].astype(np.int64)
            weights = self._routed_weights(groups, values, keep, source)

        return torch.from_numpy(weights)

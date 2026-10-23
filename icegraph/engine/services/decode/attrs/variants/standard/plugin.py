# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any, ClassVar, Iterator, Callable
import itertools
from operator import add
from functools import reduce

import numpy as np

from icegraph.statistics import StatisticService
from icegraph.common.data import AttributeDomain
from icegraph.typing.common import ArrayI
from icegraph.common.record import GlobalAttributes, Attributes, validate_attrs

from ...decoder import AttributeDecoder

from .config import StandardAttributeDecoderConfig
from .schema import ColumnMetadata, GlobalColumns, SplitMap, Stats

__all__ = ["StandardAttributeDecoder"]


class StandardAttributeDecoder(AttributeDecoder[StandardAttributeDecoderConfig]):
    name: ClassVar[str] = "standard"
    version: ClassVar[int] = 1

    def build(self) -> None:
        return

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> StandardAttributeDecoderConfig:
        return StandardAttributeDecoderConfig(**config)

    @staticmethod
    def _metadata(role: str, global_attrs: GlobalAttributes) -> ColumnMetadata | None:
        """Column metadata for a stored key, None if not found."""
        columns = validate_attrs(
            GlobalColumns, dict(global_attrs), source="the dataset global attributes"
        ).columns

        return columns.get(role)

    def _extract_columns(
            self, role: str, *,
            attrs: Callable[[], Iterator[Attributes]], global_attrs: GlobalAttributes
    ) -> list[str] | None:
        metadata = self._metadata(role, global_attrs)

        return None if metadata is None else metadata.names

    def _extract_dtypes(
            self, role: str, *,
            attrs: Callable[[], Iterator[Attributes]], global_attrs: GlobalAttributes
    ) -> list[str] | None:
        metadata = self._metadata(role, global_attrs)

        return None if metadata is None else metadata.dtypes

    def _extract_offsets(
            self, role: str, *,
            attrs: Callable[[], Iterator[Attributes]], global_attrs: GlobalAttributes
    ) -> ArrayI | None:
        metadata = self._metadata(role, global_attrs)

        return None if metadata is None else metadata.offset.astype(np.int64)

    def _extract_keys(
            self, split: int, *,
            attrs: Callable[[], Iterator[Attributes]], global_attrs: GlobalAttributes
    ) -> ArrayI:
        # load the splitmap from dataset attrs, these are ordered so this is correct
        splitmaps = (
            validate_attrs(
                SplitMap, attr[AttributeDomain.LOCAL], source=f"shard ID={attr.shard_id}"
            ).splitmap
            for attr in attrs()
        )

        # built full dataset splitmap
        splitmap = np.fromiter(itertools.chain.from_iterable(splitmaps), dtype=np.uint8)

        # build the mask and return
        return np.where(splitmap == split)[0]

    @staticmethod
    def _build_stat_service(attr: Attributes, split: int, role: str) -> StatisticService | None:
        stats = validate_attrs(
            Stats, attr[AttributeDomain.LOCAL], source=f"shard ID={attr.shard_id}"
        ).stats

        role_stats = stats.get(role)

        if role_stats is None:
            raise KeyError(
                f"Local attribute 'stats.{role}' not found in shard ID={attr.shard_id}."
            )

        struct = role_stats.get(str(split))

        if struct is None:
            # this indicates that stats are present, there are just no samples in the file
            # for this specific split, so thus no stats
            return None

        # the type of struct is responsibility of stat service to check
        return StatisticService.from_struct(struct)

    def _extract_stats(
            self, split: int, role: str, *,
            attrs: Callable[[], Iterator[Attributes]], global_attrs: GlobalAttributes
    ) -> StatisticService:
        stats = (
            s for attr in attrs()
            if (s := self._build_stat_service(attr, split, role)) is not None
        )

        try:
            first = next(stats)
        except StopIteration:
            raise RuntimeError("No shard statistics found; cannot compute aggregates.")

        # merge and return using functools reduce
        return reduce(add, stats, first)  # type: ignore[args]

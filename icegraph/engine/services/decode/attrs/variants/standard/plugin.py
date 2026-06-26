# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any, ClassVar, Iterator, Callable
import itertools
from operator import add
from functools import reduce
from collections import defaultdict

import numpy as np

from icegraph.statistics import StatisticService
from icegraph.common.data import AttributeDomain
from icegraph.typing.common import ArrayI
from icegraph.common.record import GlobalAttributes, Attributes

from ...decoder import AttributeDecoder

from .config import StandardAttributeDecoderConfig

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
    def _column_metadata(global_attrs: GlobalAttributes) -> dict[str, Any]:
        columns = global_attrs.get("columns")

        if columns is None:
            raise KeyError(
                "Missing key 'columns' in dataset global attributes."
            )

        if not isinstance(columns, dict):
            raise TypeError(
                f"Global attribute 'columns' must be a dict, "
                f"got {type(columns).__name__}."
            )

        return columns

    def _extract_columns(
            self, role: str, *,
            attrs: Callable[[], Iterator[Attributes]], global_attrs: GlobalAttributes
    ) -> list[str] | None:
        columns = self._column_metadata(global_attrs)
        metadata = columns.get(role)

        if metadata is None:
            # if no metadata is found for this role
            return None

        if not isinstance(metadata, dict):
            raise TypeError(
                f"Global attribute 'columns.{role}' must be a dict, "
                f"got {type(metadata).__name__}."
            )

        names = metadata.get("names")

        if names is None:
            # if the role is present, names need to be specified even if empty
            raise KeyError(
                f"Missing key 'columns.{role}.names' in dataset global attributes."
            )

        if not isinstance(names, list):
            raise TypeError(
                f"Global attribute 'columns.{role}.names' must be a list, "
                f"got {type(names).__name__}."
            )

        for i, item in enumerate(names):
            if not isinstance(item, str):
                raise TypeError(
                    f"Global attribute 'columns.{role}.names[{i}]' must be a str, "
                    f"got {type(item).__name__}."
                )

        return names

    def _extract_offsets(
            self, role: str, *,
            attrs: Callable[[], Iterator[Attributes]], global_attrs: GlobalAttributes
    ) -> ArrayI | None:
        columns = self._column_metadata(global_attrs)
        metadata = columns.get(role)

        if metadata is None:
            # if no metadata is found for this role
            return None

        if not isinstance(metadata, dict):
            raise TypeError(
                f"Global attribute 'columns.{role}' must be a dict, "
                f"got {type(metadata).__name__}."
            )

        offsets = metadata.get("offset")

        if offsets is None:
            raise KeyError(
                f"Missing key 'columns.{role}.offset' in dataset global attributes."
            )

        if not isinstance(offsets, np.ndarray):
            raise TypeError(
                f"Global attribute 'columns.{role}.offset' must be an ndarray, "
                f"got {type(offsets).__name__}."
            )

        if not np.issubdtype(offsets.dtype, np.integer):
            raise TypeError(
                f"Global attribute 'columns.{role}.offset' must have an integer dtype, "
                f"got {offsets.dtype}."
            )

        return offsets.astype(np.int64)

    def _extract_keys(
            self, split: int, *,
            attrs: Callable[[], Iterator[Attributes]], global_attrs: GlobalAttributes
    ) -> ArrayI:
        # load the splitmap from dataset attrs, these are ordered so this is correct
        splitmaps = (attr[AttributeDomain.LOCAL]["splitmap"] for attr in attrs())

        # built full dataset splitmap
        splitmap = np.fromiter(itertools.chain.from_iterable(splitmaps), dtype=np.uint8)

        # build the mask and return
        return np.where(splitmap == split)[0]

    @staticmethod
    def _build_stat_service(attr: Attributes, split: int, role: str) -> StatisticService | None:
        structs = attr[AttributeDomain.LOCAL].get("stats")

        if structs is None:
            raise RuntimeError(
                f"Local attribute 'stats' not found in shard ID={attr.shard_id}."
            )

        if not isinstance(structs, dict):
            raise TypeError(
                f"Local attribute 'stats' must be a dict, "
                f"got {type(structs).__name__} in shard ID={attr.shard_id}."
            )

        role_structs = structs.get(role)

        if role_structs is None:
            raise RuntimeError(
                f"Local attribute 'stats.{role}' not found in shard ID={attr.shard_id}."
            )

        if not isinstance(role_structs, dict):
            raise TypeError(
                f"Local attribute 'stats.{role}' must be a dict, "
                f"got {type(role_structs).__name__} in shard ID={attr.shard_id}."
            )

        stat_struct = role_structs.get(str(split))

        if stat_struct is None:
            # this indicates that stats are present, there are just no samples in the file
            # for this specific split, so thus no stats
            return None

        # the type of stat_struct is responsibility of stat service to check
        return StatisticService.from_struct(stat_struct)  # type: ignore

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

    def _extract_count_by_weight_group(
            self, *,
            attrs: Callable[[], Iterator[Attributes]], global_attrs: GlobalAttributes
    ) -> dict[str, int]:
        count: dict[str, int] = defaultdict(int)

        # loop over each shards attributes
        for attr in attrs():
            weight_group = attr[AttributeDomain.LOCAL].get("weight_group")

            if weight_group is None:
                raise RuntimeError(
                    f"Local attribute 'weight_group' not found in shard ID={attr.shard_id}. This error might "
                    "occur if you are trying to load simweights from a file that does not contain any."
                )

            if not isinstance(weight_group, str):
                raise TypeError(
                    f"Local attribute 'weight_group' must be a str, got "
                    f"{type(weight_group)} in shard ID={attr.shard_id}."
                )

            # count this weight group
            count[weight_group] += 1

        return count

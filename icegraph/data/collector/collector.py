# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Iterable, Iterator

import polars as pl

from icegraph.common.data import AttributeDomain
from icegraph.common.plugins import Plugin
from icegraph.utils.hashutils import CBORBlake2B

from ..envelope import Envelope
from ..quiver import QuiverIPC
from ..types import StageContext

from .config import CollectorConfig

__all__ = ["Collector"]


_LOCAL: str = AttributeDomain.LOCAL.name
_GLOBAL: str = AttributeDomain.GLOBAL.name


@dataclass
class _Block:
    """The envelopes of the files collected so far for one group."""
    parts:  list[Envelope]  = field(default_factory=list)
    nbytes: int             = 0
    events: int             = 0

    def add(self, item: Envelope) -> None:
        self.parts.append(item)
        self.nbytes += item.quiver.nbytes
        self.events += item.events.height


class Collector(Plugin[CollectorConfig, StageContext[Envelope]]):
    """
    Collects the envelopes of whole input files into blocks. Each block is passed
    downstream as one envelope and written as one shard.

    A block closes once its tables reach ``min_bytes`` and the next file of its group
    holds events, so a file without events always joins a block and is counted in it.
    """
    name: ClassVar[str] = "collector"
    version: ClassVar[int] = 1

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> CollectorConfig:
        return CollectorConfig(**config)

    def build(self) -> None:
        return

    def execute(self) -> None:
        """Collect every envelope from the source and emit the blocks downstream."""
        assert self._ctx.dst is not None

        for block in self.collect(self._ctx.src):
            self._ctx.dst.put(block)  # blocks when full, backpressure

    def collect(self, items: Iterable[Envelope]) -> Iterator[Envelope]:
        """Collect envelopes into blocks, numbering each in the order it closes."""
        blocks: dict[str, _Block] = {}
        shard = 0

        for item in items:
            group = self._group(item)
            block = blocks.setdefault(group, _Block())

            # close a full block only when a file with events arrives, so files without events always join a block
            if block.events and block.nbytes >= self.config.min_bytes and not item.events.is_empty():
                yield self._merge(block.parts, shard)
                shard += 1
                block = blocks[group] = _Block()

            block.add(item)

        for group, block in blocks.items():
            # a block without events cannot be written, so these files would go uncounted
            if not block.events:
                raise RuntimeError(
                    f"No file of group '{group}' holds events, so its {len(block.parts)} file(s) have no shard "
                    f"to be counted in. If they belong to a set whose other files are in another directory, "
                    f"set 'group_by: all'."
                )

            yield self._merge(block.parts, shard)
            shard += 1

    def _group(self, item: Envelope) -> str:
        """The group of a file, checking the extractor set what the collector reads."""
        origin = item.get_local_attr("origin")
        if origin is None:
            raise RuntimeError("The extractor must record the file of each envelope under attrs.LOCAL.origin.")

        # an events frame without columns means the extractor never set the ids
        if not item.events.columns:
            raise RuntimeError(f"The extractor set no event ids for file {origin}, see Envelope.events.")

        return str(Path(origin).parent) if self.config.group_by == "parent" else "all"

    @staticmethod
    def _merge(parts: list[Envelope], shard: int) -> Envelope:
        """One envelope holding every file of a block."""
        first = parts[0]
        origins: list[str] = [part.get_local_attr("origin") for part in parts]

        # the set id is a hash of the global attributes, so the files must share them
        hasher = CBORBlake2B()
        digest = hasher(first.attrs[_GLOBAL])
        for origin, part in zip(origins, parts):
            if hasher(part.attrs[_GLOBAL]) != digest:
                raise RuntimeError(f"File {origin} has different global attributes than {origins[0]}.")

            if part.events.columns != first.events.columns:
                raise RuntimeError(
                    f"File {origin} identifies events by {part.events.columns}, {origins[0]} by "
                    f"{first.events.columns}."
                )

        # the block is processed as one file, so no two of its events can share ids
        events = pl.concat([part.events for part in parts if not part.events.is_empty()])
        if events.is_duplicated().any():
            repeated = events.filter(events.is_duplicated()).head(1)
            holders = [
                origin for origin, part in zip(origins, parts)
                if not part.events.is_empty() and not part.events.join(repeated, on=first.ids, how="semi").is_empty()
            ]
            raise RuntimeError(
                f"Event {repeated.row(0, named=True)} appears more than once in the files {holders}, so the "
                f"ids {first.ids} do not identify an event within a block."
            )

        # each file keeps its own local attributes, keyed by the file
        sources: dict[str, dict[str, Any]] = {}
        for origin, part in zip(origins, parts):
            if origin in sources:
                raise RuntimeError(f"File {origin} was collected more than once.")

            sources[origin] = {key: value for key, value in part.attrs[_LOCAL].items() if key != "origin"}

        block = Envelope(quiver=QuiverIPC.merge(part.quiver for part in parts), events=events, shard=shard)
        block.attrs[_GLOBAL].update(first.attrs[_GLOBAL])
        block.set_local_attr("sources", sources)

        # state set by the extractor
        block.state.update(first.state)

        # time each stage spent on the block
        block.metrics = {key: sum(part.metrics.get(key, 0.0) for part in parts) for key in first.metrics}

        return block

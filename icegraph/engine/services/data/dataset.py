# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import random
from functools import cached_property
from typing import TYPE_CHECKING
from collections.abc import Iterator

from torch.utils.data import IterableDataset, get_worker_info
import torch

from icegraph.common.data import GraphData, DataRole
from icegraph.common.record import Record
from icegraph.typing.common import ArrayI

if TYPE_CHECKING:
    from icegraph.engine.services import ServiceManager

__all__ = ["GraphDataset"]


class GraphDataset(IterableDataset[GraphData]):
    """Iterable, DDP-safe dataset with two-level shuffling.

    Reads are kept chunk-granular for large files: the order of contiguous
    key-chunks is shuffled, reads within a chunk stay sequential.
    """

    def __init__(
        self,
        keys: ArrayI,
        *,
        services: ServiceManager,
        chunk_size: int,
        buffer_size: int,
        batch_size: int,
        shuffle_chunks: bool,
        exclude_roles: list[DataRole] | None = None,
    ) -> None:
        # keys stay in ascending order
        self.keys = keys
        self._services = services
        self._chunk_size = chunk_size
        self._buffer_size = buffer_size
        self._batch_size = batch_size
        self._shuffle_chunks = shuffle_chunks
        self._exclude_roles = exclude_roles if exclude_roles is not None else []

        # epoch needs to be updated for each worker
        # so epoch has to be a scalar tensor with shared memory
        self._epoch = torch.zeros(1, dtype=torch.long).share_memory_()

    def set_epoch(self, epoch: int) -> None:
        """Sets epoch, visible to all workers."""
        self._epoch[0] = epoch

    ### SEEDING

    @staticmethod
    def _seed(*parts: int) -> int:
        # merge parts into a single int. FNV-1a is deterministic across processes.
        h = 0xCBF29CE484222325
        for p in parts:
            h = ((h ^ (int(p) & 0xFFFFFFFFFFFFFFFF)) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        return h

    ### SHARDING

    def _chunks(self) -> list[tuple[int, int]]:
        count = len(self.keys) // self._chunk_size  # number of whole chunks

        # quick check, rare but very shitty to debug if not caught
        if count == 0:
            raise RuntimeError(
                f"Cannot split {len(self.keys)} items into chunks of size {self._chunk_size}."
            )

        chunks = []
        # iterate over total chunk count, dropping partial last chunk
        for i in range(count):
            start   = i * self._chunk_size
            end     = (i + 1) * self._chunk_size

            # append as a tuple range [start, end)
            chunks.append((start, end))

        return chunks

    def _assign_chunks(self) -> list[tuple[int, int]]:
        state = self._services.require("state", required_by=type(self))
        rank, world = state.rank, state.world

        chunks = self._chunks()

        # disjoint chunk partitions
        if self._shuffle_chunks:
            random.Random(self._seed(state.seed, int(self._epoch[0]))).shuffle(chunks)

        # equal chunks per rank for DDP
        # Drop the remainder so counts match across ranks
        # unequal per-rank counts cause deadlocks
        usable      = (len(chunks) // world) * world
        rank_chunks = chunks[:usable][rank::world]

        # whole chunks per worker
        # within-rank unevenness is fine, only per rank totals must match
        worker = get_worker_info()
        if worker is not None:
            rank_chunks = rank_chunks[worker.id::worker.num_workers]

        return rank_chunks

    ### SAMPLE BUILDING

    def _build(self, sample: Record) -> GraphData:
        decoder = self._services.require("decode", required_by=type(self))

        data = GraphData()
        for role in DataRole.all():
            if role == DataRole.BATCH:
                continue

            loader = getattr(decoder, f"load_{role.value}")
            setattr(
                data, role.value, loader(
                    sample, excluded=role in self._exclude_roles
                )
            )

        # excluded features carries shape [N=1, F=0] so this is fine either way
        data.num_nodes = data.features.shape[0]
        return data

    def __iter__(self) -> Iterator[GraphData]:
        records = self._services.require("record", required_by=type(self))

        # contiguous, ascending key blocks; never straddles a chunk, one read each
        blocks = (
            [self._build(s) for s in records[self.keys[start:min(start + self._batch_size, hi)]]]
            for lo, hi in self._assign_chunks()
            for start in range(lo, hi, self._batch_size)
        )

        if self._buffer_size > 1:
            yield from self._buffer_shuffle(blocks)
        else:
            #  self._buffer_size == 1 -> no fine shuffle
            yield from (item for block in blocks for item in block)

    def _buffer_shuffle(self, blocks: Iterator[list[GraphData]]) -> Iterator[GraphData]:
        state  = self._services.require("state", required_by=type(self))
        worker = get_worker_info()
        wid    = worker.id if worker is not None else 0

        # include rank and worker so disjoint shards decorrelate
        # include epoch so mixing changes each epoch
        rng = random.Random(self._seed(state.seed, int(self._epoch[0]), state.rank, wid))

        buffer: list[GraphData] = []
        for block in blocks:
            buffer.extend(block)

            # drain back down to buffer_size before pulling the next block
            while len(buffer) > self._buffer_size:
                i = rng.randrange(len(buffer))
                buffer[i], buffer[-1] = buffer[-1], buffer[i]
                yield buffer.pop()

        rng.shuffle(buffer)
        yield from buffer

    @cached_property
    def _sample_count(self) -> int:
        state = self._services.require("state", required_by=type(self))
        num_chunks = len(self.keys) // self._chunk_size

        # for inference state.world must be 1 or some items might not be processed
        return (num_chunks // state.world) * self._chunk_size

    def __len__(self) -> int:
        return self._sample_count
# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import random
import math
from functools import cached_property
from typing import TYPE_CHECKING
from collections.abc import Iterator
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future

from torch.utils.data import IterableDataset, get_worker_info
import torch
import numpy as np

from icegraph.common.data import DataRole, RawGraphBatch
from icegraph.common.record import RecordBlock
from icegraph.typing.common import ArrayI

if TYPE_CHECKING:
    from icegraph.engine.services import ServiceManager

__all__ = ["GraphDataset"]


class GraphDataset(IterableDataset[RawGraphBatch]):
    """Iterable, DDP-safe dataset with two-level shuffling.

    Reads are kept chunk-granular for large files: the order of contiguous
    key-chunks is shuffled, reads within a chunk stay sequential. Chunks are
    then gathered into groups of ~``buffer_size`` samples, permuted per record,
    and sliced into batches.
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
        buffer_refill_threshold: float,
        exclude_roles: list[DataRole] | None = None,
    ) -> None:
        # keys stay in ascending order
        self.keys = keys
        self._services = services
        self._chunk_size = chunk_size
        self._buffer_size = buffer_size
        self._batch_size = batch_size
        self._shuffle_chunks = shuffle_chunks
        self._buffer_refill_threshold = buffer_refill_threshold
        self._exclude_roles = frozenset(exclude_roles) if exclude_roles is not None else frozenset()

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

    def assemble_batch(
            self,
            block: RecordBlock,
            exclude_roles: frozenset[DataRole] = frozenset(),
    ) -> RawGraphBatch:
        """Assemble one batch from a columnar block of records."""
        decode = self._services.require("decode", required_by=type(self))

        height = block.height

        features, node_counts = decode.load_features(block, excluded=DataRole.FEATURES in exclude_roles)

        # per graph node offsets drive the batch vector and the edge shifts
        ptr = np.zeros(height + 1, dtype=np.int64)
        np.cumsum(node_counts, out=ptr[1:])

        batch = np.repeat(np.arange(height, dtype=np.int64), node_counts)

        edge_index, edge_counts = decode.load_edge_index(block, excluded=DataRole.EDGE_INDEX in exclude_roles)

        if edge_index.numel():
            # shift each graphs node ids by its node offset
            edge_index = edge_index + torch.from_numpy(np.repeat(ptr[:-1], edge_counts))

        return RawGraphBatch(
            features=features,
            targets=decode.load_targets(block, excluded=DataRole.TARGETS in exclude_roles),
            auxiliary=decode.load_auxiliary(block, excluded=DataRole.AUXILIARY in exclude_roles),
            simweights=decode.load_simweights(block, excluded=DataRole.SIMWEIGHT in exclude_roles),
            edge_index=edge_index,
            edge_attr=decode.load_edge_attr(block, excluded=DataRole.EDGE_ATTR in exclude_roles),
            batch=torch.from_numpy(batch),
            ptr=torch.from_numpy(ptr),
        )

    def __iter__(self) -> Iterator[RawGraphBatch]:
        records = self._services.require("record", required_by=type(self))
        state = self._services.require("state", required_by=type(self))
        worker = get_worker_info()
        wid = worker.id if worker is not None else 0
        # include rank and worker so disjoint shards decorrelate
        # include epoch so mixing changes each epoch
        rng = np.random.default_rng(self._seed(state.seed, int(self._epoch[0]), state.rank, wid))

        shuffle = self._buffer_size > 1
        pending = deque(self._assign_chunks())
        refill = max(1, round(self._buffer_size * (1 - self._buffer_refill_threshold)))
        refill_threshold = max(self._batch_size, (self._buffer_size - refill) * self._chunk_size)

        def claim(count: int) -> list[tuple[int, int]]:
            """Remove up to ``count`` chunk ranges from the front of the queue."""
            return [pending.popleft() for _ in range(min(count, len(pending)))]

        def read(group: list[tuple[int, int]]) -> list[RecordBlock]:
            """One sequential read per chunk, dropping empties.

            Runs on the prefetch thread: zarr and blosc release the GIL for both
            the store read and the decode, so this overlaps with shuffle/collate.
            """
            blocks: list[RecordBlock] = []
            for lo, hi in group:
                block = records.read(self.keys[lo:hi])
                if block.height:
                    blocks.append(block)
            return blocks

        io = ThreadPoolExecutor(max_workers=1, thread_name_prefix="loader-prefetch")
        try:
            inflight: Future[list[RecordBlock]] | None = (
                io.submit(read, claim(self._buffer_size)) if pending else None
            )
            carry: list[RecordBlock] = []  # survivors of the previous pool (zero or one block)

            while inflight is not None or carry:
                # --- refill
                fresh = inflight.result() if inflight is not None else []
                # hand the thread its next group before doing any work ourselves
                inflight = io.submit(read, claim(refill)) if pending else None

                parts, carry = [*carry, *fresh], []
                if not parts:
                    continue  # a group of empty chunks; not the end of the epoch

                pool = RecordBlock.concat(parts)
                order = rng.permutation(pool.height) if shuffle else np.arange(pool.height)

                # --- drain
                cursor = 0
                while cursor < pool.height:
                    remaining = pool.height - cursor

                    # stop short so survivors carry into the next pool: the mixing
                    # window slides instead of resetting at the group boundary
                    if inflight is not None and remaining < refill_threshold:
                        carry = [pool.take(order[cursor:])]
                        break

                    size = min(self._batch_size, remaining)
                    block = pool.take(order[cursor:cursor + size])
                    cursor += size

                    yield self.assemble_batch(block, self._exclude_roles)

        finally:
            io.shutdown(wait=False, cancel_futures=True)

    @cached_property
    def _batch_count(self) -> int:
        state = self._services.require("state", required_by=type(self))
        num_chunks = len(self.keys) // self._chunk_size

        # for inference state.world must be 1 or some items might not be processed
        samples = (num_chunks // state.world) * self._chunk_size

        # exact when batch_size divides the group size; otherwise per-worker
        # tail batches make this a lower bound (same approximation as before)
        return math.ceil(samples / self._batch_size)

    def __len__(self) -> int:
        return self._batch_count
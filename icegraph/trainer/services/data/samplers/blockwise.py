# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import math
from typing import Iterator

import torch
import torch.distributed as dist
from torch.utils.data import Sampler

from ..types import SizedDataset

__all__ = ["DistributedBlockwiseSampler"]


class DistributedBlockwiseSampler(Sampler[int]):
    """
    Globally shuffles block order, assigns whole blocks to ranks, and shuffles
    indices within each block.
    """

    def __init__(
            self,
            dataset: SizedDataset,
            block_size: int, *,
            num_replicas: int | None = None,
            rank: int | None = None,
            shuffle: bool = False
    ) -> None:
        super().__init__()

        # get dataset length
        self.n = len(dataset)

        # ensure block size is valid
        if block_size <= 0:
            raise ValueError("block_size must be > 0")

        self.block_size = int(block_size)
        self.shuffle = shuffle

        # set initial epoch
        self.epoch: int = 0

        # get world size and rank
        self.world  = 1 if num_replicas is None else num_replicas
        self.rank   = 0 if rank         is None else rank

        # get total block count
        self.block_count = math.ceil(self.n / self.block_size)

        # pad blocks so each rank gets the same count
        self.padded_block_count = math.ceil(self.block_count / self.world) * self.world

        # compute sample count per rank
        self.n_rankwise = (self.padded_block_count // self.world) * self.block_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.n_rankwise

    def _epoch_seed(self) -> int:
        """
        Create a single per-epoch seed shared across ranks, using the default RNG.
        """
        seed = torch.empty((), dtype=torch.int64)
        if self.rank == 0:
            seed.random_()

        # broadcast to other ranks if in ddp
        if self.world > 1:
            dist.broadcast(seed, src=0)

        # mix in epoch so each epoch reshuffles even if RNG state repeats
        return int(seed.item()) ^ (0x9E3779B97F4A7C15 * self.epoch)

    def __iter__(self) -> Iterator[int]:
        # make this epoch deterministic across ranks
        state = torch.random.get_rng_state()
        torch.manual_seed(self._epoch_seed())

        try:
            # global block shuffle if shuffle, else just return block ids in order
            blocks = torch.randperm(self.block_count).tolist() if self.shuffle else range(self.block_count)
            if self.padded_block_count > self.block_count:
                blocks += blocks[: self.padded_block_count - self.block_count]

            # get assigned blocks
            assigned_blocks = blocks[self.rank :: self.world]

            for block in assigned_blocks:
                start = block * self.block_size

                # shuffle elements within each block if shuffle is true, else yield the block
                indices = torch.randperm(self.block_size).tolist() if self.shuffle else range(self.block_size)
                for index in indices:
                    yield (start + index) % self.n
        finally:
            # restore RNG state so using this sampler doesn't perturb user RNG
            torch.random.set_rng_state(state)

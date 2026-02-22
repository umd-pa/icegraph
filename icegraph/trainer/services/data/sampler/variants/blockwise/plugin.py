# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Iterator, ClassVar, Any
import math

import torch

from ...sampler import Sampler

from .config import Config

__all__ = ["BlockwiseSampler"]


class BlockwiseSampler(Sampler[Config]):
    """
    Globally shuffles block order, assigns whole blocks to ranks, and shuffles
    indices within each block.
    """
    name: ClassVar[str] = "blockwise"
    version: ClassVar[int] = 1

    # for the type checker
    _n:                     int
    _block_count:           int
    _padded_block_count:    int
    _n_rankwise:            int

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> Config:
        return Config(**config)

    def build(self) -> None:
        # get dataset length
        self._n = len(self._dataset)

        # get total block count
        self._block_count = math.ceil(self._n / self.config.block_size)

        # pad blocks so each rank gets the same count
        self._padded_block_count = math.ceil(self._block_count / self._num_replicas) * self._num_replicas

        # compute sample count per rank
        self._n_rankwise = (self._padded_block_count // self._num_replicas) * self.config.block_size

    def __len__(self) -> int:
        return self._n_rankwise

    def __iter__(self) -> Iterator[int]:
        # make this epoch deterministic across ranks
        state = torch.random.get_rng_state()
        torch.manual_seed(self.seed())

        try:
            # global block shuffle if shuffle, else just return block ids in order
            blocks = torch.randperm(self._block_count).tolist() if self._shuffle else range(self._block_count)
            if self._padded_block_count > self._block_count:
                blocks += blocks[: self._padded_block_count - self._block_count]

            # get assigned blocks
            assigned_blocks = blocks[self._rank :: self._num_replicas]

            for block in assigned_blocks:
                start = block * self.config.block_size

                # shuffle elements within each block if shuffle is true, else yield the block
                indices = torch.randperm(self.config.block_size).tolist() if self._shuffle else range(self.config.block_size)
                for index in indices:
                    yield (start + index) % self._n
        finally:
            # restore RNG state so using this sampler doesn't perturb user RNG
            torch.random.set_rng_state(state)

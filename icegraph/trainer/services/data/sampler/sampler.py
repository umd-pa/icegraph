# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Iterator, TypeVar
from abc import abstractmethod

import torch
import torch.distributed as dist
from torch.utils.data import Sampler as _Sampler

from icegraph.types.plugins import Plugin

from ..types import SizedDataset

from .types import SamplerContext

__all__ = ["Sampler"]


C = TypeVar("C")


class Sampler(Plugin[C, SamplerContext], _Sampler[int]):

    _dataset:       SizedDataset
    _num_replicas:  int
    _rank:          int
    _shuffle:       bool
    _epoch:         int

    def on_attach(self) -> None:
        self._dataset       = self._ctx.dataset
        self._num_replicas  = self._ctx.num_replicas if self._ctx.num_replicas is not None else 1
        self._rank          = self._ctx.rank if self._ctx.rank is not None else 0
        self._shuffle       = self._ctx.shuffle

        # set initial epoch
        self._epoch         = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def seed(self) -> int:
        """Create a single per-epoch seed shared across ranks."""
        seed = torch.empty((), dtype=torch.int64)
        if self._rank == 0:
            seed.random_()

        # broadcast to other ranks if in ddp
        if self._num_replicas > 1:
            dist.broadcast(seed, src=0)

        # mix in epoch so each epoch reshuffles even if RNG state repeats
        return int(seed.item()) ^ (0x9E3779B97F4A7C15 * self._epoch)

    @abstractmethod
    def __len__(self) -> int:
        """Rank-aware dataset length."""
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[int]:
        ...

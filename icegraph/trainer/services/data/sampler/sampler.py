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

    _epoch: int

    def build(self) -> None:
        # set initial epoch
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def seed(self) -> int:
        """Create a single per-epoch seed shared across ranks."""
        device = self._ctx.device
        max_seed = 2 ** 31 - 1

        seed = torch.empty((), dtype=torch.int64, device=device)
        if self._ctx.rank == 0:
            seed.random_(0, max_seed)

        num_replicas = self._ctx.num_replicas
        if num_replicas is not None and num_replicas > 1:
            dist.broadcast(seed, src=0)

        return (int(seed.item()) + self._epoch) % max_seed

    @abstractmethod
    def __len__(self) -> int:
        """Rank-aware dataset length."""
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[int]:
        ...

# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any, ClassVar
import multiprocessing as mp
from functools import cached_property

import torch

from icegraph.common.data import Split
from icegraph.common.mapping import MemoMap

from ..service import Service

from .config import DataConfig
from .dataset import GraphDataset
from .loader import GraphDataLoader
from .sampler import Sampler, SamplerFactory, SamplerContext

__all__ = ["DataService"]


class DataService(Service[DataConfig]):
    """
    A container class for managing access to training, validation, and test datasets.
    """
    name: ClassVar[str] = "data"
    version: ClassVar[int] = 1

    deps: ClassVar[tuple[str, ...]] = ("record",)

    def build(self) -> None:
        return

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> DataConfig:
        return DataConfig(**config)

    def _build_dataset(self, split: Split) -> GraphDataset:
        return GraphDataset(split, self._ctx.services)

    @cached_property
    def _datasets(self) -> MemoMap[Split, GraphDataset]:
        return MemoMap(self._build_dataset)

    def dataset(self, split: Split) -> GraphDataset:
        return self._datasets[split]

    def _build_sampler(self, split: Split) -> Sampler[Any]:
        config = self.config.sampler

        is_train = (split == Split.TRAIN)

        # get state from the service manager
        state = self._ctx.services.require("state", required_by=type(self))

        # get world and rank from state, only ddp for training, eval runs only on main rank
        # this is enforced by the trainer, here we have to just ensure it gets a sampler with rank 0 and world 1
        rank = state.rank if is_train else 0
        world = state.world if is_train else 1

        sampler = SamplerFactory.create(config.name, **config.kwargs)

        ctx = SamplerContext(
            dataset=self.dataset(split),  # type: ignore
            num_replicas=world,
            rank=rank,
            device=state.device,
            shuffle=is_train
        )
        sampler.attach(ctx)

        return sampler

    def _build_dataloader(self, split: Split) -> GraphDataLoader:
        """Initialize the dataloader for a given split."""
        kwargs: dict[str, Any] = {
            "num_workers":  self.config.num_workers,
            "batch_size":   self.config.batch_size,
            "sampler":      self._build_sampler(split)
        }

        # if num workers is greater than 0 (multiprocessing)
        if self.config.num_workers > 0:
            kwargs.update(
                prefetch_factor=self.config.prefetch_factor,
                multiprocessing_context=mp.get_context(self.config.mp_context),
                persistent_workers=self.config.persistent_workers,
                pin_memory=torch.cuda.is_available()
            )

        return GraphDataLoader(self.dataset(split), **kwargs)

    @cached_property
    def _dataloaders(self) -> MemoMap[Split, GraphDataLoader]:
        return MemoMap(self._build_dataloader)

    def dataloader(self, split: Split) -> GraphDataLoader:
        return self._dataloaders[split]

    def set_epoch(self, epoch: int) -> None:
        # required for correct shuffling under DDP
        for loader in self._dataloaders.values():
            for name in ["sampler", "batch_sampler"]:
                sampler = getattr(loader, name, None)
                if sampler is None:
                    continue

                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)

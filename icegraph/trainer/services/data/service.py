# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Iterator, Any, ClassVar
import multiprocessing as mp
from functools import reduce
from operator import add

from torch_geometric.loader import DataLoader
import torch

from icegraph.statistics import StatisticService
from icegraph.types.data import Split, ModelInputRole
from icegraph.types.files import Source

from ..service import Service

from .types import Attributes, GlobalAttributes
from .view import DataView
from .config import DataConfig
from .store import Store, StoreFactory, StoreContext
from .module import Module, ModuleFactory, ModuleContext
from .sampler import SamplerFactory, SamplerContext

__all__ = ["DataService"]


class DataService(Service[DataView, DataConfig]):
    """
    A container class for managing access to training, validation, and test datasets.
    """
    name: ClassVar[str] = "data"
    version: ClassVar[int] = 1

    interface = DataView

    # make the type checker happy
    _source:            Source | None
    _store:             Store | None
    _datasets:          dict[Split, Module]
    _dataloaders:       dict[Split, DataLoader]
    _stats:             dict[Split, dict[ModelInputRole, StatisticService]]

    def build(self) -> None:
        """Initialize the DataService with training, validation, and test datasets."""
        self._source:   Source  | None = None
        self._store:    Store   | None = None

        # build the dataset and dataloader cache
        self._datasets:     dict[Split, Module]     = {}
        self._dataloaders:  dict[Split, DataLoader] = {}

        # statistic service cache
        self._stats: dict[Split, dict[ModelInputRole, StatisticService]] = {}

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> DataConfig:
        return DataConfig(**config)

    def on_attach(self) -> None:
        # get store config
        store_config = self.config.store

        # build the store
        self._store = StoreFactory.create(store_config.name, **store_config.kwargs)

        # attach the store
        ctx = StoreContext(source=Source(self._ctx.trainer.source))
        self._store.attach(ctx)

    @property
    def attrs(self) -> Iterator[Attributes]:
        return self._store.attrs

    @property
    def global_attrs(self) -> GlobalAttributes:
        return self._store.global_attrs

    def set_epoch(self, epoch: int) -> None:
        """
        Propagate epoch to all samplers that support epoch-based seeding.
        Required for correct shuffling under DDP.
        """
        for loader in self._dataloaders.values():
            for name in ["sampler", "batch_sampler"]:
                sampler = getattr(loader, name, None)
                if sampler is None:
                    continue

                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)

    def columns(self, role: ModelInputRole, aux: bool = False) -> list[str]:
        # pull from training dataset, all will have same columns anyway
        return self.dataset(Split.TRAIN).columns(role, aux=aux)

    def _build_dataloader(self, split: Split) -> DataLoader:
        """Initialize the dataloader for a given split."""
        # first initialize the sampler
        config = self.config.sampler

        is_train = (split == Split.TRAIN)

        # get state from the service manager
        state = self._ctx.services.require("state", required_by=DataService)

        # get world and rank from state, only ddp for training, eval runs only on main rank
        # this is enforced by the trainer, here we have to just ensure it gets a sampler with rank 0 and world 1
        rank    = state.rank    if is_train else 0
        world   = state.world   if is_train else 1

        sampler = SamplerFactory.create(config.name, **config.kwargs)

        # attach sampler
        ctx = SamplerContext(
            dataset=self.dataset(split), num_replicas=world, rank=rank, device=state.device, shuffle=is_train
        )
        sampler.attach(ctx)

        # get loader subparams
        config = self.config.loader

        # get worker count configuration
        num_workers: int = config.num_workers

        # dataloader kwargs
        kwargs: dict[str, Any] = {
            "num_workers":  num_workers,
            "batch_size":   config.batch_size,
            "sampler":      sampler
        }

        # if num workers is greater than 0 (multiprocessing)
        if num_workers > 0:
            kwargs.update(
                prefetch_factor=config.prefetch_factor,
                multiprocessing_context=mp.get_context(config.mp_context),
                persistent_workers=config.persistent_workers,
                pin_memory=torch.cuda.is_available()
            )

        # initialize the dataloader and return
        return DataLoader(self.dataset(split), **kwargs)  # type: ignore

    def dataloader(self, split: Split) -> DataLoader:
        """Get dataloader for given split."""
        dataloader = self._dataloaders.get(split)
        if dataloader is not None:
            return dataloader

        # if the dataloader has not already been built, build it
        dataloader = self._build_dataloader(split)

        # register the dataloader and return
        self._dataloaders[split] = dataloader
        return dataloader

    def _build_stats(self, split: Split, role: ModelInputRole) -> StatisticService:
        """Build an aggregate statistic service object from shards."""
        # load stat services
        # need to use generators or this will nuke memory
        stats = (attr.stats(split, role) for attr in self.attrs)

        try:
            first = next(stats)
        except StopIteration:
            raise RuntimeError("No shard statistics found; cannot compute aggregates.")

        # merge and return using functools reduce
        return reduce(add, stats, first)

    def stats(self, split: Split, role: ModelInputRole) -> StatisticService:
        # build if not previously cached
        service = self._stats.get(split, {}).get(role)
        if service is not None:
            return service

        # build if not built
        service = self._build_stats(split, role)

        # register the service and return
        self._stats.setdefault(split, {})[role] = service
        return service

    def _build_dataset(self, split: Split) -> Module:
        module_config = self.config.module

        # build the module
        dataset = ModuleFactory.create(module_config.name, **module_config.kwargs)

        # attach the module
        ctx = ModuleContext(split=split, store=self._store)
        dataset.attach(ctx)

        return dataset

    def dataset(self, split: Split) -> Module:
        """Get dataloader for given split."""
        dataset = self._datasets.get(split)
        if dataset is not None:
            return dataset

        # if the dataset has not already been built, build it
        dataset = self._build_dataset(split)

        # register the dataset and return
        self._datasets[split] = dataset
        return dataset

    def state_dict(self) -> dict[str, Any]:
        # ensure manager has already been attached before building state dict
        return {
            "config":   self.config.model_dump(mode="json"),
            "source":   self._source.to_struct(),  # purely for debug purposes, not directly restored
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        # this must be called BEFORE attach()
        if self._ctx is not None:
            raise RuntimeError("Cannot call load_state_dict after attach.")

        # update config with checkpointed one
        self.config = type(self).validate_config(state["config"])

# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Iterator, Any, ClassVar
import multiprocessing as mp
from functools import reduce
from operator import add

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch

from icegraph.statistics import StatisticService
from icegraph.types.statistics import StatisticBundleStruct
from icegraph.types.data import Split, ModelInputRole, AttributeDomain
from icegraph.types.files import Source

from ..service import Service

from .module import DatasetModule
from .samplers import DistributedBlockwiseSampler
from .readers import ReaderFactory, ShardStore
from .types import Attributes, GlobalAttributes, SizedDataset
from .view import DataView
from .config import DataConfig

__all__ = ["DataService"]


class DataService(Service[DataView, DataConfig]):
    """
    A container class for managing access to training, validation, and test datasets.
    """
    name: ClassVar[str] = "data"

    deps = ("state", "strategy")
    interface = DataView

    # make the type checker happy
    _source:            Source | None
    _store:             ShardStore | None
    _datasets:          dict[Split, DatasetModule]
    _dataloaders:       dict[Split, DataLoader]
    _stats:             dict[Split, dict[ModelInputRole, StatisticService]]
    _expect_checksum:   str | None

    def build(self) -> None:
        """Initialize the DataService with training, validation, and test datasets."""
        self._source:   Source | None = None
        self._store:    ShardStore | None = None

        # build the dataset and dataloader cache
        self._datasets:     dict[Split, DatasetModule]  = {}
        self._dataloaders:  dict[Split, DataLoader]     = {}

        # statistic service cache
        self._stats: dict[Split, dict[ModelInputRole, StatisticService]] = {}

        # expected checksum (if loading from state dict)
        self._expect_checksum: str | None = None

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> DataConfig:
        return DataConfig(**config)

    def on_attach(self) -> None:
        # cache source (build if raw)
        raw_source = self._ctx.trainer.source
        self._source = raw_source if isinstance(raw_source, Source) else Source(raw_source)

        # invalidate caches
        self._store = None
        self._dataloaders.clear()
        self._stats.clear()

        # build datasets
        self._datasets = {
            split: DatasetModule(
                split=split, store=self.store, config=self.config.module
            ) for split in Split.all()
        }

    def state_dict(self) -> dict[str, Any]:
        # ensure manager has already been attached before building state dict
        return {
            "config":   self.config.model_dump(mode="json"),
            "source":   self._source.to_struct(),  # purely for debug purposes, not directly restored
            "checksum": self.global_attrs.checksum
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        # this must be called BEFORE attach()
        if self._ctx is not None:
            raise RuntimeError("Cannot call load_state_dict after attach.")

        # update config with checkpointed one
        self.config = type(self).validate_config(state["config"])

        # store expected checksum
        self._expect_checksum = state["checksum"]

    @property
    def attrs(self) -> Iterator[Attributes]:
        return self.store.attrs

    @property
    def global_attrs(self) -> GlobalAttributes:
        return self.store.global_attrs(checksum=self._expect_checksum)

    @property
    def store(self) -> ShardStore:
        if self._source is None:
            raise RuntimeError(f"{type(self).__name__}: cannot build store before attach (source must be defined).")

        if self._store is None:
            self._store = ReaderFactory.create_store(self.config.reader.name, source=self._source)
        return self._store

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

    def _build_dataloader(self, split: Split) -> DataLoader:
        """Initialize the dataloader for a given split."""
        # get loader subparams
        loader_config = self.config.loader

        # get worker count configuration
        num_workers: int = loader_config.num_workers

        # build sampler
        is_train = (split == Split.TRAIN)

        # get state from the service manager
        state = self._ctx.services.require("state", required_by=DataService)

        # get world and rank from state, only ddp for training, eval runs only on main rank
        # this is enforced by the trainer, here we have to just ensure it gets a sampler with rank 0 and world 1
        rank    = state.rank    if is_train else 0
        world   = state.world   if is_train else 1

        sampler = DistributedBlockwiseSampler(
            self.dataset(split),
            loader_config.block_size,
            num_replicas=world,
            rank=rank,
            shuffle=is_train  # only need shuffle on training split
        )

        # dataloader kwargs
        kwargs: dict[str, Any] = {
            "num_workers":  num_workers,
            "batch_size":   loader_config.batch_size,
            "sampler":      sampler
        }

        # if num workers is greater than 0 (multiprocessing)
        if num_workers > 0:
            kwargs.update(
                prefetch_factor=loader_config.prefetch_factor,
                multiprocessing_context=mp.get_context(loader_config.mp_context),
                persistent_workers=loader_config.persistent_workers,
                pin_memory=torch.cuda.is_available()
            )

        # initialize the dataloader and return
        return DataLoader(self.dataset(split), **kwargs)

    def _build_stats(self, split: Split, role: ModelInputRole) -> StatisticService:
        """Build an aggregate statistic service object from shards."""
        def iter_structs() -> Iterator[StatisticBundleStruct]:
            """Method for iterating over StatisticBundleStruct objects stored in the dataset."""
            for attr in self.attrs:
                stats = attr.get(AttributeDomain.LOCAL)

                # ensure stats exist
                if stats is None:
                    raise RuntimeError(
                        f"Key '{AttributeDomain.LOCAL.value}' not found in attrs. "
                        f"Problematic shard: ID={attr.shard_id}."
                    )

                struct_ = stats.get(role.value, {}).get(split.value)

                # ensure struct was found
                if struct_ is None:
                    raise RuntimeError(
                        f"No stats found at attrs[{AttributeDomain.LOCAL.value}][{role.value}][{split.value}]. "
                        f"Problematic shard: ID={attr.shard_id}."
                    )

                yield struct_

        # build services from structs
        # need to use generators or this will nuke memory
        iter_services = (StatisticService.from_struct(s) for s in iter_structs())

        try:
            first = next(iter_services)
        except StopIteration:
            raise RuntimeError("No shard statistics found; cannot compute aggregates.")

        # merge and return using functools reduce
        return reduce(add, iter_services, first)

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

    def dataset(self, split: Split) -> SizedDataset[Data]:
        """Get dataset for given split."""
        if not self._datasets:
            raise RuntimeError(f"{type(self).__name__}: cannot get dataset before attach (datasets aren't built yet).")

        return self._datasets[split]

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

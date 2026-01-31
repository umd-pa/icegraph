# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Iterator, Any
import multiprocessing as mp
from functools import reduce
from operator import add

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch
from torch.utils.data import Sampler

from icegraph.statistics import StatisticService
from icegraph.types.statistics import StatisticBundleStruct
from icegraph.types.data import Split, ModelInputRole, AttributeDomain
from icegraph.types.files import Source
from icegraph.trainer.types import Params

from ..service import Service
from ..types import ServiceContext

from .module import DatasetModule
from .samplers import DistributedBlockwiseSampler
from .readers import ReaderFactory, ShardStore
from .types import Attributes, GlobalAttributes, SizedDataset
from .view import DataView


__all__ = ["DataService"]


class DataService(Service):
    """
    A container class for managing access to training, validation, and test datasets.
    """
    name = "data"
    deps = ["state"]
    view = DataView


    def __init__(self, params: Params) -> None:
        """Initialize the DataService with training, validation, and test datasets."""
        super().__init__(params)

        self.source: Source     | None = None
        self._store: ShardStore | None = None

        # build the dataset and dataloader cache
        self._datasets:     dict[Split, DatasetModule]  = {}
        self._dataloaders:  dict[Split, DataLoader]     = {}

        # statistic service cache
        self._stat_services: dict[Split, dict[ModelInputRole, StatisticService]] = {}

        # expected checksum (if loading from state dict)
        self._expect_checksum: str | None = None

    def on_attach(self, ctx: ServiceContext) -> None:
        # cache source (build if raw)
        self.source = ctx.trainer.source if isinstance(ctx.trainer.source, Source) else Source(ctx.trainer.source)

        # invalidate caches
        self._store = None
        self._dataloaders.clear()
        self._stat_services.clear()

        # build datasets
        self._datasets = {split: DatasetModule(split=split, store=self.store) for split in Split.all()}

    def state_dict(self) -> dict[str, Any]:
        # ensure manager has already been attached before building state dict
        if self.source is None:
            raise RuntimeError("Cannot call state_dict before attach (source not set).")

        state = {
            "params": self.params.to_struct(),
            "source": self.source.to_struct(),  # purely for debug purposes, not directly restored
            "checksum": self.global_attrs.checksum
        }
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        # this must be called BEFORE attach()
        if self.is_attached:
            raise RuntimeError("Cannot call load_state_dict after attach.")

        # update params with checkpointed ones
        self.params = Params.from_struct(state["params"])

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
        if self.source is None:
            raise RuntimeError(f"{type(self).__name__}: cannot build store before attach (source must be defined).")

        if self._store is None:
            reader_name = self.params.get("reader").get("name")
            self._store = ReaderFactory.create_store(reader_name, source=self.source)
        return self._store

    def _init_sampler(self, split: Split, params: Params, *, shuffle: bool = False) -> Sampler[int]:
        """Initialize and return the sampler for the split."""
        dataset = self.dataset(split)

        # required config values
        block_size: int = params.require("block_size")

        # get state from the service manager
        state = self._ctx.services.require("state", required_by=DataService)

        # get world and rank from state, only ddp for training, eval runs only on main rank
        # this is enforced by the trainer, here we have to just ensure it gets a sampler with rank 0 and world 1
        rank    = state.rank  if split == Split.TRAIN else 0
        world   = state.world if split == Split.TRAIN else 1

        return DistributedBlockwiseSampler(
            dataset, block_size, num_replicas=world, rank=rank, shuffle=shuffle
        )

    def _init_dataloader(self, split: Split) -> DataLoader:
        """Initialize the dataloader for a given split."""
        # get loader subparams
        loader_params = self.params.get("loader")

        # get worker count configuration
        num_workers: int = loader_params.require("num_workers")

        # dataloader kwargs
        kwargs: dict[str, Any] = {
            "num_workers":  num_workers,
            "batch_size":   loader_params.require("batch_size"),
            "sampler":      self._init_sampler(split, loader_params, shuffle=(split == Split.TRAIN))
        }

        # if num workers is greater than 0 (multiprocessing)
        if num_workers > 0:
            kwargs.update(
                prefetch_factor=loader_params.require("prefetch_factor"),
                multiprocessing_context=mp.get_context(loader_params.require("mp_context")),
                persistent_workers=loader_params.get("persistent_workers", True),
                pin_memory=torch.cuda.is_available()
            )

        # initialize the dataloader and return
        return DataLoader(self.dataset(split), **kwargs)

    def dataloader(self, split: Split) -> DataLoader:
        """Get dataloader for given split."""
        if (dataloader := self._dataloaders.get(split)) is None:
            # if the dataloader has not already been built, build it
            dataloader = self._init_dataloader(split)

            # register the dataloader
            self._dataloaders[split] = dataloader

        return dataloader

    def dataset(self, split: Split) -> SizedDataset[Data]:
        """Get dataset for given split."""
        if not self._datasets:
            raise RuntimeError(f"{type(self).__name__}: cannot get dataset before attach (datasets aren't built yet).")

        return self._datasets[split]

    def set_epoch(self, epoch: int) -> None:
        """
        Propagate epoch to all samplers that support epoch-based seeding.
        Required for correct shuffling under DDP.
        """
        for loader in self._dataloaders.values():
            sampler = getattr(loader, "sampler", None)

            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            # if batch_sampler is ever used
            batch_sampler = getattr(loader, "batch_sampler", None)
            if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
                batch_sampler.set_epoch(epoch)

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

    def stats(self, split: Split, role: ModelInputRole) -> StatisticService:
        # build if not previously cached
        service = self._stat_services.get(split, {}).get(role)
        if service is not None:
            return service

        # build if not built
        service = self._build_stats(split, role)

        # register the service and return
        self._stat_services.setdefault(split, {})[role] = service
        return service

# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import time

from typing import Any, ClassVar
import multiprocessing as mp
from functools import partial, cached_property

import torch

from icegraph.common.mapping import MemoMap

from ..service import Service

from .config import DataConfig
from .loader import GraphDataLoader
from .dataset import GraphDataset
from .spec import LoaderSpec

import logging
logger = logging.getLogger(__name__)

__all__ = ["DataService"]


class DataService(Service[DataConfig]):
    """
    A container class for managing access to training, validation, and test dataloaders.
    """
    name: ClassVar[str] = "data"
    version: ClassVar[int] = 1

    def build(self) -> None:
        return

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> DataConfig:
        return DataConfig(**config)

    @property
    def loader_spec(self) -> type[LoaderSpec]:
        return LoaderSpec

    @cached_property
    def _dls(self) -> MemoMap[LoaderSpec, GraphDataLoader]:
        return MemoMap(self._build_dataloader)

    def dataloader(self, spec: LoaderSpec) -> GraphDataLoader:
        return self._dls[spec]

    def _build_dataloader(self, spec: LoaderSpec) -> GraphDataLoader:
        # get new specs for each
        ds_spec = self._new_dataset()
        dl_spec = self._new_dataloader()

        # construct dataset from assembly spec
        dataset = ds_spec(keys=spec.keys, exclude_roles=spec.exclude_roles)

        # build dataloader from dataset
        dataloader = dl_spec(dataset=dataset)

        return dataloader

    def set_epoch(self, epoch: int) -> None:
        # required for correct shuffling
        for loader in self._dls.values():
            loader.set_epoch(epoch)

    def _new_dataloader(self) -> partial[GraphDataLoader]:
        """Build a new dataloader spec."""
        start = time.perf_counter()
        kwargs: dict[str, Any] = {
            "num_workers":  self.config.num_workers,
            "batch_size":   self.config.batch_size
        }

        # if num workers is greater than 0 (multiprocessing)
        if self.config.num_workers > 0:
            kwargs.update(
                prefetch_factor=self.config.prefetch_factor,
                multiprocessing_context=mp.get_context(self.config.mp_context),
                persistent_workers=self.config.persistent_workers,
                pin_memory=torch.cuda.is_available()
            )

        loader = partial(GraphDataLoader, **kwargs)
        logger.info(f"[DataService] Constructed new dataloader in {time.perf_counter() - start} s.")
        return loader

    def _new_dataset(self) -> partial[GraphDataset]:
        start = time.perf_counter()
        dataset = partial(
            GraphDataset,
            chunk_size=self.config.chunk_size,
            buffer_size=self.config.buffer_size,
            batch_size=self.config.batch_size,
            shuffle_chunks=self.config.shuffle_chunks,
            services=self._ctx.services
        )
        logger.info(f"[DataService] Constructed new dataset in {time.perf_counter() - start} s.")
        return dataset


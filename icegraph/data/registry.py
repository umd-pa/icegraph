# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Self, Union, Sequence, Optional, Dict, Any, List, Type
from pathlib import Path
import time
import os
import multiprocessing as mp

import torch_geometric as pyg
import torch
import numpy as np


from icegraph.config import IGConfig
from icegraph.data.base import DataModule
from icegraph.console import Console
from .samplers import DistributedBlockShuffleSampler
from icegraph.data.readers import LMDBDatasetShardReader

__all__ = ["DatasetRegistry"]


def dl_worker_init(_worker_id: int):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


class DatasetRegistry:
    """
    A container class for managing access to training, validation, and test datasets.
    """

    def __init__(
            self,
            module: Type,
            source: Union[str, Path, Sequence[Union[str, Path]]]
    ) -> None:
        """
        Initialize the DatasetRegistry with training, validation, and test datasets.

        Args:
            module (object): Module object to use.
            source (Union[str, Path, Sequence[Union[str, Path]]]): Path or list of paths to the LMDB file(s) containing the dataset.
        """
        # cache source for use downstream
        self.source = source

        # global attrs cache
        self._global_attrs:     Dict[str, Any]
        self._attrs:            Dict[bytes, Dict[str, Dict[str, Any]]]

        # build the datasets
        self._train_dataset =       module(source, subset="train")
        self._validation_dataset =  module(source, subset="validation")
        self._test_dataset =        module(source, subset="test")

        self._datasets = [self._train_dataset, self._validation_dataset, self._test_dataset]

        # get training params from config
        self._config = IGConfig.get()

        # build kwargs for the dataloaders
        batch_size =        self._config.user_config.training.batch_size
        num_workers =       self._config.user_config.training.num_workers
        prefetch_factor =   self._config.user_config.training.prefetch_factor
        seed =              self._config.user_config.training.seed

        self.sampler = DistributedBlockShuffleSampler(len(self._train_dataset), batch_size * 2, seed=seed, drop_last=True)

        self.dataloader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
            "multiprocessing_context": mp.get_context("fork"),
            "persistent_workers": num_workers > 0,
            "prefetch_factor": prefetch_factor if num_workers > 0 else None,
            "worker_init_fn": dl_worker_init,
            "drop_last": True
        }

        # dataloader caches
        self._train_dataloader:         Optional[pyg.loader.DataLoader] = None
        self._validation_dataloader:    Optional[pyg.loader.DataLoader] = None
        self._test_dataloader:          Optional[pyg.loader.DataLoader] = None

    def __len__(self) -> int:
        """
        Return the number of events in the full dataset.

        Returns:
            int: Number of events.
        """
        return sum(map(len, self._datasets))

    @staticmethod
    def _verify_global_attrs(attrs: List[Dict[str, Any]]) -> None:
        """Verify global attributes are consistent across shards."""
        first_seen = None
        for shard_attr in attrs:
            if first_seen is None:
                first_seen = shard_attr
                continue

            if shard_attr != first_seen:
                raise AttributeError(f"Attributes are not consistent across shards. Expected: {first_seen}, Got: {shard_attr}")

    def _load_attrs(self) -> None:
        with LMDBDatasetShardReader(self.source) as reader:
            attrs = reader.attrs()
            global_attrs = [attr["global"] for attr in attrs.values()]
            self._verify_global_attrs(global_attrs)

            self._global_attrs = global_attrs[0]  # if all are consistent (we check for this) this is safe
            self._attrs = attrs

    def profile(self, target_samples: int = 50_000, warmup_batches: int = 5) -> None:
        """
        Measure DataLoader throughput (samples/s, MB/s) for PyG batches.
        Only the time spent pulling the next batch from the loader is counted.
        Byte measurement happens off-clock so it doesn't affect the timing.

        Args:
            target_samples: stop after at least this many samples
            warmup_batches: discard first N batches to fill worker prefetch
        """

        def _bytes_of_value(v) -> int:
            if torch.is_tensor(v):
                return v.nelement() * v.element_size()
            if isinstance(v, np.ndarray):
                return v.nbytes
            if isinstance(v, (list, tuple)):
                return sum(_bytes_of_value(x) for x in v)
            if isinstance(v, dict):
                return sum(_bytes_of_value(x) for x in v.values())
            return 0

        def _pyg_bytes(data) -> int:
            # Works for both Data and Batch; uses attribute keys exposed by PyG
            total = 0
            for key in data.keys():
                try:
                    val = data[key]
                except Exception:
                    continue
                total += _bytes_of_value(val)
            return total

        loader = self.train_dataloader
        it = iter(loader)

        for _ in range(warmup_batches):
            try:
                _ = next(it)
            except StopIteration:
                break

        count = 0
        total_bytes = 0
        load_time = 0.0

        while count < target_samples:
            t0 = time.perf_counter()
            try:
                batch = next(it)  # time ONLY the loader fetch/collate/IPC
            except StopIteration:
                break
            t1 = time.perf_counter()
            load_time += (t1 - t0)

            # off clock
            batch_size = int(getattr(batch, "num_graphs", 1))
            count += batch_size
            total_bytes += _pyg_bytes(batch)

        # Guard against divide-by-zero
        load_time = max(load_time, 1e-9)

        samples_per_sec = count / load_time
        mb_per_sec = (total_bytes / (1024 ** 2)) / load_time

        Console.out(f"Effective loader throughput: {samples_per_sec:.1f} samples/s")
        Console.out(f"Effective data throughput: {mb_per_sec:.2f} MB/s (off-clock measurement)")

    @property
    def global_attrs(self) -> Dict[str, Any]:
        if getattr(self, "_global_attrs", None) is None:
            self._load_attrs()
        return self._global_attrs

    @property
    def attrs(self) -> Dict[bytes, Dict[str, Dict[str, Any]]]:
        if getattr(self, "_attrs", None) is None:
            self._load_attrs()
        return self._attrs

    @property
    def train_dataset(self) -> DataModule:
        """Getter for the training dataset."""
        return self._train_dataset

    @property
    def val_dataset(self) -> DataModule:
        """Getter for the validation dataset."""
        return self._validation_dataset

    @property
    def test_dataset(self) -> DataModule:
        """Getter for the test dataset."""
        return self._test_dataset

    @property
    def train_dataloader(self) -> pyg.loader.DataLoader:
        """
        Returns a Torch Geometric dataloader for the training split.
        """
        if self._train_dataloader is None:
            self._train_dataloader = self.train_dataset.dataloader(
                **self.dataloader_kwargs,
                sampler=self.sampler,
                shuffle=False
            )
        return self._train_dataloader

    @property
    def val_dataloader(self) -> pyg.loader.DataLoader:
        """
        Returns a Torch Geometric dataloader for the validation split.
        """
        if self._validation_dataloader is None:
            self._validation_dataloader = self.val_dataset.dataloader(
                **self.dataloader_kwargs,
                shuffle=False  # dont need shuffle on eval sets
            )
        return self._validation_dataloader

    @property
    def test_dataloader(self) -> pyg.loader.DataLoader:
        """
        Returns a Torch Geometric dataloader for the test split.
        """
        if self._test_dataloader is None:
            self._test_dataloader = self.test_dataset.dataloader(
                **self.dataloader_kwargs,
                shuffle=False  # dont need shuffle on eval sets
            )
        return self._test_dataloader

    @classmethod
    def load_from_lmdb(
            cls,
            source: Union[str, Path, Sequence[Union[str, Path]]]
    ) -> Self:
        """
        Load datasets from LMDB files and create an instance of the dataset registry.

        Args:
            source (Union[str, Path, Sequence[Union[str, Path]]]): Path or list of paths to the LMDB file(s) containing the dataset.

        Returns:
            Self: An instance of the class initialized with training, validation, and test datasets.
        """
        return cls(DataModule, source)

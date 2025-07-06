# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from icegraph.data import DatasetRegistry
from icegraph.config import IGConfig

__all__ = ["IGDataModule"]


class IGDataModule(pl.LightningDataModule):

    def __init__(self, config: IGConfig, registry: DatasetRegistry):
        super().__init__()

        # config/registry
        self.config = config
        self.registry = registry

        # instance attributes
        self.batch_size = config.user_config.training.batch_size
        self.num_workers = config.user_config.training.num_workers

    def train_dataloader(self) -> DataLoader:
        return self.registry.train_dataset.dataloader(
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return self.registry.val_dataset.dataloader(
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return self.registry.test_dataset.dataloader(
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
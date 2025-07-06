# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import pytorch_lightning as pl
import torch

from .data import IGDataModule
from icegraph.config import IGConfig
from icegraph.data import DatasetRegistry


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class IGTrainer:
    """
    Helper that wraps a Lightning Trainer with IGDataModule and provides
    a convenient method for node‐level classification.
    """
    def __init__(self, config: IGConfig, registry: DatasetRegistry):
        # seed everything before training
        pl.seed_everything(config.user_config.training.seed)

        # store config and registry
        self.config = config
        self.registry = registry

        self.datamodule = IGDataModule(
            config=config,
            registry=registry
        )

        # Build the internal Lightning Trainer
        trainer_params = {
            **config.user_config.training.trainer_params,
            "accelerator": "auto",
            "enable_progress_bar": False,
            "devices": min(1, torch.cuda.device_count())
        }
        self.trainer = pl.Trainer(**trainer_params)

        self.trainer.logger._default_hp_metric = None

    def run(self, model: pl.LightningModule):
        """
        Fit on train/val then test on test split using the DataModule.
        """
        self.trainer.fit(model, self.datamodule)
        self.trainer.test(model, self.datamodule)
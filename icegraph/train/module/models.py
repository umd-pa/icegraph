# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim

from icegraph.train.module.base import MLPModel, GNNModel
from icegraph.train.module.exceptions import InvalidModel


class NodeLevelGNN(pl.LightningModule):
    def __init__(self, model_name: str, **model_kwargs):
        super().__init__()
        # Save all hyperparameters (model_name + model_kwargs)
        self.save_hyperparameters()

        # Map model names to respective classes
        model_map = {
            "MLP": MLPModel,
            "GNN": GNNModel,
        }

        try:
            ModelCls = model_map[model_name]
        except KeyError:
            raise InvalidModel(
                f"Got '{model_name}', which is not valid; choose from {list(model_map)}"
            )

        # Instantiate with passed kwargs
        self.model = ModelCls(**model_kwargs)

        # Standard classification loss
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, edge_index=None):
        """
        Forward pass through the selected model.

        Args:
            x (Tensor): Node or event features
            edge_index (Tensor, optional): Graph connectivity; if present, used by GNNModel
        Returns:
            Tensor: Logits of shape [batch_size or num_nodes, num_classes]
        """
        if edge_index is None:
            return self.model(x)
        return self.model(x, edge_index)

    def _shared_step(self, batch, stage: str):
        """
        Common logic for train/val/test: unpack batch, forward, compute loss & acc, and log.
        """
        # IGDataModule yields (features, labels)
        x, y = batch
        # Move to correct device and types
        x = x.to(self.device)
        y = y.to(self.device).long().squeeze()

        # Forward
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)

        # Accuracy
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()

        # Logging
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=(stage!="train"))
        self.log(f"{stage}_acc",  acc,  on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        # SGD with momentum & weight decay
        return optim.SGD(
            self.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=2e-3
        )
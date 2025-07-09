# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Union
from pathlib import Path

import torch
from torch.optim import Adam
from torch_geometric.seed import seed_everything
import torch.nn.functional as F
import numpy as np

from icegraph.console import Console
from icegraph.data import DatasetRegistry
from icegraph.config import IGConfig
from .module import GravNet

__all__ = ["Trainer"]


class Trainer:
    """
    Trainer class for managing the training, validation, and testing of a PyTorch model
    using datasets registered in a `DatasetRegistry`.

    This class handles model optimization, loss computation, and RMSE reporting
    for each stage of the training lifecycle.
    """

    _model_map = {
        "gravnet": GravNet
    }

    def __init__(self, dataset_registry: DatasetRegistry, model: str = "gravnet", device: str = "cuda") -> None:
        """
        Initialize the Trainer.

        Args:
            dataset_registry (DatasetRegistry): Dataset registry containing dataloaders.
            model (str): The model to be trained and evaluated, must be one of ['gravnet'].
            device (str, optional): Preferred device for computation. Defaults to 'cuda'.
        """
        # grab global config
        self._config = IGConfig.get()

        # grab config values
        self.num_epochs = self._config.user_config.training.trainer_params.max_epochs
        hidden_channels = self._config.user_config.training.trainer_params.hidden_channels
        seed = self._config.user_config.training.seed
        layers = self._config.user_config.training.trainer_params.hidden_layers

        # optimizer dict
        optimizer_params = self._config.user_config.training.optimizer.toDict()

        # set global seed for reproducibility
        self._set_seed(seed)

        self.datasets = dataset_registry
        self.device = device if torch.cuda.is_available() else "cpu"

        # determine dimensions of input and output
        in_channels = dataset_registry.train_dataset.num_node_features
        out_channels = dataset_registry.train_dataset.num_output_features

        # get the active model
        active_model: Union[GravNet] = self._model_map[model](in_channels, hidden_channels, out_channels, layers)
        self.model = active_model.to(self.device)

        # define optimizer and the loss function
        self.optimizer = Adam(
            active_model.parameters(),
            lr=float(optimizer_params["learning_rate"]),
            betas=tuple(map(float, optimizer_params["betas"])),
            eps=float(optimizer_params["eps"]),
            weight_decay=float(optimizer_params["weight_decay"]),
            amsgrad=bool(optimizer_params["amsgrad"])
        )
        self.loss_fn = torch.nn.MSELoss()

    @staticmethod
    def _set_seed(seed: int) -> None:
        """
        Set seeds for PyTorch to ensure reproducibility.

        This method configures global random number generators and sets
        deterministic flags for PyTorch's CUDA backend to make training runs
        as reproducible as possible.

        Args:
            seed (int): The seed value to use for all random number generators.

        Notes:
            - This affects torch (CPU and GPU), NumPy, and Python's `random` module.
            - `torch.backends.cudnn.deterministic = True` ensures deterministic
              results at the cost of some performance.
            - `torch.backends.cudnn.benchmark = False` prevents dynamic algorithm
              selection that could introduce non-determinism.
        """
        # seed everything
        seed_everything(seed)

        # ensure deterministic setup
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def train(self) -> None:
        """
        Train the model for the configured number of epochs using the training dataloader.

        Logs training progress and computes both MSE and RMSE per epoch.
        """
        Console.banner("Trainer")

        # warn if falling back to CPU
        if self.device == "cpu":
            Console.out("No accelerators found, falling back to CPU training.", severity=2)

        self.model.train()

        # iterate over epochs
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            total_rmse = 0.0
            total = 0

            Console.out(f"[Train] Epoch {epoch + 1}/{self.num_epochs}")
            # iterate over batches in the dataloader
            for batch in Console.progress_bar(self.datasets.train_dataloader):
                batch = batch.to(self.device)

                self.optimizer.zero_grad()
                out = self.model(batch.x, batch.batch)
                target = batch.y.view(-1, 1)
                loss = self.loss_fn(out, target)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * batch.y.size(0)
                total_rmse += F.mse_loss(out, target, reduction="sum").sqrt().item()
                total += batch.y.size(0)

            # compute and display RMSE and MSE
            avg_loss = total_loss / total
            rmse = total_rmse / total
            Console.out(f" --> MSE: {avg_loss:.4f} | RMSE: {rmse:.4f}")

    def validate(self) -> None:
        """
        Evaluate the model using the validation dataloader.

        Reports MSE and RMSE on the validation set without updating model weights.
        """
        self.model.eval()
        total_loss = 0.0
        total_rmse = 0.0
        total = 0

        Console.out("[Validation]")
        with torch.no_grad():
            for batch in Console.progress_bar(self.datasets.val_dataloader):
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.batch)
                target = batch.y.view(-1, 1)
                loss = self.loss_fn(out, target)

                total_loss += loss.item() * batch.y.size(0)
                total_rmse += F.mse_loss(out, target, reduction="sum").sqrt().item()
                total += batch.y.size(0)

        # compute and display RMSE and MSE
        avg_loss = total_loss / total
        rmse = total_rmse / total
        Console.out(f" --> MSE: {avg_loss:.4f} | RMSE: {rmse:.4f}")

    def test(self) -> None:
        """
        Evaluate the final model performance on the test dataset.

        Reports RMSE only, without computing or logging loss.
        """
        self.model.eval()
        total_rmse = 0.0
        total = 0

        Console.out("[Test]")
        with torch.no_grad():
            for batch in Console.progress_bar(self.datasets.test_dataloader):
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.batch)
                target = batch.y.view(-1, 1)
                total_rmse += F.mse_loss(out, target, reduction="sum").sqrt().item()
                total += batch.y.size(0)

        # compute and display RMSE
        rmse = total_rmse / total
        Console.out(f" --> RMSE: {rmse:.4f}")

    def save(self, outfile: Union[str, Path]) -> None:
        """
        Save the model.

        Args:
            outfile (Union[str, Path]): Path to save the model.
        """
        torch.save(self.model.state_dict(), outfile)

    def run(self) -> None:
        """
        Run the full training pipeline including training, validation, and testing.
        """
        self.train()
        self.validate()
        self.test()
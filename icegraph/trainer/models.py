# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Union, Optional
from pathlib import Path

import torch
from torch.optim import Adam
from torch_geometric.seed import seed_everything
import torch.nn.functional as F
import torch_scatter

from icegraph.console import Console
from icegraph.data import DatasetRegistry
from icegraph.config import IGConfig
from .module import GravNet
from icegraph.pathutils import PathResolver
from .tensorboard import TensorBoard

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

    def __init__(
        self,
        dataset_registry: DatasetRegistry,
        outfile: Optional[Union[str, Path]] = None,
        model: str = "gravnet",
        device: str = "cuda"
    ) -> None:
        """
        Initialize the Trainer.

        Args:
            dataset_registry (DatasetRegistry): Dataset registry containing dataloaders.
            outfile (Optional[Union[str, Path]]): Path to save the trained model.
            model (str): The model to be trained and evaluated, must be one of ['gravnet'].
            device (str, optional): Preferred device for computation. Defaults to 'cuda'.
        """
        # grab global config
        self._config = IGConfig.get()

        resolver = PathResolver(path=outfile, origin=None, extension="pt", stage="trainer")
        self.outfile = resolver.resolve()

        # place log dir next to run files
        self.log_dir = self.outfile.parent / "logs"

        # grab config values
        self.num_epochs = self._config.user_config.training.trainer_params.max_epochs
        self.test_interval_epochs = self._config.user_config.training.trainer_params.test_interval_epochs

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

        self._tensorboard: Optional[TensorBoard] = None

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

    def train(self, tensorboard: bool = False) -> None:
        """
        Train the model for the configured number of epochs using the training dataloader.

        Logs training progress and computes both MSE and RMSE per epoch.

        Args:
            tensorboard (bool): Whether to start TensorBoard, defaults to False.
        """
        Console.banner("Trainer")
        Console.out(f"Model save path: {self.outfile}")

        # warn if falling back to CPU
        if self.device == "cpu":
            Console.out("No accelerators found, falling back to CPU training.", severity=2)

        if tensorboard:
            self._tensorboard = TensorBoard(self.log_dir)
            self._tensorboard.launch()

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
                target = torch_scatter.scatter_mean(batch.y, batch.batch, dim=0).view(-1, 1)
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

            if self._tensorboard is not None:
                self._tensorboard.writer.add_scalar("Train/MSE", avg_loss, epoch + 1)
                self._tensorboard.writer.add_scalar("Train/RMSE", rmse, epoch + 1)

            # save the model after every epoch and run test
            self.save(epoch=epoch)

            # only run on specified intervals
            if self.test_interval_epochs > 0 and (epoch + 1) % self.test_interval_epochs == 0:
                self.test(epoch=epoch)

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
                target = torch_scatter.scatter_mean(batch.y, batch.batch, dim=0).view(-1, 1)
                loss = self.loss_fn(out, target)

                total_loss += loss.item() * batch.y.size(0)
                total_rmse += F.mse_loss(out, target, reduction="sum").sqrt().item()
                total += batch.y.size(0)

        # compute and display RMSE and MSE
        avg_loss = total_loss / total
        rmse = total_rmse / total
        Console.out(f" --> MSE: {avg_loss:.4f} | RMSE: {rmse:.4f}")

        # tensorboard
        if self._tensorboard is not None:
            self._tensorboard.writer.add_scalar("Validation/MSE", avg_loss, self.num_epochs - 1)
            self._tensorboard.writer.add_scalar("Validation/RMSE", rmse, self.num_epochs - 1)

    def test(self, epoch: Optional[int] = None) -> None:
        """
        Evaluate the final model performance on the test dataset.

        Reports RMSE only, without computing or logging loss.

        Args:
            epoch (int): The current epoch.
        """
        self.model.eval()
        total_rmse = 0.0
        total = 0

        Console.out("[Test]")
        with torch.no_grad():
            for batch in Console.progress_bar(self.datasets.test_dataloader):
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.batch)
                target = torch_scatter.scatter_mean(batch.y, batch.batch, dim=0).view(-1, 1)
                total_rmse += F.mse_loss(out, target, reduction="sum").sqrt().item()
                total += batch.y.size(0)

        # compute and display RMSE
        rmse = total_rmse / total
        Console.out(f" --> RMSE: {rmse:.4f}")

        # tensorboard
        if self._tensorboard is not None and epoch is not None:
            self._tensorboard.writer.add_scalar("Test/RMSE", rmse, epoch + 1)

    def save(self, epoch: Optional[int] = None) -> None:
        """
        Save the model.
        """
        if epoch:
            Console.out(f"[Epoch {epoch + 1}] Saving model...")
            save_path = self.outfile.with_name(f"{self.outfile.stem}_{epoch}{self.outfile.suffix}")
        else:
            Console.out("Saving model...")
            save_path = self.outfile

        try:
            torch.save(self.model.state_dict(), save_path)
            Console.out("Saved successfully.")
        except Exception as e:
            Console.out(f"Failed to save model: {e}", severity=3)

    def run(self, tensorboard: bool = False) -> None:
        """
        Run the full training pipeline including training, validation, and testing.

        Args:
            tensorboard (bool): Whether to start TensorBoard, defaults to False.
        """
        self.train(tensorboard)
        self.validate()

        if self._tensorboard is not None:
            self._tensorboard.writer.close()
            self._tensorboard.shutdown()

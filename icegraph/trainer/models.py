# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Union, Optional
from pathlib import Path
import math
from dataclasses import dataclass

import torch
from torch.optim import Adam
from torch_geometric.seed import seed_everything
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
import torch_scatter

from icegraph.console import Console
from icegraph.data import DatasetRegistry
from icegraph.config import IGConfig
from .config import TrainerConfig
from .arch import ModelFactory
from icegraph.pathutils import PathResolver
from .callbacks.base import Callback
from .callbacks import ConsoleCallback, CheckpointCallback, TensorBoardCallback
from .base.exceptions import EmptyDataLoaderError

__all__ = ["Trainer"]


class Trainer:
    """
    Trainer class for managing the training, validation, and testing of a PyTorch model
    using datasets registered in a `DatasetRegistry`.

    This class handles model optimization, loss computation, and RMSE reporting
    for each stage of the training lifecycle.
    """

    # define a metrics dataclass
    @dataclass
    class Metrics:
        samples: int
        sse_sum: Union[float, int]

        _avg_loss: Optional[Union[float, int]] = None
        _rmse: Optional[Union[float, int]] = None

        @property
        def avg_loss(self) -> Union[float, int]:
            if self._avg_loss is None:
                self._avg_loss = self.sse_sum / self.samples
            return self._avg_loss

        @property
        def rmse(self) -> Union[float, int]:
            if self._rmse is None:
                self._rmse = math.sqrt(self.avg_loss)
            return self._rmse


    def __init__(
        self,
        dataset_registry: DatasetRegistry,
        callbacks: Optional[list[Callback]] = None,
        trainer_config: Optional[TrainerConfig] = None,
        outfile: Optional[Union[str, Path]] = None,
        model: str = "gravnet",
        device: str = "cuda"
    ) -> None:
        """
        Initialize the Trainer.

        Sets up global configuration, reproducibility, datasets, model, optimizer,
        loss function, and metrics tracking.

        Args:
            dataset_registry (DatasetRegistry): Dataset registry containing dataloaders.
            callbacks (Optional[list[Callback]]): List of callbacks to pass to the trainer. If none are passed,
                defaults to Console, Checkpoint and TensorBoard callbacks.
            trainer_config (TrainerConfig): A TrainerConfig instance with training params.
            outfile (Optional[Union[str, Path]]): Path to save the trained model.
            model (str): The model to be trained and evaluated, must be one of ['gravnet'].
            device (str, optional): Preferred device for computation. Defaults to 'cuda'.
        """
        # grab global config and generate local trainer config
        self._config = IGConfig.get()
        self.trainer_config = TrainerConfig.from_config(self._config) if trainer_config is None else trainer_config

        # resolve the output path
        resolver = PathResolver(path=outfile, origin=None, extension="pt", stage="trainer")
        self.outfile = resolver.resolve()

        # place log dir next to run files
        self.log_dir = self.outfile.parent / "logs"

        # set global seed for reproducibility
        self._set_seed(self.trainer_config.seed)

        # load datasets and device
        self.datasets = dataset_registry
        self.device: torch.device = (
            torch.device("cuda")
            if torch.cuda.is_available() and device == "cuda"
            else torch.device("cpu")
        )

        # grab callbacks
        self.callbacks = callbacks or [ConsoleCallback(), CheckpointCallback(), TensorBoardCallback()]

        # determine dimensions of input and output
        in_channels = dataset_registry.train_dataset.num_node_features
        out_channels = dataset_registry.train_dataset.num_output_features

        # get the active model
        active_model = ModelFactory.create(
            model, in_channels, self.trainer_config.hidden_channels, out_channels, self.trainer_config.hidden_layers
        )
        self.model = active_model.to(self.device)

        # define optimizer and the loss function
        self.optimizer = Adam(
            active_model.parameters(),
            lr=self.trainer_config.lr,
            betas=self.trainer_config.betas,
            eps=self.trainer_config.eps,
            weight_decay=self.trainer_config.weight_decay,
            amsgrad=self.trainer_config.amsgrad
        )
        self.loss_fn = torch.nn.MSELoss(reduction="mean")

        # init metric dicts
        self._train_metrics: dict[int, Trainer.Metrics] = {}
        self._val_metrics: dict[int, Trainer.Metrics] = {}
        self._test_metrics: dict[int, Trainer.Metrics] = {}

        # get config values
        self._max_epochs = self.trainer_config.max_epochs

        self._fire("on_init")

    def _fire(self, hook_name: str, *args, **kwargs):
        """
        Invoke a hook on every registered callback.

        Args:
            hook_name (str): The name of the callback method to call.
            *args: Positional arguments to forward into the callback.
            **kwargs: Keyword arguments to forward into the callback.
        """
        for cb in self.callbacks:
            fn = getattr(cb, hook_name)
            fn(self, *args, **kwargs)

    @staticmethod
    def _set_seed(seed: int) -> None:
        """
        Configure seeds and deterministic flags for full reproducibility.

        Args:
            seed (int): Seed for torch, NumPy, and Python random.
        """
        # seed everything
        seed_everything(seed)

        # ensure deterministic setup
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _forward_and_target(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass and align targets to batch-aggregated outputs.

        Args:
            batch (Batch): A PyG batch.

        Returns:
            out (Tensor): Model predictions per graph in the batch.
            target (Tensor): True values aggregated by graph, shaped to match `out`.
        """
        out = self.model(batch.x, batch.batch)  # type: ignore[arg-type]
        # if y is a 1‑D tensor of node‑level scalars, collect it per graph
        if batch.y.dim() == 1 and batch.y.size(0) == batch.batch.size(0):  # type: ignore[arg-type]
            target = torch_scatter.scatter_mean(batch.y, batch.batch, dim=0)  # type: ignore[arg-type]
        else:
            # graph‑level truth already: just use it directly
            target = batch.y  # type: ignore[arg-type]
        return out, target

    def _train_batchwise(self, dataloader: DataLoader) -> Metrics:
        """
        Run one epoch of training over `dataloader` and collect SSE and sample counts.

        Args:
            dataloader (DataLoader): Yields batches for training.

        Returns:
            Metrics: Contains total samples and sum of squared errors for the epoch.
        """
        metrics = self.Metrics(0, 0)

        # make sure correct mode is active
        self.model.train()

        # iterate over each batch
        for batch in Console.progress_bar(dataloader):
            self._fire("on_batch_begin", batch)

            batch = batch.to(self.device)
            out, target = self._forward_and_target(batch)
            batch_size = out.size(0)

            self.optimizer.zero_grad()
            loss = self.loss_fn(out, target)

            loss.backward()
            self.optimizer.step()

            metrics.sse_sum += loss.item() * batch_size
            metrics.samples += batch_size

            self._fire("on_batch_end", batch, loss.item(), metrics)

        return metrics

    def _evaluate_batchwise(self, dataloader: DataLoader) -> Metrics:
        """
        Run one pass of evaluation (no gradient) over `dataloader`.

        Args:
            dataloader (DataLoader): Yields batches for validation or testing.

        Returns:
            Metrics: Contains total samples and sum of squared errors for the run.
        """
        metrics = self.Metrics(0, 0)

        # make sure correct mode is active
        self.model.eval()

        # use no_grad on eval loops
        with torch.no_grad():
            # iterate over each batch
            for batch in Console.progress_bar(dataloader):
                self._fire("on_batch_begin", batch)

                batch = batch.to(self.device)
                out, target = self._forward_and_target(batch)
                batch_size = out.size(0)

                loss = self.loss_fn(out, target)

                metrics.sse_sum += loss.item() * batch_size
                metrics.samples += batch_size

                self._fire("on_batch_end", batch, loss.item(), metrics)

        return metrics

    @property
    def train_metrics(self) -> dict[int, Metrics]:
        """Getter for training metrics."""
        return self._train_metrics

    @property
    def val_metrics(self) -> dict[int, Metrics]:
        """Getter for validation metrics."""
        return self._val_metrics

    @property
    def test_metrics(self) -> dict[int, Metrics]:
        """Getter for testing metrics."""
        return self._test_metrics

    def train(self) -> None:
        """
        Train the model for the configured number of epochs.

        Loops over epochs, logs MSE/RMSE, writes TensorBoard scalars if enabled,
        and saves both latest and best checkpoints after each epoch.
        """
        self._fire("on_train_begin")

        # iterate over epochs
        for epoch in range(self._max_epochs):  # type: ignore[arg-type]
            self._fire("on_epoch_begin", epoch)

            metrics = self._train_batchwise(self.datasets.train_dataloader)

            if metrics.samples == 0:
                raise EmptyDataLoaderError("No data in dataloader; cannot train/validate.")

            self._train_metrics[epoch + 1] = metrics

            self._fire("on_epoch_end", epoch, metrics)

            # save the model after every epoch and run test
            self.save(epoch=epoch, metrics=metrics)

            # only run on specified intervals
            test_interval = self.trainer_config.test_interval
            if test_interval > 0 and (epoch + 1) % test_interval == 0:
                self.test(epoch=epoch)

        self._fire("on_train_end")

    def validate(self, epoch: int) -> None:
        """
        Compute validation metrics without altering model weights.

        Args:
            epoch (int): The epoch index for logging/scalar steps.
        """
        self._fire("on_validation_begin", epoch)
        metrics = self._evaluate_batchwise(self.datasets.val_dataloader)

        if metrics.samples == 0:
            raise EmptyDataLoaderError("No data in dataloader; cannot train/validate.")

        self._val_metrics[epoch + 1] = metrics

        self._fire("on_validation_end", epoch, metrics)

    def test(self, epoch: int) -> None:
        """
        Compute test metrics using the final model (no weight updates).

        Args:
            epoch (int): The epoch index for logging/scalar steps.
        """
        self._fire("on_test_begin", epoch)
        metrics = self._evaluate_batchwise(self.datasets.test_dataloader)

        if metrics.samples == 0:
            raise EmptyDataLoaderError("No data in dataloader; cannot train/validate.")

        self._test_metrics[epoch + 1] = metrics

        self._fire("on_test_end", epoch, metrics)

    def save(self, epoch: Optional[int] = None, metrics: Optional[Metrics] = None) -> None:
        """
        Saves the model if there is an associated callback.
        """
        self._fire("on_save", epoch, metrics)

    def run(self) -> None:
        """
        Execute the full pipeline: training, testing at set intervals, final validation, and teardown.
        """
        Console.banner("Trainer")

        self.train()
        self.validate(self._max_epochs - 1)

        self._fire("on_teardown")

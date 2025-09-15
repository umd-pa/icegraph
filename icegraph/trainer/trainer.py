# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Union, Optional, List, Self, Type, Dict
from pathlib import Path

import torch
from torch.optim import Adam
from torch_geometric.seed import seed_everything
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from icegraph.console import Console
from icegraph.data import DatasetRegistry
from icegraph.config import IGConfig
from ._config import TrainerConfig
from .arch import ModelFactory
from icegraph.utils.pathutils import PathResolver
from .callbacks.base import Callback, Normalizer
from .callbacks import ConsoleCallback, ExportCallback, TensorBoardCallback, resolve_normalizer
from .base.exceptions import EmptyDataLoaderError, TrainerError
from .protocols.base import TaskStrategy
from .protocols import resolve_strategy

__all__ = ["Trainer"]


class Trainer(torch.nn.Module):
    """
    Trainer class for managing the training, validation, and testing of a PyTorch model
    using datasets registered in a `DatasetRegistry`.

    This class handles model optimization, loss computation, and RMSE reporting
    for each stage of the training lifecycle.
    """

    def __init__(
        self,
        dataset_registry: DatasetRegistry, *,
        callbacks: Optional[list[Callback]] = None,
        trainer_config: Optional[TrainerConfig] = None,
        outdir: Optional[Union[str, Path]] = None,
        model: str = "gravnet",
        device: str = "cuda",
        normalizer: Optional[Normalizer] = None
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
            outdir (Optional[Union[str, Path]]): Path to save the trained model and any other generated files.
            model (str): The model to be trained and evaluated, must be one of ['gravnet'].
            device (str): Preferred device for computation. Defaults to 'cuda'.
            normalizer (Optional[normalizer]): Normalizer to use, overrides any normalizer specified in config.
            strategy (Optional[TaskStrategy]): Strategy to use, overrides any strategy specified in config.
        """
        super().__init__()

        # grab global config and generate local trainer config
        Console.banner("Trainer")

        self._config = IGConfig.get()
        self.trainer_config = trainer_config or TrainerConfig.from_config(self._config)

        # resolve the output path
        resolver = PathResolver(path=outdir, origin=None, extension=None, stage="trainer")
        self.outdir = resolver.resolve(return_dir=True)

        # place log dir next to run files
        self.log_dir = self.outdir / "logs"

        # set global seed for reproducibility
        self._set_seed(self.trainer_config.seed)

        # load datasets and device
        self.datasets = dataset_registry

        # load strategy
        strategy_selection = self._config.user_config.training.strategy.task
        self.strategy_kwargs = self._config.user_config.training.strategy.kwargs.toDict()

        strategy_spec = resolve_strategy(strategy_selection, call=False)
        self.strategy = strategy_spec(**self.strategy_kwargs)

        # get the device selection
        self.device: torch.device = (
            torch.device("cuda")
            if torch.cuda.is_available() and device == "cuda"
            else torch.device("cpu")
        )

        # grab callbacks
        default_callbacks = [
            ConsoleCallback(),
            ExportCallback(),
            TensorBoardCallback()
        ]
        self.callbacks = callbacks or default_callbacks

        # grab normalizer
        # make sure the user didnt pass any normalizers in callbacks
        norm_selection = self._config.user_config.training.normalizer
        self.normalizer = normalizer or resolve_normalizer(norm_selection)
        self._ensure_single_normalizer()

        # determine dimensions of input and output
        in_channels = dataset_registry.train_dataset.num_node_features
        out_channels = dataset_registry.train_dataset.num_output_features

        # get the model
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

        # loss function
        self.loss_fn = self.strategy.loss_function()
        self.strategy.post_init_check(self.model)  # run post init check if any

        # init metric dicts
        self._train_metrics: Dict[int, Dict[str, float]] = {}
        self._val_metrics: Dict[int, Dict[str, float]] = {}
        self._test_metrics: Dict[int, Dict[str, float]] = {}

        # setup stash dict
        self._last_eval = {
            "val": {"preds": None, "targets": None, "includes": None},
            "test": {"preds": None, "targets": None, "includes": None},
        }

        self._fire("on_init")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._fire("on_teardown")

    def register_callback(self, callback: Type[Callback]) -> None:
        """Register a callback."""
        if not issubclass(callback, Callback):
            raise TypeError("callback must be a subclass of 'Callback'")
        self.callbacks.append(callback())

        # make sure user didnt pass a normalizer
        self._ensure_single_normalizer()

        # need to initialize this new callback as it wasn't caught in trainer's __init__
        self.callbacks[-1].on_init(self)

    @property
    def val_predictions(self) -> Optional[torch.Tensor]:
        return self._last_eval["val"]["preds"]

    @property
    def val_targets(self) -> Optional[torch.Tensor]:
        return self._last_eval["val"]["targets"]

    @property
    def val_includes(self) -> Optional[torch.Tensor]:
        return self._last_eval["val"]["includes"]

    @property
    def test_predictions(self) -> Optional[torch.Tensor]:
        return self._last_eval["test"]["preds"]

    @property
    def test_targets(self) -> Optional[torch.Tensor]:
        return self._last_eval["test"]["targets"]

    @property
    def test_includes(self) -> Optional[torch.Tensor]:
        return self._last_eval["test"]["includes"]

    def _ensure_single_normalizer(self) -> None:
        _normalizers: List[Normalizer] = [cb for cb in self.callbacks + [self.normalizer] if isinstance(cb, Normalizer)]

        if len(_normalizers) == 0:
            raise TrainerError(
                "No normalizer found. Exactly one normalizer is required, which must be an instance of 'Normalizer'."
            )
        if len(_normalizers) > 1:
            names = ", ".join(type(cb).__name__ for cb in _normalizers)
            raise TrainerError(f"Multiple normalizers found ({names}). Exactly one normalizer is allowed.")

    def _fire(self, hook_name: str, *args, **kwargs):
        """
        Invoke a hook on every registered callback.

        Args:
            hook_name (str): The name of the callback method to call.
            *args: Positional arguments to forward into the callback.
            **kwargs: Keyword arguments to forward into the callback.
        """
        for cb in self.callbacks + [self.normalizer]:
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
        target = self.strategy.adapt_targets(batch, out)

        return out, target

    def _train_batchwise(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Run one epoch of training over `dataloader` and collect SSE and sample counts.

        Args:
            dataloader (DataLoader): Yields batches for training.

        Returns:
            Dict[str, float]: Contains metrics for the epoch.
        """
        metrics = self.strategy.make_metrics()

        # make sure correct mode is active
        self.model.train()

        # iterate over each batch
        for batch in Console.progress_bar(dataloader):
            self._fire("on_batch_begin", batch)

            print(batch.y)
            batch = batch.to(self.device)
            self._fire("on_batch_transfer", batch)

            out, target = self._forward_and_target(batch)

            self.optimizer.zero_grad()
            loss = self.loss_fn(out, target)

            loss.backward()
            self.optimizer.step()

            # task-agnostic accumulation
            metrics.update(out.detach(), target.detach(), loss.detach(), mask=None)

            self._fire("on_batch_end", batch, out.detach(), target.detach(), loss.item(), metrics)

        return metrics.compute()

    def _evaluate_batchwise(self, dataloader: DataLoader, *, stash: Optional[str] = None) -> Dict[str, float]:
        """
        Run one pass of evaluation (no gradient) over `dataloader`.

        Args:
            dataloader (DataLoader): Yields batches for validation or testing.
            stash (Optional[str]): Whether to stash results.

        Returns:
            Metrics: Contains metrics for the run.
        """
        metrics = self.strategy.make_metrics()

        # make sure correct mode is active
        self.model.eval()

        # stashing
        collect = stash in {"val", "test"}
        outs: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        includes: list[torch.Tensor] = []

        # use no_grad on eval loops
        with torch.no_grad():
            # iterate over each batch
            for batch in Console.progress_bar(dataloader):
                self._fire("on_batch_begin", batch)

                batch = batch.to(self.device)
                self._fire("on_batch_transfer", batch)

                out, target = self._forward_and_target(batch)
                out, target, mask = self.strategy.filter_eval(out, target)

                loss = self.loss_fn(out, target)

                metrics.update(out.detach(), target.detach(), loss.detach(), mask=mask)

                self._fire("on_batch_end", batch, out.detach(), target.detach(), loss.item(), metrics)

                if collect:
                    outs.append(out.detach().cpu().clone())
                    targets.append(target.detach().cpu().clone())

                    # grab includes if present
                    if hasattr(batch, "include_labels"):
                        includes.append(batch.include_labels.detach().cpu().clone())

            if collect:
                self._last_eval[stash]["preds"] = torch.cat(outs, dim=0) if outs else None
                self._last_eval[stash]["targets"] = torch.cat(targets, dim=0) if targets else None
                self._last_eval[stash]["includes"] = torch.cat(includes, dim=0) if includes else None

        return metrics.compute()

    @property
    def train_metrics(self) -> dict[int, Dict[str, float]]:
        """Getter for training metrics."""
        return self._train_metrics

    @property
    def val_metrics(self) -> dict[int, Dict[str, float]]:
        """Getter for validation metrics."""
        return self._val_metrics

    @property
    def test_metrics(self) -> dict[int, Dict[str, float]]:
        """Getter for testing metrics."""
        return self._test_metrics

    def _train(self, **kwargs) -> None:
        """
        Train the model for the configured number of epochs.

        Loops over epochs, logs MSE/RMSE.
        """
        self._fire("on_train_begin")

        # iterate over epochs
        for epoch in range(self.trainer_config.max_epochs):  # type: ignore[arg-type]
            self._fire("on_epoch_begin", epoch)

            metrics_dict = self._train_batchwise(self.datasets.train_dataloader)

            if not metrics_dict or all(v != v for v in metrics_dict.values()):  # all NaN
                raise EmptyDataLoaderError("No data in dataloader; cannot train/validate.")

            self._train_metrics[epoch + 1] = metrics_dict

            self._fire("on_epoch_end", epoch, metrics_dict)

            # save the model after every epoch and run test
            self.save(epoch=epoch, metrics=metrics_dict)

            # only run on specified intervals
            val_interval = self.trainer_config.val_interval
            if val_interval > 0 and (epoch + 1) % val_interval == 0:
                self._validate(epoch=epoch)

        self._fire("on_train_end")

    def _validate(self, epoch: int) -> None:
        """
        Compute validation metrics without altering model weights.

        Args:
            epoch (int): The epoch index for logging/scalar steps.
        """
        self._fire("on_validation_begin", epoch)
        metrics_dict = self._evaluate_batchwise(self.datasets.val_dataloader, stash="val")

        if not metrics_dict or all(v != v for v in metrics_dict.values()):
            raise EmptyDataLoaderError("No data in dataloader; cannot train/validate.")

        self._val_metrics[epoch + 1] = metrics_dict

        self._fire("on_validation_end", epoch, metrics_dict)

    def _test(self, epoch: int) -> None:
        """
        Compute test metrics using the final model (no weight updates).

        Args:
            epoch (int): The epoch index for logging/scalar steps.
        """
        self._fire("on_test_begin", epoch)
        metrics_dict = self._evaluate_batchwise(self.datasets.test_dataloader, stash="test")

        if not metrics_dict or all(v != v for v in metrics_dict.values()):
            raise EmptyDataLoaderError("No data in dataloader; cannot train/validate.")

        self._test_metrics[epoch + 1] = metrics_dict

        self._fire("on_test_end", epoch, metrics_dict)

    def save(self, epoch: Optional[int] = None, metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Saves the model if there is an associated callback.
        """
        self._fire("on_save", epoch, metrics)

    def execute(self) -> None:
        """
        Execute the full pipeline: training, validation at set intervals, final testing, and teardown.
        """
        self._train()
        self._test(self.trainer_config.max_epochs - 1)

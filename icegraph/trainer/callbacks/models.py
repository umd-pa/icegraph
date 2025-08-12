# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Optional, TYPE_CHECKING

import torch
from torch_geometric.data import Batch

from .base import Callback
from icegraph.console import Console
from icegraph.trainer.tensorboard import TensorBoard
from icegraph.utils import Statistics
from icegraph.data.readers import LMDBConfiguredShardReader, LMDBReader

if TYPE_CHECKING:
    from .. import Trainer
else:
    Trainer = None

__all__ = ["TensorBoardCallback", "CheckpointCallback", "ConsoleCallback", "MinMaxNormCallback"]


class MinMaxNormCallback(Callback):
    def __init__(self):
        self._x_off:   torch.Tensor | None = None  # features: min
        self._x_scale: torch.Tensor | None = None  # features: 1/(max-min)

        self._y_off:   torch.Tensor | None = None  # labels (selected, after any log transform)
        self._y_scale: torch.Tensor | None = None
        self._y_logmask: torch.Tensor | None = None  # bool mask over selected labels

        self._on_device: bool = False

    def on_init(self, trainer: Trainer):
        # Build global stats once
        map_df = LMDBReader(trainer.datasets.map_file).to_pandas()
        LMDBConfiguredShardReader.configure(trainer.datasets.source, max_open_envs=4, map_df=map_df)
        with LMDBConfiguredShardReader() as reader:
            f_stats, t_stats = reader.stats  # tuple[Statistics, Statistics]

        self._set_minmax_from_stats(
            f_stats=f_stats,
            t_stats=t_stats,
            target_labels=trainer.target_labels,
            log_labels=set(trainer.apply_log_scaling or []),  # names to log-scale
        )

    def _set_minmax_from_stats(
        self,
        *,
        f_stats: Statistics,
        t_stats: Statistics,
        target_labels: list[str],
        log_labels: set[str],
        eps: float = 1e-8,
    ) -> None:
        # Features
        f_min = torch.tensor(f_stats.min, dtype=torch.float32)
        f_max = torch.tensor(f_stats.max, dtype=torch.float32)
        self._x_off   = f_min
        self._x_scale = (f_max - f_min).clamp_min(eps).reciprocal()

        # Truth: select labels in target order; build log mask for those
        if target_labels:
            idx = [t_stats.columns.index(name) for name in target_labels]
            t_min = torch.tensor(t_stats.min, dtype=torch.float32)[idx]
            t_max = torch.tensor(t_stats.max, dtype=torch.float32)[idx]

            logmask = torch.tensor([name in log_labels for name in target_labels], dtype=torch.bool)

            # Validate: disable log for labels with negative mins
            if torch.any(logmask & (t_min < 0)):
                bad = [name for name, m in zip(target_labels, (logmask & (t_min < 0)).tolist()) if m]
                Console.out(f"Warning: negative values in log-scaled labels {bad}; "
                            f"disabling log for those.", severity=2)
                # turn off log for those problematic labels
                for i, name in enumerate(target_labels):
                    if name in bad:
                        logmask[i] = False

            # Compute offsets/scales:
            # - for log labels, use log1p(min/max) to define the range in log-space
            y_min = t_min.clone()
            y_max = t_max.clone()

            if torch.any(logmask):
                y_min_log = torch.log1p(t_min[logmask])
                y_max_log = torch.log1p(t_max[logmask])
                y_min[logmask] = y_min_log
                y_max[logmask] = y_max_log

            self._y_off   = y_min
            self._y_scale = (y_max - y_min).clamp_min(eps).reciprocal()
            self._y_logmask = logmask
        else:
            self._y_off = self._y_scale = self._y_logmask = None

        self._on_device = False  # move lazily

    def on_batch_transfer(self, trainer: Trainer, batch: Batch):
        self._normalize_inplace(trainer, batch)

    def _ensure_on_device(self, device: torch.device):
        if self._on_device:
            return
        if self._x_off is not None:
            self._x_off   = self._x_off.to(device, non_blocking=True)
            self._x_scale = self._x_scale.to(device, non_blocking=True)
        if self._y_off is not None:
            self._y_off      = self._y_off.to(device, non_blocking=True)
            self._y_scale    = self._y_scale.to(device, non_blocking=True)
        if self._y_logmask is not None:
            self._y_logmask  = self._y_logmask.to(device)
        self._on_device = True

    def _normalize_inplace(self, trainer: Trainer, batch: Batch):
        self._ensure_on_device(trainer.device)

        x = getattr(batch, "x", None)
        if x is not None and self._x_off is not None:
            if x.dtype is not torch.float32:
                batch.x = x = x.float()
            x.add_(-self._x_off).mul_(self._x_scale)

        y = getattr(batch, "y", None)
        if y is not None and self._y_off is not None:
            if not torch.is_floating_point(y):
                y = y.float()

            # Ensure shape [B, L] for broadcast; keep single-label case efficient
            if y.ndim == 1:
                if self._y_off.numel() == 1:
                    # Optional log1p for that single label
                    if self._y_logmask is not None and self._y_logmask.numel() == 1 and bool(self._y_logmask[0]):
                        y = torch.log1p(y)
                    y = y.add(-self._y_off).mul(self._y_scale)
                else:
                    y = y.unsqueeze(1)
                    # fall through to 2D path
            if y.ndim == 2:
                if self._y_logmask is not None and torch.any(self._y_logmask):
                    # Apply log1p only to selected label columns
                    cols = torch.nonzero(self._y_logmask, as_tuple=False).squeeze(1)
                    y[:, cols] = torch.log1p(y[:, cols])
                y = y.add(-self._y_off).mul(self._y_scale)
                if y.shape[1] == 1:
                    y = y.squeeze(1)

            batch.y = y


class TensorBoardCallback(Callback):

    def __init__(self):
        self._tb: Optional[TensorBoard] = None

    def on_init(self, trainer):
        if self._tb is None:
            self._tb = TensorBoard(trainer.log_dir)
        self._tb.launch()

    def on_epoch_end(self, trainer, epoch, metrics):
        self._tb.writer.add_scalar("Train/MSE", metrics.avg_loss, epoch + 1)
        self._tb.writer.add_scalar("Train/RMSE", metrics.rmse, epoch + 1)

    def on_validation_end(self, trainer, epoch, metrics):
        self._tb.writer.add_scalar("Validation/MSE", metrics.avg_loss, epoch + 1)
        self._tb.writer.add_scalar("Validation/RMSE", metrics.rmse, epoch + 1)

    def on_test_end(self, trainer, epoch, metrics):
        self._tb.writer.add_scalar("Test/MSE", metrics.avg_loss, epoch + 1)
        self._tb.writer.add_scalar("Test/RMSE", metrics.rmse, epoch + 1)

    def on_teardown(self, cls):
        self._tb.writer.close()
        self._tb.shutdown()


class ConsoleCallback(Callback):

    def on_train_begin(self, trainer):
        Console.out(f"Model save path: {trainer.outfile}")

        # warn if falling back to CPU
        if trainer.device.type == "cpu":
            Console.out("No accelerators found, falling back to CPU training.", severity=2)

    def on_epoch_begin(self, trainer, epoch):
        Console.out(f"[Train] Epoch {epoch + 1}/{trainer.trainer_config.max_epochs}")

    def on_validation_begin(self, trainer, epoch):
        Console.out(f"[Validation] Epoch {epoch + 1}/{trainer.trainer_config.max_epochs}")

    def on_test_begin(self, trainer, epoch):
        Console.out(f"[Test] Epoch {epoch + 1}/{trainer.trainer_config.max_epochs}")

    def display_loss(self, trainer, epoch, metrics):
        Console.out(f" --> MSE: {metrics.avg_loss:.4f} | RMSE: {metrics.rmse:.4f}")

    on_validation_end = on_test_end = on_epoch_end = display_loss


class CheckpointCallback(Callback):

    def __init__(self):
        self._best_rmse: float = float("inf")

    def on_save(self, trainer, epoch, metrics):
        # get paths for latest and best
        stem, suffix = trainer.outfile.stem, trainer.outfile.suffix
        latest_path = trainer.outfile.with_name(f"{stem}_latest{suffix}")
        best_path = trainer.outfile.with_name(f"{stem}_best{suffix}")

        # save latest model
        label = f"[Epoch {epoch + 1}]" if epoch is not None else ""
        Console.out(f"{label} Saving latest model to {latest_path}...")
        payload = {
            "epoch": epoch,
            "model_state": trainer.model.state_dict(),
            "optim_state": trainer.optimizer.state_dict(),
        }
        try:
            torch.save(payload, latest_path)
        except Exception as e:
            Console.out(f"Failed to save latest model: {e}", severity=3)

        # save best if metrics are favorable
        if metrics is not None:
            current_rmse = metrics.rmse
            if current_rmse < self._best_rmse:
                Console.out(
                    f"New best RMSE {current_rmse:.4f} < {self._best_rmse:.4f}; "
                    f"saving best model to {best_path}...",
                    severity=1
                )
                self._best_rmse = current_rmse
                try:
                    torch.save(payload, best_path)
                except Exception as e:
                    Console.out(f"Failed to save best model: {e}", severity=3)

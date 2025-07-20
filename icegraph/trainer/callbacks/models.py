# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Optional

import torch

from .base import Callback
from icegraph.console import Console
from icegraph.trainer.tensorboard import TensorBoard

__all__ = ["TensorBoardCallback", "CheckpointCallback", "ConsoleCallback"]


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

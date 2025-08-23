# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from datetime import datetime
from typing import Optional, TYPE_CHECKING, List

import torch

from .base import Callback
from icegraph.console import Console
from icegraph.data.base import IGData
from icegraph.inference import CoreModel
from icegraph.trainer.tensorboard import TensorBoard
from icegraph.renderer import ParityPlot
from icegraph._version import __version__

__all__ = ["TensorBoardCallback", "ExportCallback", "ConsoleCallback", "RegressionMetricsCallback"]

if TYPE_CHECKING:
    from .. import Trainer
else:
    class Trainer:
        class Metrics:
            pass


class TensorBoardCallback(Callback):

    def __init__(self) -> None:
        self._tb: Optional[TensorBoard] = None

    def on_init(self, trainer) -> None:
        if self._tb is None:
            self._tb = TensorBoard(trainer.log_dir)
        self._tb.launch()

    def on_epoch_end(self, trainer, epoch, metrics) -> None:
        self._tb.writer.add_scalar("Train/MSE", metrics.avg_loss, epoch + 1)
        self._tb.writer.add_scalar("Train/RMSE", metrics.rmse, epoch + 1)

    def on_validation_end(self, trainer, epoch, metrics) -> None:
        self._tb.writer.add_scalar("Validation/MSE", metrics.avg_loss, epoch + 1)
        self._tb.writer.add_scalar("Validation/RMSE", metrics.rmse, epoch + 1)

    def on_test_end(self, trainer, epoch, metrics) -> None:
        self._tb.writer.add_scalar("Test/MSE", metrics.avg_loss, epoch + 1)
        self._tb.writer.add_scalar("Test/RMSE", metrics.rmse, epoch + 1)

    def on_teardown(self, trainer) -> None:
        self._tb.writer.close()
        self._tb.shutdown()


class ConsoleCallback(Callback):

    def on_train_begin(self, trainer) -> None:
        Console.out(f"Trainer output directory: {trainer.outdir}")

        # warn if falling back to CPU
        if trainer.device.type == "cpu":
            Console.out("No accelerators found, falling back to CPU training.", severity=2)

    def on_epoch_begin(self, trainer, epoch) -> None:
        Console.out(f"[Train] Epoch {epoch + 1}/{trainer.trainer_config.max_epochs}")

    def on_validation_begin(self, trainer, epoch) -> None:
        Console.out(f"[Validation] Epoch {epoch + 1}/{trainer.trainer_config.max_epochs}")

    def on_test_begin(self, trainer, epoch) -> None:
        Console.out(f"[Test] Epoch {epoch + 1}/{trainer.trainer_config.max_epochs}")

    def display_loss(self, trainer, epoch, metrics) -> None:
        Console.out(f" --> MSE: {metrics.avg_loss:.4f} | RMSE: {metrics.rmse:.4f}")

    on_validation_end = on_test_end = on_epoch_end = display_loss


class ExportCallback(Callback):
    def __init__(self) -> None:
        self._best_rmse: float = float("inf")

    def on_save(self, trainer, epoch, metrics) -> None:
        latest_path = trainer.outdir / "model_latest.pt"
        best_path = trainer.outdir / "model_best.pt"

        # Build CoreModel for export
        export_model = CoreModel(
            net=trainer.model,
            normalizer=trainer.normalizer,
            metadata={
                **IGData.attrs,
                "model": {
                    "version": __version__,
                    "timestamp": datetime.now().timestamp()
                }
            }
        )

        label = f"[Epoch {epoch + 1}]" if epoch is not None else ""
        Console.out(f"{label} Saving latest model to {latest_path}...")

        try:
            torch.save(export_model, latest_path)
        except Exception as e:
            Console.out(f"Failed to save model: {e}", severity=3)

        # Save best model if improved
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
                    torch.save(export_model, best_path)
                except Exception as e:
                    Console.out(f"Failed to save model: {e}", severity=3)


class RegressionMetricsCallback(Callback):

    def __init__(self) -> None:
        self._y_asinh_mask: Optional[List[str]] = None
        self._target_labels: Optional[List[str]] = None

    def on_init(self, trainer: Trainer) -> None:
        self._y_asinh_mask = IGData.attrs[0]["global"]["apply_log_scaling_y"]
        self._target_labels = IGData.attrs[0]["global"]["target_labels"]

    def on_test_end(self, trainer: Trainer, epoch: int, metrics: Trainer.Metrics) -> None:
        test_pred = trainer.test_predictions
        test_targ = trainer.test_targets

        n_cols = test_pred.shape[1]

        for i in range(n_cols):
            label = self._target_labels[i]

            pred = test_pred[:, i]
            targ = test_targ[:, i]

            axis_title = label

            if self._target_labels[i] in self._y_asinh_mask:
                pred = torch.log10(pred)
                targ = torch.log10(targ)

                axis_title = r"log_{10}(\text{%s})$" % axis_title

            plot = ParityPlot()
            plot.plot(
                x=targ,
                y=pred,
                title=f"{label} Parity [Epoch {epoch + 1}]",
                save_path=f"/data/i3store/users/tstjean/{label}.parity.{epoch + 1}.html",
                yaxis_title=r"$\text{Predicted }" + axis_title,
                xaxis_title=r"$\text{True }" + axis_title,
            )

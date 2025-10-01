# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Optional

from .callback import Callback
from icegraph.trainer.loggers.tensorboard import TensorBoard

__all__ = ["TensorBoardCallback"]


class TensorBoardCallback(Callback):

    def __init__(self) -> None:
        self._tb: Optional[TensorBoard] = None

    def on_init(self, trainer) -> None:
        if self._tb is None:
            self._tb = TensorBoard(trainer.log_dir)
        pid, port = self._tb.launch()

        trainer.console.log(f"TensorBoard started with PID {pid} at http://localhost:{port}")

    def on_train_end(self, trainer, epoch, metrics) -> None:
        for metric, value in metrics.items():
            self._tb.writer.add_scalar(f"Train/{metric.upper().split(':')[-1]}", value, epoch + 1)

    def on_validation_end(self, trainer, epoch, metrics) -> None:
        for metric, value in metrics.items():
            self._tb.writer.add_scalar(f"Validation/{metric.upper().split(':')[-1]}", value, epoch + 1)

    def on_test_end(self, trainer, epoch, metrics) -> None:
        for metric, value in metrics.items():
            self._tb.writer.add_scalar(f"Test/{metric.upper().split(':')[-1]}", value, epoch + 1)

    def on_teardown(self, trainer) -> None:
        self._tb.writer.close()
        self._tb.shutdown()

# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TYPE_CHECKING

# local package
from icegraph.trainer.callbacks import Callback
from icegraph.types.data import Split

# local subpackage
from .service import TensorBoardService

if TYPE_CHECKING:
    from .. import context
    from icegraph.trainer.services.metrics import ComputedMetric

__all__ = ["TensorBoardCallback"]

# module logger
import logging
logger = logging.getLogger(__name__)


class TensorBoardCallback(Callback):

    def __init__(self, port: int) -> None:
        super().__init__()

        self._port: int = port

        self._service: TensorBoardService | None = None

    def on_init(self, ctx: context.InitContext) -> None:
        if self._service is None:
            self._service = TensorBoardService(ctx.trainer.logdir, port=self._port)
        self._service.launch()

    def _log(self, metrics: list[ComputedMetric], split: Split, epoch: int) -> None:
        for metric in metrics:
            self._service.writer.add_scalar(f"{split.value.upper()}/{metric.name.upper()}", metric.value, epoch + 1)

    def on_train_end(self, ctx: context.TrainEndContext) -> None:
        trainer = ctx.trainer

        metrics = trainer.metrics.compute()
        epoch = trainer.current_epoch

        self._log(metrics, trainer.split, epoch)

    def on_validation_end(self, ctx: context.ValidationEndContext) -> None:
        trainer = ctx.trainer

        metrics = trainer.metrics.compute()
        epoch = trainer.current_epoch

        self._log(metrics, trainer.split, epoch)

    def on_test_end(self, ctx: context.TestEndContext) -> None:
        trainer = ctx.trainer

        metrics = trainer.metrics.compute()
        epoch = trainer.current_epoch

        self._log(metrics, trainer.split, epoch)

    def on_teardown(self, ctx: context.TeardownContext) -> None:
        self._service.close()

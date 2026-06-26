# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path
from functools import cached_property

# local package
from icegraph.common.data import Split

from ..callback import TrainerCallback

# local subpackage
from .service import TensorBoardService

if TYPE_CHECKING:
    from .. import context
    from icegraph.engine.services.metrics import ComputedMetric

__all__ = ["TensorBoardCallback"]

# module logger
import logging
logger = logging.getLogger(__name__)


class TensorBoardCallback(TrainerCallback):

    def __init__(self, port: int) -> None:
        super().__init__()

        self._port: int = port
        self._logdir: str | Path | None = None

    def on_init(self, ctx: context.InitContext) -> None:
        self._logdir = ctx.engine.logdir

    @cached_property
    def service(self) -> TensorBoardService:
        if self._logdir is None:
            raise RuntimeError("Cannot start TensorBoardService before init.")

        # build and launch tensorboard service
        service = TensorBoardService(self._logdir, port=self._port)
        service.launch()

        return service

    def _log(self, metrics: list[ComputedMetric], split: Split, epoch: int) -> None:
        for metric in metrics:
            self.service.writer.add_scalar(f"{split.value.upper()}/{metric.repr.upper()}", metric.value, epoch + 1)

    def on_train_end(self, ctx: context.TrainEndContext) -> None:
        trainer = ctx.engine

        metrics = trainer.metrics.compute()
        epoch = trainer.current_epoch

        self._log(metrics, trainer.split, epoch)

    def on_validation_end(self, ctx: context.ValidationEndContext) -> None:
        trainer = ctx.engine

        metrics = trainer.metrics.compute()
        epoch = trainer.current_epoch

        self._log(metrics, trainer.split, epoch)

    def on_test_end(self, ctx: context.TestEndContext) -> None:
        trainer = ctx.engine

        metrics = trainer.metrics.compute()
        epoch = trainer.current_epoch

        self._log(metrics, trainer.split, epoch)

    def on_teardown(self, ctx: context.TeardownContext) -> None:
        self.service.close()

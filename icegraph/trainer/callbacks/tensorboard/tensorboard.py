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
    from icegraph.engine.services.metrics import MetricValue

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

    def _log(self, ctx: context.TrainEndContext | context.ValidationEndContext | context.TestEndContext) -> None:
        step = ctx.engine.current_epoch + 1
        split = ctx.engine.split
        prefix = split.value.upper()

        for metric in ctx.engine.metrics.compute(split):
            name = metric.repr.upper()

            for head, values in enumerate(metric.value):
                if values is None:
                    continue

                if values.numel() == 1:
                    self.service.writer.add_scalar(f"{prefix}/{name}/{head}", values.item(), step)
                else:
                    for i, v in enumerate(values):
                        self.service.writer.add_scalar(f"{prefix}/{name}/{head}/{i}", v.item(), step)

    on_train_end = on_validation_end = on_test_end = _log

    def on_teardown(self, ctx: context.TeardownContext) -> None:
        self.service.close()

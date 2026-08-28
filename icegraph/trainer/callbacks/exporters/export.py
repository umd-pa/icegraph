# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from pathlib import Path

import torch

# local package
from icegraph._version import __version__
from icegraph.common.engine import ComponentKind
from icegraph.engine.components import Component

# local subpackage
from ..callback import TrainerCallback

if TYPE_CHECKING:
    from .. import context
    from icegraph.trainer import Trainer

__all__ = ["ExportCallback"]

# module logger
import logging
logger = logging.getLogger(__name__)


class ExportCallback(TrainerCallback):
    outdir: Path

    def __init__(self, save_interval: int = 10) -> None:
        self._save_interval = save_interval

    def on_init(self, ctx: context.InitContext) -> None:
        # define model dir and make sure it exists
        self.outdir = ctx.engine.outdir / "models"
        self.outdir.mkdir(parents=True, exist_ok=True)

    def _gather_state(self, trainer: Trainer) -> dict[str, Any]:
        kinds = ComponentKind.inference()

        states: dict[str, Any] = {}
        for kind in kinds:
            component = trainer.components.require(kind, required_by=type(self))

            if kind is ComponentKind.MODEL:
                # @TODO: model is not a Component[Any] (actually BoundModel or DistributedDataParallel), this works at runtime but needs to be fixed
                component = component.module  # pyright: ignore[reportAttributeAccessIssue]

            states[kind.value] = component.state_dict()

        # carry the config of the same components, so inference can rebuild them
        config = trainer.config.model_dump(
            mode="json",
            include={"components": {kind.value for kind in kinds}},
        )

        return {
            "states": states,
            "config": config,
            "metadata": {
                "version": __version__,
                "timestamp": datetime.now().timestamp(),
                "global_attrs": dict(trainer.record.global_attrs)
            }
        }

    def on_epoch_end(self, ctx: context.EpochEndContext) -> None:
        epoch = ctx.engine.current_epoch

        # if current interval is a save interval, save a persistent copy
        if (epoch + 1) % self._save_interval != 0:
            return

        # build model for export
        export_model = self._gather_state(ctx.engine)

        persistent_path = self.outdir / f"model.epoch_{epoch + 1}.pt"

        try:
            torch.save(export_model, persistent_path)
            logger.info("exported model saved: %s", str(persistent_path))
        except Exception:
            logger.exception("failed to save persistent model", exc_info=True)
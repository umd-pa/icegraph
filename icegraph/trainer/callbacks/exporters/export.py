# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from pathlib import Path

import torch

# local package
from icegraph._version import __version__
from icegraph.engine.components import Component, ComponentContext

# local subpackage
from ..callback import Callback

if TYPE_CHECKING:
    from .. import context
    from icegraph.trainer import Trainer

__all__ = ["ExportCallback"]

# module logger
import logging
logger = logging.getLogger(__name__)


class ExportCallback(Callback):
    outdir: Path

    def __init__(self, save_interval: int = 10) -> None:
        self._save_interval = save_interval

    def on_init(self, ctx: context.InitContext) -> None:
        # define model dir and make sure it exists
        self.outdir = ctx.trainer.outdir / "models"
        self.outdir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _component_state(component: Component[Any, ComponentContext]) -> tuple[str, dict[str, Any]]:
        return (
            component.__class__.__name__,
            component.state_dict()
        )

    def _gather_state(self, trainer: Trainer) -> dict[str, Any]:
        adapter     = trainer.adapter
        transformer = trainer.transformer
        normalizer  = trainer.normalizer
        model       = trainer.model.module if hasattr(trainer.model, "module") else trainer.model

        # load required configs
        config = trainer.config.model_dump(mode="json", include={"components", "policy"})

        return {
            "adapter": self._component_state(adapter),
            "transformer": self._component_state(transformer),
            "normalizer": self._component_state(normalizer),
            "model": self._component_state(model),
            "config": config,
            "metadata": {
                "version": __version__,
                "timestamp": datetime.now().timestamp(),
                "global_attrs": dict(trainer.record.global_attrs)
            }
        }

    def on_epoch_end(self, ctx: context.EpochEndContext) -> None:
        epoch = ctx.trainer.current_epoch

        # if current interval is a save interval, save a persistent copy
        if (epoch + 1) % self._save_interval != 0:
            return

        # build model for export
        export_model = self._gather_state(ctx.trainer)

        persistent_path = self.outdir / f"model.epoch_{epoch + 1}.pt"

        try:
            torch.save(export_model, persistent_path)
            logger.info("exported model saved: %s", str(persistent_path))
        except Exception:
            logger.exception("failed to save persistent model", exc_info=True)
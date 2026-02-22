# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from pathlib import Path

import torch

# local package
from icegraph._version import __version__

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
    def __init__(self) -> None:
        super().__init__()
        # cache for best loss, start at infinity
        self._best_loss: float = float("inf")

        # model dir
        self.models_dir: Path | None = None

    def on_init(self, ctx: context.InitContext) -> None:
        # define model dir and make sure it exists
        self.models_dir = ctx.trainer.outdir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _gather_state(trainer: Trainer) -> dict:
        attrs   = trainer.data.attrs
        norm    = trainer.normalizer
        net     = trainer.model.module if hasattr(trainer.model, "module") else trainer.model

        state = dict(
            network=(net.__class__.__name__, net.state_dict()),
            normalizer=(norm.__class__.__name__, norm.state_dict()),
            metadata={
                **attrs,
                "model": {
                    "version": __version__,
                    "timestamp": datetime.now().timestamp()
                }
            }
        )
        return state

    def _export(self, ctx: context.ValidationEndContext | context.TestEndContext) -> None:
        trainer = ctx.trainer
        loss = ctx.loss

        latest_path = self.models_dir / "model_latest.pt"
        best_path = self.models_dir / "model_best.pt"

        # Build model for export
        export_model = self._gather_state(trainer)

        try:
            torch.save(export_model, latest_path)
            logger.debug("latest model saved: %s", str(latest_path))
        except Exception:
            logger.exception(f"failed to save latest model", exc_info=True)

        if loss >= self._best_loss:
            # break out if performance has deteriorated
            return

        self._best_loss = loss
        try:
            torch.save(export_model, best_path)
            logger.info("new best model (loss=%.5g) saved: %s", loss, str(best_path))
        except Exception:
            logger.exception(f"failed to save best model", exc_info=True)

    # run both on validation and test, not on train
    on_validation_end = on_test_end = _export

    def on_epoch_end(self, ctx: context.EpochEndContext) -> None:
        trainer = ctx.trainer

        epoch = trainer.current_epoch

        # if current interval is a save interval, save a persistent copy
        if (epoch + 1) % trainer.config.trainer.save_interval != 0:
            return

        # build model for export
        export_model = self._gather_state(trainer)

        persistent_path = trainer.outdir / "models" / f"model.epoch_{epoch + 1}.pt"

        try:
            torch.save(export_model, persistent_path)
            logger.debug("persistent model saved: %s", str(persistent_path))
        except Exception:
            logger.exception(f"failed to save persistent model", exc_info=True)
# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from datetime import datetime
from typing import TYPE_CHECKING, Dict
from pathlib import Path

import torch

from .callback import Callback
from icegraph.types import ComputedMetrics
from icegraph._version import __version__

__all__ = ["ExportCallback"]

if TYPE_CHECKING:
    from .. import Trainer
else:
    class Trainer:
        pass


class ExportCallback(Callback):
    def __init__(self, **_) -> None:
        super().__init__()
        self._best_loss: float = float("inf")

    @staticmethod
    def _strip_attrs(attrs: dict) -> dict:
        return attrs

    def _gather_state(self, trainer: Trainer) -> Dict:
        # TODO: only pass necessary attrs, registry.attrs can be very large and needs to be stripped
        attrs = self._strip_attrs(trainer.registry.attrs)

        norm = trainer.normalizer
        net = trainer.model.module if hasattr(trainer.model, "module") else trainer.model

        state = dict(
            network=(net.__class__.__name__, net.state_dict(), ),
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

    def _export(self, trainer: Trainer, epoch: int, metrics: ComputedMetrics) -> None:
        models_dir: Path = trainer.outdir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        latest_path = models_dir / "model_latest.pt"
        best_path = models_dir / "model_best.pt"

        # Build CoreModel for export
        export_model = self._gather_state(trainer)

        trainer.console.log(f"Saving latest model to {latest_path}")
        try:
            torch.save(export_model, latest_path)
        except Exception as e:
            trainer.console.log(f"Failed to save model: {e}")

        # Save best model if improved
        if metrics is None:
            return

        loss_key = next((k for k in metrics if k.startswith("loss")), None)
        loss = metrics[loss_key]

        if loss >= self._best_loss:
            return

        trainer.console.log(
            f"New best {loss_key.split(':')[1].upper()} {loss:.4f} < {self._best_loss:.4f}; "
            f"saving model to {best_path}"
        )
        self._best_loss = loss
        try:
            torch.save(export_model, best_path)
        except Exception as e:
            trainer.console.log(f"Failed to save model: {e}")

    # run both on validation and test, not on train
    on_validation_end = on_test_end = _export

    def on_epoch_end(self, trainer, epoch) -> None:
        # if current interval is a save interval, save a persistent copy
        if (epoch + 1) % trainer.trainer_config.save_interval != 0:
            return

        export_model = self._gather_state(trainer)
        persistent_path = trainer.outdir / "models" / f"model.epoch_{epoch + 1}.pt"
        trainer.console.log(f"Saving persistent model to {persistent_path}")

        try:
            torch.save(export_model, persistent_path)
        except Exception as e:
            trainer.console.log(f"Failed to save model: {e}", severity=3)
# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override
from collections.abc import Mapping

import torch
from torch import Tensor

# local package
from icegraph.statistics import StatisticService
from icegraph.renderer import Line2D
from icegraph.common.histogram import Histogram

# local subpackage
from ..base import BHistogramReducer
from ._ovr import project_ovr, iter_ovr, bin_curve

if TYPE_CHECKING:
    from icegraph.trainer import Trainer
    from icegraph.common.data import GraphBatch

__all__ = ["PrecisionRecallPlotter"]

# module logger
import logging
logger = logging.getLogger(__name__)


class PrecisionRecallPlotter(BHistogramReducer):

    @override
    def _build_bounds(self, stats: StatisticService, label: str) -> tuple[Tensor, Tensor]:
        mins = torch.as_tensor([0], dtype=torch.float32)
        maxs = torch.as_tensor([1], dtype=torch.float32)
        return mins, maxs

    @override
    def _build_bins(self) -> Tensor:
        return torch.tensor([5000])

    @override
    def project(self, batch: GraphBatch, out: Tensor, label: str) -> tuple[Tensor, Tensor]:
        target = batch.targets.block([label])  # this head's target

        # one-vs-rest precision-recall for all classes
        return project_ovr(out, target)

    @override
    def reduce(self, counts: Mapping[int, Tensor], label: str) -> dict[str, Tensor]:
        curves: dict[str, Tensor] = {}

        for c, neg, pos in iter_ovr(counts):
            # true/false positives at each threshold, high score -> positive prediction
            tp = pos.flip(0).cumsum(0).flip(0)
            fp = neg.flip(0).cumsum(0).flip(0)

            precision = tp / (tp + fp).clamp_min(1.0)
            recall = tp / pos.sum().clamp_min(1.0)

            # resample precision over recall
            # precision should be non-increasing
            # in recall, so a right-to-left running max gives the upper envelope
            pr = bin_curve(recall, precision, pos.numel())
            curves[self.group_name(c, label)] = torch.cummax(pr.flip(0), dim=0).values.flip(0)

        return curves

    @override
    def emit(self, trainer: Trainer, artifacts: dict[str, Histogram], label: str) -> None:
        epoch = trainer.current_epoch

        # building a 2d line plot
        plot = Line2D()

        title = (
            f"<b>Precision-Recall</b>: {label} "
            f"[Epoch {trainer.current_epoch + 1} - {trainer.split.value.upper()}]"
        )
        plot.set_title(title)

        plot.set_xlabel(r"$\mathrm{Recall}$")
        plot.set_ylabel(r"$\mathrm{Precision}$")

        plot.set_legend_location(x=0.02, y=0.02, xanchor="left", yanchor="bottom")

        path = trainer.plotdir / "precision_recall" / f"{label}.PR.{epoch + 1}.html"
        plot.plot(artifacts, path)
        logger.info("new precision-recall plot saved: %s", str(path))

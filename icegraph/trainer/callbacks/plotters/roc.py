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
from icegraph.renderer import Line2D, OneToOne
from icegraph.common.histogram import Histogram

# local subpackage
from ..base import BHistogramReducer
from ._ovr import project_ovr, iter_ovr, bin_curve

if TYPE_CHECKING:
    from icegraph.trainer import Trainer
    from icegraph.common.data import GraphBatch

__all__ = ["ROCPlotter"]

# module logger
import logging
logger = logging.getLogger(__name__)


class ROCPlotter(BHistogramReducer):

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

        # one-vs-rest ROC for all classes
        return project_ovr(out, target)

    @override
    def reduce(self, counts: Mapping[int, Tensor], label: str) -> dict[str, Tensor]:
        curves: dict[str, Tensor] = {}

        for c, neg, pos in iter_ovr(counts):
            # true/false positives at each threshold, high score -> positive prediction
            tp = pos.flip(0).cumsum(0).flip(0)
            fp = neg.flip(0).cumsum(0).flip(0)

            # convert to rates
            tpr = tp / pos.sum().clamp_min(1.0)
            fpr = fp / neg.sum().clamp_min(1.0)

            # resample tpr over fpr, then take the running max to smooth
            roc = bin_curve(fpr, tpr, pos.numel())
            curves[self.group_name(c, label)] = torch.cummax(roc, dim=0).values

        return curves

    @override
    def emit(self, trainer: Trainer, artifacts: dict[str, Histogram], label: str) -> None:
        epoch = trainer.current_epoch

        # building a 2d line plot
        plot = Line2D()

        # add overlays
        plot.add_module(OneToOne())

        title = (
            f"<b>Receiver Operating Characteristic Curve</b>: {label} "
            f"[Epoch {trainer.current_epoch + 1} - {trainer.split.value.upper()}]"
        )
        plot.set_title(title)

        plot.set_xlabel(r"$\mathrm{False\;Positive\;Rate}$")
        plot.set_ylabel(r"$\mathrm{True\;Positive\;Rate}$")

        plot.set_legend_location(x=0.98, y=0.02, xanchor="right", yanchor="bottom")

        path = trainer.plotdir / "ROC" / f"{label}.roc.{epoch + 1}.html"
        plot.plot(artifacts, path)

        logger.info("new ROC plot saved: %s", str(path))

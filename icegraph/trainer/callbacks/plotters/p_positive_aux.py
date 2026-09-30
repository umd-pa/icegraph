# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

import torch
from torch import Tensor

# local package
from icegraph.statistics import StatisticService
from icegraph.common.data import Split, DataRole
from icegraph.renderer import Histogram2D
from icegraph.common.histogram import Histogram

# local subpackage
from ..base import BHistogramReducer

if TYPE_CHECKING:
    from icegraph.trainer import Trainer
    from icegraph.common.data import GraphBatch

__all__ = ["PPositiveAuxPlotter"]

# module logger
import logging
logger = logging.getLogger(__name__)


class PPositiveAuxPlotter(BHistogramReducer):
    """One-vs-rest ``p(y = c | x)`` against the auxiliary column named by the ``column`` kwarg, one heatmap per class."""

    @property
    def column(self) -> str:
        return self._kwargs["column"]

    @override
    def _build_bounds(self, stats: StatisticService, label: str) -> tuple[Tensor, Tensor]:
        # stats handed in are for targets, take the auxiliary range from its own train stats
        decode = self._ctx.engine.decode
        aux_stats = decode.get_stats(Split.TRAIN, DataRole.AUXILIARY)

        # stats are per physical column, locate the columns start
        index = decode.get_columns(DataRole.AUXILIARY).index(self.column)
        physical = int(decode.get_offsets(DataRole.AUXILIARY)[index])

        # mins/maxs
        mins = torch.as_tensor([0, aux_stats.get("min")[physical]], dtype=torch.float32)
        maxs = torch.as_tensor([1, aux_stats.get("max")[physical]], dtype=torch.float32)

        return mins, maxs

    @override
    def _build_bins(self) -> Tensor:
        return torch.tensor([100, 100])

    @override
    def project(self, batch: GraphBatch, out: Tensor, label: str) -> tuple[Tensor, Tensor]:
        probs = out.softmax(dim=-1)  # [B, C]
        aux = batch.auxiliary.block([self.column]).to(probs).expand_as(probs)  # [B, 1] -> [B, C]

        # one row per (sample, class), grouped by class
        rows = torch.stack((probs.flatten(), aux.flatten()), dim=1)  # [B * C, 2]
        groups = torch.arange(probs.size(1), device=out.device).repeat(probs.size(0))  # [B * C]

        return rows, groups

    @override
    def emit(self, trainer: Trainer, artifacts: dict[str, Histogram], label: str) -> None:
        epoch = trainer.current_epoch

        # building a 2d histogram, one class shown at a time
        plot = Histogram2D(exclusive=True)

        # update title
        title = (
            f"<b>One-vs-Rest Probability vs. {self.column}</b>: {label} "
            f"[Epoch {trainer.current_epoch + 1} - {trainer.split.value.upper()}]"
        )
        plot.set_title(title)

        # update axis labels
        xlabel = r"$%s$" % self.scale[0].format_repr(r"p_\theta(y = c \mid x)")
        ylabel = r"$%s$" % self.scale[1].format_repr(r"\mathrm{%s}" % self.column.replace("_", r"\_"))

        plot.set_xlabel(xlabel)
        plot.set_ylabel(ylabel)

        # plot
        path = trainer.plotdir / "p_positive_aux" / f"{label}.p_positive_aux.{self.column}.{epoch + 1}.html"
        plot.plot(artifacts, path)
        logger.info("new positive-class probability vs. %s plot saved: %s", self.column, str(path))

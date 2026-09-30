# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override
from collections.abc import Mapping

import torch
from torch import Tensor

# local package
from icegraph.common.data import DataRole
from icegraph.common.transforms import TransformSpace
from icegraph.renderer import Histogram2D
from icegraph.common.histogram import Histogram

# local subpackage
from ..base import BHistogramReducer, BoundsConstructorContext

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
    def _build_bounds(self, ctx: BoundsConstructorContext, label: str) -> tuple[Tensor, Tensor]:
        # auxiliary stats, and the column within them
        aux_stats = ctx.get_stats(DataRole.AUXILIARY)
        physical = ctx.physical_index(self.column, DataRole.AUXILIARY)

        # a log axis starts at the smallest positive value, the linear min may be <= 0 and send the bound to -inf
        if self.scale[1] == TransformSpace.LOG:
            lo = 10 ** aux_stats.get("min", space=TransformSpace.LOG)[physical]
        else:
            lo = aux_stats.get("min")[physical]

        # mins/maxs
        mins = torch.as_tensor([0, lo], dtype=torch.float32)
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
    def reduce(self, counts: Mapping[int, Tensor], label: str) -> dict[str, Tensor]:
        # one series per class, log counts if required, empty bins become nan so they render as gaps
        log = self._kwargs.get("log_count", False)
        return {
            name: t.log10().masked_fill(t == 0, torch.nan) if log else t
            for name, t in super().reduce(counts, label).items()
        }

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

        # plotly rotates latex colorbar titles, so use html to keep the title horizontal on top
        if self._kwargs.get("log_count", False):
            plot.set_zlabel("log<sub>10</sub>(Count)")

        # plot
        path = trainer.plotdir / "p_positive_aux" / f"{label}.p_positive_aux.{self.column}.{epoch + 1}.html"
        plot.plot(artifacts, path)
        logger.info("new positive-class probability vs. %s plot saved: %s", self.column, str(path))

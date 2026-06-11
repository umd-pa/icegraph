# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

import torch
from torch import Tensor

# local package
from icegraph.statistics import StatisticService
from icegraph.renderer import Histogram2D, MedianQuantileBand
from icegraph.common.histogram import Histogram

# local subpackage
from ..base import BHistogramReducer, HistogramAccumulator

if TYPE_CHECKING:
    from .. import context
    from icegraph.trainer import Trainer

__all__ = ["BiasPlotter"]

# module logger
import logging
logger = logging.getLogger(__name__)


class BiasPlotter(BHistogramReducer):

    def _build_bounds(self, stats: StatisticService, label: str) -> tuple[Tensor, Tensor]:
        # get label index in stat array
        index = stats.index_of(label)

        # mins/maxs
        mins = torch.as_tensor([stats.get("min")[index], -5], dtype=torch.float32)
        maxs = torch.as_tensor([stats.get("max")[index], 5], dtype=torch.float32)

        return mins, maxs

    def _build_bins(self) -> tuple[int, ...]:
        return 150, 150

    def _reduce(self, out: Tensor, target: Tensor, ctx: context.BatchEndContext) -> Tensor:
        # compute bias
        bias = torch.where(target != 0, (out - target) / target, torch.zeros_like(target))

        # return bias plotted as function of target
        return torch.cat((target, bias), dim=1)

    def _postprocess_accumulator(self, data: dict[int, HistogramAccumulator], label: str) -> dict[int | str, HistogramAccumulator]:
        return {"Data": list(data.values())[0]}  # only one so this is fine

    def _dispatch(self, trainer: Trainer, data: dict[int | str, Histogram], label: str, save_dir: Path) -> None:
        # only one histogram, so pull from dict
        hist = list(data.values())[0]

        # building a 2d histogram
        plot = Histogram2D()

        # add overlays
        plot.add_module(MedianQuantileBand())

        # update title
        title = f"<b>Bias</b>: {label} [Epoch {trainer.current_epoch + 1} - {trainer.split.value.upper()}]"
        plot.set_title(title)

        # update axis labels
        xlabel = r"$\mathrm{Target}\;%s$" % hist.space[0].format_repr(r"\mathrm{%s}" % label)
        ylabel = r"$\mathrm{(Predicted - Target)/Target}\;%s$" % hist.space[1].format_repr(r"\mathrm{%s}" % label)

        plot.set_xlabel(xlabel)
        plot.set_ylabel(ylabel)

        # plot
        path = save_dir / "bias" / f"{label}.bias.{trainer.current_epoch + 1}.html"
        plot.plot(data, path)
        logger.info(f"new bias plot saved: %s", str(path))

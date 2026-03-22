# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

# local package
from icegraph.statistics import StatisticService
from icegraph.renderer import ParityPlot
from icegraph.common.histogram import Histogram

# local subpackage
from ..base import BinnedHistogramReducer

if TYPE_CHECKING:
    from .. import context
    from icegraph.trainer import Trainer

__all__ = ["ParityPlotter"]

# module logger
import logging
logger = logging.getLogger(__name__)


class ParityPlotter(BinnedHistogramReducer):

    def _build_bounds(self, stats: StatisticService) -> Tensor:
        # filter stats
        stats.filter_to(self._target_labels).align_to(self._target_labels)

        # mins/maxs
        mins = torch.as_tensor(stats.get("min"), dtype=torch.float32)
        maxs = torch.as_tensor(stats.get("max"), dtype=torch.float32)

        bounds = torch.stack((mins, maxs), dim=1)

        # reshape to match convention
        bounds = bounds.unsqueeze(-1).expand(-1, -1, 2)

        return bounds

    def _build_margin(self) -> Tensor:
        return torch.tensor(5, dtype=torch.long)

    def _build_bins(self) -> Tensor:
        return torch.tensor((100, 100), dtype=torch.long)

    def _reduce(self, ctx: context.BatchEndContext) -> Tensor:
        # stack data by axis
        return torch.stack((ctx.target, ctx.out), dim=-1)

    def _dispatch(self, trainer: Trainer, data: Histogram) -> None:
        epoch = trainer.current_epoch

        save_path = self._save_dir / f"{data.name}.parity.{epoch + 1}.html"

        plot = ParityPlot(data, epoch=epoch)

        # update title
        title = "%s <b>Parity</b> [Epoch %d - %s]" % (data.name, trainer.current_epoch + 1, trainer.split.value.title())
        plot.set_title(title)

        # update axis labels
        xlabel = r"$\mathrm{Target}\;%s$" % data.space[0].format_repr(r"\mathrm{%s}" % data.name)
        ylabel = r"$\mathrm{Predicted}\;%s$" % data.space[1].format_repr(r"\mathrm{%s}" % data.name)

        plot.set_xlabel(xlabel)
        plot.set_ylabel(ylabel)

        # plot
        plot.plot(save_path=save_path)

        logger.info(f"new parity plot saved: %s", str(save_path))

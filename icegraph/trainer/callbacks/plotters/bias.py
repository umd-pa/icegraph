# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar
from typing_extensions import override
from collections.abc import Mapping

import torch
from torch import Tensor

# local package
from icegraph.common.data import DataRole
from icegraph.common.transforms import Space
from icegraph.renderer import Histogram2D, MedianQuantileBand
from icegraph.common.histogram import Histogram

# local subpackage
from ..base import BHistogramReducer, BoundsConstructorContext, AxisLayout

if TYPE_CHECKING:
    from icegraph.trainer import Trainer
    from icegraph.common.data import GraphBatch

__all__ = ["BiasPlotter"]

# module logger
import logging
logger = logging.getLogger(__name__)


class BiasPlotter(BHistogramReducer):
    ndim: ClassVar[int] = 2

    @override
    def default_axis_layout(self) -> AxisLayout:
        return AxisLayout(margins=(0.1,), bins=(150,))

    @override
    def build_bounds(self, ctx: BoundsConstructorContext, label: str) -> tuple[Tensor, Tensor]:
        # target stats, and the label's column within them
        stats = ctx.get_stats(DataRole.TARGETS)
        index = ctx.physical_index(label, DataRole.TARGETS)

        # a log axis starts at the smallest positive value, the linear min may be <= 0 and send the bound to -inf
        if self.axis_layout[label].spaces[0] == Space.LOG:
            lo = 10 ** stats.get("min", space=Space.LOG)[index]
        else:
            lo = stats.get("min")[index]

        # mins/maxs
        mins = torch.as_tensor([lo, -5], dtype=torch.float32)
        maxs = torch.as_tensor([stats.get("max")[index], 5], dtype=torch.float32)

        return mins, maxs

    @override
    def project(self, batch: GraphBatch, out: Tensor, label: str) -> Tensor:
        target = batch.targets.block([label])  # this head's target

        # compute bias
        bias = torch.where(target != 0, (out - target) / target, torch.zeros_like(target))

        # return bias plotted as function of target
        return torch.cat((target, bias), dim=1)

    @override
    def reduce(self, counts: Mapping[int, Tensor], label: str) -> dict[str, Tensor]:
        return {"Data": counts[0]}  # single group

    @override
    def emit(self, trainer: Trainer, artifacts: dict[str, Histogram], label: str) -> None:
        # building a 2d histogram
        plot = Histogram2D()

        # add overlays
        plot.add_module(MedianQuantileBand())

        # update title
        title = f"<b>Bias</b>: {label} [Epoch {trainer.current_epoch + 1} - {trainer.split.value.upper()}]"
        plot.set_title(title)

        # update axis labels
        scale = self.axis_layout[label].spaces
        xlabel = r"$\mathrm{Target}\;%s$" % scale[0].format_repr(r"\mathrm{%s}" % label.replace("_", r"\_"))
        ylabel = r"$\mathrm{(Predicted - Target)/Target}\;%s$" % scale[1].format_repr(r"\mathrm{%s}" % label.replace("_", r"\_"))

        plot.set_xlabel(xlabel)
        plot.set_ylabel(ylabel)

        # set count label
        plot.set_zlabel(self.get_count_label(label))

        # plot
        path = trainer.plotdir / "bias" / f"{label}.bias.{trainer.current_epoch + 1}.html"
        plot.plot(artifacts, path, log_count=self.axis_layout[label].log_count)
        logger.info(f"new bias plot saved: %s", str(path))

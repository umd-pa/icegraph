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
from icegraph.renderer import Histogram2D, OneToOne, MedianQuantileBand
from icegraph.common.histogram import Histogram

# local subpackage
from ..base import BHistogramReducer, BoundsConstructorContext, AxisLayout

if TYPE_CHECKING:
    from icegraph.trainer import Trainer
    from icegraph.common.data import GraphBatch

__all__ = ["ParityPlotter"]

# module logger
import logging
logger = logging.getLogger(__name__)


class ParityPlotter(BHistogramReducer):
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
        lo = [
            10 ** stats.get("min", space=Space.LOG)[index]
            if space == Space.LOG else stats.get("min")[index]
            for space in self.axis_layout[label].spaces
        ]

        # mins/maxs
        mins = torch.as_tensor(lo, dtype=torch.float32)
        maxs = torch.as_tensor(stats.get("max")[index], dtype=torch.float32).repeat(2)

        return mins, maxs

    @override
    def project(self, batch: GraphBatch, out: Tensor, label: str) -> Tensor:
        target = batch.targets.block([label])  # this head's target

        # cat data by axis
        return torch.cat((target, out), dim=1)   # shape [B, 2]

    @override
    def reduce(self, counts: Mapping[int, Tensor], label: str) -> dict[str, Tensor]:
        return {"Data": counts[0]}  # single group

    @override
    def emit(self, trainer: Trainer, artifacts: dict[str, Histogram], label: str) -> None:
        epoch = trainer.current_epoch

        # building a 2d histogram
        plot = Histogram2D()

        # add overlays
        plot.add_module([OneToOne(), MedianQuantileBand()])

        # update title
        title = f"<b>Parity</b>: {label} [Epoch {trainer.current_epoch + 1} - {trainer.split.value.upper()}]"
        plot.set_title(title)

        # update axis labels
        scale = self.axis_layout[label].spaces
        xlabel = r"$\mathrm{Target}\;%s$" % scale[0].format_repr(r"\mathrm{%s}" % label.replace("_", r"\_"))
        ylabel = r"$\mathrm{Predicted}\;%s$" % scale[1].format_repr(r"\mathrm{%s}" % label.replace("_", r"\_"))

        plot.set_xlabel(xlabel)
        plot.set_ylabel(ylabel)

        # set count label
        plot.set_zlabel(self.get_count_label(label))

        # plot
        path = trainer.plotdir / "parity" / f"{label}.parity.{epoch + 1}.html"
        plot.plot(artifacts, path, log_count=self.axis_layout[label].log_count)
        logger.info(f"new parity plot saved: %s", str(path))

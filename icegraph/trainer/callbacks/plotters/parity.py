# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override
from pathlib import Path
from collections.abc import Mapping

import torch
from torch import Tensor

# local package
from icegraph.common.transforms import TransformSpace
from icegraph.statistics import StatisticService
from icegraph.renderer import Histogram2D, OneToOne, MedianQuantileBand
from icegraph.common.histogram import Histogram

# local subpackage
from ..base import BHistogramReducer, HistogramAccumulator

if TYPE_CHECKING:
    from .. import context
    from icegraph.trainer import Trainer

__all__ = ["ParityPlotter"]

# module logger
import logging
logger = logging.getLogger(__name__)


class ParityPlotter(BHistogramReducer):

    @override
    def _build_bounds(self, stats: StatisticService, label: str) -> tuple[Tensor, Tensor]:
        # get label index in stat array
        index = stats.index_of(label)

        # mins/maxs
        mins = torch.as_tensor(stats.get("min")[index], dtype=torch.float32).repeat(2)
        maxs = torch.as_tensor(stats.get("max")[index], dtype=torch.float32).repeat(2)

        return mins, maxs

    @override
    def _build_bins(self) -> Tensor:
        return torch.tensor([150, 150])

    @override
    def _reduce(self, out: Tensor, target: Tensor, ctx: context.BatchEndContext) -> Tensor:
        # cat data by axis
        return torch.cat((target, out), dim=1)   # shape [B, 2]

    @override
    def _postprocess_accumulator(self, data: Mapping[int, HistogramAccumulator], label: str) -> dict[str, HistogramAccumulator]:
        return {"Data": list(data.values())[0]}  # only one so this is fine

    @override
    def _dispatch(
            self, trainer: Trainer, data: dict[int | str, Histogram], space: tuple[TransformSpace, ...], label: str
    ) -> None:
        epoch = trainer.current_epoch

        # building a 2d histogram
        plot = Histogram2D()

        # add overlays
        plot.add_module([OneToOne(), MedianQuantileBand()])

        # update title
        title = f"<b>Parity</b>: {label} [Epoch {trainer.current_epoch + 1} - {trainer.split.value.upper()}]"
        plot.set_title(title)

        # update axis labels
        xlabel = r"$\mathrm{Target}\;%s$" % space[0].format_repr(r"\mathrm{%s}" % label)
        ylabel = r"$\mathrm{Predicted}\;%s$" % space[1].format_repr(r"\mathrm{%s}" % label)

        plot.set_xlabel(xlabel)
        plot.set_ylabel(ylabel)

        # plot
        path = trainer.plotdir / "parity" / f"{label}.parity.{epoch + 1}.html"
        plot.plot(data, path)
        logger.info(f"new parity plot saved: %s", str(path))

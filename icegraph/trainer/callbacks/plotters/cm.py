# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override
from collections.abc import Mapping

import torch
from torch import Tensor

# local package
from icegraph.renderer import Histogram2D, Labels
from icegraph.common.histogram import Histogram
from icegraph.common.transforms import TransformSpace

# local subpackage
from ..base import CHistogramReducer, HistogramAccumulator

if TYPE_CHECKING:
    from .. import context
    from icegraph.trainer import Trainer

__all__ = ["CMPlotter"]

# module logger
import logging
logger = logging.getLogger(__name__)


class CMPlotter(CHistogramReducer):

    @override
    def _build_bins(self) -> Tensor:
        return torch.tensor([2, 2])

    @override
    def _reduce(self, out: Tensor, target: Tensor, ctx: context.BatchEndContext) -> Tensor:
        # stack data by axis
        return torch.cat((target, out.argmax(dim=-1, keepdim=True)), dim=1)

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
        plot.add_module(Labels())

        # update title
        title = f"<b>Confusion Matrix</b>: {label} [Epoch {trainer.current_epoch + 1} - {trainer.split.value.upper()}]"
        plot.set_title(title)

        # update axis labels
        ylabel = r"$\mathrm{Predicted}\;\mathrm{%s}$" % label
        xlabel = r"$\mathrm{Target}\;\mathrm{%s}$" % label

        plot.set_xlabel(xlabel)
        plot.set_ylabel(ylabel)

        # plot
        path = trainer.plotdir / "confusion_matrix" / f"{label}.CM.{epoch + 1}.html"
        plot.plot(data, path)
        logger.info(f"new CM plot saved: %s", str(path))

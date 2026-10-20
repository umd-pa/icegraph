# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override
from collections.abc import Mapping

import torch
from torch import Tensor

# local package
from icegraph.common.transforms import Space
from icegraph.renderer import Histogram1D
from icegraph.common.histogram import Histogram

# local subpackage
from ..base import BHistogramReducer, BoundsConstructorContext, AxisLayout

if TYPE_CHECKING:
    from icegraph.trainer import Trainer
    from icegraph.common.data import GraphBatch

__all__ = ["BinaryPPositivePlotter"]

# module logger
import logging
logger = logging.getLogger(__name__)


class BinaryPPositivePlotter(BHistogramReducer):

    @override
    def default_axis_layout(self) -> AxisLayout:
        return AxisLayout(margins=(0.,), bins=(100,))

    @override
    def build_bounds(self, ctx: BoundsConstructorContext, label: str) -> tuple[Tensor, Tensor]:
        # mins/maxs
        mins = torch.as_tensor([0], dtype=torch.float32)
        maxs = torch.as_tensor([1], dtype=torch.float32)

        return mins, maxs

    @override
    def project(self, batch: GraphBatch, out: Tensor, label: str) -> tuple[Tensor, Tensor]:
        target = batch.targets.block([label])  # this head's target

        if out.ndim != 2 or out.size(-1) != 2:
            raise ValueError(
                f"{type(self).__name__} expects out with shape [N, 2], "
                f"but got shape {tuple(out.shape)}."
            )

        # get probability assigned to the positive class
        probs = out.softmax(dim=-1)[:, 1:]

        # group by target class
        return probs, target.squeeze(1)

    @override
    def reduce(self, counts: Mapping[int, Tensor], label: str) -> dict[str, Tensor]:
        # one series per class, log counts if required
        log = self._kwargs.get("log_count", False)
        return {name: t.log10() if log else t for name, t in super().reduce(counts, label).items()}

    @override
    def emit(self, trainer: Trainer, artifacts: dict[str, Histogram], label: str) -> None:
        epoch = trainer.current_epoch

        # building a 1d histogram
        plot = Histogram1D()

        # update title
        title = (
            f"<b>Binary Positive-Class Probability</b>: {label} "
            f"[Epoch {trainer.current_epoch + 1} - {trainer.split.value.upper()}]"
        )
        plot.set_title(title)

        # update axis labels
        plot.set_xlabel(r"$p_\theta(y = 1 \mid x)$")

        # set count label
        plot.set_ylabel(self.get_count_label(label))

        # plot
        path = trainer.plotdir / "p_positive" / f"{label}.p_positive.{epoch + 1}.html"
        plot.plot(artifacts, path, log_count=self.axis_layout[label].log_count)
        logger.info(f"new positive-class probability plot saved: %s", str(path))

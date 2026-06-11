# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

import torch
from torch import Tensor

# local package
from icegraph.statistics import StatisticService
from icegraph.common.transforms import TransformSpace
from icegraph.renderer import Histogram1D
from icegraph.common.histogram import Histogram

# local subpackage
from ..base import BHistogramReducer, HistogramAccumulator

if TYPE_CHECKING:
    from .. import context
    from icegraph.trainer import Trainer

__all__ = ["BinaryPPositivePlotter"]

# module logger
import logging
logger = logging.getLogger(__name__)


class BinaryPPositivePlotter(BHistogramReducer):

    def _build_bounds(self, stats: StatisticService, label: str) -> tuple[Tensor, Tensor]:
        # mins/maxs
        mins = torch.as_tensor([0], dtype=torch.float32)
        maxs = torch.as_tensor([1], dtype=torch.float32)

        return mins, maxs

    def _build_bins(self) -> tuple[int, ...]:
        return 100,

    def _reduce(self, out: Tensor, target: Tensor, ctx: context.BatchEndContext) -> tuple[Tensor, Tensor]:
        if out.ndim != 2 or out.size(-1) != 2:
            raise ValueError(
                f"{type(self).__name__} expects out with shape [N, 2], "
                f"but got shape {tuple(out.shape)}."
            )

        # get probability assigned to the positive class
        index = torch.ones_like(target)
        probs = out.softmax(dim=-1).gather(dim=1, index=index)

        # return with mapping defined by target
        return probs, target.squeeze(1)

    def _postprocess_accumulator(self, data: dict[int, HistogramAccumulator], label: str) -> dict[int | str, HistogramAccumulator]:
        processed: dict[str, HistogramAccumulator] = {}

        # load class name map
        class_name_map: dict[int, str] = self._kwargs.get("class_name_map", {}).get(label, {})

        # process each accumulator
        for c, acc in data.items():
            # log counts if required
            if self._kwargs.get("log_count", False):
                acc.data = torch.log10(acc.data)

            # rename
            name = class_name_map.get(c, f"Class {c}")
            processed[name] = acc

        return processed

    def _dispatch(self, trainer: Trainer, data: dict[int | str, Histogram], label: str, save_dir: Path) -> None:
        epoch = trainer.current_epoch

        # building a 2d histogram
        plot = Histogram1D()

        # update title
        title = (
            f"<b>Binary Positive-Class Probability</b>: {label} "
            f"[Epoch {trainer.current_epoch + 1} - {trainer.split.value.upper()}]"
        )
        plot.set_title(title)

        # update axis labels
        plot.set_xlabel(r"$p_\theta(y = 1 \mid x)$")
        if self._kwargs.get("log_count", False):
            plot.set_ylabel("$%s$" % TransformSpace.LOG.format_repr(r'\mathrm{Count}'))
        else:
            plot.set_ylabel("Count")

        # plot
        path = save_dir / "p_positive" / f"{label}.p_positive.{epoch + 1}.html"
        plot.plot(data, path)
        logger.info(f"new positive-class probability plot saved: %s", str(path))

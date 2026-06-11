# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

import torch
from torch import Tensor

# local package
from icegraph.statistics import StatisticService
from icegraph.renderer import Line2D, OneToOne
from icegraph.common.histogram import Histogram

# local subpackage
from ..base import BHistogramReducer, HistogramAccumulator

if TYPE_CHECKING:
    from .. import context
    from icegraph.trainer import Trainer

__all__ = ["ROCPlotter"]

# module logger
import logging
logger = logging.getLogger(__name__)


class ROCPlotter(BHistogramReducer):

    def _build_bounds(self, stats: StatisticService, label: str) -> tuple[Tensor, Tensor]:
        mins = torch.as_tensor([0], dtype=torch.float32)
        maxs = torch.as_tensor([1], dtype=torch.float32)
        return mins, maxs

    def _build_bins(self) -> tuple[int, ...]:
        return 5000,

    def _reduce(self, out: Tensor, target: Tensor, ctx: context.BatchEndContext) -> tuple[Tensor, Tensor]:
        # one-vs-rest ROC for all N classes
        probs = out.softmax(dim=-1) # [B, C]

        # build list of classes from probs shape
        classes = torch.arange(probs.shape[1], device=out.device).unsqueeze(0)  # [1, C]

        # for each sample, emit one score per class
        # key = 2*c     -> negatives for class c
        # key = 2*c + 1 -> positives for class c
        is_pos = target.eq(classes)  # from broadcasting [B, 1] x [1, C] = [B, C]
        keys = 2 * classes + is_pos.long()

        return probs.flatten().unsqueeze(1), keys.flatten()

    def _postprocess_accumulator(self, data: dict[int, HistogramAccumulator], label: str) -> dict[str, HistogramAccumulator]:
        processed: dict[str, HistogramAccumulator] = {}

        # load class count
        n_classes = (max(data.keys()) // 2) + 1

        # load class name map
        class_name_map: dict[int, str] = self._kwargs.get("class_name_map", {}).get(label, {})

        for c in range(n_classes):
            neg_key = 2 * c
            pos_key = 2 * c + 1

            if neg_key not in data or pos_key not in data:
                # skip if either is missing
                continue

            # load negative and positive accumulators for this class
            neg_acc = data[neg_key]
            pos_acc = data[pos_key]

            # flatten and convert to float (not strictly necessary, but will be done anyway later)
            neg = neg_acc.data.float().flatten()
            pos = pos_acc.data.float().flatten()

            # compute true positive and false positive for each threshold
            tp = pos.flip(0).cumsum(0).flip(0)
            fp = neg.flip(0).cumsum(0).flip(0)

            # convert tp and fp to rates
            tpr = tp / pos.sum().clamp_min(1.0)
            fpr = fp / neg.sum().clamp_min(1.0)

            # convert tpr/fpr to a histogram
            roc = torch.zeros_like(pos)
            fpr_bin = torch.clamp((fpr * roc.numel()).long(), 0, roc.numel() - 1)

            for b, y in zip(fpr_bin, tpr):
                # take the largest tpr for each bin
                roc[b] = torch.maximum(roc[b], y)

            # smooth the roc
            roc = torch.cummax(roc, dim=0).values

            # reassign to accumulator
            acc = pos_acc
            acc.data = roc

            # rename
            name = class_name_map.get(c, f"Class {c}")
            processed[name] = acc

        return processed

    def _dispatch(self, trainer: Trainer, data: dict[int | str, Histogram], label: str, save_dir: Path) -> None:
        epoch = trainer.current_epoch

        # building a 2d line plot
        plot = Line2D()

        # add overlays
        plot.add_module(OneToOne())

        title = (
            f"<b>Receiver Operating Characteristic Curve</b>: {label} "
            f"[Epoch {trainer.current_epoch + 1} - {trainer.split.value.upper()}]"
        )
        plot.set_title(title)

        plot.set_xlabel(r"$\mathrm{False\;Positive\;Rate}$")
        plot.set_ylabel(r"$\mathrm{True\;Positive\;Rate}$")

        plot.set_legend_location(x=0.98, y=0.02, xanchor="right", yanchor="bottom")

        path = save_dir / "ROC" / f"{label}.roc.{epoch + 1}.html"
        plot.plot(data, path)

        logger.info("new ROC plot saved: %s", str(path))

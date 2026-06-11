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

__all__ = ["PrecisionRecallPlotter"]

# module logger
import logging
logger = logging.getLogger(__name__)


class PrecisionRecallPlotter(BHistogramReducer):

    def _build_bounds(self, stats: StatisticService, label: str) -> tuple[Tensor, Tensor]:
        mins = torch.as_tensor([0], dtype=torch.float32)
        maxs = torch.as_tensor([1], dtype=torch.float32)
        return mins, maxs

    def _build_bins(self) -> tuple[int, ...]:
        return 5000,

    def _reduce(self, out: Tensor, target: Tensor, ctx: context.BatchEndContext) -> tuple[Tensor, Tensor]:
        # one-vs-rest precision-recall for all N classes
        probs = out.softmax(dim=-1)  # [B, C]

        # build list of classes from probs shape
        classes = torch.arange(probs.shape[1], device=out.device).unsqueeze(0)  # [1, C]

        # for each sample, emit one score per class
        # key = 2*c     -> negatives for class c
        # key = 2*c + 1 -> positives for class c
        is_pos = target.eq(classes)  # broadcasting [B, 1] x [1, C] = [B, C]
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

            # flatten and convert to float
            neg = neg_acc.data.float().flatten()
            pos = pos_acc.data.float().flatten()

            # compute true positives and false positives for each threshold
            # threshold direction is high score -> positive prediction
            tp = pos.flip(0).cumsum(0).flip(0)
            fp = neg.flip(0).cumsum(0).flip(0)

            # compute precision and recall
            precision = tp / (tp + fp).clamp_min(1.0)
            recall = tp / pos.sum().clamp_min(1.0)

            # convert precision/recall curve to histogram-like line data
            # x-axis is recall, y-axis is precision
            pr = torch.zeros_like(pos)

            recall_bin = torch.clamp(
                (recall * pr.numel()).long(),
                0,
                pr.numel() - 1,
            )

            for b, y in zip(recall_bin, precision):
                # take the largest precision for each recall bin
                pr[b] = torch.maximum(pr[b], y)

            # precision should be monotonically non-increasing with recall.
            # Filling from right to left gives the upper envelope.
            pr = torch.flip(torch.cummax(torch.flip(pr, dims=(0,)), dim=0).values, dims=(0,))

            # reassign to accumulator
            acc = pos_acc
            acc.data = pr

            # rename
            name = class_name_map.get(c, f"Class {c}")
            processed[name] = acc

        return processed

    def _dispatch(self, trainer: Trainer, data: dict[int | str, Histogram], label: str, save_dir: Path) -> None:
        epoch = trainer.current_epoch

        # building a 2d line plot
        plot = Line2D()

        title = (
            f"<b>Precision-Recall</b>: {label} "
            f"[Epoch {trainer.current_epoch + 1} - {trainer.split.value.upper()}]"
        )
        plot.set_title(title)

        plot.set_xlabel(r"$\mathrm{Recall}$")
        plot.set_ylabel(r"$\mathrm{Precision}$")

        plot.set_legend_location(x=0.02, y=0.02, xanchor="left", yanchor="bottom")

        path = save_dir / "precision_recall" / f"{label}.PR.{epoch + 1}.html"
        plot.plot(data, path)

        logger.info("new precision-recall plot saved: %s", str(path))
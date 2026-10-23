# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor
import numpy as np

# local package
from icegraph.engine.services.metrics import MetricValue
from icegraph.common.data import Split
from icegraph.renderer import Line2D, HLine
from icegraph.common.histogram import Histogram

# local subpackage
from ..callback import TrainerCallback

if TYPE_CHECKING:
    from .. import context

__all__ = ["MetricsPlotter"]

# module logger
import logging
logger = logging.getLogger(__name__)


def _pad(values: tuple[Tensor | None, ...]) -> Tensor:
    """Stack per-head values into [L, W], missing entries are nan."""
    width = max((0 if v is None else v.numel() for v in values), default=0)

    out = torch.full((len(values), width), torch.nan)

    for h, v in enumerate(values):
        if v is not None:
            out[h, :v.numel()] = v.reshape(-1)

    return out


@dataclass(slots=True)
class MetricSeries:
    values: Tensor      # [E, L, W]
    epoch: list[int]
    ema: Tensor         # [E, L, W]
    span: int
    optimum: float

    def append(self, mv: MetricValue, epoch: int) -> MetricSeries:
        if self.span != mv.span:
            raise ValueError(f"Metric series span ({self.span}) is not equal to appended metric value span ({mv.span}).")

        if self.optimum != mv.optimum:
            raise ValueError(f"Metric series optimum ({self.optimum}) is not equal to appended metric value optimum ({mv.optimum}).")

        return MetricSeries(
            values=torch.concat([self.values, _pad(mv.value).unsqueeze(0)]),
            epoch=self.epoch + [epoch + 1],
            ema=torch.concat([self.ema, _pad(mv.ema).unsqueeze(0)]),
            span=self.span,
            optimum=self.optimum
        )

    @classmethod
    def from_metric_value(cls, mv: MetricValue, epoch: int) -> MetricSeries:
        return cls(
            values=_pad(mv.value).unsqueeze(0),
            epoch=[epoch + 1],
            ema=_pad(mv.ema).unsqueeze(0),
            span=mv.span,
            optimum=mv.optimum
        )


class MetricsPlotter(TrainerCallback):

    def __init__(self) -> None:
        self._metrics: dict[tuple[Split, str], MetricSeries] = {}

    def update_metrics(
            self, ctx: context.TrainEndContext | context.ValidationEndContext | context.TestEndContext
    ) -> None:
        for mv in ctx.engine.metrics.compute(ctx.engine.split):
            # no batches were seen for this split, nothing to record
            if not mv.value:
                continue

            key = (ctx.engine.split, mv.repr)

            if self._metrics.get(key) is None:
                self._metrics[key] = MetricSeries.from_metric_value(mv, ctx.engine.current_epoch)
                continue

            self._metrics[key] = self._metrics[key].append(mv, ctx.engine.current_epoch)

    on_train_end = on_validation_end = on_test_end = update_metrics

    def on_epoch_end(self, ctx: context.EpochEndContext) -> None:
        # generate the plot
        outdir = ctx.engine.plotdir / "metrics"
        outdir.mkdir(parents=True, exist_ok=True)

        for (split, name), series in self._metrics.items():

            # one histogram per (head, entry), slots that never carried a value are skipped
            _, heads, entries = series.values.shape
            data: dict[str, Histogram] = {}

            for h in range(heads):
                for e in range(entries):
                    column = series.values[:, h, e]

                    if torch.isnan(column).all():
                        continue

                    label = f"Head {h}" if entries == 1 else f"Head {h} [{e}]"

                    column_numpy = column.numpy()

                    # one bin per recorded epoch, centered on it
                    epochs = series.epoch
                    step = (epochs[-1] - epochs[0]) / (len(epochs) - 1) if len(epochs) > 1 else 1.0

                    bounds = np.array(
                        [[epochs[0] - step / 2], [epochs[-1] + step / 2]],
                        dtype=np.float64,
                    )

                    data[label] = Histogram(column_numpy, bounds=bounds)

            # building a line plot
            plot = Line2D()

            # add overlays
            plot.add_module(
                HLine(
                    y=series.optimum,
                    line_dash="dash",
                    line_color="black",
                    annotation_text="Optimum",
                    annotation_position="bottom right"
                )
            )

            # update title
            title = f"<b>Metric</b>: {name} [{split.value.upper()}]"
            plot.set_title(title)

            # update axis labels
            plot.set_xlabel("Epoch")
            plot.set_ylabel("Metric Value")

            # plot (overwrite on each save intentionally, previous data is not lost)
            path = outdir / f"metric.{name}.{split.value}.html"
            plot.plot(data, path)
            logger.info(f"new metric plot ({name}.{split.value}) saved: %s", str(path))
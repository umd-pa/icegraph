# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from torch import Tensor

from icegraph.trainer.types import Params

from ..service import Service
from ..types import ServiceContext

from .metric import Metric
from .factory import MetricFactory
from .types import ComputedMetric
from .view import MetricView

__all__ = ["MetricService"]


class MetricService(Service):
    name = "metrics"
    deps = ["strategy"]
    view = MetricView

    def __init__(self, params: Params) -> None:
        super().__init__(params)

        self._metrics: list[Metric] = []

    def on_attach(self, ctx: ServiceContext) -> None:
        strategy = ctx.services.require("strategy", required_by=MetricService)
        mode = strategy.mode

        # load user metric selections
        selection_list = self.params.require("select")
        for selection in selection_list:
            # build metric, structure of config is enforced by pydantic
            metric = MetricFactory.create(selection["name"], **selection["kwargs"])

            # verify metric compatibility
            if mode not in metric.compatible:
                raise RuntimeError(f"Metric '{type(metric).__name__}' is not compatible with strategy '{mode}'.")

            self._metrics.append(metric)

    def update(self, out: Tensor, target: Tensor) -> None:
        """Update each metric."""
        for metric in self._metrics:
            metric.update(out, target)

    def compute(self) -> list[ComputedMetric]:
        """Return dict of computed metrics."""
        computed_metrics = [
            metric.compute() for metric in self._metrics
        ]
        return computed_metrics

    def update_summaries(self) -> None:
        for metric in self._metrics:
            metric.update_summaries()

    def reset(self) -> None:
        for metric in self._metrics:
            metric.reset()

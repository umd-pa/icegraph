# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any, ClassVar

from torch import Tensor

from ..service import Service

from .metric import Metric, MetricFactory, MetricContext
from .types import ComputedMetric
from .view import MetricView
from .config import MetricConfig

__all__ = ["MetricService"]


class MetricService(Service[MetricView, MetricConfig]):
    name: ClassVar[str] = "metrics"
    version: ClassVar[int] = 1

    deps = ("strategy",)
    interface = MetricView

    # make the type checker happy
    _metrics: list[Metric]

    def build(self) -> None:
        self._metrics = []

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> MetricConfig:
        return MetricConfig(**config)

    def on_attach(self) -> None:
        strategy = self._ctx.services.require("strategy", required_by=MetricService)

        # load user metric selections
        selection_list = self.config.select
        for selection in selection_list:
            # build metric, structure of config is enforced by pydantic
            metric = MetricFactory.create(selection.name, **selection.kwargs)

            # attach the metric
            ctx = MetricContext()
            metric.attach(ctx)

            # verify metric compatibility
            strategy.ensure_compatible(metric)

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

    def state_dict(self) -> dict[str, Any]:
        return {"config": self.config.model_dump(mode="json")}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.config = type(self).validate_config(state["config"])

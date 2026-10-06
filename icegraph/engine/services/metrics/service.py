# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any, ClassVar

from icegraph.common.data import Split
from icegraph.common.tensors import SegmentedTensor

from ..service import Service

from .metric import Metric, MetricFactory, MetricContext
from .types import MetricValue
from .config import MetricConfig

__all__ = ["MetricService"]


class MetricService(Service[MetricConfig]):
    name: ClassVar[str] = "metrics"
    version: ClassVar[int] = 1

    # built by setup()
    _metrics: list[Metric]

    def build(self) -> None:
        return

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> MetricConfig:
        return MetricConfig(**config)

    def __getstate__(self) -> dict[str, Any]:
        # spawn workers pickle the whole service manager, metric state lives on
        # the accelerator after the first epoch and must not pull CUDA into workers
        state = self.__dict__.copy()
        state.pop("_metrics", None)
        return state

    def setup(self) -> None:
        # load user metric selections
        self._metrics = []
        for selection in self.config.select:
            # build metric, structure of config is enforced by pydantic
            metric = MetricFactory.create(selection.name, **selection.kwargs)

            # attach the metric
            ctx = MetricContext()
            metric.attach(ctx)

            self._metrics.append(metric)

    def update(self, out: SegmentedTensor, target: SegmentedTensor, split: Split) -> None:
        """Update each metric."""
        for metric in self._metrics:
            metric.update(out, target, split)

    def compute(self, split: Split) -> list[MetricValue]:
        """Return list of computed metrics."""
        return [metric.compute(split) for metric in self._metrics]

    def update_summaries(self, split: Split) -> None:
        for metric in self._metrics:
            metric.update_summaries(split)

    def reset(self, split: Split) -> None:
        for metric in self._metrics:
            metric.reset(split)

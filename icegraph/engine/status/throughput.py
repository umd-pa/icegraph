# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, TypeVar, Any

import torch
from torch import Tensor
from torch.utils.flop_counter import FlopCounterMode

from .meter import Meter
from .types import ThroughputStats

if TYPE_CHECKING:
    from icegraph.common.data import GraphBatch

    from .status import Status

__all__ = ["Throughput"]


T = TypeVar("T")

# marks an exhausted loader
_END = object()


def _matmul_flops(a: Tensor, b: Tensor) -> int:
    """FLOPs of ``a @ b``, counting a sparse operand by its stored values. The default
    formulas size a product from shapes alone, so a sparse ``[N, N]``
    adjacency would be counted as a dense ``N x N`` product, inflating the count by
    orders of magnitude.
    """
    if a.layout != torch.strided:
        return 2 * a._nnz() * b.shape[-1]

    if b.layout != torch.strided:
        return 2 * a.shape[0] * b._nnz()

    return 2 * a.shape[0] * a.shape[1] * b.shape[-1]


def _mm_flops(a: Tensor, b: Tensor, *args: Any, out_val: Any = None, **kwargs: Any) -> int:
    return _matmul_flops(a, b)


def _addmm_flops(bias: Tensor, a: Tensor, b: Tensor, *args: Any, out_val: Any = None, **kwargs: Any) -> int:
    return _matmul_flops(a, b)


# formulas marked raw receive the tensors rather than their shapes so they can see the layout
_mm_flops._get_raw = True  # pyright: ignore[reportFunctionMemberAccess]
_addmm_flops._get_raw = True  # pyright: ignore[reportFunctionMemberAccess]

_FLOP_FORMULAS = {
    torch.ops.aten.mm:      _mm_flops,
    torch.ops.aten.addmm:   _addmm_flops,
}


class Throughput:
    """
    Rates for one stream of batches, e.g. a split: samples and batches per second,
    FLOP/s, and the fraction of time spent waiting on the loader.

    The engine iterates its loader through ``iterate`` and wraps the processing of
    each batch in ``step``, everything else is measured here.
    """

    def __init__(self, status: Status, name: str) -> None:
        window = status.config.window

        self.name:      str     = name
        self.samples:   Meter   = Meter(window)
        self.batches:   Meter   = Meter(window)
        self.flops:     Meter   = Meter(window)

        # seconds blocked waiting on the loader, its rate is the fraction of time spent waiting
        self.waiting:   Meter   = Meter(window)

        # counted once on the stream's first batch, then scaled by node count
        self.flops_per_node: float | None = None

        self._status = status

    def iterate(self, loader: Iterable[T]) -> Iterator[T]:
        """
        Iterate a loader, reporting the wait for its first batch as a task, which covers
        worker startup and the first buffer fill, and metering time blocked on data.
        """
        # the pause since the stream last ran does not count
        for meter in (self.samples, self.batches, self.flops, self.waiting):
            meter.restart()

        start = time.perf_counter()
        with self._status.task(f"Waiting for first {self.name} batch", level=logging.DEBUG):
            iterator = iter(loader)
            item = next(iterator, _END)

        while item is not _END:  # pyright: ignore[reportPossiblyUnboundVariable]
            self.waiting.add(time.perf_counter() - start)
            yield item  # pyright: ignore[reportPossiblyUnboundVariable, reportReturnType]

            start = time.perf_counter()
            item = next(iterator, _END)  # pyright: ignore[reportPossiblyUnboundVariable]

    @contextmanager
    def step(self, batch: GraphBatch) -> Iterator[None]:
        """Account for processing one batch, counting its FLOPs if it is the stream's first."""
        if self.flops_per_node is None and self._status.config.profile_flops:
            counter = FlopCounterMode(display=False, custom_mapping=_FLOP_FORMULAS)
            with counter:
                yield

            # matmul-like ops only, scatter aggregation is not counted
            self.flops_per_node = counter.get_total_flops() / max(batch.num_nodes, 1)
        else:
            yield

        # shapes only, so nothing here waits on the device
        self.samples.add(batch.num_graphs)
        self.batches.add(1)
        if self.flops_per_node:
            self.flops.add(self.flops_per_node * batch.num_nodes)

    def stats(self) -> ThroughputStats:
        return ThroughputStats(
            samples=self.samples.rate,
            batches=self.batches.rate,
            flops=self.flops.rate,
            waiting=min(self.waiting.rate, 1.)
        )

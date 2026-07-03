# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TypeVar, Generic
from abc import abstractmethod
from pathlib import Path
from threading import Timer
import time

from icegraph.common.plugins import Plugin

from .types import StageContext
from .envelope import Envelope

__all__ = ["Stage"]

# module logger
import logging
logger = logging.getLogger(__name__)


C = TypeVar("C")
I = TypeVar("I", bound="Envelope |  Path")


class Stage(Plugin[C, StageContext[I]], Generic[C, I]):
    """
    Abstract base class for a single stage in the data pipeline.

    A Stage consumes input items, processes them, and emits results to an
    output queue managed by a parent Pipeline.
    """

    # warning timer if an item is taking too long
    warning_timeout: float = 30.0

    @staticmethod
    def _warn(stage: str, i: int, t: float) -> None:
        logger.warning("Stage %s item %d running > %.1fs", stage, i, t)

    def execute(self) -> None:
        """Consume inputs, process them, and emit results downstream."""
        logger.debug("stage %s starting", self.name)


        try:
            for i, item in enumerate(self._ctx.src):
                # if we are on last stage, need to devivify the env before processing
                if self._ctx.index == self._ctx.total - 1:
                    if isinstance(item, Path):
                        raise TypeError("Cannot devivify a Path.")
                    item.devivify()

                # start the timer
                timer = Timer(self.warning_timeout, self._warn, args=(type(self).name, i, self.warning_timeout))
                timer.daemon = True
                timer.start()

                try:
                    # get start time
                    start = time.perf_counter()

                    # run processing
                    out = self._process(item)

                    # log time elapsed
                    elapsed = (time.perf_counter() - start) * 1000
                    logger.debug(
                        "stage=%s, item=%d: completed in %.3fms",
                        self.name,
                        i,
                        elapsed,
                    )

                    if out is not None:
                        # record metrics
                        out.metrics[f"{self._ctx.index}_{self.name}"] = elapsed

                finally:
                    timer.cancel()

                # continue if dropped
                if out is None:
                    logger.warning("envelope dropped at stage %s", self.name)
                    continue

                if self._ctx.dst is None:
                    logger.warning("stage %s has no destination queue", self.name)
                else:
                    self._ctx.dst.put(out)  # blocks when full, backpressure

        finally:
            logger.debug("stage %s finished, sentinel emitted and queue closed", self.name)

    @abstractmethod
    def _process(self, item: I) -> Envelope | None:
        """
        Apply processing to input item, always emitting Envelope (or None to drop).

        Args:
            item (I): Input item.
        """
        ...

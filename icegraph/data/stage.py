# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TypeVar
from abc import abstractmethod
from pathlib import Path
from threading import Timer
import time

from icegraph.types.plugins import Plugin
from icegraph.types.data import AttributeDomain

from .types import Envelope, StageContext

__all__ = ["Stage"]

# module logger
import logging
logger = logging.getLogger(__name__)


C = TypeVar("C")


class Stage(Plugin[C, StageContext]):
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

        # cache stage info
        cls = type(self)
        stage_info = {"name": cls.name, "version": cls.version, "plugin": f"{cls.__module__}.{cls.__qualname__}"}

        try:
            for i, item in enumerate(self._ctx.src):
                # if we are on last stage, need to devivify the env before processing
                if self._ctx.index == self._ctx.total - 1:
                    item.devivify()

                # start the timer
                timer = Timer(self.warning_timeout, self._warn, args=(cls.name, i, self.warning_timeout))
                timer.daemon = True
                timer.start()

                try:
                    # get start time
                    start = time.perf_counter()

                    # run processing
                    out = self._process(item)

                    # log time elapsed
                    logger.debug(
                        "stage=%s, item=%d: completed in %.3fms",
                        self.name,
                        i,
                        (time.perf_counter() - start) * 1000,
                    )

                finally:
                    timer.cancel()

                # continue if dropped
                if out is None:
                    continue

                # log plugin version info
                out.attrs[AttributeDomain.GLOBAL.name].setdefault("stage_manifest", []).append(stage_info.copy())

                self._ctx.dst.put(out)  # blocks when full, backpressure
        finally:
            # Always signal end-of-stream downstream
            self._ctx.dst.close()
            logger.debug("stage %s finished, sentinel emitted and queue closed", self.name)

    @abstractmethod
    def _process(self, item: Path | Envelope) -> Envelope | None:
        """
        Apply processing to input item, always emitting Envelope (or None to drop).

        Args:
            item (Path | Envelope): Input item.
        """
        ...

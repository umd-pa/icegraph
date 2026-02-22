# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TypeVar
from abc import abstractmethod
from pathlib import Path

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

    def execute(self) -> None:
        """Consume inputs, process them, and emit results downstream."""
        logger.debug("stage %s starting", type(self).__name__)

        # cache stage info
        cls = type(self)
        stage_info = {"name": cls.name, "version": cls.version, "plugin": f"{cls.__module__}.{cls.__qualname__}"}

        try:
            for item in self._ctx.src:
                # if we are on last stage, need to devivify the env before processing
                if self._ctx.index == self._ctx.total - 1:
                    item.devivify()

                out = self._process(item)

                # continue if dropped
                if out is None:
                    continue

                # log plugin version info
                out.attrs[AttributeDomain.GLOBAL.name].setdefault("stage_manifest", []).append(stage_info.copy())

                self._ctx.dst.put(out)  # blocks when full, backpressure
        finally:
            # Always signal end-of-stream downstream
            self._ctx.dst.close()
            logger.debug("stage %s finished; sentinel emitted and queue closed", self.__class__.__name__)

    @abstractmethod
    def _process(self, item: Path | Envelope) -> Envelope | None:
        """
        Apply processing to input item, always emitting Envelope (or None to drop).

        Args:
            item (Path | Envelope): Input item.
        """
        ...

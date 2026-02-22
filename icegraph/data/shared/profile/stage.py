# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Callable
import functools
import time
from pathlib import Path

from icegraph.data.stage import Stage
from icegraph.data.types import Envelope

__all__ = ["profile_stage"]


def profile_stage():
    """
    Measures runtime (ms) of a stage process, stores time in env.metrics keyed by stage.
    """

    def profiler(func: Callable[[Stage, Path | Envelope], Envelope | None]):
        @functools.wraps(func)
        def inner(self, item: Path | Envelope) -> Envelope | None:
            # run and profile func
            start = time.perf_counter()
            env = func(self, item)
            end = time.perf_counter()

            if env is None:
                # if env is none, this item was dropped
                return None

            # calc elapsed time, save to metrics
            env.metrics[type(self).__name__] = (end - start) * 1000  # elapsed time in ms

            return env

        return inner

    return profiler
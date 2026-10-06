# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import time
import threading
from collections import deque

__all__ = ["Meter"]


class Meter:
    """Running total of a quantity, plus its rate over the most recent updates."""

    def __init__(self, window: int) -> None:
        self.total: float = 0

        self._window: deque[tuple[float, float]] = deque(maxlen=window)
        self._lock = threading.Lock()

    def add(self, n: float) -> None:
        now = time.perf_counter()
        with self._lock:
            self.total += n
            self._window.append((now, n))

    @property
    def rate(self) -> float:
        """
        Per second over the window, 0 until two updates are in it.

        The first update in the window only marks when it opened, since the
        work it counts was done before then.
        """
        with self._lock:
            if len(self._window) < 2:
                return 0.

            span = self._window[-1][0] - self._window[0][0]
            amount = sum(n for _, n in self._window) - self._window[0][1]

        return amount / span if span > 0 else 0.

    def restart(self) -> None:
        """Forget the window so the rate covers only what follows, e.g. after a pause. The total is kept."""
        with self._lock:
            self._window.clear()

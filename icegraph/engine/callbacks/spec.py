# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Any

from .callback import Callback

__all__ = ["CallbackSpec"]


@dataclass(frozen=True)
class CallbackSpec:
    callback:   type[Callback]
    kwargs:     dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (inspect.isclass(self.callback) and issubclass(self.callback, Callback)):
            raise TypeError(
                f"'callback' must be a subclass of "
                f"'icegraph.engine.callbacks.Callback', got {self.callback!r}."
            )

    def __call__(self) -> Callback:
        return self.callback(**self.kwargs)

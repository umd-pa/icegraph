# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from dataclasses import dataclass, field
import inspect
from typing import Any

from .callback import Callback

__all__ = ["CallbackSpec"]


@dataclass
class CallbackSpec:
    callback:   type[Callback]
    kwargs:     dict[str, Any] = field(default_factory=dict)

    def __call__(self) -> Callback:
        if inspect.isclass(self.callback) and not issubclass(self.callback, Callback):
            raise TypeError("Expected callback to be a class inheriting from 'Callback'.")

        return self.callback(**self.kwargs)

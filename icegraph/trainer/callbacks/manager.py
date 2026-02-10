# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TYPE_CHECKING

from .types import CallbackSpec
from .callback import Callback
from . import context

if TYPE_CHECKING:
    from icegraph.trainer import Trainer

__all__ = ["CallbackManager"]

# module logger
import logging
logger = logging.getLogger(__name__)


class CallbackManager:

    def __init__(self, trainer: Trainer) -> None:
        super().__init__()

        # cache trainer
        self.trainer = trainer

        self.callbacks: list[Callback] = []

    def register(self, spec: CallbackSpec) -> None:
        """Register a callback spec to the manager."""
        if not isinstance(spec, CallbackSpec):
            raise TypeError("Parameter 'spec' must be an instance of CallbackSpec.")

        # initialize the callback
        callback = spec()

        # initialize each instance and register
        callback.on_init(context.InitContext(self.trainer))
        self.callbacks.append(callback)

        logger.debug("registered callback %s", type(callback).__name__)

    def fire(self, hook_name: str, ctx: context.Context) -> None:
        """
        Invoke a hook on every registered callback.

        Args:
            hook_name (str): The name of the callback method to call.
            ctx: Context to forward into each callback.
        """
        for cb in self.callbacks:
            getattr(cb, hook_name)(ctx)
